"""Curriculum gates reject censored evidence; simulator boundaries stay authoritative."""

import importlib
from copy import deepcopy

import pytest
import torch

from colonist_rl.opponents import HeuristicPolicy, RandomPolicy
from colonist_rl.policy import GraphPolicy
from colonist_rl.training import PPOConfig, RolloutCollector, ppo_update, save_checkpoint


def curriculum():
    assert importlib.util.find_spec("colonist_rl.curriculum") is not None, "curriculum missing"
    return importlib.import_module("colonist_rl.curriculum")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def evidence(baseline="heuristic", vp=5, win=True, boards=(62000000, 62000001)):
    return {
        "status": "complete",
        "baseline": baseline,
        "candidate": "graph",
        "deterministic": True,
        "aggregation_scope": "complete_blocks",
        "requested_board_seeds": list(boards),
        "completed_board_seeds": list(boards),
        "partial_outcomes": [],
        "interrupted_game": None,
        "outcomes": [
            {
                "board_seed": b,
                "seat": s,
                "won": win if s == 0 else False,
                "victory_points": vp,
                "terminated": True,
                "truncated": False,
                "winner": 0,
            }
            for b in boards
            for s in range(4)
        ],
    }


def test_gate_uses_resolved_rows_and_exact_lineup_not_optimistic_summary():
    m = curriculum()
    initial, fixed, stage = evidence(vp=5), evidence(vp=4), evidence("random", vp=4)
    stage.update(win_rate=1.0, mean_victory_points=10.0)
    assert m.promotion_gate("random", stage, fixed, initial)["promote"]
    for mutation in (
        "missing",
        "truncated",
        "partial",
        "wrong_lineup",
        "regression",
        "mismatch",
        "low_win",
        "low_vp",
        "duplicate_requested",
        "initial_censored",
    ):
        a, b, c = deepcopy(stage), deepcopy(fixed), deepcopy(initial)
        if mutation == "missing":
            a["outcomes"].pop()
        elif mutation == "truncated":
            a["outcomes"][0].update(truncated=True, won=None)
        elif mutation == "partial":
            a["status"] = "incomplete"
        elif mutation == "wrong_lineup":
            a["baseline"] = "mixed"
        elif mutation == "regression":
            for row in b["outcomes"]:
                row["victory_points"] = 3.9
        elif mutation == "mismatch":
            c = evidence(boards=(1, 2))
        elif mutation == "low_win":
            for row in a["outcomes"]:
                row["won"] = False
        elif mutation == "low_vp":
            for row in a["outcomes"]:
                row["victory_points"] = 3.9
        elif mutation == "duplicate_requested":
            a["requested_board_seeds"].append(a["requested_board_seeds"][0])
        else:
            c["outcomes"][0]["won"] = None
        assert not m.promotion_gate("random", a, b, c)["promote"], mutation
    assert not m.promotion_gate("random", evidence("random", boards=(1,)), fixed, initial)[
        "promote"
    ]
    assert not m.promotion_gate("random", None, fixed, initial)["promote"]


def test_selection_rejects_censored_regressed_and_final_seeds_and_prefers_earliest():
    m = curriculum()
    initial = evidence(vp=5)

    def item(name, order, vp=5):
        return {
            "id": name,
            "order": order,
            "fixed_result": evidence(vp=vp),
            "initial_fixed_result": initial,
        }

    earlier, later = item("earlier", 0), item("later", 1)
    censored, regressed = item("censored", 2, 9), item("regressed", 3, 3)
    censored["fixed_result"]["outcomes"][0]["won"] = None
    reserved = item("reserved", 4, 10)
    reserved["fixed_result"] = evidence(vp=10, boards=(71001000, 71001001))
    assert m.select_checkpoint([later, censored, regressed, reserved, earlier])["id"] == "earlier"
    assert m.select_checkpoint([])["id"] is None


def test_explicit_lineup_and_frozen_pool_change_only_at_new_episode():
    policy, frozen = GraphPolicy(16), GraphPolicy(16)
    collector = RolloutCollector(
        policy,
        historical_policies=[frozen],
        opponent_lineup=("heuristic", "random", "random"),
        board_seeds=[101, 102],
        max_game_steps=1,
    )
    assert sum(isinstance(p, HeuristicPolicy) for p in collector.driver.policies.values()) == 1
    collector.opponent_lineup = ("heuristic", "heuristic", "historical")
    assert sum(type(p) is RandomPolicy for p in collector.driver.policies.values()) == 2
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    records = collector.collect(2)
    assert len(records) == 1  # seat 1 exhausts its one-step board before reaching learner
    assert collector.actual_board_seeds == [101, 102]
    assert collector.stop_reason == "board_schedule_exhausted"
    assert all(torch.equal(before[k], v) for k, v in frozen.state_dict().items())
    assert all(not p.requires_grad for p in collector.historical_policies[0].parameters())
    assert not collector.historical_policies[0].training
    assert collector.collect(1) == []


def test_explicit_schedule_validated_and_legacy_mixed_stays_two_heuristics():
    policy = GraphPolicy(16)
    for boards in ([], [1, 1], [True], [1.5]):
        with pytest.raises(ValueError):
            RolloutCollector(policy, board_seeds=boards)
    legacy = RolloutCollector(policy, seed=4)
    assert sum(isinstance(p, HeuristicPolicy) for p in legacy.driver.policies.values()) == 2
    assert isinstance(legacy.collect(1), list)


def test_expired_constructor_and_collect_do_not_start_or_mutate_simulator(monkeypatch):
    from colonist_rl import training

    clock = [10.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    collector = RolloutCollector(GraphPolicy(16), board_seeds=[101], deadline=9)
    assert collector.actual_board_seeds == [] and collector.env is None
    assert collector.collect(2, deadline=9) == []
    active = RolloutCollector(GraphPolicy(16), board_seeds=[101])
    version, memory = active.env.version, active.memory
    assert active.collect(2, deadline=9) == []
    assert active.env.version == version and active.memory is memory


def test_expiry_in_learner_forward_prevents_engine_action(monkeypatch):
    from colonist_rl import training

    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    policy = GraphPolicy(16)
    collector = RolloutCollector(policy, board_seeds=[101])
    version = collector.env.version

    def expire(module, args, output):
        clock[0] = 10.0

    handle = policy.register_forward_hook(expire)
    assert collector.collect(1, deadline=5) == []
    handle.remove()
    assert collector.env.version == version
    assert collector.dropped_partial_intervals == 0


def test_expiry_after_learner_action_drops_interval_and_stops(monkeypatch):
    from colonist_rl import training

    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    collector = RolloutCollector(GraphPolicy(16), board_seeds=[101])
    original = collector.driver.step

    def expire_after_step(action):
        result = original(action)
        clock[0] = 10.0
        return result

    monkeypatch.setattr(collector.driver, "step", expire_after_step)
    version = collector.env.version
    assert collector.collect(1, deadline=5) == []
    assert collector.env.version == version + 1
    assert collector.dropped_partial_intervals == 1
    assert collector.stop_reason == "deadline"
    clock[0] = 0
    assert collector.collect(1, deadline=5) == []


def test_expiry_during_opponent_actor_or_proposal_prevents_engine_mutation(monkeypatch):
    from colonist_rl import training
    from colonist_rl.simulator import FourPlayerGame
    from colonist_rl.training import _MatchDriver

    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    env = FourPlayerGame()
    env.reset(seed=5)
    driver = _MatchDriver(env, 3, 5)
    original = driver.policies[env.current_player].act

    def slow_actor(view):
        result = original(view)
        clock[0] = 10
        return result

    monkeypatch.setattr(driver.policies[env.current_player], "act", slow_actor)
    version = env.version
    action = driver.opponent_action(deadline=5)
    assert action is None and env.version == version
    # Real game progressed by heuristics to a proposal opportunity.
    clock[0] = 0
    teacher = HeuristicPolicy(8)
    for _ in range(1000):
        view = env.observe()
        if view.player_id != 3 and view.context.kind == "PLAY_TURN" and view.context.has_rolled:
            break
        env.step(teacher.act(view))

    def slow_proposal(view):
        clock[0] = 10
        return object()  # must never be passed to engine after expiration

    monkeypatch.setattr(driver.policies[env.current_player], "propose", slow_proposal)
    version = env.version
    driver.maybe_offer(deadline=5)
    assert env.version == version


def test_ppo_deadline_before_optimizer_step_preserves_parameters_and_numeric_metrics(monkeypatch):
    from colonist_rl import training

    policy = GraphPolicy(16)
    records = RolloutCollector(policy, seed=4).collect(2)
    optimizer = torch.optim.Adam(policy.parameters())
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])

    def expire(module, args, output):
        clock[0] = 10

    handle = policy.register_forward_hook(expire)
    metrics = ppo_update(policy, optimizer, records, PPOConfig(epochs=1), deadline=5)
    handle.remove()
    assert metrics["completed_updates"] == 0 and metrics["deadline_reached"] == 1
    assert all(isinstance(v, (int, float)) for v in metrics.values())
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
    assert optimizer.state == {}


def test_tiny_runner_reserves_final_seeds_persists_real_evaluation_and_deadline(
    tmp_path, monkeypatch
):
    m = curriculum()
    initial = {}
    for seed in (11, 29):
        initial[seed] = tmp_path / f"initial-{seed}.pt"
        save_checkpoint(initial[seed], GraphPolicy(16))
    monkeypatch.setattr(
        m,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )
    config = m.CurriculumConfig(
        compute_seconds=0.02,
        wall_seconds=1,
        max_decisions_per_seed=2,
        rollout_size=2,
        evaluation_interval=1,
        max_game_steps=1,
    )
    original_inputs = {seed: path.read_bytes() for seed, path in initial.items()}
    external_rng = torch.get_rng_state().clone()
    result = m.run_curriculum(tmp_path / "run", initial_checkpoints=initial, config=config)
    assert torch.equal(external_rng, torch.get_rng_state())
    assert {seed: path.read_bytes() for seed, path in initial.items()} == original_inputs
    assert all(row["initial_checkpoint_metadata"] == {} for row in result["seeds"].values())
    assert result["status"] in ("complete", "partial")
    assert result["manifest"]["seed_splits"]["final_test_reserved"] == list(
        range(71001000, 71001008)
    )
    assert result["manifest"]["compute_seconds_per_seed"] == 0.01
    assert set(result["seeds"]) == {"11", "29"}
    assert all(row["stage"] == "random" for row in result["seeds"].values())
    assert all(row["decisions"] <= 2 for row in result["seeds"].values())
    assert (tmp_path / "run" / "results.json").exists()
    assert not any(
        71001000 <= b <= 71001007
        for row in result["seeds"].values()
        for b in row["actual_board_seeds"]
    )


def test_real_frozen_actor_memory_and_parameters_stay_separate_from_learner():
    import hashlib

    policy, frozen = GraphPolicy(16), GraphPolicy(16)
    collector = RolloutCollector(
        policy,
        seed=11,
        board_seeds=[111, 112],
        historical_policies=[frozen],
        opponent_lineup=("historical", "heuristic", "random"),
        max_game_steps=100,
    )
    snapshot = collector.historical_policies[0]

    def digest(model):
        return hashlib.sha256(
            b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())
        ).hexdigest()

    before = digest(snapshot)
    records = collector.collect(12)
    optimizer = torch.optim.Adam(policy.parameters())
    assert records
    assert collector.driver.memories
    assert all(not memory.requires_grad for memory in collector.driver.memories.values())
    assert all(memory is not collector.memory for memory in collector.driver.memories.values())
    ppo_update(policy, optimizer, records, PPOConfig(epochs=1, sequence_length=4))
    assert digest(snapshot) == before
    assert all(parameter.grad is None for parameter in snapshot.parameters())


def test_ppo_counts_finished_steps_when_next_update_hits_deadline(monkeypatch):
    from colonist_rl import training

    policy = GraphPolicy(16)
    records = RolloutCollector(policy, seed=3).collect(2)
    optimizer = torch.optim.Adam(policy.parameters())
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    original = optimizer.step

    def finish_step_then_expire(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 10.0
        return value

    monkeypatch.setattr(optimizer, "step", finish_step_then_expire)
    metrics = ppo_update(
        policy, optimizer, records, PPOConfig(epochs=2, sequence_length=1), deadline=5
    )
    assert metrics["completed_updates"] == 1
    assert metrics["requested_updates"] == 4
    assert metrics["deadline_reached"] == 1
    assert optimizer.state


@pytest.mark.parametrize("actor_seconds, allocations", [(0.001, 3), (1.0, 1)])
def test_runner_failure_keeps_later_allocations_and_stops_after_compute_overrun(
    tmp_path, monkeypatch, actor_seconds, allocations
):
    m = curriculum()
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, GraphPolicy(16))
    monkeypatch.setattr(
        m,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )
    clock = [0.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    evaluations = []

    def resolved_evaluation(**kwargs):
        evaluations.append(tuple(kwargs["opponent_lineup"]))
        value = evidence(kwargs["baseline"], vp=3, boards=kwargs["board_seeds"])
        value["opponent_lineup"] = list(kwargs["opponent_lineup"])
        for row in value["outcomes"]:
            kwargs["on_game_end"](row)
        return value

    monkeypatch.setattr(m, "evaluate_blocks", resolved_evaluation)
    original = GraphPolicy.forward

    def slow_forward(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        clock[0] += actor_seconds  # work progresses time, not deadline queries
        return value

    monkeypatch.setattr(GraphPolicy, "forward", slow_forward)
    result = m.run_curriculum(
        tmp_path / "run",
        initial_checkpoints={11: initial},
        config=m.CurriculumConfig(compute_seconds=0.3, wall_seconds=100, max_decisions_per_seed=6),
    )
    row = result["seeds"]["11"]
    assert len(row["allocations"]) == allocations
    assert row["stage"] == "random"
    assert all(a["stage"] == "random" for a in row["allocations"])
    assert all(lineup in (("random",) * 3, ("heuristic",) * 3) for lineup in evaluations)
    if actor_seconds == 1.0:
        assert row["compute_seconds"] == 1.0 and row["decisions"] == 0
    else:
        assert row["decisions"] == 6
    assert row["selection"]["id"] == "11/initial"


def test_strict_evaluator_exposes_explicit_current_stage_lineup():
    from colonist_rl.evaluation import evaluate_blocks

    result = evaluate_blocks(
        GraphPolicy(16),
        board_seeds=[101, 102],
        max_game_steps=1,
        opponent_lineup=("heuristic", "random", "random"),
    )
    assert result["opponent_lineup"] == ["heuristic", "random", "random"]
    assert result["completed_board_seeds"] == [101, 102]
    assert result["win_rate"] is None


def test_inflight_bootstrap_is_retained_and_reports_deadline(monkeypatch):
    from colonist_rl import training

    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    policy = GraphPolicy(16)
    collector = RolloutCollector(policy, board_seeds=[101])
    version = collector.env.version

    def expire_after_bootstrap(module, args, output):
        if collector.env.version > version:
            clock[0] = 10

    handle = policy.register_forward_hook(expire_after_bootstrap)
    records = collector.collect(1, deadline=5)
    handle.remove()
    assert len(records) == 1 and not records[0].boundary
    with torch.no_grad():
        expected = policy(collector.env.observe(collector.learner), collector.memory).value.item()
    assert records[0].next_value == expected
    assert collector.dropped_partial_intervals == 0
    assert collector.stop_reason == "deadline"


def test_runner_decision_cap_counts_applied_actions_even_when_interval_is_dropped(
    tmp_path, monkeypatch
):
    from colonist_rl.training import _MatchDriver

    m = curriculum()
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, GraphPolicy(16))
    monkeypatch.setattr(
        m,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )
    clock, actions = [0.0], []
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])

    def fixed_evaluation(**kwargs):
        value = evidence(kwargs["baseline"], vp=3, boards=kwargs["board_seeds"])
        value["opponent_lineup"] = list(kwargs["opponent_lineup"])
        return value

    monkeypatch.setattr(m, "evaluate_blocks", fixed_evaluation)
    original = _MatchDriver.step

    def slow_real_step(self, action):
        if self.env.current_player == self.learner:
            actions.append(action)
        value = original(self, action)
        clock[0] += 40
        return value

    monkeypatch.setattr(_MatchDriver, "step", slow_real_step)
    result = m.run_curriculum(
        tmp_path / "run",
        initial_checkpoints={11: initial},
        config=m.CurriculumConfig(compute_seconds=100, wall_seconds=1000, max_decisions_per_seed=1),
    )
    row = result["seeds"]["11"]
    assert len(actions) == 1
    assert row["decisions"] == 1
    assert row["completed_intervals"] == 0
    assert row["dropped_partial_intervals"] == 1
