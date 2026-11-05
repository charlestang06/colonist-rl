"""Deterministic toy episodes test recurrence and provenance without game studies."""

import importlib
import random
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colonist_rl import training
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer
from colonist_rl.simulator import StepResult
from colonist_rl.training import PPOConfig, RolloutCollector, Transition, ppo_update

BOARD = BoardSpec.from_hexes([(0, 0)])
RULES = RulesConfig.standard()


def view(version, seat=0, forced=True, passive=False):
    return Observation(
        seat,
        BOARD,
        RULES,
        DecisionContext("PLAY_TURN", seat, seat, version),
        tuple(PublicPlayer(p, 2, 0, 0) for p in range(4)),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (),
        (),
        0,
        ()
        if passive
        else (Action("END_TURN"),)
        if forced
        else (Action("END_TURN"), Action("ROLL")),
    )


class ToyPolicy(torch.nn.Module):
    """Scalar recurrence has a hand-computable full-prefix oracle."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.value_weight = torch.nn.Parameter(torch.tensor(0.3))
        self.calls = []
        self.expire = None

    @property
    def device(self):
        return self.weight.device

    def recurrent_step(self, observation, memory=None):
        self.calls.append(
            (
                observation.context.version,
                None if memory is None else memory.detach().item(),
                torch.is_grad_enabled(),
                self.weight.detach().item(),
            )
        )
        if self.expire:
            self.expire()
        return self.weight * ((0 if memory is None else memory) + observation.context.version + 1)

    def forward(self, observation, memory=None):
        hidden = self.recurrent_step(observation, memory)
        logits = (
            torch.stack((-hidden, hidden))
            if len(observation.legal_actions) == 2
            else hidden.expand(len(observation.legal_actions))
        )
        return SimpleNamespace(logits=logits, value=hidden * self.value_weight, memory=hidden)


class ToyGame:
    def __init__(self, actors, rewards, *, winner=1, truncated=False):
        self.actors, self.rewards = actors, rewards
        self.outcome, self.cutoff = winner, truncated

    def reset(self, seed):
        self.version = 0
        self._terminated = self._truncated = False
        self.winner = None
        return self.observe()

    @property
    def current_player(self):
        return self.actors[min(self.version, len(self.actors) - 1)]

    def observe(self, player=None):
        observation = view(
            self.version,
            self.current_player if player is None else player,
            forced=self.version != 2,
            passive=self._terminated or self._truncated,
        )
        # Nonterminal learner-0 shaping potential follows the literal toy rewards.
        points = round(2 + 50 * sum(r[0] for r in self.rewards[: self.version]))
        return replace(
            observation,
            players=(
                replace(observation.players[0], victory_points=points),
                *observation.players[1:],
            ),
        )

    def step(self, action):
        assert action in self.observe().legal_actions
        reward = self.rewards[self.version]
        self.version += 1
        if self.version == len(self.actors):
            self._terminated = not self.cutoff
            self._truncated = self.cutoff
            self.winner = None if self.cutoff else self.outcome
        return StepResult(
            self.observe(),
            reward,
            self._terminated,
            self._truncated,
            self.winner,
            {"version": self.version},
        )


class ToyDriver:
    def __init__(self, env, learner, *args, **kwargs):
        self.env, self.learner = env, learner
        self.offers = 0

    @property
    def view(self):
        return self.env.observe()

    def maybe_offer(self, **kwargs):
        pass

    def opponent_action(self, **kwargs):
        return self.view.legal_actions[0]

    def step(self, action):
        return self.env.step(action)


@pytest.fixture
def toy(monkeypatch):
    # Its driver binding must be imported before the toy patches training.
    importlib.import_module("colonist_rl.evaluation")

    def install(
        actors=(0, 1, 0, 1, 0),
        rewards=(0.02, -0.04, 0.04, -0.02, -0.04),
        *,
        winner=1,
        truncated=False,
    ):
        vectors = tuple((r, 0.0, 0.0, 0.0) for r in rewards)
        monkeypatch.setattr(
            training,
            "FourPlayerGame",
            lambda **kw: ToyGame(actors, vectors, winner=winner, truncated=truncated),
        )
        monkeypatch.setattr(training, "_MatchDriver", ToyDriver)

    install()
    return install


def audited(policy, **kwargs):
    return RolloutCollector(policy, audited_history=True, board_seeds=[101], **kwargs)


def oracle(weight, versions):
    hidden = None
    for version in versions:
        hidden = weight * ((0 if hidden is None else hidden) + version + 1)
    return hidden


def test_original_collector_stale_context_is_a_real_weight_change(toy):
    policy = ToyPolicy()
    collector = RolloutCollector(policy, board_seeds=[101])
    first = collector.collect(1)[0]
    # A real optimizer update to the recurrent parameter, between collect calls.
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    optimizer.zero_grad()
    (-policy.weight).backward()
    optimizer.step()
    second = collector.collect(1)[0]
    assert first.memory is None
    assert second.memory.item() == pytest.approx(0.4)
    assert second.memory.item() != pytest.approx(oracle(policy.weight.item(), [0]))


def test_original_ppo_uses_poisoned_stored_memory(toy):
    policy = ToyPolicy()
    records = RolloutCollector(policy, board_seeds=[101]).collect(2)
    poisoned = [replace(records[1], memory=torch.tensor(99.0))]
    policy.calls.clear()
    ppo_update(policy, torch.optim.SGD(policy.parameters(), lr=0.01), poisoned, PPOConfig(epochs=1))
    assert policy.calls[0][1] == 99.0


def test_collector_refreshes_full_history_after_update_and_keeps_snapshots(toy):
    policy = ToyPolicy()
    collector = audited(policy, run_id="refresh")
    first = collector.collect(1)[0]
    snapshots = collector.episode_traces
    before = snapshots[first.episode_key]
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    optimizer.zero_grad()
    (-policy.weight).backward()
    optimizer.step()
    policy.calls.clear()
    second = collector.collect(1)[0]
    expected = oracle(policy.weight.item(), [0])
    assert second.memory.item() == pytest.approx(expected)
    hidden = policy.weight.item() * (expected + 3)
    assert second.value == pytest.approx(hidden * policy.value_weight.item())
    logits = torch.tensor([-hidden, hidden])
    assert second.log_probability == pytest.approx(
        torch.log_softmax(logits, 0)[second.action_index].item(), abs=1e-7
    )
    assert [(v, grad) for v, _, grad, _ in policy.calls] == [(0, False), (2, False), (4, False)]
    after = collector.episode_traces[first.episode_key]
    assert [o.context.version for o in before.observations] == [0]
    assert [o.context.version for o in after.observations] == [0, 2]
    assert len(before.transitions) == 1
    assert first.observation is after.observations[0]
    assert after.transitions[1].memory is None  # no mutable tensors inside snapshots
    with pytest.raises(TypeError):
        snapshots["mutated"] = after
    with pytest.raises(FrozenInstanceError):
        after.winner = 0


@pytest.mark.parametrize("fallback", [False, True])
def test_ppo_replays_current_prefix_before_every_chunk_ignoring_poison(toy, monkeypatch, fallback):
    policy = ToyPolicy()
    collector = audited(policy, run_id="chunks")
    records = collector.collect(3)
    records = [replace(r, memory=torch.tensor(999.0)) for r in records]
    if fallback:
        # This policy exposes only forward; exercise the documented fallback.
        policy.recurrent_step_saved = policy.recurrent_step
        policy.recurrent_step = None

        def forward(observation, memory=None):
            hidden = policy.recurrent_step_saved(observation, memory)
            logits = (
                torch.stack((-hidden, hidden))
                if len(observation.legal_actions) == 2
                else hidden.expand(len(observation.legal_actions))
            )
            return SimpleNamespace(logits=logits, value=hidden * policy.value_weight, memory=hidden)

        monkeypatch.setattr(policy, "forward", forward)
    monkeypatch.setattr(training.random, "shuffle", lambda values: values.reverse())
    policy.calls.clear()
    old = policy.weight.item()
    metrics = ppo_update(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.02),
        records,
        PPOConfig(epochs=2, sequence_length=1),
        episode_traces=collector.episode_traces,
        require_current_history=True,
    )
    assert metrics["completed_updates"] == 6
    assert policy.weight.item() != old
    assert len({w for _, _, grad, w in policy.calls if grad}) > 1
    for version, memory, grad, weight in policy.calls:
        if grad:
            assert memory == pytest.approx(oracle(weight, [v for v in (0, 2, 4) if v < version]))
        elif version == 0:
            assert memory is None
    assert sum(not grad for _, _, grad, _ in policy.calls) == 6


def test_resolved_negative_shaped_suffix_spans_calls_and_off_turn_rewards(toy):
    collector = audited(ToyPolicy(), run_id="returns")
    first = collector.collect(1)[0]
    pending = collector.episode_traces[first.episode_key]
    with pytest.raises(ValueError):
        training.resolved_returns(pending)
    rest = collector.collect(2)
    trace = collector.episode_traces[first.episode_key]
    assert [r.reward for r in trace.transitions] == pytest.approx([-0.02, 0.02, -0.04])
    assert training.resolved_returns(trace) == pytest.approx((-0.04, -0.02, -0.04))
    # Independent unchanged-reward telescope: an opponent win removes own potential.
    assert training.resolved_returns(trace) == pytest.approx(
        tuple(
            -0.2 * observation.own_victory_points / observation.rules.victory_points
            for observation in trace.observations
        )
    )
    assert trace.winner == 1 and trace.learner_seat == 0
    assert trace.terminated and not trace.truncated and trace.finalized
    assert trace.initial_reward == 0.0
    assert [(r.step_start, r.step_end) for r in trace.transitions] == [(0, 2), (2, 4), (4, 5)]
    assert [r.observation_index for r in [first, *rest]] == [0, 1, 2]
    assert trace.final_version == 5
    assert rest[-1].next_value == 0.0
    assert collector.collect(1) == []
    assert collector.episode_traces[first.episode_key] == trace


def test_zero_decision_terminal_and_separate_initial_reward_are_retained(toy):
    toy(actors=(1, 1), rewards=(0.02, -0.06), winner=1)
    collector = audited(ToyPolicy(), run_id="empty")
    assert collector.collect(1) == []
    (trace,) = collector.episode_traces.values()
    assert trace.observations == trace.transitions == ()
    assert trace.initial_reward == pytest.approx(-0.04)
    assert (trace.initial_step_start, trace.initial_step_end, trace.final_version) == (0, 2, 2)
    assert trace.winner == 1 and trace.terminated
    assert training.resolved_returns(trace) == ()
    assert collector.episodes == 1


def test_initial_reward_is_excluded_from_first_decision_suffix(toy):
    toy(actors=(1, 0), rewards=(0.2, -0.04), winner=1)
    collector = audited(ToyPolicy())
    (record,) = collector.collect(1)
    trace = collector.episode_traces[record.episode_key]
    assert trace.initial_reward == 0.2
    assert record.reward == -0.04
    assert training.resolved_returns(trace) == (-0.04,)
    assert (trace.initial_step_end, record.step_start) == (1, 1)


@pytest.mark.parametrize("damage", ["prefix", "ordinal", "key", "seat", "interval", "reward"])
def test_bad_bindings_fail_before_any_optimizer_mutation(toy, damage):
    policy = ToyPolicy()
    collector = audited(policy)
    records = collector.collect(3)
    traces = dict(collector.episode_traces)
    key = records[0].episode_key
    trace = traces[key]
    if damage == "prefix":
        traces[key] = replace(trace, observations=trace.observations[1:])
    elif damage == "ordinal":
        records[1] = replace(records[1], observation_index=2)
    elif damage == "key":
        records[1] = replace(records[1], episode_key="another:0")
    elif damage == "seat":
        traces[key] = replace(trace, learner_seat=1)
    elif damage == "interval":
        records[1] = replace(records[1], step_start=1)
    else:
        records[1] = replace(records[1], reward=10.0)
    optimizer = torch.optim.Adam(policy.parameters())
    state = {k: v.clone() for k, v in policy.state_dict().items()}
    with pytest.raises(ValueError):
        ppo_update(policy, optimizer, records, episode_traces=traces, require_current_history=True)
    assert optimizer.state == {}
    assert all(torch.equal(state[k], v) for k, v in policy.state_dict().items())


def test_two_local_episode_zero_keys_separate_gae_and_recurrence(toy, monkeypatch):
    policy = ToyPolicy()
    one, two = audited(policy), audited(policy)
    a, b = one.collect(1)[0], two.collect(1)[0]
    assert a.episode == b.episode == 0 and a.episode_key != b.episode_key
    # A changed reward in the second episode must not enter the first GAE target.
    b = replace(b, reward=20.0)
    trace_b = replace(two.episode_traces[b.episode_key], transitions=(replace(b, memory=None),))
    captured = []
    real_advantages = training.advantages

    def capture(*args, **kwargs):
        result = real_advantages(*args, **kwargs)
        captured.append(result[1])
        return result

    monkeypatch.setattr(training, "advantages", capture)
    policy.calls.clear()
    ppo_update(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.0),
        [a, b],
        PPOConfig(epochs=1, sequence_length=10),
        episode_traces={**one.episode_traces, b.episode_key: trace_b},
        require_current_history=True,
    )
    assert captured[0][0] == pytest.approx(a.reward + a.next_value)
    assert [(v, m) for v, m, grad, _ in policy.calls if grad] == [(0, None), (0, None)]


def test_deadline_in_collector_prefix_is_resumable_and_starts_no_action(toy, monkeypatch):
    policy = ToyPolicy()
    collector = audited(policy)
    (record,) = collector.collect(1)
    trace = collector.episode_traces[record.episode_key]
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    policy.expire = lambda: clock.__setitem__(0, 10.0)
    assert collector.collect(1, deadline=5.0) == []
    assert collector.env.version == 2
    assert collector.dropped_partial_intervals == 0
    assert collector.episode_traces[record.episode_key].observations == trace.observations
    policy.expire = None
    clock[0] = 0.0
    assert collector.collect(1, deadline=5.0)[0].observation_index == 1


def test_deadline_in_ppo_prefix_never_updates(toy, monkeypatch):
    policy = ToyPolicy()
    collector = audited(policy)
    records = collector.collect(2)
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    policy.expire = lambda: clock.__setitem__(0, 10.0)
    optimizer = torch.optim.Adam(policy.parameters())
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    metrics = ppo_update(
        policy,
        optimizer,
        records[1:],
        PPOConfig(epochs=1),
        deadline=5.0,
        episode_traces=collector.episode_traces,
        require_current_history=True,
    )
    assert metrics["completed_updates"] == 0 and metrics["deadline_reached"] == 1
    assert optimizer.state == {}
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())


def test_dropped_interval_seals_trace_and_rejects_resolved_returns(toy, monkeypatch):
    collector = audited(ToyPolicy())
    (record,) = collector.collect(1)
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    original = collector.driver.step

    def expire(action):
        result = original(action)
        clock[0] = 10.0
        return result

    monkeypatch.setattr(collector.driver, "step", expire)
    assert collector.collect(1, deadline=5.0) == []
    trace = collector.episode_traces[record.episode_key]
    assert trace.dropped_intervals == 1 and trace.finalized
    assert trace.incomplete_reason == "deadline_dropped_interval"
    with pytest.raises(ValueError):
        training.resolved_returns(trace)
    clock[0] = 0.0
    assert collector.collect(1) == []


def test_truncation_retains_real_bootstrap_and_is_not_resolved(toy):
    toy(actors=(0,), rewards=(0.02,), truncated=True)
    policy = ToyPolicy()
    collector = audited(policy)
    (record,) = collector.collect(1)
    assert record.truncated and not record.terminated
    assert record.next_value == pytest.approx(0.4 * (0.4 + 2) * 0.3)
    trace = collector.episode_traces[record.episode_key]
    with pytest.raises(ValueError):
        training.resolved_returns(trace)
    assert len(trace.observations) == 1


@pytest.mark.parametrize("damage", ["unknown", "gap", "prefix", "bootstrap", "crossed"])
def test_resolved_returns_rejects_false_complete_provenance(toy, damage):
    collector = audited(ToyPolicy())
    records = collector.collect(3)
    trace = collector.episode_traces[records[0].episode_key]
    if damage == "unknown":
        trace = replace(trace, winner=None)
    elif damage == "gap":
        trace = replace(trace, final_version=6)
    elif damage == "prefix":
        trace = replace(trace, began_at_reset=False)
    elif damage == "bootstrap":
        trace = replace(
            trace,
            transitions=(*trace.transitions[:-1], replace(trace.transitions[-1], next_value=1.0)),
        )
    else:
        trace = replace(
            trace,
            transitions=(
                replace(trace.transitions[0], episode_key="crossed"),
                *trace.transitions[1:],
            ),
        )
    with pytest.raises(ValueError):
        training.resolved_returns(trace)


def test_terminal_interval_on_deadline_retains_actual_outcome(toy, monkeypatch):
    toy(actors=(0,), rewards=(-0.04,), winner=1)
    collector = audited(ToyPolicy())
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    original = collector.driver.step

    def expire(action):
        result = original(action)
        clock[0] = 10.0
        return result

    monkeypatch.setattr(collector.driver, "step", expire)
    (record,) = collector.collect(1, deadline=5.0)
    trace = collector.episode_traces[record.episode_key]
    assert trace.terminated and trace.winner == 1
    assert training.resolved_returns(trace) == (-0.04,)
    assert collector.dropped_partial_intervals == 0


def test_legacy_constructors_default_rng_and_audited_id_validation(toy):
    record = Transition(view(0), 0, 0.0, 0.0, 0.0, 0.0, False, False, False, None, 0)
    assert record.episode_key is None and record.observation_index is None
    policy = ToyPolicy()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    audited(policy)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    for kwargs in (
        {"audited_history": 1},
        {"audited_history": True, "run_id": ""},
        {"audited_history": True, "run_id": 1},
        {"run_id": "unusable"},
    ):
        with pytest.raises(ValueError):
            RolloutCollector(policy, **kwargs)
    with pytest.raises(ValueError):
        ppo_update(
            policy,
            torch.optim.SGD(policy.parameters(), lr=0),
            [record],
            require_current_history=True,
        )


@pytest.mark.parametrize(
    "damage", ["history", "memory", "initial_nan", "initial_type", "reward_type", "board", "rules"]
)
def test_required_history_rejects_malformed_trace_before_forward(toy, damage):
    policy = ToyPolicy()
    collector = audited(policy)
    records = collector.collect(3)
    trace = collector.episode_traces[records[0].episode_key]
    if damage == "history":
        trace = replace(trace, history_complete=False)
    elif damage == "memory":
        trace = replace(
            trace,
            transitions=(
                replace(trace.transitions[0], memory=torch.tensor(1.0)),
                *trace.transitions[1:],
            ),
        )
    elif damage == "initial_nan":
        trace = replace(trace, initial_reward=float("nan"))
    elif damage == "initial_type":
        trace = replace(trace, initial_reward=None)
    elif damage == "reward_type":
        trace = replace(
            trace, transitions=(replace(trace.transitions[0], reward=None), *trace.transitions[1:])
        )
    else:
        observation = replace(
            trace.observations[1],
            **{
                damage: BoardSpec.from_hexes([(0, 0), (1, 0)])
                if damage == "board"
                else replace(RULES, victory_points=11)
            },
        )
        transitions = (
            trace.transitions[0],
            replace(trace.transitions[1], observation=observation),
            trace.transitions[2],
        )
        trace = replace(
            trace,
            transitions=transitions,
            observations=(trace.observations[0], observation, trace.observations[2]),
        )
        records[1] = replace(records[1], observation=observation)
    optimizer = torch.optim.Adam(policy.parameters())
    policy.calls.clear()
    with pytest.raises(ValueError):
        ppo_update(
            policy,
            optimizer,
            records,
            episode_traces={trace.episode_key: trace},
            require_current_history=True,
        )
    assert not policy.calls and not optimizer.state


def test_initial_deadline_progress_resumes_without_losing_initial_rewards(toy, monkeypatch):
    toy(actors=(1, 0), rewards=(0.2, -0.04))
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    original = ToyDriver.step

    def expire(driver, action):
        result = original(driver, action)
        clock[0] = 10.0
        return result

    monkeypatch.setattr(ToyDriver, "step", expire)
    collector = audited(ToyPolicy(), deadline=5.0)
    (pending,) = collector.episode_traces.values()
    assert pending.initial_reward == 0.2 and pending.initial_step_end == 1
    assert pending.incomplete_reason == "deadline" and not pending.finalized
    monkeypatch.setattr(ToyDriver, "step", original)
    clock[0] = 0.0
    (record,) = collector.collect(1, deadline=5.0)
    trace = collector.episode_traces[record.episode_key]
    assert trace.initial_reward == 0.2 and record.step_start == 1
    assert training.resolved_returns(trace) == (-0.04,)


def test_finalized_and_empty_traces_survive_reset_and_board_rotation(toy):
    toy(actors=(0,), rewards=(-0.04,), winner=0)
    collector = RolloutCollector(
        ToyPolicy(), audited_history=True, run_id="rotation", board_seeds=[101, 102]
    )
    (record,) = collector.collect(3)
    traces = tuple(collector.episode_traces.values())
    assert len(traces) == 2 and collector.episodes == 2
    assert [(t.board_seed, t.learner_seat, t.winner) for t in traces] == [(101, 0, 0), (102, 1, 0)]
    assert traces[0].episode_key == record.episode_key
    assert traces[1].transitions == () and traces[1].terminated
    assert training.resolved_returns(traces[1]) == ()


def test_gapped_or_reentered_update_episodes_fail_before_step(toy):
    policy = ToyPolicy()
    one, two = audited(policy), audited(policy)
    a, b = one.collect(3), two.collect(1)
    traces = {**one.episode_traces, **two.episode_traces}
    for records in ([a[0], a[2]], [a[0], b[0], a[1]]):
        optimizer = torch.optim.Adam(policy.parameters())
        with pytest.raises(ValueError):
            ppo_update(
                policy, optimizer, records, episode_traces=traces, require_current_history=True
            )
        assert not optimizer.state


def test_legacy_default_and_explicit_false_have_identical_records_and_rng(toy):
    policy = ToyPolicy()
    torch.manual_seed(73)
    state = torch.get_rng_state()
    default = RolloutCollector(policy, board_seeds=[101]).collect(3)
    end = torch.get_rng_state()
    torch.set_rng_state(state)
    explicit_collector = RolloutCollector(policy, board_seeds=[101], audited_history=False)
    explicit = explicit_collector.collect(3)
    assert all(
        replace(a, memory=None) == replace(b, memory=None) for a, b in zip(default, explicit)
    )
    assert torch.equal(torch.get_rng_state(), end)
    assert not explicit_collector.episode_traces


def test_ppo_default_and_explicit_false_keep_parameters_and_rng_identical(toy):
    policy = ToyPolicy()
    records = RolloutCollector(policy, board_seeds=[101]).collect(3)
    state, rng = policy.state_dict(), random.getstate()
    other = ToyPolicy()
    other.load_state_dict(state)
    first = ppo_update(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.01),
        records,
        PPOConfig(epochs=2, sequence_length=1),
    )
    end = random.getstate()
    random.setstate(rng)
    second = ppo_update(
        other,
        torch.optim.SGD(other.parameters(), lr=0.01),
        records,
        PPOConfig(epochs=2, sequence_length=1),
        require_current_history=False,
    )
    assert first == second and random.getstate() == end
    assert all(
        torch.equal(value, other.state_dict()[key]) for key, value in policy.state_dict().items()
    )


def test_real_policy_audited_prefix_matches_forward_after_real_update():
    from colonist_rl.policy import GraphPolicy

    torch.manual_seed(24)
    policy = GraphPolicy(16)
    collector = RolloutCollector(policy, seed=4, audited_history=True, run_id="real-policy")
    first = collector.collect(2)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
    ppo_update(
        policy,
        optimizer,
        first,
        PPOConfig(epochs=1, sequence_length=1),
        episode_traces=collector.episode_traces,
        require_current_history=True,
    )
    trace = collector.episode_traces[first[0].episode_key]
    expected = None
    with torch.no_grad():
        for observation in trace.observations:
            expected = policy(observation, expected).memory
    (following,) = collector.collect(1)
    assert following.episode_key == first[0].episode_key
    assert torch.equal(following.memory, expected)
    with torch.no_grad():
        output = policy(following.observation, expected)
    assert following.value == output.value.item()
