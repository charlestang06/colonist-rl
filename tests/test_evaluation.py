"""Protocol checks use tiny real games; they do not measure playing competence."""

import json
import random
import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from colonist_rl.contracts import Observation
from colonist_rl.evaluation import evaluate_blocks, paired_block_summary, validate_seed_splits
from colonist_rl.policy import GraphPolicy
from colonist_rl.training import evaluate


@pytest.fixture(autouse=True)
def one_torch_thread():
    torch.set_num_threads(1)


def without_timings(value):
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items() if "latency" not in k}
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    return value


@pytest.mark.parametrize("splits", [([1], [1], [3]), ([1], [2], [2])])
def test_split_overlap_is_rejected(splits):
    with pytest.raises(ValueError, match="disjoint"):
        validate_seed_splits(*splits)


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [1.0], ["1"], "12", {1, 2}])
def test_invalid_seed_lists_are_rejected(seeds):
    with pytest.raises(ValueError, match="seed"):
        evaluate_blocks(board_seeds=seeds, max_game_steps=1)
    with pytest.raises(ValueError, match="seed"):
        validate_seed_splits(seeds, [21], [22])


def test_seed_splits_accept_disjoint_explicit_lists():
    assert validate_seed_splits([1, 2], [3], [4, 5]) is None


def test_two_complete_real_blocks_repeat_exactly_except_latency():
    result = evaluate_blocks(board_seeds=[701, 702], max_game_steps=24)
    repeated = evaluate_blocks(board_seeds=[701, 702], max_game_steps=24)
    assert result["status"] == "complete"
    assert result["completed_board_seeds"] == [701, 702]
    assert result["games"] == result["requested_games"] == 8
    assert {(r["board_seed"], r["seat"]) for r in result["outcomes"]} == {
        (701, 0),
        (701, 1),
        (701, 2),
        (701, 3),
        (702, 0),
        (702, 1),
        (702, 2),
        (702, 3),
    }
    assert all(r["truncated"] and not r["terminated"] for r in result["outcomes"])
    assert result["candidate"] == "heuristic"
    assert result["action_mode"] == "seeded_heuristic"
    for row in result["outcomes"]:
        assert sum(row["phase_counts"].values()) == row["learner_decisions"]
        assert sum(row["action_kind_counts"].values()) == row["learner_decisions"]
        assert 0 < row["non_forced_decisions"] <= row["learner_decisions"]
        assert row["decision_latency_ms_mean"] >= 0
    assert without_timings(result) == without_timings(repeated)
    json.dumps(result, allow_nan=False)


def test_random_candidates_and_legacy_entrypoint_share_reproducible_matches():
    result = evaluate_blocks(board_seeds=[17], candidate="random", max_game_steps=8)
    legacy = evaluate(None, 4, 17, "heuristic", False, 8, candidate="random")
    assert result["candidate"] == "random"
    assert result["action_mode"] == "seeded_random"
    assert without_timings(result["outcomes"]) == without_timings(legacy["outcomes"])
    assert evaluate(games=1, max_game_steps=1)["games"] == 1


class RecordingPolicy(GraphPolicy):
    def __init__(self, *, fail=False):
        super().__init__(hidden_size=8)
        self.fail = fail
        self.initial_memories = []
        self.action_modes = []

    def act(self, view, memory=None, *, deterministic=False):
        assert isinstance(view, Observation)
        self.initial_memories.append(memory is None)
        self.action_modes.append(deterministic)
        random.random()
        np.random.random()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("injected decision failure")
        return super().act(view, memory, deterministic=deterministic)


@pytest.mark.parametrize("deterministic", [True, False])
def test_graph_candidate_resets_memory_and_records_action_mode(deterministic):
    policy = RecordingPolicy()
    result = evaluate_blocks(
        policy, board_seeds=[32], max_game_steps=8, deterministic=deterministic
    )
    assert policy.initial_memories.count(True) == 4
    assert policy.action_modes == [deterministic] * len(policy.action_modes)
    assert result["candidate"] == "graph"
    assert result["action_mode"] == ("deterministic" if deterministic else "stochastic")
    assert policy.training
    repeated = evaluate_blocks(
        policy, board_seeds=[32], max_game_steps=8, deterministic=deterministic
    )
    assert without_timings(result) == without_timings(repeated)


@pytest.mark.parametrize("fail", [True, False])
def test_raising_policy_restores_all_external_rng_and_module_modes(fail):
    policy = RecordingPolicy(fail=fail).train()
    policy.heads.eval()
    modes = [module.training for module in policy.modules()]
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            evaluate_blocks(policy, board_seeds=[43], max_game_steps=8)
    else:
        evaluate_blocks(policy, board_seeds=[43], max_game_steps=8)
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
    assert restored_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert [module.training for module in policy.modules()] == modes


def synthetic_result(wins, points):
    return {
        "status": "complete",
        "outcomes": [
            {
                "board_seed": board,
                "seat": seat,
                "won": seat in wins[board],
                "victory_points": points[board][seat],
            }
            for board in [51, 52]
            for seat in range(4)
        ],
    }


def test_paired_summary_aggregates_differences_by_board():
    reference = synthetic_result({51: [0], 52: [0, 1]}, {51: [2, 2, 2, 2], 52: [4, 4, 4, 4]})
    candidate = synthetic_result({51: [0, 1, 2], 52: [0]}, {51: [3, 3, 3, 3], 52: [2, 2, 2, 2]})
    summary = paired_block_summary(reference, candidate)
    assert summary["boards"] == [
        {"board_seed": 51, "win_rate_difference": 0.5, "mean_victory_points_difference": 1.0},
        {"board_seed": 52, "win_rate_difference": -0.25, "mean_victory_points_difference": -2.0},
    ]
    assert summary["win_rate_difference"] == 0.125
    assert summary["mean_victory_points_difference"] == -0.5
    assert summary["paired_games"] == 8
    assert "significance" not in summary
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "different", "partial"])
def test_paired_summary_rejects_invalid_pairs(corruption):
    reference = synthetic_result({51: [], 52: []}, {51: [2] * 4, 52: [2] * 4})
    candidate = deepcopy(reference)
    if corruption == "duplicate":
        candidate["outcomes"].append(candidate["outcomes"][0])
    elif corruption == "missing":
        candidate["outcomes"].pop()
    elif corruption == "different":
        for row in candidate["outcomes"]:
            if row["board_seed"] == 51:
                row["board_seed"] = 53
    else:
        candidate["status"] = "incomplete"
    with pytest.raises(ValueError):
        paired_block_summary(reference, candidate)


def test_expired_deadline_has_no_complete_blocks():
    result = evaluate_blocks(board_seeds=[701, 702], deadline=time.monotonic() - 1)
    assert result["status"] == "incomplete"
    assert result["incomplete_reason"] == "deadline"
    assert result["requested_games"] == 8
    assert result["games"] == 0
    assert result["outcomes"] == result["partial_outcomes"] == []
    assert result["completed_board_seeds"] == []


def test_deadline_mid_block_retains_partial_accounting(monkeypatch):
    # Only the deadline clock is controlled; simulator and actor decisions are real.
    from colonist_rl import evaluation

    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 12 else 2.0

    monkeypatch.setattr(evaluation.time, "monotonic", clock)
    result = evaluate_blocks(board_seeds=[701, 702], max_game_steps=4, deadline=1.0)
    assert result["status"] == "incomplete"
    assert result["games"] == 0
    assert result["outcomes"] == []
    assert 0 < len(result["partial_outcomes"]) < 4
    assert result["finished_games"] == len(result["partial_outcomes"])
    assert result["interrupted_game"]["board_seed"] == 701
    assert result["interrupted_game"]["learner_decisions"] >= 0
    assert result["completed_board_seeds"] == []


def test_real_termination_is_distinct_from_a_step_limit(monkeypatch):
    from colonist_rl import evaluation
    from colonist_rl.board import RulesConfig
    from colonist_rl.simulator import FourPlayerGame

    # Lower the target through the real simulator's public rules configuration.
    def short_game(*, max_steps):
        return FourPlayerGame(
            replace(RulesConfig.standard(), victory_points=1), max_steps=max_steps
        )

    monkeypatch.setattr(evaluation, "FourPlayerGame", short_game)
    result = evaluate_blocks(board_seeds=[71], max_game_steps=8)
    assert result["status"] == "complete"
    assert result["terminations"] == 4
    assert result["truncations"] == 0
    assert result["wins"] == 1
    assert all(row["terminated"] and not row["truncated"] for row in result["outcomes"])


def test_deadline_retains_completed_blocks_and_separates_partial_games(monkeypatch):
    from colonist_rl import evaluation

    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 27 else 2.0

    monkeypatch.setattr(evaluation.time, "monotonic", clock)
    result = evaluate_blocks(board_seeds=[701, 702], max_game_steps=4, deadline=1.0)
    assert result["status"] == "incomplete"
    assert result["completed_board_seeds"] == [701]
    assert result["games"] == 4
    assert result["finished_games"] == 5
    assert len(result["partial_outcomes"]) == 1
    assert all(row["board_seed"] == 701 for row in result["outcomes"])
    assert result["partial_outcomes"][0]["board_seed"] == 702
    assert result["interrupted_game"]["board_seed"] == 702
    assert result["truncations"] == 4
