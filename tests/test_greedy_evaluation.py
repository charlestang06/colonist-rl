"""Bounded routing checks; these games provide no empirical performance evidence."""

import json
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colonist_rl.cli import main, parser
from colonist_rl.contracts import Action
from colonist_rl.evaluation import evaluate_blocks
from colonist_rl.greedy_no_trade import GreedyNoTrade
from colonist_rl.opponents import HeuristicPolicy, RandomPolicy
from colonist_rl.policy import GraphPolicy
from colonist_rl.simulator import FourPlayerGame
from colonist_rl.training import _MatchDriver, evaluate


@pytest.fixture(autouse=True)
def one_torch_thread():
    torch.set_num_threads(1)


def without_timings(value):
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items() if "latency" not in k}
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    return value


def test_named_candidate_matches_supplied_greedy_and_legacy_games():
    named = evaluate_blocks(board_seeds=[413], candidate="greedy_no_trade", max_game_steps=32)
    supplied = evaluate_blocks(GreedyNoTrade(), board_seeds=[413], max_game_steps=32)
    legacy = evaluate(
        games=4, seed=413, candidate="greedy_no_trade", max_game_steps=32, deterministic=True
    )
    assert named["candidate"] == "greedy_no_trade"
    assert named["candidate_identity"] == "plain-greedy-no-trade-v1"
    assert named["action_mode"] == "deterministic_greedy_no_trade"
    assert named["baseline"] == "heuristic"
    assert named["opponent_lineup"] is None
    assert "baseline_identity" not in named
    assert named["completed_board_seeds"] == [413]
    assert named["games"] == named["truncations"] == 4
    assert named["learner_decisions"] > 0
    assert named["action_kind_counts"].get("ACCEPT_TRADE", 0) == 0
    assert named["action_kind_counts"].get("MARITIME_TRADE", 0) == 0
    assert without_timings(named) == without_timings(supplied)
    assert without_timings(named["outcomes"]) == without_timings(legacy["outcomes"])
    json.dumps(named, allow_nan=False)


@pytest.mark.parametrize("deterministic", [True, False])
def test_greedy_mode_is_deterministic_even_when_sampling_requested(deterministic):
    result = evaluate_blocks(
        board_seeds=[414],
        candidate="greedy_no_trade",
        deterministic=deterministic,
        baseline="greedy_no_trade",
        max_game_steps=24,
    )
    assert result["action_mode"] == "deterministic_greedy_no_trade"
    assert result["baseline"] == "greedy_no_trade"
    assert result["baseline_identity"] == "plain-greedy-no-trade-v1"
    assert result["offers"] == 0
    assert result["games"] == 4


@pytest.mark.parametrize("baseline", ["heuristic", "greedy_no_trade"])
def test_explicit_mixed_lineup_metadata_does_not_replace_fixed_default_panel(baseline):
    result = evaluate_blocks(
        board_seeds=[415],
        baseline=baseline,
        opponent_lineup=("greedy_no_trade", "random", "heuristic"),
        max_game_steps=24,
    )
    assert result["candidate"] == "heuristic"
    assert result["action_mode"] == "seeded_heuristic"
    assert result["baseline"] == baseline
    assert result["opponent_lineup"] == ["greedy_no_trade", "random", "heuristic"]
    assert result["opponent_lineup_identities"] == ["plain-greedy-no-trade-v1", None, None]
    assert "candidate_identity" not in result
    assert "baseline_identity" not in result


@pytest.mark.parametrize("learner", range(4))
def test_driver_explicit_lineup_is_relative_to_each_learner_seat(learner):
    env = FourPlayerGame(max_steps=8)
    env.reset(seed=416)
    driver = _MatchDriver(
        env, learner, 416, opponent_lineup=("greedy_no_trade", "random", "heuristic")
    )
    assert isinstance(driver.policies[(learner + 1) % 4], GreedyNoTrade)
    assert isinstance(driver.policies[(learner + 2) % 4], RandomPolicy)
    assert isinstance(driver.policies[(learner + 3) % 4], HeuristicPolicy)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize(
    "menu, expected",
    [
        (
            (
                Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
                Action("END_TURN"),
            ),
            "END_TURN",
        ),
        (
            (Action("ACCEPT_TRADE", offer_id="gift"), Action("REJECT_TRADE", offer_id="gift")),
            "REJECT_TRADE",
        ),
    ],
)
def test_driver_greedy_routes_literal_trade_menus_to_an_action(explicit, menu, expected):
    observation = replace(FourPlayerGame().reset(seed=417), legal_actions=menu)
    learner = (observation.player_id + 3) % 4
    driver = _MatchDriver(
        SimpleNamespace(observe=lambda: observation),
        learner,
        417,
        baseline="greedy_no_trade",
        opponent_lineup=("greedy_no_trade", "random", "heuristic") if explicit else None,
    )
    selected = driver.opponent_action()
    assert isinstance(selected, Action)
    assert selected.kind == expected
    assert any(selected is legal for legal in menu)


def test_driver_keeps_historical_graph_tuple_interface_with_greedy_lineup():
    observation = FourPlayerGame().reset(seed=418)
    learner = (observation.player_id + 1) % 4
    graph = GraphPolicy(8).eval()
    driver = _MatchDriver(
        SimpleNamespace(observe=lambda: observation),
        learner,
        418,
        (graph,),
        opponent_lineup=("greedy_no_trade", "random", "historical"),
    )
    action = driver.opponent_action()
    assert action in observation.legal_actions
    assert driver.memories[observation.player_id] is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate": "greedy_no_trade", "policy": GraphPolicy(8)},
        {"candidate": "unknown"},
        {"baseline": "unknown"},
        {"opponent_lineup": ("greedy_no_trade", "unknown", "heuristic")},
        {"opponent_lineup": ("greedy_no_trade", "historical", "heuristic")},
        {"opponent_lineup": ("greedy_no_trade", "random")},
        {"baseline": "greedy_no_trade", "historical_policies": (GraphPolicy(8),)},
    ],
)
def test_invalid_greedy_route_combinations_fail_before_game_creation(monkeypatch, kwargs):
    from colonist_rl import evaluation

    def unexpected_game(*args, **kwargs):
        pytest.fail("invalid configuration created a game")

    monkeypatch.setattr(evaluation, "FourPlayerGame", unexpected_game)
    with pytest.raises(ValueError):
        evaluate_blocks(board_seeds=[419], max_game_steps=1, **kwargs)


def test_driver_rejects_implicit_historical_override_of_greedy_baseline():
    with pytest.raises(ValueError, match="explicit.*lineup"):
        _MatchDriver(SimpleNamespace(), 0, 419, (GraphPolicy(8),), baseline="greedy_no_trade")


def test_explicit_historical_mixed_lineup_can_override_greedy_baseline():
    result = evaluate_blocks(
        board_seeds=[419],
        baseline="greedy_no_trade",
        historical_policies=(GraphPolicy(8),),
        opponent_lineup=("greedy_no_trade", "historical", "random"),
        max_game_steps=24,
    )
    assert result["games"] == 4
    assert "baseline_identity" not in result
    assert result["opponent_lineup"] == ["greedy_no_trade", "historical", "random"]
    assert result["opponent_lineup_identities"] == ["plain-greedy-no-trade-v1", None, None]


@pytest.mark.parametrize("fail", [False, True])
def test_supplied_greedy_restores_rng_and_mode_even_on_failure(fail):
    class ObservedGreedy(GreedyNoTrade):
        def act(self, view, memory=None, *, deterministic=False):
            assert not self.training
            random.random()
            np.random.random()
            torch.rand(1)
            if fail:
                raise RuntimeError("greedy decision failure")
            return super().act(view, memory, deterministic=deterministic)

    policy = ObservedGreedy().train()
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="greedy decision failure"):
            evaluate_blocks(policy, board_seeds=[420], max_game_steps=8)
    else:
        result = evaluate_blocks(policy, board_seeds=[420], max_game_steps=8)
        assert result["candidate_identity"] == "plain-greedy-no-trade-v1"
    assert policy.training
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
    assert restored_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("reject_trades", [False, True])
def test_candidate_routes_bank_and_response_menus_without_model_forward(
    monkeypatch, supplied, reject_trades
):
    from colonist_rl import evaluation

    base_view = FourPlayerGame().reset(seed=421)
    bank_menu = (
        Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
        Action("END_TURN"),
    )
    response_menu = (
        Action("ACCEPT_TRADE", offer_id="gift"),
        Action("REJECT_TRADE", offer_id="gift"),
    )

    class MenuGame:
        # Only the environment boundary is scripted: the evaluator and driver
        # must select and step actual Actions through both distinct menus.
        def __init__(self, *, max_steps):
            self.version = 0
            self.current_player = 0

        def reset(self, *, seed):
            return self.observe()

        def observe(self):
            return replace(
                base_view,
                player_id=0,
                context=replace(
                    base_view.context,
                    actor=0,
                    kind=("PLAY_TURN" if self.version == 0 else "TRADE_RESPONSE"),
                ),
                legal_actions=bank_menu if self.version == 0 else response_menu,
            )

        def step(self, action):
            assert action is (bank_menu[1] if self.version == 0 else response_menu[1])
            self.version += 1
            return SimpleNamespace(terminated=self.version == 2, truncated=False, winner=0)

        def _actual_vp(self):
            return [10, 2, 2, 2]

    monkeypatch.setattr(evaluation, "FourPlayerGame", MenuGame)
    result = evaluate(
        GreedyNoTrade() if supplied else None,
        candidate="greedy_no_trade",
        games=1,
        seed=421,
        max_game_steps=8,
        reject_trades=reject_trades,
    )
    assert result["action_kind_counts"] == {"END_TURN": 1, "REJECT_TRADE": 1}
    assert result["trade_responses"] == 1
    assert result["terminations"] == 1


@pytest.mark.parametrize("command", ["simulate", "evaluate"])
def test_cli_accepts_opt_in_greedy_baseline_and_keeps_heuristic_default(command, capsys):
    assert parser().parse_args([command]).baseline == "heuristic"
    assert (
        main([command, "--baseline", "greedy_no_trade", "--games", "1", "--max-game-steps", "8"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["candidate"] == "heuristic"
    assert result["baseline"] == "greedy_no_trade"
    assert result["baseline_identity"] == "plain-greedy-no-trade-v1"
    assert result["games"] == 1
