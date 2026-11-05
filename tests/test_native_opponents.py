"""Fixture-only gateway checks. Never invoke an authentic VF/AB decision."""

import copy
import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from colonist_rl.contracts import RESOURCES, Action


@dataclass(frozen=True)
class NativeAction:
    color: str
    action_type: str
    value: int = 0


class FixtureEnv:
    def __init__(self, seat=2):
        self.version = 4
        self._rng_state = random.Random(19).getstate()
        self.game = SimpleNamespace(
            state=SimpleNamespace(
                colors=["blue", "red", "orange", "white"],
                action_records=[{"secret": [1]}],
                development_listdeck=["KNIGHT", "MONOPOLY"],
            ),
            playable_actions=[
                NativeAction("orange", "BUILD_ROAD", 7),
                NativeAction("orange", "BUILD_ROAD", 3),
            ],
        )
        self.view = SimpleNamespace(
            player_id=seat,
            context=SimpleNamespace(actor=seat, kind="PLAY_TURN", version=4, discard_remaining=0),
            legal_actions=(Action("BUILD_ROAD", target=103), Action("BUILD_ROAD", target=107)),
            hand=(3, 2, 1, 2, 2),
        )

    def observe(self):
        return self.view

    def _convert(self, action):
        return Action(action.action_type, target=100 + action.value)

    def _randomness(self):
        raise AssertionError("planning must never enter true environment RNG")


@pytest.fixture
def gateway(monkeypatch):
    from colonist_rl import native_opponents as native

    calls = []

    class Player:
        def __init__(self, color, **kwargs):
            calls.append(("init", color, kwargs))

        def decide(self, game, actions):
            calls.append(("decide", game, list(actions), random.random()))
            chosen = actions[0]
            game.playable_actions.clear()
            game.state.action_records[0]["secret"].append(9)
            game.state.development_listdeck.pop()
            return chosen

    monkeypatch.setattr(
        native, "verify_native_installation", lambda: {"commit": native.ENGINE_REVISION}
    )
    monkeypatch.setattr(native, "_native_player_class", lambda identity: Player)
    return native, calls


@pytest.mark.parametrize("identity", ["native_vf_82aae93", "native_ab2_82aae93"])
def test_exact_delegation_color_native_order_copy_and_rng(gateway, identity):
    native, calls = gateway
    env = FixtureEnv()
    original = copy.deepcopy(env.game)
    external, engine_rng = random.getstate(), env._rng_state
    runner = native.NativeOpponentRunner(env, 2, identity, planning_seed=71)
    action = runner.select_action()
    assert action is env.view.legal_actions[1]
    assert calls[0][1] == "orange"
    assert calls[0][2] == (
        {"depth": 2, "prunning": False, "value_fn_builder_name": None, "epsilon": None}
        if "ab2" in identity
        else {"value_fn_builder_name": None, "params": None, "epsilon": None}
    )
    assert calls[1][1] is not env.game
    assert calls[1][2] == original.playable_actions
    assert calls[1][3] == random.Random(71).random()
    assert env.game == original
    assert random.getstate() == external
    assert env._rng_state == engine_rng
    assert runner.metrics()["native_decisions"] == 1
    assert runner.metrics()["search_nodes"] is None
    assert "privileged" in runner.metadata()["information_regime"]


def test_illegal_native_return_and_error_restore_rng(gateway, monkeypatch):
    native, _ = gateway
    env = FixtureEnv()
    runner = native.NativeOpponentRunner(env, 2, native.NATIVE_IDS[0], planning_seed=3)
    external = random.getstate()

    def bad(game, actions):
        random.random()
        return NativeAction("orange", "BUILD_ROAD", 99)

    monkeypatch.setattr(runner._player, "decide", bad)
    with pytest.raises(ValueError, match="native legal"):
        runner.select_action()
    assert random.getstate() == external

    def raises(game, actions):
        random.random()
        raise RuntimeError("native failed")

    monkeypatch.setattr(runner._player, "decide", raises)
    with pytest.raises(RuntimeError, match="native failed"):
        runner.select_action()
    assert random.getstate() == external


def test_conversion_requires_actual_wrapper_member_and_actor(gateway):
    native, _ = gateway
    env = FixtureEnv()
    runner = native.NativeOpponentRunner(env, 2, native.NATIVE_IDS[0], planning_seed=3)
    env.view.legal_actions = (Action("ROLL"),)
    with pytest.raises(ValueError, match="wrapper legal"):
        runner.select_action()
    env.view.player_id = 1
    with pytest.raises(ValueError, match="seat"):
        runner.select_action()


def test_bulk_discard_cache_only_hand_and_reject_trade(gateway):
    native, calls = gateway
    env = FixtureEnv()
    env.view.context.kind = "DISCARD"
    env.view.context.discard_remaining = 5
    env.view.legal_actions = tuple(Action("DISCARD_RESOURCE", resource=r) for r in RESOURCES)
    runner = native.NativeOpponentRunner(env, 2, native.NATIVE_IDS[0], planning_seed=17)
    expected = random.Random(17).sample(
        [r for r, n in zip(RESOURCES, env.view.hand) for _ in range(n)], 5
    )
    actual = []
    for remaining in range(5, 0, -1):
        env.view.context.discard_remaining = remaining
        actual.append(runner.select_action().resource)
        env.version += 1
        env.view.context.version += 1
    assert actual == expected
    assert len(calls) == 1  # no fabricated native discard search decision
    env.view.context.kind = "TRADE_RESPONSE"
    reject = Action("REJECT_TRADE", offer_id="fixture")
    env.view.legal_actions = (Action("ACCEPT_TRADE", offer_id="fixture"), reject)
    assert runner.select_action() is reject
    assert runner.metadata()["discard_adapter"] == "native_random_bulk_to_per_card"


def test_native_not_learner_or_training_lineup(gateway):
    native, _ = gateway
    from colonist_rl.evaluation import evaluate_blocks
    from colonist_rl.training import _lineup

    for identity in native.NATIVE_IDS:
        with pytest.raises(ValueError, match="candidate"):
            evaluate_blocks(board_seeds=[1], candidate=identity)
        with pytest.raises(ValueError, match="opponent_lineup"):
            _lineup((identity, "random", "heuristic"))
        assert _lineup((identity, "random", "heuristic"), allow_native=True)[0] == identity


def test_metadata_verification_checks_installed_audited_hashes():
    from colonist_rl.native_opponents import verify_native_installation

    result = verify_native_installation()
    assert len(result["source_sha256"]) == 12
    assert result["version"] == "3.3.0"


def test_driver_explicit_native_evaluation_dispatch_and_default_parity(gateway):
    native, calls = gateway
    from colonist_rl.opponents import HeuristicPolicy
    from colonist_rl.training import _MatchDriver

    env = FixtureEnv()
    default = _MatchDriver(env, 0, 3, baseline="heuristic")
    assert not default.native_opponents
    assert all(isinstance(p, HeuristicPolicy) for p in default.policies.values())
    for learner in range(4):
        driver = _MatchDriver(
            env,
            learner,
            3,
            allow_native=True,
            opponent_lineup=(native.NATIVE_IDS[0], "random", "heuristic"),
        )
        native_seat = (learner + 1) % 4
        assert list(driver.native_opponents) == [native_seat]
        assert calls[-1][1] == env.game.state.colors[native_seat]
    driver = _MatchDriver(env, 0, 3, baseline=native.NATIVE_IDS[0], allow_native=True)
    driver.maybe_offer()  # no proposals from any seat under native protocol
    assert driver.opponent_action() is env.view.legal_actions[1]


def test_evaluation_native_identity_and_budget_validation_without_games(gateway):
    native, _ = gateway
    from colonist_rl.evaluation import evaluate_blocks

    with pytest.raises(ValueError, match="deadline"):
        evaluate_blocks(board_seeds=[1], baseline=native.NATIVE_IDS[0])
    result = evaluate_blocks(board_seeds=[1], baseline=native.NATIVE_IDS[0], deadline=0)
    assert result["status"] == "incomplete"
    assert result["baseline_identity"] == native.NATIVE_IDS[0]
    assert result["native_protocol"]["full_rule_equivalence"] is False
    assert result["native_protocol"]["domestic_trades"] == "disabled_all_seats"
    assert result["native_protocol"]["information_regime"] == (
        "public_only_candidate_vs_privileged_native_opponents"
    )
    assert result["outcomes"] == []


def test_cli_native_selectors_parse_without_running_policy():
    from colonist_rl.cli import parser

    args = parser().parse_args(
        [
            "evaluate",
            "--baseline",
            "native_ab2_82aae93",
            "--candidate",
            "greedy_no_trade",
            "--wall-seconds",
            "30",
            "--opponent-lineup",
            "native_vf_82aae93",
            "random",
            "native_ab2_82aae93",
        ]
    )
    assert args.baseline == "native_ab2_82aae93"
    assert args.candidate == "greedy_no_trade"
    assert args.wall_seconds == 30


@pytest.mark.parametrize(
    "kind",
    [
        "BUILD_SETTLEMENT",
        "BUILD_CITY",
        "BUILD_ROAD",
        "MOVE_ROBBER",
        "MARITIME_TRADE",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY",
        "BUY_DEVELOPMENT_CARD",
        "ROLL",
        "END_TURN",
    ],
)
def test_every_native_phase_uses_authoritative_conversion(gateway, kind):
    native, _ = gateway
    env = FixtureEnv()
    env.game.playable_actions = [NativeAction("orange", kind, 7)]
    expected = Action(kind, target=9, partner=1)
    converted = []

    def convert(action):
        converted.append(action)
        return expected

    env._convert = convert
    env.view.legal_actions = (expected,)
    runner = native.NativeOpponentRunner(env, 2, native.NATIVE_IDS[0], planning_seed=2)
    assert runner.select_action() is expected
    assert converted == env.game.playable_actions


def test_native_runner_cannot_be_supplied_as_candidate_and_metadata_is_public(gateway):
    import json

    native, _ = gateway
    from colonist_rl.evaluation import evaluate_blocks

    runner = native.NativeOpponentRunner(FixtureEnv(), 2, native.NATIVE_IDS[0], planning_seed=2)
    with pytest.raises(TypeError, match="candidate"):
        evaluate_blocks(runner, board_seeds=[1])
    assert "secret" not in json.dumps({"metadata": runner.metadata(), "metrics": runner.metrics()})


def test_bad_native_source_hash_is_rejected(monkeypatch):
    from colonist_rl import native_opponents as native

    monkeypatch.setattr(native, "_SOURCE_SHA256", {"players/value.py": "0" * 64})
    with pytest.raises(RuntimeError, match="source identity mismatch"):
        native.verify_native_installation()
