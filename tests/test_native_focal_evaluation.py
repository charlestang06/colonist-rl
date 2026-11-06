"""Focal-reference RED tests: all games/actors/native decisions below are fakes.

Root owns importing/running this module under its existing CPU/RSS guards. This
file does not authorize any authentic native decision, model forward or rollout.
Missing APIs fail as assertions inside tests, rather than collection/import errors.
"""
from __future__ import annotations

import inspect
import json
import random
from types import SimpleNamespace as NS

import pytest

from colonist_rl import evaluation as ev
from colonist_rl import native_opponents as native_module
from colonist_rl import simulator as simulator_module
from colonist_rl.contracts import Action
from colonist_rl.native_opponents import NATIVE_IDS


BACKGROUND = ("greedy_no_trade",) * 3
SECRET = "PRIVATE_NATIVE_SENTINEL_MUST_NOT_ESCAPE"


def reference_api():
    api = getattr(ev, "evaluate_native_reference_blocks", None)
    assert callable(api), "missing evaluation-only focal native reference API"
    return api


def public_no_offer_api():
    assert "no_offers" in inspect.signature(ev.evaluate_blocks).parameters, (
        "missing evaluation-only no_offers keyword on public block evaluation"
    )
    return ev.evaluate_blocks


@pytest.fixture
def world(monkeypatch):
    def forbidden_side_effect(*args, **kwargs):
        raise AssertionError("real process/signal/engine/native/NN boundary reached in focal fixtures")

    class InertTimer:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def cancel(self):
            pass

    for target in ("subprocess.Popen", "os._exit", "os.kill", "os.killpg",
                   "signal.signal", "signal.raise_signal", "signal.pthread_kill"):
        monkeypatch.setattr(target, forbidden_side_effect)
    monkeypatch.setattr("threading.Timer", InertTimer)
    monkeypatch.setattr(ev.torch.nn.Module, "__call__", forbidden_side_effect)
    for name in ("__init__", "select_action"):
        monkeypatch.setattr(native_module.NativeOpponentRunner, name, forbidden_side_effect)
    for name in ("__init__", "reset", "step"):
        monkeypatch.setattr(simulator_module.FourPlayerGame, name, forbidden_side_effect)
    state = NS(
        now=1000.0, envs=[], drivers=[], runners=[], steps=[], actor_constructions=0,
        public_calls=0, offer_calls=0, forbid_public=False, poison_offers=False,
        overrun_seat=None, raise_native=False,
    )

    class Game:
        def __init__(self, *, max_steps):
            self.max_steps = max_steps
            self.version = 0
            self.actor = 0
            self.focal = 0
            self.actions = (Action("END_TURN"), Action("BUILD_ROAD", target=9))
            self.game = NS(state=NS(colors=("blue", "red", "orange", "white"),
                                    hidden=[SECRET], ordered_deck=[SECRET]))
            state.envs.append(self)

        @property
        def current_player(self):
            return self.actor

        def reset(self, *, seed):
            self.seed = seed
            self.version = 0

        def observe(self, seat=None):
            player = self.actor if seat is None else seat
            return NS(player_id=player, legal_actions=self.actions,
                      context=NS(actor=self.actor, kind="PLAY_TURN", has_rolled=True,
                                 free_roads=0, turn_number=0, turn_owner=self.focal,
                                 version=self.version))

        def _actual_vp(self):
            return tuple(10 if seat == self.focal else 2 for seat in range(4))

    class Driver:
        def __init__(self, env, learner, seed, historical=(), **kwargs):
            self.env, self.learner, self.seed = env, learner, seed
            self.kwargs = kwargs
            self.env.focal = self.env.actor = learner
            self.native_opponents = {}
            self.trade_responses = self.offers = 0
            state.drivers.append(self)

        @property
        def view(self):
            return self.env.observe()

        def maybe_offer(self, *, deadline=None):
            state.offer_calls += 1
            if state.poison_offers:
                raise AssertionError("matched no-offer protocol reached proposal selection")

        def opponent_action(self, *, deadline=None):
            raise AssertionError("one-action fixture never has an opponent turn")

        def step(self, action):
            if not any(action is member for member in self.env.actions):
                raise ValueError("fixture requires the exact original legal member")
            state.steps.append((self.env.seed, self.learner, action))
            self.env.version += 1
            self.env.actor = (self.learner + 1) % 4
            return NS(terminated=True, truncated=False, winner=self.learner)

    class Native:
        def __init__(self, env, seat, identity, *, planning_seed):
            self.env, self.seat, self.identity = env, seat, identity
            self.planning_seed = planning_seed
            self.color = env.game.state.colors[seat]
            self.calls = 0
            self.private = env.game
            state.runners.append(self)

        def select_action(self):
            view = self.env.observe()
            if view.player_id != self.seat or view.context.actor != self.seat:
                raise ValueError("focal native bound to the wrong acting seat")
            self.calls += 1
            if state.raise_native:
                random.random()
                raise RuntimeError("native fixture decision failed")
            if self.seat == state.overrun_seat:
                state.now += 10
            return view.legal_actions[1]

        def metadata(self):
            return {
                "identity": self.identity,
                "adapter_identity": self.identity + "+atomic_discard_reject_trade_v1",
                "seat": self.seat,
                "planning_seed": self.planning_seed,
                "configuration": ({"depth": 2, "prunning": False,
                                   "value_fn_builder_name": None, "epsilon": None}
                                  if self.identity == NATIVE_IDS[1]
                                  else {"value_fn_builder_name": None, "params": None,
                                        "epsilon": None}),
                "effective_weights": "upstream_base_DEFAULT_WEIGHTS",
                "information_regime": "public_only_candidate_vs_privileged_native_opponents",
                "discard_adapter": "native_random_bulk_to_per_card",
                "trade_adapter": "reject_synthetic_trade_response",
                "proposal_support": "none",
                "root_action_order": "native_original",
                "chance_model": "native82_unmodified_Action.value_ActionRecord.result_mismatch",
                "full_rule_equivalence": False,
            }

        def metrics(self):
            return {"native_decisions": self.calls, "native_nonforced_decisions": self.calls,
                    "phase_counts": {"PLAY_TURN": self.calls},
                    "phase_planning_seconds": {"PLAY_TURN": 0.0},
                    "total_planning_seconds": 0.0, "search_nodes": None,
                    "search_expansions": None, "internal_timeout_count": None}

    class Public:
        def __init__(self, *args, **kwargs):
            state.actor_constructions += 1
            if state.forbid_public:
                raise AssertionError("native reference constructed a public focal actor")

        def act(self, view):
            state.public_calls += 1
            return view.legal_actions[0]

        def select_action(self, view):
            return self.act(view)

    class Greedy(Public):
        identity = "greedy_no_trade_fixture"

        def __init__(self, *args, **kwargs):
            pass  # Metadata identity construction is not a public focal actor.

    monkeypatch.setattr(ev, "FourPlayerGame", Game)
    monkeypatch.setattr(ev, "_MatchDriver", Driver)
    monkeypatch.setattr(ev, "NativeOpponentRunner", Native)
    monkeypatch.setattr(ev, "HeuristicPolicy", Public)
    monkeypatch.setattr(ev, "RandomPolicy", Public)
    monkeypatch.setattr(ev, "GreedyNoTrade", Greedy)
    monkeypatch.setattr(ev.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(ev.time, "perf_counter", lambda: state.now)
    state.Native = Native
    return state


def reference_args(**changes):
    return {"reference": NATIVE_IDS[0], "board_seeds": [91001],
            "opponent_lineup": BACKGROUND, "deadline": 1100.0, **changes}


def test_reference_dispatches_exact_runner_for_both_ids_and_all_focal_seats(world):
    api = reference_api()
    world.forbid_public = True
    for identity in NATIVE_IDS:
        result = api(**reference_args(reference=identity))
        assert result["status"] == "complete"
        assert result["requested_games"] == result["games"] == 4
        assert result["focal_role"] == "native_reference"
        assert result["focal_identity"] == identity
        assert result["evaluation_only"] is True
        assert result["training_labels_allowed"] is False
        assert [row["seat"] for row in result["outcomes"]] == [0, 1, 2, 3]
        latest = world.runners[-4:]
        assert [runner.seat for runner in latest] == [0, 1, 2, 3]
        assert [runner.color for runner in latest] == ["blue", "red", "orange", "white"]
        assert all(runner.identity == identity for runner in latest)
        assert [runner.planning_seed for runner in latest] == [91001 * 97 + s for s in range(4)]
        assert all(driver.kwargs["opponent_lineup"] == BACKGROUND for driver in world.drivers[-4:])
        assert all(action is runner.env.actions[1]
                   for (_, _, action), runner in zip(world.steps[-4:], latest))
    assert world.actor_constructions == world.public_calls == 0


def test_focal_native_enforces_no_offers_without_any_native_background(world):
    api = reference_api()
    world.poison_offers = True
    result = api(**reference_args())
    assert world.offer_calls == 0
    assert result["no_offers"] is True
    assert result["offers"] == 0
    assert result["native_protocol"]["domestic_trades"] == "disabled_all_seats"
    assert "focal" in result["native_protocol"]["information_regime"]
    assert all(not driver.native_opponents for driver in world.drivers)


def test_public_no_offers_is_opt_in_and_legacy_default_still_calls_proposals(world):
    api = public_no_offer_api()
    first = api(board_seeds=[91001], candidate="heuristic", opponent_lineup=BACKGROUND)
    assert world.offer_calls == 4
    assert first["no_offers"] is False
    world.poison_offers = True
    matched = api(board_seeds=[91001], candidate="heuristic", opponent_lineup=BACKGROUND,
                  no_offers=True)
    assert world.offer_calls == 4
    assert matched["no_offers"] is True
    assert matched["games"] == 4


def test_reference_rejects_invalid_identity_protocol_policy_and_label_callback_before_reset(world):
    api = reference_api()
    for changes in ({"reference": "heuristic"}, {"reference": "unknown"},
                    {"deadline": None}, {"deadline": float("nan")},
                    {"deadline": float("inf")}, {"deadline": True},
                    {"opponent_lineup": None}, {"opponent_lineup": BACKGROUND[:2]},
                    {"no_offers": False}, {"no_offers": 1}):
        with pytest.raises((ValueError, TypeError)):
            api(**reference_args(**changes))
    for forbidden in ({"policy": object()}, {"on_candidate_step": lambda step: None},
                      {"historical_policies": (object(),)}):
        with pytest.raises((ValueError, TypeError)):
            api(**reference_args(**forbidden))
    assert world.envs == world.runners == world.steps == []


def test_existing_public_legacy_and_training_native_gates_stay_closed(world):
    from colonist_rl.training import _lineup, evaluate

    for identity in NATIVE_IDS:
        with pytest.raises(ValueError, match="candidate"):
            ev.evaluate_blocks(board_seeds=[91001], candidate=identity, deadline=1100)
        with pytest.raises(ValueError, match="candidate"):
            evaluate(board_seeds=[91001], candidate=identity, deadline=1100)
        with pytest.raises(ValueError, match="opponent_lineup"):
            _lineup((identity, "random", "heuristic"))
    supplied = world.Native.__new__(world.Native)
    with pytest.raises(TypeError, match="candidate"):
        ev.evaluate_blocks(supplied, board_seeds=[91001], deadline=1100)
    assert world.envs == world.steps == []


def test_expired_reference_deadline_constructs_no_game_or_native_runner(world):
    result = reference_api()(**reference_args(deadline=1000.0))
    assert result["status"] == "incomplete"
    assert result["incomplete_reason"] == "deadline"
    assert result["requested_games"] == 4
    assert result["games"] == result["finished_games"] == 0
    assert result["outcomes"] == result["partial_outcomes"] == []
    assert result["win_rate"] is None
    assert result["focal_identity"] == NATIVE_IDS[0]
    assert world.envs == world.runners == world.steps == []


def test_native_decision_overrun_never_steps_and_retains_interrupted_focal_metrics(world):
    world.overrun_seat = 0
    result = reference_api()(**reference_args(deadline=1005.0))
    assert result["status"] == "incomplete"
    assert result["finished_games"] == 0
    assert result["outcomes"] == result["partial_outcomes"] == []
    row = result["interrupted_game"]
    assert row["board_seed"] == 91001 and row["seat"] == 0
    assert row.get("won") is None
    assert row["native_focal"]["metadata"]["identity"] == NATIVE_IDS[0]
    assert row["native_focal"]["metrics"]["native_decisions"] == 1
    assert row["native_focal"]["metrics"]["search_nodes"] is None
    assert world.steps == []


def test_mid_block_interrupt_retains_completed_callback_and_partial_outcome(world):
    world.overrun_seat = 1
    callbacks = []
    result = reference_api()(**reference_args(deadline=1005.0, on_game_end=callbacks.append))
    assert result["status"] == "incomplete"
    assert result["games"] == 0 and result["finished_games"] == 1
    assert result["outcomes"] == []
    assert [r["seat"] for r in result["partial_outcomes"]] == [0]
    assert [r["seat"] for r in callbacks] == [0]
    assert result["interrupted_game"]["seat"] == 1
    assert result["interrupted_game"]["native_focal"]["metrics"]["native_decisions"] == 1
    assert result["win_rate"] is None
    assert len(world.steps) == 1


def test_reference_callback_is_detached_primitive_diagnostics_not_private_labels(world):
    callbacks = []
    def observe(row):
        callbacks.append(json.loads(json.dumps(row)))
        row["won"] = "callback_poison"
        row["native_focal"]["metrics"]["native_decisions"] = -999

    result = reference_api()(**reference_args(on_game_end=observe))
    assert len(callbacks) == 4
    assert all(row["won"] is True for row in result["outcomes"])
    assert all(row["native_focal"]["metrics"]["native_decisions"] == 1 for row in result["outcomes"])
    for row in result["outcomes"]:
        metadata = row["native_focal"]["metadata"]
        assert metadata["seat"] == row["seat"]
        assert metadata["seat_role"] == "focal_reference"
        assert "privileged" in metadata["information_regime"] and "focal" in metadata["information_regime"]
        assert metadata["full_rule_equivalence"] is False
        assert metadata["adapter_identity"].endswith("+atomic_discard_reject_trade_v1")
        assert row.get("native_opponents", []) == []
        assert row["native_focal"]["metrics"]["internal_timeout_count"] is None
    serialized = json.dumps(result)
    assert SECRET not in serialized
    assert "teacher_action_index" not in serialized
    assert "native_root_actions" not in serialized


def test_public_candidate_step_callback_stays_public_and_keeps_exact_members(world):
    callbacks = []
    result = public_no_offer_api()(board_seeds=[91001], candidate="heuristic",
                                   opponent_lineup=BACKGROUND, no_offers=True,
                                   on_candidate_step=callbacks.append)
    assert result["games"] == 4
    assert len(callbacks) == 4
    for step, env in zip(callbacks, world.envs):
        assert step.action is env.actions[0]
        assert step.before.player_id == step.after.player_id == env.focal
        assert not hasattr(step.before, "game") and not hasattr(step.after, "game")
        assert step.terminated and not step.truncated
    assert world.runners == []


def test_native_exception_propagates_without_public_fallback_and_restores_rng(world):
    api = reference_api()
    before = random.getstate()
    world.raise_native = world.forbid_public = True
    with pytest.raises(RuntimeError, match="native fixture decision failed"):
        api(**reference_args())
    assert random.getstate() == before
    assert world.actor_constructions == world.public_calls == 0
    assert world.steps == []


def test_public_no_offers_requires_a_strict_boolean_before_game_creation(world):
    api = public_no_offer_api()
    for invalid in (None, 0, 1, "no_offers"):
        with pytest.raises((ValueError, TypeError)):
            api(board_seeds=[91001], opponent_lineup=BACKGROUND, no_offers=invalid)
    assert world.envs == world.steps == []
