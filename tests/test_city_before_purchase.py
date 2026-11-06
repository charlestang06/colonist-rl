"""One-trade public city projections test contracts, never card value or strength."""

import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from colonist_rl.board import BoardSpec, Port, RulesConfig
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, TradeOffer
from colonist_rl import public_turn_planner as p


def implementation():
    assert importlib.util.find_spec("colonist_rl.city_before_purchase") is not None, (
        "CityBeforePurchasePolicy is not implemented"
    )
    from colonist_rl.city_before_purchase import CityBeforePurchasePolicy
    return CityBeforePurchasePolicy


def trade(resource=0, ratio=4, receive=4):
    return Action("MARITIME_TRADE", give=tuple(ratio if i == resource else 0 for i in range(5)),
                  receive=tuple(int(i == receive) for i in range(5)))


def view(hand=(4, 0, 1, 2, 2)):
    board = BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,))
    return Observation(10, board, RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        (PublicPlayer(10, 1, sum(hand), 0, roads_left=0, settlements_left=0),
         PublicPlayer(20, 2, 30, 0), PublicPlayer(30, 0, 0, 0), PublicPlayer(40, 0, 0, 0)),
        hand, (0, 0, 0, 0, 0), (Building(0, 10),), (), 99,
        (Action("BUY_DEVELOPMENT_CARD"), trade(), Action("END_TURN")))


class Parent:
    def __init__(self, observation, *, config=None, **changes):
        self.config = config or p.TurnPlannerConfig()
        self.evaluator = None
        self.gain_margin_micro = 250_000
        self.calls = self.proposals = 0
        self.offer = object()
        self.result = p.TurnPlan(observation.legal_actions[0], 0, (), 0, 0, 0, 0,
                                 "unsupported_heuristic_action", "not_searched")
        self.result = replace(self.result, **changes)

    def plan(self, observation):
        self.calls += 1
        return self.result

    def propose(self, observation):
        self.proposals += 1
        return self.offer


def make(observation=None, **kwargs):
    cls = implementation()
    observation = observation or view()
    parent = Parent(observation, **kwargs)
    return observation, parent, cls(parent, enabled=True)


def forbid_extra_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("wrapper queried heuristic/planner/evaluator")
    monkeypatch.setattr(p, "plan_turn", forbidden)
    monkeypatch.setattr(p.HeuristicViewPolicy, "act", forbidden)


def test_default_disabled_preserves_exact_parent_identity_and_input(monkeypatch):
    cls = implementation()
    observation = view()
    parent = Parent(observation)
    original = deepcopy(observation)
    forbid_extra_queries(monkeypatch)
    wrapper = cls(parent)
    assert wrapper.plan(observation) is parent.result
    assert parent.calls == 1 and observation == original
    assert wrapper.last_diagnostics["enabled"] is False
    assert wrapper.last_diagnostics["nodes"] == wrapper.last_diagnostics["geometry_visits"] == 0


@pytest.mark.parametrize("enabled", [0, 1, None, "yes", [], object()])
def test_flag_is_strict_bool(enabled):
    cls = implementation()
    with pytest.raises(ValueError, match="bool"):
        cls(Parent(view()), enabled=enabled)


def test_funding_trade_city_arithmetic_identity_and_no_extra_queries(monkeypatch):
    observation, parent, wrapper = make()
    original = deepcopy(observation)
    forbid_extra_queries(monkeypatch)
    plan = wrapper.plan(observation)
    assert parent.calls == 1
    assert plan.first_action is observation.legal_actions[1]
    assert plan.root_index == 1 and plan.reason == "city_before_purchase"
    assert plan.search_stop == "one_trade_city_complete"
    assert plan.steps == (p.PlannedStep("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1)),
                          p.PlannedStep("BUILD_CITY", 0))
    work = p._Work(parent.config)
    geometry = p._Geometry(observation, work)
    start = p._initial_projection(observation)
    after_trade = p._root_transition(observation, start, plan.first_action, geometry.describe(start))
    completed = p._transition(observation, after_trade, plan.steps[1], geometry.describe(after_trade))
    assert completed.hand == (0, 0, 1, 0, 0)
    assert completed.cities_left == start.cities_left - 1
    assert completed.settlements_left == start.settlements_left + 1
    assert completed.buildings == (Building(0, 10, 2),)
    assert completed.bank_lower == tuple(x + y - z for x, y, z in zip(start.bank_lower, start.hand, completed.hand))
    assert completed.bank_upper == tuple(x + y - z for x, y, z in zip(start.bank_upper, start.hand, completed.hand))
    assert plan.evaluator_calls == plan.evaluator_time_ns == 0
    assert wrapper.last_diagnostics["residual_hand"] == (0, 0, 1, 0, 0)
    assert wrapper.last_diagnostics["vp_stratum"] == "economic"
    assert wrapper.last_diagnostics["candidate_trades"] == 1
    assert wrapper.last_diagnostics["funded_trades"] == wrapper.last_diagnostics["completed_targets"] == 1
    assert plan.nodes == wrapper.last_diagnostics["nodes"] == 2
    assert observation == original


def test_multiple_trades_prefer_more_retained_resources_and_stronger_city_site():
    implementation()
    observation = view(hand=(2, 4, 1, 2, 2))
    board = BoardSpec.from_hexes(((0, 0), (4, 0)), ("WHEAT", "WHEAT"), (2, 6))
    low, high = board.tile_vertices[0][0], board.tile_vertices[1][0]
    board = replace(board, ports=(Port(tuple(board.tile_vertices[0][:2]), "WOOD", 2),))
    observation = replace(observation, board=board, buildings=(Building(low, 10), Building(high, 10)),
        legal_actions=(observation.legal_actions[0], trade(1), trade(0, 2), Action("END_TURN")))
    _, parent, wrapper = make(observation)
    plan = wrapper.plan(observation)
    # Wood2 retains four brick; brick4 retains only two wood. Same build goals.
    # Six has five times the production probability of two on the same resource.
    assert plan.root_index == 2 and plan.steps[-1].target == high
    assert wrapper.last_diagnostics["residual_hand"] == (0, 4, 1, 0, 0)
    assert wrapper.last_diagnostics["completed_targets"] == 4 and parent.calls == 1


def test_canonical_ties_then_original_root_index_for_equal_actions():
    implementation()
    observation = view(hand=(4, 4, 1, 2, 2))
    observation = replace(observation, buildings=(Building(3, 10), Building(0, 10)),
        legal_actions=(observation.legal_actions[0], trade(0), trade(1), trade(1)))
    _, _, wrapper = make(observation)
    plan = wrapper.plan(observation)
    assert plan.root_index == 2  # Canonical give tuple brick before wood, duplicate index first.
    assert plan.first_action is observation.legal_actions[2]
    assert plan.first_action is not observation.legal_actions[3]
    assert plan.steps[-1].target == 0
    assert wrapper.last_diagnostics["completed_targets"] == 6


@pytest.mark.parametrize("case", ["no_trade", "nonfunding", "two_trades", "city_stock", "no_settlement",
    "unrolled", "nonown", "discard", "offer", "free_roads", "schema", "board_version", "rules", "port",
    "direct_city", "direct_settlement", "missing_own"])
def test_ineligible_context_stock_and_funding_return_parent_identity(case):
    implementation()
    observation = view()
    if case == "no_trade":
        observation = replace(observation, legal_actions=(observation.legal_actions[0], Action("END_TURN")))
    elif case in ("nonfunding", "two_trades"):
        hand = (8, 0, 1, 1, 2) if case == "two_trades" else (4, 0, 1, 1, 1)
        observation = view(hand)
    elif case == "city_stock":
        observation = replace(observation, players=(replace(observation.players[0], cities_left=0), *observation.players[1:]))
    elif case == "no_settlement":
        observation = replace(observation, buildings=(Building(0, 10, 2),))
    elif case in ("unrolled", "nonown", "discard", "offer", "free_roads"):
        changes = {"unrolled": {"has_rolled": False}, "nonown": {"actor": 20},
            "discard": {"discard_remaining": 1}, "free_roads": {"free_roads": 1},
            "offer": {"offer": TradeOffer("x", 10, 20, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))}}[case]
        observation = replace(observation, context=replace(observation.context, **changes))
    elif case == "schema":
        observation = replace(observation, schema_version=2)
    elif case == "board_version":
        observation = replace(observation, board=replace(observation.board, version="other"))
    elif case == "rules":
        observation = replace(observation, rules=replace(observation.rules, victory_points=11))
    elif case == "port":
        observation = replace(observation, board=replace(observation.board, ports=(Port((0, 1), "WOOD", 1),)))
    elif case.startswith("direct_"):
        kind = "BUILD_CITY" if case == "direct_city" else "BUILD_SETTLEMENT"
        observation = replace(observation, legal_actions=(*observation.legal_actions, Action(kind, 0)))
    elif case == "missing_own":
        observation = replace(observation, players=observation.players[1:])
    _, parent, wrapper = make(observation)
    assert wrapper.plan(observation) is parent.result
    assert parent.calls == 1
    assert wrapper.last_diagnostics["selected_root_index"] is None


@pytest.mark.parametrize("changes", [{"reason": "insufficient_gain"}, {"reason": "geometry_limit"},
    {"nodes": 1}, {"geometry_visits": 1}, {"evaluator_calls": 1}, {"evaluator_time_ns": 1},
    {"root_index": 1}, {"first_action": Action("BUY_DEVELOPMENT_CARD")}, {"first_action": Action("END_TURN")}])
def test_only_exact_ordinary_zero_work_buy_parent_is_eligible(changes):
    observation, parent, wrapper = make(**changes)
    assert wrapper.plan(observation) is parent.result
    assert parent.calls == 1


def test_evaluator_parent_is_rejected_without_calling_evaluator():
    observation, parent, wrapper = make()
    def forbidden(*args):
        raise AssertionError("evaluator called")
    parent.evaluator = forbidden
    assert wrapper.plan(observation) is parent.result
    assert parent.calls == 1


@pytest.mark.parametrize("known_vp,success,stratum", [(8, True, "economic"), (9, True, "near_win"), (10, False, "finished"), (11, False, "finished")])
def test_known_hidden_vp_counts_toward_before_only_boundary(known_vp, success, stratum):
    implementation()
    observation = view()
    observation = replace(observation, players=(replace(observation.players[0], victory_points=7), *observation.players[1:]),
                          development=(0, 0, 0, 0, known_vp - 7))
    _, parent, wrapper = make(observation)
    plan = wrapper.plan(observation)
    assert (plan is not parent.result) is success
    assert wrapper.last_diagnostics["known_own_vp"] == known_vp
    assert wrapper.last_diagnostics["vp_stratum"] == stratum


@pytest.mark.parametrize("case", ["bad_ratio", "extra_field", "insufficient_give", "bank", "resource_count"])
def test_malformed_public_contract_fails_closed_even_after_good_candidate(case):
    implementation()
    observation = view()
    if case in ("bad_ratio", "extra_field", "insufficient_give"):
        bad = trade(ratio=3) if case == "bad_ratio" else replace(trade(), resource="WOOD") if case == "extra_field" else trade(1)
        observation = replace(observation, legal_actions=(*observation.legal_actions, bad))
    elif case == "bank":
        observation = replace(observation, bank=(19, 19, 19, 19, 19))
    else:
        observation = replace(observation, players=(replace(observation.players[0], resource_count=99), *observation.players[1:]))
    _, parent, wrapper = make(observation)
    assert wrapper.plan(observation) is parent.result
    assert wrapper.last_diagnostics["status"] == "public_contract_mismatch"
    if case in ("bad_ratio", "extra_field", "insufficient_give"):
        assert wrapper.last_diagnostics["nodes"] > 0
        assert wrapper.last_diagnostics["geometry_visits"] > 0


def test_partial_bank_duplicate_menu_evidence_never_sums_withdrawals():
    implementation()
    observation = view()
    observation = replace(observation, legal_actions=(*observation.legal_actions, trade()))
    initial = p._initial_projection(observation)
    assert initial.bank_lower[4] == 1
    _, _, wrapper = make(observation)
    assert wrapper.plan(observation).reason == "city_before_purchase"
    assert wrapper.last_diagnostics["candidate_trades"] == 2
    assert wrapper.last_diagnostics["completed_targets"] == 2


@pytest.mark.parametrize("cap", [0, 1, 3])
def test_node_cap_requires_full_completion_even_after_partial_winner(cap):
    implementation()
    observation = view()
    observation = replace(observation, legal_actions=(*observation.legal_actions, trade()))
    _, parent, wrapper = make(observation, config=p.TurnPlannerConfig(max_nodes=cap))
    assert wrapper.plan(observation) is parent.result
    assert wrapper.last_diagnostics["status"] == "node_limit"
    assert wrapper.last_diagnostics["nodes"] == cap
    assert wrapper.last_diagnostics["selected_root_index"] is None
    if cap == 3:
        assert wrapper.last_diagnostics["completed_targets"] == 1
        assert wrapper.last_diagnostics["geometry_visits"] > 0


def test_geometry_zero_and_partial_caps_and_full_boundary():
    observation, parent, wrapper = make()
    assert wrapper.plan(observation).reason == "city_before_purchase"
    full = wrapper.last_diagnostics["geometry_visits"]
    for cap in (0, full - 1, full):
        _, parent, capped = make(observation, config=p.TurnPlannerConfig(max_geometry_visits=cap))
        result = capped.plan(observation)
        assert (result is parent.result) is (cap < full)
        assert capped.last_diagnostics["geometry_visits"] == cap
        if cap < full:
            assert capped.last_diagnostics["status"] == "geometry_limit"


@pytest.mark.parametrize("depth", [0, 1])
def test_depth_below_two_abstains(depth):
    observation, parent, wrapper = make(config=p.TurnPlannerConfig(max_depth=depth))
    assert wrapper.plan(observation) is parent.result
    assert wrapper.last_diagnostics["nodes"] == 0


def test_declared_margin_is_root_quality_floor():
    observation, parent, wrapper = make()
    parent.gain_margin_micro = 10**15
    assert wrapper.plan(observation) is parent.result
    assert wrapper.last_diagnostics["completed_targets"] == 1
    assert wrapper.last_diagnostics["status"] == "insufficient_gain"


def test_unexpected_projection_and_parent_errors_propagate(monkeypatch):
    cls = implementation()
    from colonist_rl import city_before_purchase as module
    observation, parent, wrapper = make()
    def broken(*args):
        raise RuntimeError("programming fault")
    monkeypatch.setattr(module, "_root_transition", broken)
    with pytest.raises(RuntimeError, match="programming fault"):
        wrapper.plan(observation)
    assert parent.calls == 1 and wrapper.last_diagnostics["nodes"] == 1
    assert wrapper.last_diagnostics["geometry_visits"] > 0
    parent.plan = broken
    with pytest.raises(RuntimeError, match="programming fault"):
        cls(parent, enabled=True).plan(observation)


def test_diagnostics_reset_act_replans_and_propose_delegates_once():
    observation, parent, wrapper = make()
    assert wrapper.act(observation) is observation.legal_actions[1]
    first = wrapper.last_diagnostics.copy()
    changed = replace(observation, legal_actions=(observation.legal_actions[0], Action("END_TURN")))
    assert wrapper.act(changed) is parent.result.first_action
    assert parent.calls == 2
    assert wrapper.last_diagnostics["nodes"] == 0
    assert wrapper.last_diagnostics["selected_root_index"] is None
    assert first["selected_root_index"] == 1
    assert wrapper.propose(changed) is parent.offer
    assert parent.proposals == 1 and parent.calls == 2


def test_exact_train_fixture_envelope_and_real_public_funding_route():
    implementation()
    from colonist_rl.adapters.observation_io import observation_from_dict
    envelope = json.loads((Path(__file__).parent / "fixtures/train001-seat0-own55-before.json").read_text())
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert sha256(canonical).hexdigest() == "6e2e0ebf58b7c0d66627c71a21b6d9584c7efcc7cdce91c42cfced1bf57bcdc6"
    observation = observation_from_dict(envelope)
    buy_index = next(i for i, a in enumerate(observation.legal_actions) if a.kind == "BUY_DEVELOPMENT_CARD")
    parent = Parent(observation, first_action=observation.legal_actions[buy_index], root_index=buy_index)
    wrapper = implementation()(parent, enabled=True)
    plan = wrapper.plan(observation)
    assert plan.reason == "city_before_purchase"
    assert plan.first_action is observation.legal_actions[plan.root_index]
    assert plan.first_action.give == (0, 0, 0, 4, 0)
    assert plan.first_action.receive == (0, 0, 0, 0, 1)
    assert plan.steps[-1].target in (46, 49)
    assert wrapper.last_diagnostics["vp_stratum"] == "economic"


@pytest.mark.parametrize("name,value", [("max_nodes", -1), ("max_nodes", True), ("max_depth", -1),
    ("max_geometry_visits", 1.5), ("beam_width", 0), ("seed", False)])
def test_instrumented_parent_invalid_config_is_rejected_at_construction(name, value):
    cls = implementation()
    from dataclasses import asdict
    from types import SimpleNamespace
    parent = Parent(view())
    values = asdict(parent.config)
    values[name] = value
    parent.config = SimpleNamespace(**values)
    with pytest.raises(ValueError):
        cls(parent, enabled=True)


def test_expected_contract_exception_preserves_consumed_work(monkeypatch):
    implementation()
    from colonist_rl import city_before_purchase as module
    observation, parent, wrapper = make()
    def rejected(*args):
        raise p._PublicContract("test public contract")
    monkeypatch.setattr(module, "_root_transition", rejected)
    assert wrapper.plan(observation) is parent.result
    assert wrapper.last_diagnostics["status"] == "public_contract_mismatch"
    assert wrapper.last_diagnostics["nodes"] == 1
    assert wrapper.last_diagnostics["geometry_visits"] > 0
