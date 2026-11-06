"""Public option comparator contracts; pure fixtures only, never game outcomes."""

import importlib.util
from copy import deepcopy
from dataclasses import replace

import pytest

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, Road
from colonist_rl.public_turn_planner import PlannedStep, TurnPlan, TurnPlannerConfig


def implementation():
    assert importlib.util.find_spec("colonist_rl.public_option_purchase") is not None, (
        "PublicOptionPurchasePolicy is not implemented"
    )
    from colonist_rl.public_option_purchase import PublicOptionPurchasePolicy
    return PublicOptionPurchasePolicy


def trade(give=0, receive=4, ratio=4):
    return Action(
        "MARITIME_TRADE",
        give=tuple(ratio if i == give else 0 for i in range(5)),
        receive=tuple(int(i == receive) for i in range(5)),
    )


def view(*, vp=4, hand=(4, 0, 1, 2, 2), actions=None, settlements_left=0,
         cities_left=4, development=(0, 0, 0, 0, 0), board=None,
         buildings=None, roads=(), roads_left=0, context=None, rules=None):
    board = board or BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,))
    if actions is None:
        actions = (Action("BUY_DEVELOPMENT_CARD"), trade(), Action("END_TURN"))
    if buildings is None:
        buildings = (Building(0, 10),)
    context = context or DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True)
    return Observation(
        10, board, rules or RulesConfig.standard(), context,
        (PublicPlayer(10, vp, sum(hand), sum(development), roads_left=roads_left,
                      settlements_left=settlements_left, cities_left=cities_left),
         PublicPlayer(20, 2, 30, 0), PublicPlayer(30, 0, 0, 0),
         PublicPlayer(40, 0, 0, 0)),
        hand, development, buildings, roads, 99, actions,
    )


class Parent:
    def __init__(self, observation, *, config=None, **changes):
        self.config = config or TurnPlannerConfig()
        self.evaluator = None
        self.gain_margin_micro = 10_000
        self.calls = self.proposals = 0
        self.offer = object()
        self.result = replace(TurnPlan(
            observation.legal_actions[0], 0, (), 0, 0, 0, 0,
            "unsupported_heuristic_action", "not_searched",
        ), **changes)

    def plan(self, observation):
        self.calls += 1
        return self.result

    def propose(self, observation):
        self.proposals += 1
        return self.offer


def test_default_disabled_returns_exact_parent_plan_without_mutating_view():
    observation = view()
    original = deepcopy(observation)
    parent = Parent(observation)
    policy = implementation()(parent)
    assert policy.plan(observation) is parent.result
    assert parent.calls == 1
    assert observation == original
    assert policy.last_diagnostics["status"] == "disabled"


@pytest.mark.parametrize("value", (0, 1, None, "yes", [], object()))
def test_enabled_requires_a_real_boolean(value):
    observation = view()
    with pytest.raises(ValueError, match="bool"):
        implementation()(Parent(observation), enabled=value)


def test_early_buy_preserved_in_a_city_funding_menu_by_card_option_credit():
    observation = view(vp=2)
    original = deepcopy(observation)
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    plan = policy.plan(observation)
    assert plan is parent.result
    assert parent.calls == 1 and observation == original
    assert policy.last_diagnostics["status"] == "buy_preferred"
    assert policy.last_diagnostics["card_option_credit_micro"] == 123_200_000
    assert policy.last_diagnostics["buy_hand_after_cost"] == (4, 0, 0, 1, 1)
    assert policy.last_diagnostics["pass_reserve_micro"] == 70_000_000
    assert policy.last_diagnostics["completed_targets"] == 1


def test_late_pass_preserves_city_resources_when_no_trade_can_build():
    actions = (Action("BUY_DEVELOPMENT_CARD"), Action("END_TURN"))
    observation = view(vp=8, hand=(0, 0, 1, 1, 3), actions=actions)
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    plan = policy.plan(observation)
    assert plan is not parent.result
    assert plan.first_action is actions[1] and plan.root_index == 1
    assert plan.reason == "public_option_pass"
    assert plan.steps == (PlannedStep("END_TURN"),)
    assert policy.last_diagnostics["card_option_credit_micro"] == 61_600_000
    assert policy.last_diagnostics["pass_reserve_micro"] == 70_000_000
    assert policy.last_diagnostics["buy_hand_after_cost"] == (0, 0, 0, 0, 2)


def test_late_trade_city_beats_both_card_and_pass_without_demoting_early_buy():
    observation = view(vp=8)
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    plan = policy.plan(observation)
    assert plan is not parent.result
    assert plan.first_action is observation.legal_actions[1]
    assert plan.root_index == 1 and plan.reason == "public_option_trade_build"
    assert plan.steps == (
        PlannedStep("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1)),
        PlannedStep("BUILD_CITY", 0),
    )
    assert policy.last_diagnostics["selected_build_kind"] == "BUILD_CITY"
    assert policy.last_diagnostics["selected_build_target"] == 0
    assert policy.last_diagnostics["residual_hand"] == (0, 0, 1, 0, 0)
    assert policy.last_diagnostics["score_trade_build_micro"] > max(
        policy.last_diagnostics["score_buy_micro"],
        policy.last_diagnostics["score_pass_micro"],
    )


def test_one_trade_settlement_uses_a_connected_public_site_and_original_root():
    board = BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,))
    roads = (Road(0, 10), Road(1, 10))
    actions = (Action("BUY_DEVELOPMENT_CARD"), trade(give=4, receive=0),
               Action("END_TURN"))
    observation = view(
        vp=8, hand=(0, 1, 1, 1, 4), actions=actions, board=board,
        cities_left=0, settlements_left=1, roads=roads, roads_left=13,
    )
    parent = Parent(observation)
    plan = implementation()(parent, enabled=True).plan(observation)
    assert plan.first_action is actions[1] and plan.root_index == 1
    assert plan.steps == (
        PlannedStep("MARITIME_TRADE", give=(0, 0, 0, 0, 4), receive=(1, 0, 0, 0, 0)),
        PlannedStep("BUILD_SETTLEMENT", 2),
    )
    assert parent.calls == 1


def test_duplicate_equal_trade_roots_break_ties_by_original_menu_index():
    first, second = trade(), trade()
    assert first == second and first is not second
    actions = (Action("BUY_DEVELOPMENT_CARD"), first, second, Action("END_TURN"))
    observation = view(vp=8, actions=actions)
    plan = implementation()(Parent(observation), enabled=True).plan(observation)
    assert plan.first_action is first and plan.root_index == 1


def test_currently_legal_known_winning_city_is_selected_before_purchase():
    actions = (Action("BUY_DEVELOPMENT_CARD"), Action("BUILD_CITY", target=0),
               Action("END_TURN"))
    observation = view(vp=8, development=(0, 0, 0, 0, 1),
                       hand=(0, 0, 1, 2, 3), actions=actions)
    parent = Parent(observation)
    plan = implementation()(parent, enabled=True).plan(observation)
    assert plan.first_action is actions[1] and plan.root_index == 1
    assert plan.reason == "public_option_legal_win"


def test_malformed_current_winning_build_root_abstains_instead_of_executing_it():
    actions = (Action("BUY_DEVELOPMENT_CARD"), Action("BUILD_CITY", target=None),
               Action("END_TURN"))
    observation = view(vp=9, hand=(0, 0, 1, 2, 3), actions=actions)
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    assert policy.plan(observation) is parent.result
    assert policy.last_diagnostics["status"] == "public_contract_mismatch"


def test_direct_nonwinning_build_menu_abstains_from_narrow_comparison():
    actions = (Action("BUY_DEVELOPMENT_CARD"), Action("BUILD_CITY", target=0),
               Action("END_TURN"))
    observation = view(vp=4, hand=(0, 0, 1, 2, 3), actions=actions)
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    assert policy.plan(observation) is parent.result
    assert policy.last_diagnostics["status"] == "direct_build_available"


def test_incomplete_scan_and_bad_bank_contract_preserve_exact_parent_plan():
    observation = view(vp=8)
    parent = Parent(observation, config=TurnPlannerConfig(max_nodes=2))
    policy = implementation()(parent, enabled=True)
    assert policy.plan(observation) is parent.result
    assert policy.last_diagnostics["status"] == "node_limit"
    geometry_parent = Parent(observation, config=TurnPlannerConfig(max_geometry_visits=0))
    geometry_policy = implementation()(geometry_parent, enabled=True)
    assert geometry_policy.plan(observation) is geometry_parent.result
    assert geometry_policy.last_diagnostics["status"] == "geometry_limit"
    bad = view(vp=8, actions=(Action("BUY_DEVELOPMENT_CARD"), trade(ratio=3),
                              Action("END_TURN")))
    bad_parent = Parent(bad)
    bad_policy = implementation()(bad_parent, enabled=True)
    assert bad_policy.plan(bad) is bad_parent.result
    assert bad_policy.last_diagnostics["status"] == "public_contract_mismatch"


def test_node_cap_after_finding_one_route_does_not_publish_partial_best():
    actions = (Action("BUY_DEVELOPMENT_CARD"), trade(), trade(),
               Action("END_TURN"))
    observation = view(vp=8, actions=actions)
    parent = Parent(observation, config=TurnPlannerConfig(max_nodes=4))
    policy = implementation()(parent, enabled=True)
    assert policy.plan(observation) is parent.result
    assert policy.last_diagnostics["status"] == "node_limit"
    assert policy.last_diagnostics["completed_targets"] == 1


def test_missing_current_player_in_public_roster_abstains():
    observation = view(vp=8)
    observation = replace(observation, players=observation.players[1:])
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    assert policy.plan(observation) is parent.result
    assert policy.last_diagnostics["status"] == "missing_own_player"


def test_unsupported_parent_and_context_abstain_without_projection():
    observation = view()
    parent = Parent(observation, reason="planned")
    policy = implementation()(parent, enabled=True)
    assert policy.plan(observation) is parent.result
    assert policy.last_diagnostics["status"] == "ineligible_parent"
    unrolled = view(context=DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=False))
    parent = Parent(unrolled)
    policy = implementation()(parent, enabled=True)
    assert policy.plan(unrolled) is parent.result
    assert policy.last_diagnostics["status"] == "unsupported_context"


def test_act_and_propose_delegate_through_public_policy_surface():
    observation = view(vp=8)
    parent = Parent(observation)
    policy = implementation()(parent, enabled=True)
    assert policy.act(observation) is observation.legal_actions[1]
    assert policy.propose(observation) is parent.offer
    assert parent.calls == parent.proposals == 1
    assert policy.last_diagnostics["status"] == "parent_proposal"
