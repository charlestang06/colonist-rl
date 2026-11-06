"""Opt-in future pruning may save attempts, but cannot remove a feasible projection."""

from dataclasses import replace

import pytest

from colonist_rl import public_turn_planner as planner
from colonist_rl.board import BoardSpec, Port, RulesConfig
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, Road


def _trade(give: int, receive: int, ratio: int) -> Action:
    return Action(
        "MARITIME_TRADE",
        give=tuple(ratio if index == give else 0 for index in range(5)),
        receive=tuple(int(index == receive) for index in range(5)),
    )


def _view(hand, *, roads=(), ratio=4, actions=None) -> Observation:
    rules = RulesConfig.standard()
    board = BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,))
    if ratio == 2:
        board = replace(board, ports=(Port((0, 1), "WOOD", 2),))
    elif ratio == 3:
        board = replace(board, ports=(Port((0, 1), None, 3),))
    return Observation(
        10,
        board,
        rules,
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        (
            PublicPlayer(10, 1, sum(hand), 0),
            PublicPlayer(20, 0, 0, 0),
            PublicPlayer(30, 0, 0, 0),
            PublicPlayer(40, 0, 0, 0),
        ),
        hand,
        (0, 0, 0, 0, 0),
        (Building(0, 10),),
        tuple(Road(edge, 10) for edge in roads),
        99,
        (Action("END_TURN"),) if actions is None else actions,
        bank=tuple(supply - held for supply, held in zip(rules.bank_resources, hand)),
    )


def _state_shape(view):
    state = planner._initial_projection(view)
    work = planner._Work(planner.TurnPlannerConfig())
    shape = planner._Geometry(view, work).describe(state)
    return state, shape


def test_pruning_is_disabled_by_default_and_requires_a_bool():
    assert planner.TurnPlannerConfig().prune_invalid_future is False
    assert planner.TurnPlannerConfig(prune_invalid_future=True).prune_invalid_future is True
    with pytest.raises(ValueError, match="prune_invalid_future must be a bool"):
        planner.TurnPlannerConfig(prune_invalid_future=1)


@pytest.mark.parametrize(
    ("hand", "roads", "kind"),
    (
        ((0, 0, 0, 2, 3), (), "BUILD_CITY"),
        ((1, 1, 1, 1, 0), (0, 1), "BUILD_SETTLEMENT"),
        ((1, 1, 0, 0, 0), (), "BUILD_ROAD"),
    ),
)
def test_all_feasible_generated_builds_survive_the_necessary_filter(hand, roads, kind):
    view = _view(hand, roads=roads)
    state, shape = _state_shape(view)
    candidates = list(planner._candidates(view, state, shape))
    feasible_builds = [
        step for step in candidates
        if step.kind == kind and planner._transition(view, state, step, shape) is not None
    ]
    assert feasible_builds
    assert all(not planner._future_certainly_invalid(state, step) for step in feasible_builds)
    assert all(
        planner._transition(view, state, step, shape) is None
        for step in candidates if planner._future_certainly_invalid(state, step)
    )


@pytest.mark.parametrize("ratio", (2, 3, 4))
def test_all_feasible_maritime_ratios_survive_the_necessary_filter(ratio):
    view = _view((ratio, 0, 0, 0, 0), ratio=ratio)
    state, shape = _state_shape(view)
    candidates = list(planner._candidates(view, state, shape))
    expected = planner.PlannedStep(
        "MARITIME_TRADE", give=_trade(0, 1, ratio).give,
        receive=_trade(0, 1, ratio).receive,
    )
    assert expected in candidates
    assert planner._transition(view, state, expected, shape) is not None
    assert not planner._future_certainly_invalid(state, expected)
    assert all(
        not planner._future_certainly_invalid(state, step)
        for step in candidates if planner._transition(view, state, step, shape) is not None
    )


@pytest.mark.parametrize("ratio", (2, 3, 4))
def test_zero_bank_lower_prunes_a_trade_despite_an_affordable_give(ratio):
    view = _view((ratio, 0, 0, 0, 0), ratio=ratio)
    state, shape = _state_shape(view)
    state = replace(
        state,
        bank_lower=tuple(0 if index == 1 else count
                         for index, count in enumerate(state.bank_lower)),
    )
    step = planner.PlannedStep(
        "MARITIME_TRADE", give=_trade(0, 1, ratio).give,
        receive=_trade(0, 1, ratio).receive,
    )
    assert step in tuple(planner._candidates(view, state, shape))
    assert state.hand[0] == ratio and state.bank_upper[1] > 0
    assert planner._future_certainly_invalid(state, step)
    assert planner._transition(view, state, step, shape) is None


def test_uncapped_pruning_preserves_root_plan_score_and_feasible_coverage():
    actions = (_trade(0, 2, 4), Action("BUILD_ROAD", 0), Action("END_TURN"))
    view = _view((8, 4, 0, 0, 0), actions=actions)
    base = planner.TurnPlannerConfig(max_depth=3, max_nodes=10_000)
    original = planner.plan_turn(view, base)
    pruned = planner.plan_turn(view, replace(base, prune_invalid_future=True))
    assert original.search_stop != "node_limit"
    assert pruned.search_stop == original.search_stop
    assert (pruned.first_action, pruned.root_index, pruned.steps) == (
        original.first_action, original.root_index, original.steps
    )
    assert pruned.first_action is original.first_action
    assert (
        pruned.score_before_micro, pruned.score_after_micro, pruned.reason,
        pruned.geometry_visits, pruned.evaluator_calls,
    ) == (
        original.score_before_micro, original.score_after_micro, original.reason,
        original.geometry_visits, original.evaluator_calls,
    )
    old, new = original.search_diagnostics, pruned.search_diagnostics
    assert old is not None and new is not None
    assert new.pruned_invalid_candidates > 0
    assert (new.root_attempts, new.feasible_transitions, new.unique_successors,
            new.duplicate_successors, new.max_feasible_depth, new.selected_plan_depth,
            new.beam_truncations) == (
        old.root_attempts, old.feasible_transitions, old.unique_successors,
        old.duplicate_successors, old.max_feasible_depth, old.selected_plan_depth,
        old.beam_truncations,
    )
    assert pruned.nodes < original.nodes
    assert new.root_attempts + new.future_attempts == pruned.nodes
    assert new.feasible_transitions + new.invalid_transitions == pruned.nodes
    assert new.unique_successors + new.duplicate_successors == new.feasible_transitions


def test_capped_pruning_changes_feasible_coverage_without_changing_root_budget():
    root = _trade(0, 2, 4)
    view = _view((8, 0, 0, 0, 0), actions=(root,))
    base = planner.TurnPlannerConfig(max_nodes=2, max_depth=2)
    original = planner.plan_turn(view, base)
    pruned = planner.plan_turn(view, replace(base, prune_invalid_future=True))
    old, new = original.search_diagnostics, pruned.search_diagnostics
    assert old is not None and new is not None
    assert original.first_action is pruned.first_action is root
    assert original.search_stop == pruned.search_stop == "node_limit"
    assert original.nodes == pruned.nodes == 2
    assert old.root_attempts == new.root_attempts == 1
    assert old.future_attempts == new.future_attempts == 1
    assert old.feasible_transitions == 1 and old.invalid_transitions == 1
    assert new.feasible_transitions == 2 and new.invalid_transitions == 0
    assert new.pruned_invalid_candidates > 0
    assert new.unique_successors + new.duplicate_successors == new.feasible_transitions


def test_root_contract_validation_is_unchanged_when_future_pruning_is_enabled():
    invalid_root = _trade(0, 2, 3)
    view = _view((4, 0, 0, 0, 0), actions=(invalid_root,))
    original = planner.plan_turn(view)
    pruned = planner.plan_turn(view, planner.TurnPlannerConfig(prune_invalid_future=True))
    assert original.first_action is pruned.first_action is invalid_root
    assert original.reason == pruned.reason == "public_contract_mismatch"
    assert original.nodes == pruned.nodes == 1
    assert original.search_diagnostics is not None
    assert pruned.search_diagnostics is not None
    assert pruned.search_diagnostics.root_attempts == 1
    assert pruned.search_diagnostics.invalid_transitions == 1
    assert pruned.search_diagnostics.pruned_invalid_candidates == 0
