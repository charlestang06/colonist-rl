"""Exact accounting for the planner's public, bounded search."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from colonist_rl import public_turn_planner as planner
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer


def _trade(give: int, receive: int, *, ratio: int = 4) -> Action:
    return Action(
        "MARITIME_TRADE",
        give=tuple(ratio if i == give else 0 for i in range(5)),
        receive=tuple(int(i == receive) for i in range(5)),
    )


def _view(hand: tuple[int, ...], actions: tuple[Action, ...]) -> Observation:
    rules = RulesConfig.standard()
    return Observation(
        10,
        BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,)),
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
        (),
        99,
        actions,
        bank=tuple(supply - held for supply, held in zip(rules.bank_resources, hand)),
    )


def _counts(plan: planner.TurnPlan) -> dict[str, int]:
    assert plan.search_diagnostics is not None
    counts = asdict(plan.search_diagnostics)
    assert counts["root_attempts"] + counts["future_attempts"] == plan.nodes
    assert counts["feasible_transitions"] + counts["invalid_transitions"] == plan.nodes
    assert counts["unique_successors"] + counts["duplicate_successors"] == counts[
        "feasible_transitions"
    ]
    return counts


def _commuting_trades(_view, state, _shape):
    if state.hand[0] >= 4:
        yield planner.PlannedStep(
            "MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)
        )
    if state.hand[1] >= 4:
        yield planner.PlannedStep(
            "MARITIME_TRADE", give=(0, 4, 0, 0, 0), receive=(0, 0, 0, 0, 1)
        )


def test_old_positional_turn_plan_constructor_keeps_optional_immutable_diagnostics():
    legacy = planner.TurnPlan(
        Action("END_TURN"), 0, (), 0, 0, 0, 0, "fallback", "not_searched", 0, 0
    )
    assert legacy.search_diagnostics is None
    diagnostics = planner.SearchDiagnostics()
    assert legacy == replace(legacy, search_diagnostics=diagnostics)
    with pytest.raises(FrozenInstanceError):
        diagnostics.root_attempts = 1


def test_pre_search_fallback_has_zeroed_search_diagnostics():
    observation = _view((0, 0, 0, 0, 0), (Action("END_TURN"),))
    observation = replace(
        observation, context=replace(observation.context, has_rolled=False)
    )
    plan = planner.plan_turn(observation)
    assert plan.reason == "unsupported_context"
    assert _counts(plan) == dict.fromkeys(
        (
            "root_attempts", "future_attempts", "feasible_transitions", "invalid_transitions",
            "unique_successors", "duplicate_successors", "max_feasible_depth",
            "selected_plan_depth", "beam_truncations", "pruned_invalid_candidates",
        ),
        0,
    )


def test_root_node_cap_counts_the_attempted_projection_before_validation_stops():
    observation = _view((4, 0, 0, 0, 0), (_trade(0, 2), Action("END_TURN")))
    plan = planner.plan_turn(observation, planner.TurnPlannerConfig(max_nodes=1))
    assert (plan.reason, plan.search_stop, plan.nodes) == ("root_validation_limit", "node_limit", 1)
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 0,
        "feasible_transitions": 1,
        "invalid_transitions": 0,
        "unique_successors": 1,
        "duplicate_successors": 0,
        "max_feasible_depth": 1,
        "selected_plan_depth": 0,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }


def test_invalid_future_candidate_consumes_the_last_available_node():
    observation = _view((4, 0, 0, 0, 0), (_trade(0, 2),))
    plan = planner.plan_turn(observation, planner.TurnPlannerConfig(
        max_nodes=2, max_depth=3, prune_invalid_future=False,
    ))
    assert (plan.search_stop, plan.nodes) == ("node_limit", 2)
    # The first continuation is an unaffordable city at vertex 0.
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 1,
        "feasible_transitions": 1,
        "invalid_transitions": 1,
        "unique_successors": 1,
        "duplicate_successors": 0,
        "max_feasible_depth": 1,
        "selected_plan_depth": 0,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }


def test_invalid_root_projection_is_charged_before_contract_fallback():
    observation = _view((4, 0, 0, 0, 0), (_trade(0, 2, ratio=3),))
    plan = planner.plan_turn(observation)
    assert plan.reason == "public_contract_mismatch"
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 0,
        "feasible_transitions": 0,
        "invalid_transitions": 1,
        "unique_successors": 0,
        "duplicate_successors": 0,
        "max_feasible_depth": 0,
        "selected_plan_depth": 0,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }


def test_commuting_futures_share_one_successor_at_the_same_depth(monkeypatch):
    monkeypatch.setattr(planner, "_candidates", _commuting_trades)
    observation = _view((8, 4, 0, 0, 0), (_trade(0, 2),))
    plan = planner.plan_turn(observation, planner.TurnPlannerConfig(max_depth=3, beam_width=2))
    assert plan.nodes == 5
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 4,
        "feasible_transitions": 5,
        "invalid_transitions": 0,
        "unique_successors": 4,
        "duplicate_successors": 1,
        "max_feasible_depth": 3,
        "selected_plan_depth": 0,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }


def test_beam_truncations_count_successors_dropped_by_width(monkeypatch):
    monkeypatch.setattr(planner, "_candidates", _commuting_trades)
    observation = _view((8, 4, 0, 0, 0), (_trade(0, 2),))
    plan = planner.plan_turn(observation, planner.TurnPlannerConfig(max_depth=2, beam_width=1))
    assert plan.nodes == 3
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 2,
        "feasible_transitions": 3,
        "invalid_transitions": 0,
        "unique_successors": 3,
        "duplicate_successors": 0,
        "max_feasible_depth": 2,
        "selected_plan_depth": 0,
        "beam_truncations": 1,
        "pruned_invalid_candidates": 0,
    }


def test_end_turn_only_reaches_an_empty_beam_after_one_feasible_root():
    observation = _view((0, 0, 0, 0, 0), (Action("END_TURN"),))
    plan = planner.plan_turn(observation, planner.TurnPlannerConfig(max_depth=2))
    assert plan.search_stop == "empty_beam"
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 0,
        "feasible_transitions": 1,
        "invalid_transitions": 0,
        "unique_successors": 1,
        "duplicate_successors": 0,
        "max_feasible_depth": 1,
        "selected_plan_depth": 0,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }


def test_geometry_cap_after_projection_preserves_feasible_root_accounting():
    observation = _view((4, 0, 0, 0, 0), (_trade(0, 2),))
    work = planner._Work(planner.TurnPlannerConfig(max_geometry_visits=1_000_000))
    geometry = planner._Geometry(observation, work)
    geometry.describe(planner._initial_projection(observation))
    plan = planner.plan_turn(
        observation, planner.TurnPlannerConfig(max_geometry_visits=work.geometry_visits)
    )
    assert (plan.reason, plan.search_stop) == ("geometry_limit", "geometry_limit")
    assert plan.geometry_visits == work.geometry_visits
    assert _counts(plan) == {
        "root_attempts": 1,
        "future_attempts": 0,
        "feasible_transitions": 1,
        "invalid_transitions": 0,
        "unique_successors": 1,
        "duplicate_successors": 0,
        "max_feasible_depth": 1,
        "selected_plan_depth": 0,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }


def test_selected_plan_depth_is_one_for_a_direct_city():
    city = Action("BUILD_CITY", 0)
    observation = _view((0, 0, 0, 2, 3), (city, Action("END_TURN")))
    observation = replace(
        observation,
        development=(0, 0, 0, 0, 1),
        players=(
            replace(observation.players[0], victory_points=8, development_count=1),
            *observation.players[1:],
        ),
    )
    plan = planner.plan_turn(observation, planner.TurnPlannerConfig(max_depth=1))
    assert plan.reason == "planned"
    assert plan.first_action is city
    assert _counts(plan) == {
        "root_attempts": 2,
        "future_attempts": 0,
        "feasible_transitions": 2,
        "invalid_transitions": 0,
        "unique_successors": 2,
        "duplicate_successors": 0,
        "max_feasible_depth": 1,
        "selected_plan_depth": 1,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 0,
    }
