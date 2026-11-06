"""Paired public gameplay accounting with synthetic evaluation results only."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from colonist_rl.contracts import Action
from colonist_rl import paired_gameplay as paired


def _view(seat, version, actions, *, hand=(1, 2, 3, 4, 5), buildings=()):
    return NS(
        player_id=seat,
        context=NS(version=version),
        legal_actions=actions,
        hand=hand,
        own_victory_points=4,
        buildings=buildings,
    )


def _row(seed, seat, kinds, *, won=False, truncated=False):
    return {
        "game": seat,
        "board_seed": seed,
        "seat": seat,
        "won": None if truncated else won,
        "winner": None if truncated else (seat if won else (seat + 1) % 4),
        "victory_points": 8 if won else 5,
        "terminated": not truncated,
        "truncated": truncated,
        "steps": 7,
        "trade_responses": 0,
        "offers": 0,
        "learner_decisions": sum(kinds.values()),
        "non_forced_decisions": sum(kinds.values()),
        "phase_counts": {"PLAY_TURN": sum(kinds.values())},
        "action_kind_counts": dict(kinds),
        "decision_latency_ms_mean": 0.1,
        "decision_latency_ms_p95": 0.1,
    }


class Planner:
    def __init__(self, action_kind):
        self.action_kind = action_kind
        self.calls = 0

    def plan(self, view):
        self.calls += 1
        index = next(i for i, action in enumerate(view.legal_actions)
                     if action.kind == self.action_kind)
        return NS(
            first_action=view.legal_actions[index],
            root_index=index,
            reason="planned",
            search_stop="complete",
            nodes=3,
            geometry_visits=5,
            score_before_micro=100,
            score_after_micro=200,
        )


def test_fresh_policy_per_arm_board_and_complete_four_seat_accounting(monkeypatch):
    instances, calls, persisted = [], [], []

    def factory(kind):
        def make():
            policy = Planner(kind)
            instances.append(policy)
            return policy
        return make

    def evaluate(policy, **kwargs):
        calls.append(kwargs)
        assert kwargs["candidate"] == "graph"
        assert kwargs["no_offers"] is True
        assert kwargs["on_candidate_step"] is None
        assert kwargs["opponent_lineup"] == ("heuristic",) * 3
        seed = kwargs["board_seeds"][0]
        assert len(kwargs["board_seeds"]) == 1
        rows = []
        for seat in range(4):
            actions = (Action("END_TURN"), Action("BUILD_ROAD", target=9))
            action, memory = policy.act(_view(seat, seat, actions), deterministic=True)
            assert action is next(a for a in actions if a.kind == policy.policy.action_kind)
            assert memory is None
            row = _row(seed, seat, {action.kind: 1}, won=seat == 0)
            if kwargs["on_game_end"] is not None:
                kwargs["on_game_end"](deepcopy(row))
            rows.append(row)
        return {"status": "complete", "incomplete_reason": None, "outcomes": rows,
                "partial_outcomes": [], "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel(
        {"plain": factory("END_TURN"), "road": factory("BUILD_ROAD")},
        board_seeds=[31, 47], on_game_end=persisted.append,
    )
    assert result["status"] == "complete"
    assert result["paired_complete_board_seeds"] == [31, 47]
    assert result["paired_resolved_board_seeds"] == [31, 47]
    assert len(instances) == len(calls) == 4
    assert all(policy.calls == 4 for policy in instances)
    for arm, kind in (("plain", "END_TURN"), ("road", "BUILD_ROAD")):
        report = result["arms"][arm]
        assert report["wins"] == 2 and report["resolved_games"] == 8
        assert report["truncated_games"] == 0
        assert report["action_kind_counts"] == {kind: 8}
        assert report["plan_reason_counts"] == {"planned": 8}
        assert report["search_stop_counts"] == {"complete": 8}
        assert report["plan_nodes"] == 24 and report["geometry_visits"] == 40
        assert report["selection_latency_ms"]["count"] == 8
        assert [board["board_seed"] for board in report["boards"]] == [31, 47]
        assert all([row["seat"] for row in board["rows"]] == [0, 1, 2, 3]
                   for board in report["boards"])
    assert len(persisted) == 16
    assert {(row["arm"], row["board_seed"], row["seat"]) for row in persisted} == {
        (arm, seed, seat) for arm in ("plain", "road") for seed in (31, 47)
        for seat in range(4)
    }
    json.dumps(result, allow_nan=False)


def test_partial_block_is_reported_without_claiming_a_paired_board(monkeypatch):
    persisted = []

    def evaluate(policy, **kwargs):
        seed = kwargs["board_seeds"][0]
        if policy.policy.action_kind == "BUILD_ROAD":
            row = _row(seed, 0, {"BUILD_ROAD": 1}, won=True)
            kwargs["on_game_end"](deepcopy(row))
            return {"status": "incomplete", "incomplete_reason": "deadline",
                    "outcomes": [], "partial_outcomes": [row],
                    "interrupted_game": {"board_seed": seed, "seat": 1, "steps": 3}}
        rows = [_row(seed, seat, {"END_TURN": 1}, won=seat == 0) for seat in range(4)]
        for row in rows:
            kwargs["on_game_end"](deepcopy(row))
        return {"status": "complete", "incomplete_reason": None, "outcomes": rows,
                "partial_outcomes": [], "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel(
        {"plain": lambda: Planner("END_TURN"), "road": lambda: Planner("BUILD_ROAD")},
        board_seeds=[31, 47], on_game_end=persisted.append,
    )
    assert result["status"] == "incomplete" and result["incomplete_reason"] == "deadline"
    assert result["paired_complete_board_seeds"] == []
    assert [b["status"] for b in result["arms"]["plain"]["boards"]] == [
        "complete", "not_run"]
    assert [b["status"] for b in result["arms"]["road"]["boards"]] == [
        "incomplete", "not_run"]
    partial = result["arms"]["road"]["boards"][0]
    assert partial["rows"] == [] and [r["seat"] for r in partial["partial_rows"]] == [0]
    assert partial["interrupted_game"]["seat"] == 1
    assert result["arms"]["road"]["games"] == 0
    assert result["arms"]["road"]["finished_games"] == 1
    assert len(persisted) == 5


class CityPlanner:
    def __init__(self):
        self.calls = 0
        self.last_diagnostics = {}

    def plan(self, view):
        self.calls += 1
        if view.context.version == 10:
            index, reason, status = 1, "city_before_purchase", "city_before_purchase"
            parent_kind = "BUY_DEVELOPMENT_CARD"
            parent_index = 0
        else:
            index, reason, status = 0, "planned", "no_maritime_trade"
            parent_kind = view.legal_actions[index].kind
            parent_index = index
        self.last_diagnostics = {
            "status": status, "parent_root_index": parent_index,
            "parent_kind": parent_kind, "selected_city_target": 7 if index else None,
            "residual_hand": (0, 1, 2, 3, 4),
        }
        return NS(
            first_action=view.legal_actions[index], root_index=index,
            reason=reason, search_stop="one_trade_city_complete" if index else "complete",
            nodes=2, geometry_visits=4,
            score_before_micro=100, score_after_micro=250,
        )


def test_changed_city_detail_tracks_next_executed_city_via_optional_step_callback(monkeypatch):
    instances, forwarded = [], []

    def factory():
        policy = CityPlanner()
        instances.append(policy)
        return policy

    def evaluate(adapter, **kwargs):
        assert kwargs["on_candidate_step"] is not None
        rows = []
        for seat in range(4):
            if seat == 0:
                before = _view(seat, 10, (Action("BUY_DEVELOPMENT_CARD"),
                                          Action("MARITIME_TRADE")))
                selected, _ = adapter.act(before)
                after = _view(seat, 11, (Action("BUILD_CITY", target=7),),
                              hand=(0, 1, 2, 3, 4), buildings=(NS(vertex=7, owner=0, level=1),))
                first_step = NS(before=before, action=selected, after=after,
                                terminated=False, truncated=False)
                kwargs["on_candidate_step"](first_step)
                built, _ = adapter.act(after)
                final = _view(seat, 12, (Action("END_TURN"),),
                              hand=(0, 1, 2, 1, 1), buildings=(NS(vertex=7, owner=0, level=2),))
                second_step = NS(before=after, action=built, after=final,
                                 terminated=True, truncated=False)
                kwargs["on_candidate_step"](second_step)
                kinds = {"MARITIME_TRADE": 1, "BUILD_CITY": 1}
            else:
                before = _view(seat, 20 + seat, (Action("END_TURN"),))
                selected, _ = adapter.act(before)
                after = _view(seat, 21 + seat, (Action("END_TURN"),))
                kwargs["on_candidate_step"](NS(before=before, action=selected, after=after,
                                                 terminated=True, truncated=False))
                kinds = {"END_TURN": 1}
            row = _row(31, seat, kinds, won=seat == 0)
            if kwargs["on_game_end"] is not None:
                kwargs["on_game_end"](deepcopy(row))
            rows.append(row)
        return {"status": "complete", "outcomes": rows, "partial_outcomes": [],
                "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel({"city": factory}, board_seeds=[31],
                                     on_candidate_step=forwarded.append)
    board = result["arms"]["city"]["boards"][0]
    assert instances[0].calls == 5
    assert len(forwarded) == 5
    assert board["wrapper_status_counts"] == {"city_before_purchase": 1,
                                               "no_maritime_trade": 4}
    assert board["wrapper_changed_count"] == 1
    assert board["plan_reason_counts"] == {"city_before_purchase": 1, "planned": 4}
    assert board["action_kind_counts"] == {"MARITIME_TRADE": 1, "BUILD_CITY": 1,
                                          "END_TURN": 3}
    detail = board["changed_decision_details"][0]
    assert detail["board_seed"] == 31 and detail["seat"] == 0
    assert detail["input_version"] == 10 and detail["input_hand"] == [1, 2, 3, 4, 5]
    assert detail["known_own_vp"] == 4
    assert detail["selected_legal_action"]["kind"] == "MARITIME_TRADE"
    assert detail["intended_city_target"] == 7
    assert detail["score_before_micro"] == 100 and detail["score_after_micro"] == 250
    assert detail["status"] == "city_before_purchase"
    assert detail["selected_action_executed"] is True
    assert detail["next_selected_action"]["kind"] == "BUILD_CITY"
    assert detail["next_actual_action"]["kind"] == "BUILD_CITY"
    assert detail["city_built_next_own_action"] is True
    assert detail["wrapper_diagnostics"]["residual_hand"] == [0, 1, 2, 3, 4]
    json.dumps(result, allow_nan=False)


def test_wrapper_projection_work_is_counted_when_fallback_returns_parent_plan(monkeypatch):
    class FallbackPlanner:
        def plan(self, view):
            self.last_diagnostics = {
                "status": "insufficient_gain",
                "parent_root_index": 0,
                "parent_kind": "BUY_DEVELOPMENT_CARD",
                "nodes": 7,
                "geometry_visits": 9,
            }
            return NS(
                first_action=view.legal_actions[0], root_index=0,
                reason="unsupported_heuristic_action", search_stop="not_searched",
                nodes=0, geometry_visits=0,
            )

    def evaluate(adapter, **kwargs):
        rows = []
        for seat in range(4):
            action = Action("BUY_DEVELOPMENT_CARD")
            selected, _ = adapter.act(_view(seat, seat, (action,)))
            assert selected is action
            rows.append(_row(31, seat, {action.kind: 1}, won=seat == 0))
        return {"status": "complete", "outcomes": rows, "partial_outcomes": [],
                "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel({"fallback": FallbackPlanner}, board_seeds=[31])
    board = result["arms"]["fallback"]["boards"][0]
    assert board["plan_nodes"] == board["geometry_visits"] == 0
    assert board["wrapper_nodes"] == 28
    assert board["wrapper_geometry_visits"] == 36
    assert board["wrapper_status_counts"] == {"insufficient_gain": 4}
    assert board["wrapper_changed_count"] == 0


@pytest.mark.parametrize("arms,seeds,no_offers", [
    ({}, [1], True),
    ({"plain": lambda: Planner("END_TURN")}, [], True),
    ({"plain": lambda: Planner("END_TURN")}, [1, 1], True),
    ({"plain": lambda: Planner("END_TURN")}, [True], True),
    ({"plain": lambda: Planner("END_TURN")}, [1], False),
])
def test_invalid_panel_inputs_rejected_before_evaluation(monkeypatch, arms, seeds, no_offers):
    monkeypatch.setattr(paired, "evaluate_blocks", lambda *args, **kwargs: pytest.fail(
        "invalid input reached evaluation"))
    with pytest.raises(ValueError):
        paired.run_paired_panel(arms, board_seeds=seeds, no_offers=no_offers)
