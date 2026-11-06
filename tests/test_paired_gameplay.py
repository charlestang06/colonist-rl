"""Paired public gameplay accounting with synthetic evaluation results only."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest

from colonist_rl import paired_gameplay as paired
from colonist_rl.contracts import Action
from colonist_rl.public_turn_planner import PlannedStep, SearchDiagnostics, TurnPlan


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
    assert result["paired_summary"]["paired_resolved"] is True
    assert result["paired_summary"]["arms"]["plain"]["games"] == 8
    assert result["paired_summary"]["arms"]["road"]["wins"] == 2
    assert result["paired_summary"]["contrast_vs_reference"]["road"] == {
        "win_rate_difference": 0.0, "mean_victory_points_difference": 0.0}
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
        if seed == 47 and policy.policy.action_kind == "BUILD_ROAD":
            for seat in (0, 1):
                policy.act(_view(seat, seat, (Action("BUILD_ROAD", target=9),)))
            row = _row(seed, 0, {"BUILD_ROAD": 1}, won=True)
            kwargs["on_game_end"](deepcopy(row))
            return {"status": "incomplete", "incomplete_reason": "deadline",
                    "outcomes": [], "partial_outcomes": [row],
                    "interrupted_game": {"board_seed": seed, "seat": 1, "steps": 3}}
        kind = policy.policy.action_kind
        rows = []
        for seat in range(4):
            policy.act(_view(seat, seat, (Action(kind, target=9 if kind == "BUILD_ROAD" else None),)))
            row = _row(seed, seat, {kind: 1}, won=seat == 0)
            kwargs["on_game_end"](deepcopy(row))
            rows.append(row)
        return {"status": "complete", "incomplete_reason": None, "outcomes": rows,
                "partial_outcomes": [], "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel(
        {"plain": lambda: Planner("END_TURN"), "road": lambda: Planner("BUILD_ROAD")},
        board_seeds=[31, 47, 59], on_game_end=persisted.append,
    )
    assert result["status"] == "incomplete" and result["incomplete_reason"] == "deadline"
    assert result["paired_complete_board_seeds"] == [31]
    assert [b["status"] for b in result["arms"]["plain"]["boards"]] == [
        "complete", "complete", "not_run"]
    assert [b["status"] for b in result["arms"]["road"]["boards"]] == [
        "complete", "incomplete", "not_run"]
    partial = result["arms"]["road"]["boards"][1]
    assert partial["rows"] == [] and [r["seat"] for r in partial["partial_rows"]] == [0]
    assert partial["interrupted_game"]["seat"] == 1
    assert partial["action_kind_counts"] == {}
    assert partial["partial_action_kind_counts"] == {"BUILD_ROAD": 1}
    assert partial["attempted_action_kind_counts"] == {"BUILD_ROAD": 2}
    assert partial["plan_reason_counts"] == {"planned": 2}
    assert partial["selection_latency_ms"]["count"] == 2
    assert partial["action_kind_counts_scope"] == "complete_four_seat_blocks"
    assert partial["planning_metrics_scope"] == "attempted_policy_selections"
    assert result["arms"]["plain"]["games"] == 8
    assert result["arms"]["plain"]["wins"] == 2
    assert result["arms"]["plain"]["totals_scope"] == "descriptive_completed_arm_boards"
    assert result["arms"]["road"]["games"] == 4
    assert result["arms"]["road"]["finished_games"] == 5
    assert result["arms"]["road"]["partial_action_kind_counts"] == {"BUILD_ROAD": 1}
    matched = result["paired_summary"]
    assert matched["board_seeds"] == [31]
    assert matched["arms"]["plain"]["games"] == matched["arms"]["road"]["games"] == 4
    assert matched["arms"]["plain"]["wins"] == matched["arms"]["road"]["wins"] == 1
    assert matched["paired_resolved"] is False and matched["contrast_vs_reference"] is None
    assert len(persisted) == 13


def test_searched_selection_latency_excludes_unsearched_decisions_per_board_and_arm(monkeypatch):
    durations = {31: (1, 10, 3, 20), 47: (5, 30, 40, 50)}
    ticks = []
    cursor = 0.0
    for seed in (31, 47):
        for milliseconds in durations[seed]:
            ticks.extend((cursor, cursor + milliseconds / 1000))
            cursor += 1.0
    clock = iter(ticks)
    monkeypatch.setattr(paired, "time", NS(perf_counter=lambda: next(clock)))

    class MixedPlanner:
        def plan(self, view):
            seed, seat = divmod(view.context.version, 10)
            searched = seat == 0 or (seed == 31 and seat == 2)
            return NS(
                first_action=view.legal_actions[0], root_index=0,
                reason=("planned" if seat == 0 else "insufficient_gain")
                if searched else "unsupported_context",
                search_stop=("node_limit" if seat == 2 else "complete")
                if searched else "not_searched",
                nodes=2 if searched else 0, geometry_visits=0,
            )

    def evaluate(adapter, **kwargs):
        seed = kwargs["board_seeds"][0]
        rows = []
        for seat in range(4):
            action = Action("END_TURN")
            selected, _ = adapter.act(_view(seat, seed * 10 + seat, (action,)))
            assert selected is action
            rows.append(_row(seed, seat, {"END_TURN": 1}, won=seat == 0))
        return {"status": "complete", "outcomes": rows, "partial_outcomes": [],
                "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel({"mixed": MixedPlanner}, board_seeds=[31, 47])
    arm = result["arms"]["mixed"]
    board31, board47 = arm["boards"]
    assert board31["selection_latency_ms"]["count"] == 4
    assert board47["selection_latency_ms"]["count"] == 4
    assert arm["selection_latency_ms"]["count"] == 8
    assert board31["selection_latency_ms"]["mean"] == pytest.approx(8.5)
    assert arm["selection_latency_ms"]["mean"] == pytest.approx(19.875)
    assert arm["searched_selection_latency_scope"] == (
        "attempted_planner_selections_with_search_stop_other_than_not_searched")
    assert board31["searched_selection_latency_ms"] == pytest.approx(
        {"count": 2, "mean": 2.0, "p95": 2.9}
    )
    assert board47["searched_selection_latency_ms"] == pytest.approx(
        {"count": 1, "mean": 5.0, "p95": 5.0}
    )
    assert arm["searched_selection_latency_ms"] == pytest.approx(
        {"count": 3, "mean": 3.0, "p95": 4.8}
    )
    assert board31["searched_plan_reason_counts"] == {
        "planned": 1, "insufficient_gain": 1}
    assert board47["searched_plan_reason_counts"] == {"planned": 1}
    assert arm["searched_plan_reason_counts"] == {"planned": 2, "insufficient_gain": 1}


def test_candidate_step_callback_identifies_arm_and_board_in_multiarm_panel(monkeypatch):
    raw_steps, received = [], []

    def evaluate(adapter, **kwargs):
        rows = []
        for seat in range(4):
            view = _view(seat, seat, (Action("END_TURN"), Action("BUILD_ROAD", target=9)))
            action, _ = adapter.act(view)
            step = NS(before=view, action=action,
                      after=_view(seat, seat + 1, (Action("END_TURN"),)),
                      terminated=True, truncated=False)
            raw_steps.append(step)
            kwargs["on_candidate_step"](step)
            rows.append(_row(31, seat, {action.kind: 1}, won=seat == 0))
        return {"status": "complete", "outcomes": rows, "partial_outcomes": [],
                "interrupted_game": None}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    paired.run_paired_panel(
        {"plain": lambda: Planner("END_TURN"), "road": lambda: Planner("BUILD_ROAD")},
        board_seeds=[31], on_candidate_step=received.append,
    )
    assert [(event.arm, event.board_seed) for event in received] == [
        ("plain", 31)] * 4 + [("road", 31)] * 4
    assert all(event.step is raw for event, raw in zip(received, raw_steps))


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
    assert all(isinstance(event, paired.PairedCandidateStep) and event.arm == "city"
               and event.board_seed == 31 for event in forwarded)
    assert [event.step.action.kind for event in forwarded[:2]] == [
        "MARITIME_TRADE", "BUILD_CITY"]
    assert board["wrapper_status_counts"] == {"city_before_purchase": 1,
                                               "no_maritime_trade": 4}
    assert board["search_diagnostics_record_count"] == 0
    assert board["search_diagnostics_unavailable_count"] == 5
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


def test_public_option_changed_details_confirm_trade_build_and_pass_from_public_steps():
    class OptionPlanner:
        def plan(self, view):
            version = view.context.version
            if version == 10:
                index, status = 1, "public_option_trade_build"
            elif version == 20:
                index, status = 1, "public_option_pass"
            else:
                index, status = 0, "ineligible_parent"
            self.last_diagnostics = {
                "status": status,
                "parent_root_index": 0,
                "parent_kind": "BUY_DEVELOPMENT_CARD" if index else view.legal_actions[0].kind,
                "selected_build_kind": "BUILD_SETTLEMENT" if version == 10 else None,
                "selected_build_target": 7 if version == 10 else None,
                "card_option_credit_micro": 61_600_000,
                "pass_reserve_micro": 70_000_000,
                "score_buy_micro": 500_000_000,
                "score_pass_micro": 510_000_000,
                "score_trade_build_micro": 530_000_000 if version == 10 else None,
            }
            return NS(
                first_action=view.legal_actions[index], root_index=index,
                reason=status, search_stop="one_trade_build_complete" if version == 10
                else "pass_complete", nodes=4, geometry_visits=9,
                score_before_micro=450_000_000,
                score_after_micro=530_000_000 if version == 10 else 510_000_000,
            )

    adapter = paired._PlannedPolicyAdapter(OptionPlanner(), board_seed=31)
    before_trade = _view(0, 10, (Action("BUY_DEVELOPMENT_CARD"),
                                 Action("MARITIME_TRADE")))
    selected_trade, _ = adapter.act(before_trade)
    after_trade = _view(0, 11, (Action("BUILD_SETTLEMENT", target=7),),
                        hand=(0, 1, 2, 3, 4))
    adapter.record_candidate_step(NS(before=before_trade, action=selected_trade,
                                     after=after_trade))
    selected_build, _ = adapter.act(after_trade)
    after_build = _view(0, 12, (Action("END_TURN"),),
                        buildings=(NS(vertex=7, owner=0, level=1),))
    adapter.record_candidate_step(NS(before=after_trade, action=selected_build,
                                     after=after_build))

    before_pass = _view(0, 20, (Action("BUY_DEVELOPMENT_CARD"), Action("END_TURN")))
    selected_pass, _ = adapter.act(before_pass)
    after_pass = _view(0, 21, (Action("END_TURN"),), hand=(1, 2, 3, 4, 5))
    adapter.record_candidate_step(NS(before=before_pass, action=selected_pass,
                                     after=after_pass))

    trade_detail, pass_detail = adapter.changed_details
    assert trade_detail["status"] == "public_option_trade_build"
    assert trade_detail["selected_legal_action"]["kind"] == "MARITIME_TRADE"
    assert trade_detail["intended_build_kind"] == "BUILD_SETTLEMENT"
    assert trade_detail["intended_build_target"] == 7
    assert trade_detail["selected_action_executed"] is True
    assert trade_detail["selected_after"]["version"] == 11
    assert trade_detail["next_selected_action"]["kind"] == "BUILD_SETTLEMENT"
    assert trade_detail["next_actual_action"]["target"] == 7
    assert trade_detail["city_built_next_own_action"] is None
    assert trade_detail["build_confirmed_next_own_action"] is True
    assert trade_detail["card_option_credit_micro"] == 61_600_000
    assert trade_detail["score_trade_build_micro"] == 530_000_000
    assert pass_detail["status"] == "public_option_pass"
    assert pass_detail["selected_legal_action"]["kind"] == "END_TURN"
    assert pass_detail["selected_action_executed"] is True
    assert pass_detail["selected_after"]["version"] == 21
    assert pass_detail["pass_reserve_micro"] == 70_000_000
    assert pass_detail["score_buy_micro"] == 500_000_000
    assert pass_detail["score_pass_micro"] == 510_000_000
    assert pass_detail["next_selected_action"] is None
    adapter.act(_view(0, 30, (Action("END_TURN"),)))
    assert pass_detail["next_selected_action"] is None
    for _ in range(paired._CHANGED_DETAIL_LIMIT):
        adapter.act(before_pass)
    assert len(adapter.changed_details) == paired._CHANGED_DETAIL_LIMIT
    assert adapter.changed_detail_omitted_count == 2
    json.dumps(adapter.changed_details, allow_nan=False)


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


def test_graph_policy_path_delegates_mode_and_memory_without_planner_calls(monkeypatch):
    class Graph:
        def __init__(self):
            self.training = True
            self.head = NS(training=True)
            self.memories = []
            self.calls = 0
            self.clone_action = False

        def modules(self):
            return (self, self.head)

        def eval(self):
            self.training = self.head.training = False
            return self

        def act(self, view, memory=None, *, deterministic=False):
            assert not self.training and not self.head.training
            assert deterministic is True
            self.calls += 1
            self.memories.append(memory)
            action = Action("END_TURN") if self.clone_action else view.legal_actions[0]
            return action, (memory or 0) + 1

    graph = Graph()

    def evaluate(adapter, **kwargs):
        modes = [(module, module.training) for module in adapter.modules()]
        adapter.eval()
        try:
            rows = []
            for seat in range(4):
                memory = None
                for version in (0, 1):
                    action = Action("END_TURN")
                    selected, memory = adapter.act(_view(seat, version, (action,)), memory,
                                                   deterministic=True)
                    assert selected is action
                rows.append(_row(31, seat, {"END_TURN": 2}, won=seat == 0))
            return {"status": "complete", "outcomes": rows, "partial_outcomes": [],
                    "interrupted_game": None}
        finally:
            for module, training in modes:
                module.training = training

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel({"graph": lambda: graph}, board_seeds=[31])
    board = result["arms"]["graph"]["boards"][0]
    assert graph.calls == 8 and graph.memories == [None, 1] * 4
    assert graph.training and graph.head.training
    assert board["action_kind_counts"] == board["attempted_action_kind_counts"] == {
        "END_TURN": 8}
    assert board["plan_reason_counts"] == board["search_stop_counts"] == {}
    assert board["selection_latency_ms"]["count"] == 8
    assert board["searched_selection_latency_ms"]["count"] == 0
    assert board["search_diagnostics_record_count"] == 0
    assert board["search_diagnostics_unavailable_count"] == 8
    assert board["search_diagnostics_totals"]["attempted_transitions"] == 0
    assert board["search_diagnostics_depth_histograms"] == {
        "selected_plan_depth": {}, "max_feasible_depth": {}}
    graph.clone_action = True
    with pytest.raises(ValueError, match="original legal action"):
        paired.run_paired_panel({"graph": lambda: graph}, board_seeds=[31])
    assert graph.training and graph.head.training


def test_search_diagnostics_capture_and_aggregate_attempted_selections(monkeypatch):
    diagnostics = {
        310: SearchDiagnostics(
            root_attempts=1, future_attempts=4, feasible_transitions=5,
            unique_successors=4, duplicate_successors=1, max_feasible_depth=3,
            selected_plan_depth=2, beam_truncations=1,
            pruned_invalid_candidates=3,
        ),
        311: SearchDiagnostics(
            root_attempts=2, feasible_transitions=1, invalid_transitions=1,
            unique_successors=1, max_feasible_depth=1,
        ),
        313: SearchDiagnostics(),
        470: SearchDiagnostics(
            root_attempts=1, future_attempts=2, feasible_transitions=2,
            invalid_transitions=1, unique_successors=2, max_feasible_depth=2,
            selected_plan_depth=1, pruned_invalid_candidates=2,
        ),
    }
    missing_nodes = {312: 7, 471: 9}

    class InstrumentedPlanner:
        def plan(self, view):
            info = diagnostics.get(view.context.version)
            nodes = (
                info.root_attempts + info.future_attempts
                if info is not None else missing_nodes[view.context.version]
            )
            steps = () if info is None else (
                PlannedStep("BUILD_ROAD", 9), PlannedStep("BUILD_CITY", 0)
            )[:info.selected_plan_depth]
            return TurnPlan(
                view.legal_actions[0], 0, steps, 0, 0, nodes, 0,
                "planned" if steps else "insufficient_gain", "complete",
                search_diagnostics=info,
            )

    def evaluate(adapter, **kwargs):
        seed = kwargs["board_seeds"][0]
        seats = range(4) if seed == 31 else (0, 1)
        rows = []
        for seat in seats:
            version = seed * 10 + seat
            action = Action("BUILD_ROAD", target=9)
            selected, _ = adapter.act(_view(seat, version, (action,)))
            assert selected is action
            captured = adapter.records[-1]["search_diagnostics"]
            assert captured == (asdict(diagnostics[version]) if version in diagnostics else None)
            rows.append(_row(seed, seat, {"BUILD_ROAD": 1}, won=seat == 0))
        if seed == 31:
            return {"status": "complete", "outcomes": rows, "partial_outcomes": [],
                    "interrupted_game": None}
        return {"status": "incomplete", "incomplete_reason": "deadline",
                "outcomes": [], "partial_outcomes": rows[:1],
                "interrupted_game": {"board_seed": seed, "seat": 1, "steps": 1}}

    monkeypatch.setattr(paired, "evaluate_blocks", evaluate)
    result = paired.run_paired_panel({"instrumented": InstrumentedPlanner},
                                     board_seeds=[31, 47])
    board31, board47 = result["arms"]["instrumented"]["boards"]
    arm = result["arms"]["instrumented"]
    assert board31["search_diagnostics_scope"] == (
        "attempted_policy_selections_with_search_diagnostics")
    assert (board31["search_diagnostics_record_count"],
            board31["search_diagnostics_unavailable_count"]) == (3, 1)
    assert board31["plan_nodes"] == 14
    assert board31["search_diagnostics_totals"] == {
        "attempted_transitions": 7,
        "root_attempts": 3,
        "future_attempts": 4,
        "feasible_transitions": 6,
        "invalid_transitions": 1,
        "unique_successors": 5,
        "duplicate_successors": 1,
        "selected_plan_depth_sum": 2,
        "max_feasible_depth_sum": 4,
        "beam_truncations": 1,
        "pruned_invalid_candidates": 3,
    }
    assert board31["search_diagnostics_depth_histograms"] == {
        "selected_plan_depth": {"0": 2, "2": 1},
        "max_feasible_depth": {"0": 1, "1": 1, "3": 1},
    }
    assert board47["status"] == "incomplete"
    assert (board47["search_diagnostics_record_count"],
            board47["search_diagnostics_unavailable_count"]) == (1, 1)
    assert board47["plan_nodes"] == 12
    assert board47["search_diagnostics_totals"] == {
        "attempted_transitions": 3,
        "root_attempts": 1,
        "future_attempts": 2,
        "feasible_transitions": 2,
        "invalid_transitions": 1,
        "unique_successors": 2,
        "duplicate_successors": 0,
        "selected_plan_depth_sum": 1,
        "max_feasible_depth_sum": 2,
        "beam_truncations": 0,
        "pruned_invalid_candidates": 2,
    }
    assert board47["search_diagnostics_depth_histograms"] == {
        "selected_plan_depth": {"1": 1}, "max_feasible_depth": {"2": 1}}
    assert (arm["search_diagnostics_record_count"],
            arm["search_diagnostics_unavailable_count"]) == (4, 2)
    assert arm["plan_nodes"] == 26
    assert arm["search_diagnostics_totals"] == {
        "attempted_transitions": 10,
        "root_attempts": 4,
        "future_attempts": 6,
        "feasible_transitions": 8,
        "invalid_transitions": 2,
        "unique_successors": 7,
        "duplicate_successors": 1,
        "selected_plan_depth_sum": 3,
        "max_feasible_depth_sum": 6,
        "beam_truncations": 1,
        "pruned_invalid_candidates": 5,
    }
    assert arm["search_diagnostics_depth_histograms"] == {
        "selected_plan_depth": {"0": 2, "1": 1, "2": 1},
        "max_feasible_depth": {"0": 1, "1": 1, "2": 1, "3": 1},
    }
    json.dumps(result, allow_nan=False)


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
