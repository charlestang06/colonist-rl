"""Small paired-gameplay runner for policies that plan from public observations.

Each arm/board job owns a fresh planner. The existing block evaluator owns all
gameplay, seat rotation, RNG restoration, deadlines, and final game rows.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict

from .evaluation import evaluate_blocks


_CHANGED_DETAIL_LIMIT = 64


def _json_copy(value):
    """Detach evaluator records and turn their tuples into JSON arrays."""
    return json.loads(json.dumps(value, allow_nan=False))


def _action_record(action):
    return _json_copy(asdict(action))


def _latency_summary(records):
    values = sorted(record["selection_latency_ms"] for record in records)
    if not values:
        return {"count": 0, "mean": 0.0, "p95": 0.0}
    rank = (len(values) - 1) * 0.95
    lower = int(rank)
    p95 = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        rank - lower
    )
    return {"count": len(values), "mean": sum(values) / len(values), "p95": p95}


class _PlannedPolicyAdapter:
    """Expose the evaluator's graph-policy surface without a model forward."""

    def __init__(self, policy, board_seed):
        if not callable(getattr(policy, "plan", None)):
            raise TypeError("arm factory must return a policy with plan(view)")
        self.policy = policy
        self.board_seed = board_seed
        self.records = []
        self.changed_details = []
        self.changed_detail_omitted_count = 0
        self._pending_changed = {}
        self._awaiting_next_step = {}
        self._last_record = None
        self._last_action = None

    def eval(self):
        return self

    def modules(self):
        return ()

    def act(self, view, memory=None, deterministic=True):
        start = time.perf_counter()
        plan = self.policy.plan(view)
        index = plan.root_index
        if type(index) is not int or index not in range(len(view.legal_actions)):
            raise ValueError("plan root_index must name a current legal action")
        action = view.legal_actions[index]
        if plan.first_action is not action:
            raise ValueError("plan must return the original legal root action")
        latency_ms = (time.perf_counter() - start) * 1000
        diagnostics = getattr(self.policy, "last_diagnostics", None)
        diagnostics = _json_copy(diagnostics) if diagnostics is not None else None
        changed = bool(
            isinstance(diagnostics, dict)
            and diagnostics.get("parent_root_index") is not None
            and diagnostics.get("parent_kind") is not None
            and (index != diagnostics["parent_root_index"]
                 or action.kind != diagnostics["parent_kind"])
        )
        record = {
            "seat": view.player_id,
            "input_version": view.context.version,
            "action_kind": action.kind,
            "plan_reason": plan.reason,
            "search_stop": plan.search_stop,
            "nodes": plan.nodes,
            "geometry_visits": plan.geometry_visits,
            "selection_latency_ms": latency_ms,
            "wrapper_diagnostics": diagnostics,
            "wrapper_changed": changed,
        }
        previous = self._pending_changed.pop(view.player_id, None)
        if previous is not None:
            previous["next_selected_action"] = _action_record(action)
            self._awaiting_next_step[view.player_id] = previous
        if changed and diagnostics["status"] == "city_before_purchase":
            if len(self.changed_details) < _CHANGED_DETAIL_LIMIT:
                detail = {
                    "board_seed": self.board_seed,
                    "seat": view.player_id,
                    "input_version": view.context.version,
                    "input_hand": list(view.hand),
                    "known_own_vp": view.own_victory_points,
                    "selected_legal_action": _action_record(action),
                    "intended_city_target": diagnostics.get("selected_city_target"),
                    "score_before_micro": plan.score_before_micro,
                    "score_after_micro": plan.score_after_micro,
                    "status": diagnostics["status"],
                    "plan_reason": plan.reason,
                    "search_stop": plan.search_stop,
                    "nodes": plan.nodes,
                    "geometry_visits": plan.geometry_visits,
                    "selection_latency_ms": latency_ms,
                    "wrapper_diagnostics": diagnostics,
                    "selected_action_executed": None,
                    "selected_after": None,
                    "next_selected_action": None,
                    "next_actual_action": None,
                    "city_built_next_own_action": None,
                }
                self.changed_details.append(detail)
                self._pending_changed[view.player_id] = detail
                record["_detail"] = detail
            else:
                self.changed_detail_omitted_count += 1
        self.records.append(record)
        self._last_record = record
        self._last_action = action
        return action, memory

    def record_candidate_step(self, step):
        """Use the evaluator's optional public after-step observation only."""
        if self._last_record is None or step.action is not self._last_action:
            raise ValueError("candidate step did not execute the selected root action")
        detail = self._last_record.get("_detail")
        if detail is not None:
            detail["selected_action_executed"] = True
            detail["selected_after"] = {
                "version": step.after.context.version,
                "hand": list(step.after.hand),
                "known_own_vp": step.after.own_victory_points,
            }
        previous = self._awaiting_next_step.pop(step.before.player_id, None)
        if previous is not None:
            previous["next_actual_action"] = _action_record(step.action)
            target = previous["intended_city_target"]
            previous["city_built_next_own_action"] = bool(
                step.action.kind == "BUILD_CITY"
                and step.action.target == target
                and any(
                    building.owner == step.before.player_id
                    and building.vertex == target
                    and building.level == 2
                    for building in step.after.buildings
                )
            )


def _metrics(rows, records):
    actions = Counter()
    for row in rows:
        actions.update(row.get("action_kind_counts", {}))
    wrapper_diagnostics = [record["wrapper_diagnostics"] for record in records
                           if isinstance(record["wrapper_diagnostics"], dict)]
    statuses = Counter(
        diagnostics["status"] for diagnostics in wrapper_diagnostics
        if diagnostics.get("status") is not None
    )
    return {
        "wins": sum(row.get("won") is True for row in rows),
        "resolved_games": sum(type(row.get("won")) is bool for row in rows),
        "truncated_games": sum(bool(row.get("truncated")) for row in rows),
        "action_kind_counts": dict(actions),
        "plan_reason_counts": dict(Counter(record["plan_reason"] for record in records)),
        "search_stop_counts": dict(Counter(record["search_stop"] for record in records)),
        "wrapper_status_counts": dict(statuses),
        "wrapper_changed_count": sum(record["wrapper_changed"] for record in records),
        "plan_nodes": sum(record["nodes"] for record in records),
        "geometry_visits": sum(record["geometry_visits"] for record in records),
        "wrapper_nodes": sum(diagnostics.get("nodes") or 0
                             for diagnostics in wrapper_diagnostics),
        "wrapper_geometry_visits": sum(diagnostics.get("geometry_visits") or 0
                                       for diagnostics in wrapper_diagnostics),
        "selection_latency_ms": _latency_summary(records),
    }


def _board_report(seed, result, adapter):
    rows = _json_copy(result.get("outcomes", []))
    partial = _json_copy(result.get("partial_outcomes", []))
    interrupted = _json_copy(result.get("interrupted_game"))
    seen = set()
    for row in rows + partial:
        seat = row.get("seat")
        if row.get("board_seed") != seed or type(seat) is not int or seat not in range(4):
            raise ValueError("evaluation returned an invalid board/seat row")
        if seat in seen:
            raise ValueError("evaluation returned a duplicate board/seat row")
        seen.add(seat)
    complete = (
        result.get("status") == "complete"
        and {row["seat"] for row in rows} == {0, 1, 2, 3}
        and not partial
        and interrupted is None
    )
    if not complete:
        partial = rows + partial
        rows = []
    return {
        "board_seed": seed,
        "status": "complete" if complete else "incomplete",
        "incomplete_reason": None if complete else (
            result.get("incomplete_reason") or "missing_four_seat_block"
        ),
        "rows": rows,
        "partial_rows": partial,
        "interrupted_game": interrupted,
        "games": len(rows),
        "finished_games": len(rows) + len(partial),
        "decision_metrics_scope": "attempted_selections",
        **_metrics(rows, adapter.records),
        "changed_decision_details": adapter.changed_details,
        "changed_detail_omitted_count": adapter.changed_detail_omitted_count,
    }


def _not_run_board(seed, reason):
    return {
        "board_seed": seed,
        "status": "not_run",
        "incomplete_reason": reason,
        "rows": [],
        "partial_rows": [],
        "interrupted_game": None,
        "games": 0,
        "finished_games": 0,
        "decision_metrics_scope": "attempted_selections",
        **_metrics([], []),
        "changed_decision_details": [],
        "changed_detail_omitted_count": 0,
    }


def run_paired_panel(
    arm_factories,
    *,
    board_seeds,
    opponent_lineup=("heuristic",) * 3,
    max_game_steps=20000,
    deadline=None,
    no_offers=True,
    on_game_end=None,
    on_candidate_step=None,
):
    """Run matching four-seat boards for each public planner arm.

    ``deadline`` is absolute monotonic time. Completed protocol games that
    reach the step cap remain in ``rows`` with unresolved wins. Incomplete
    blocks keep finished rows in ``partial_rows`` and never count as paired.
    Candidate-step capture is opt-in because it requests an extra public after
    observation from the evaluator for every executed candidate action.
    """
    if not isinstance(arm_factories, Mapping) or not arm_factories:
        raise ValueError("arm_factories must be a nonempty mapping")
    factories = list(arm_factories.items())
    if any(type(name) is not str or not name or not callable(factory)
           for name, factory in factories):
        raise ValueError("arms require nonempty string names and callable factories")
    if not isinstance(board_seeds, (list, tuple)) or not board_seeds or any(
        type(seed) is not int for seed in board_seeds
    ) or len(set(board_seeds)) != len(board_seeds):
        raise ValueError("board_seeds must be an explicit nonempty list of unique integers")
    seeds = list(board_seeds)
    if type(no_offers) is not bool or not no_offers:
        raise ValueError("paired public gameplay requires no_offers=True")
    if not isinstance(opponent_lineup, (list, tuple)) or len(opponent_lineup) != 3:
        raise ValueError("opponent_lineup must contain three opponent names")
    lineup = tuple(opponent_lineup)
    if type(max_game_steps) is not int or max_game_steps < 1:
        raise ValueError("max_game_steps must be a positive integer")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be finite absolute monotonic time")
    if on_game_end is not None and not callable(on_game_end):
        raise ValueError("on_game_end must be callable or None")
    if on_candidate_step is not None and not callable(on_candidate_step):
        raise ValueError("on_candidate_step must be callable or None")

    boards = {name: [] for name, _ in factories}
    records = {name: [] for name, _ in factories}
    stop_reason = None
    for seed in seeds:
        for name, factory in factories:
            if stop_reason is None and deadline is not None and time.monotonic() >= deadline:
                stop_reason = "deadline"
            if stop_reason is not None:
                boards[name].append(_not_run_board(seed, stop_reason))
                continue
            adapter = _PlannedPolicyAdapter(factory(), seed)

            def game_end(row, *, arm=name):
                on_game_end({"arm": arm, **row})

            def candidate_step(step, *, current=adapter):
                current.record_candidate_step(step)
                on_candidate_step(step)

            result = evaluate_blocks(
                adapter,
                board_seeds=[seed],
                candidate="graph",
                baseline="heuristic",
                opponent_lineup=lineup,
                max_game_steps=max_game_steps,
                deterministic=True,
                deadline=deadline,
                no_offers=True,
                on_game_end=game_end if on_game_end is not None else None,
                on_candidate_step=candidate_step if on_candidate_step is not None else None,
            )
            board = _board_report(seed, result, adapter)
            boards[name].append(board)
            records[name].extend(adapter.records)
            if board["status"] != "complete":
                stop_reason = board["incomplete_reason"]

    arms = {}
    for name, _ in factories:
        complete_rows = [row for board in boards[name] for row in board["rows"]]
        arms[name] = {
            "status": "complete" if all(board["status"] == "complete"
                                        for board in boards[name]) else "incomplete",
            "completed_board_seeds": [board["board_seed"] for board in boards[name]
                                      if board["status"] == "complete"],
            "boards": boards[name],
            "games": len(complete_rows),
            "finished_games": sum(board["finished_games"] for board in boards[name]),
            "decision_metrics_scope": "attempted_selections",
            **_metrics(complete_rows, records[name]),
        }
    paired = [seed for index, seed in enumerate(seeds) if all(
        boards[name][index]["status"] == "complete" for name, _ in factories
    )]
    resolved = [seed for index, seed in enumerate(seeds) if seed in paired and all(
        boards[name][index]["resolved_games"] == 4
        and boards[name][index]["truncated_games"] == 0 for name, _ in factories
    )]
    return {
        "status": "complete" if stop_reason is None else "incomplete",
        "incomplete_reason": stop_reason,
        "requested_board_seeds": seeds,
        "paired_complete_board_seeds": paired,
        "paired_resolved_board_seeds": resolved,
        "opponent_lineup": list(lineup),
        "max_game_steps": max_game_steps,
        "deadline": deadline,
        "no_offers": True,
        "changed_detail_limit_per_board": _CHANGED_DETAIL_LIMIT,
        "arms": arms,
    }
