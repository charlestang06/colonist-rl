"""Small paired-gameplay runner for policies using public observations.

Each arm/board job owns a fresh policy. The existing block evaluator owns all
gameplay, seat rotation, RNG restoration, deadlines, and final game rows.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .evaluation import CandidatePublicStep, evaluate_blocks
from .greedy_no_trade import GreedyNoTrade
from .opponents import HeuristicPolicy

_CHANGED_DETAIL_LIMIT = 64
_SEARCH_DIAGNOSTIC_COUNT_FIELDS = (
    "root_attempts",
    "future_attempts",
    "feasible_transitions",
    "invalid_transitions",
    "unique_successors",
    "duplicate_successors",
    "beam_truncations",
    "pruned_invalid_candidates",
)


@dataclass(frozen=True)
class PairedCandidateStep:
    """A public candidate transition labeled with its arm and board."""

    arm: str
    board_seed: int
    step: CandidatePublicStep


@dataclass(frozen=True)
class BaselineFocalSpec:
    """Select an existing public baseline as a paired focal arm."""

    kind: str

    def __post_init__(self):
        if self.kind not in ("heuristic", "greedy_no_trade"):
            raise ValueError("unknown public baseline focal kind")


class _SeededBaseline:
    """Keep one public baseline instance per focal game and original seed rule."""

    def __init__(self, kind, board_seed):
        self.kind, self.board_seed = kind, board_seed
        self._actors = {}

    def eval(self):
        return self

    def modules(self):
        return ()

    def act(self, view, memory=None, *, deterministic=True):
        seat = view.player_id
        if seat not in self._actors:
            self._actors[seat] = (
                HeuristicPolicy(self.board_seed + 31 + seat)
                if self.kind == "heuristic" else GreedyNoTrade()
            )
        actor = self._actors[seat]
        action = (
            actor.act(view) if self.kind == "heuristic"
            else actor.select_action(view)
        )
        return action, None


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
        search_diagnostics = getattr(plan, "search_diagnostics", None)
        search_diagnostics = (
            _json_copy(asdict(search_diagnostics)) if search_diagnostics is not None else None
        )
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
            "search_diagnostics": search_diagnostics,
            "wrapper_changed": changed,
        }
        previous = self._pending_changed.pop(view.player_id, None)
        if previous is not None:
            previous["next_selected_action"] = _action_record(action)
            self._awaiting_next_step[view.player_id] = previous
        if changed and diagnostics["status"] in (
            "city_before_purchase", "public_option_trade_build", "public_option_pass"
        ):
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
                if diagnostics["status"].startswith("public_option_"):
                    detail.update({
                        "selected_legal_index": index,
                        "intended_build_kind": diagnostics.get("selected_build_kind"),
                        "intended_build_target": diagnostics.get("selected_build_target"),
                        "card_option_credit_micro": diagnostics.get("card_option_credit_micro"),
                        "pass_reserve_micro": diagnostics.get("pass_reserve_micro"),
                        "score_buy_micro": diagnostics.get("score_buy_micro"),
                        "score_pass_micro": diagnostics.get("score_pass_micro"),
                        "score_trade_build_micro": diagnostics.get("score_trade_build_micro"),
                        "build_confirmed_next_own_action": None,
                    })
                self.changed_details.append(detail)
                if diagnostics["status"] != "public_option_pass":
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
            if target is not None:
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
            kind = previous.get("intended_build_kind")
            if kind in ("BUILD_CITY", "BUILD_SETTLEMENT"):
                target = previous["intended_build_target"]
                level = 2 if kind == "BUILD_CITY" else 1
                previous["build_confirmed_next_own_action"] = bool(
                    step.action.kind == kind
                    and step.action.target == target
                    and any(
                        building.owner == step.before.player_id
                        and building.vertex == target
                        and building.level == level
                        for building in step.after.buildings
                    )
                )


class _GraphPolicyAdapter(_PlannedPolicyAdapter):
    """Preserve graph-policy mode and recurrent memory semantics."""

    def eval(self):
        self.policy.eval()
        return self

    def modules(self):
        return self.policy.modules()

    def act(self, view, memory=None, deterministic=True):
        start = time.perf_counter()
        action, next_memory = self.policy.act(view, memory, deterministic=deterministic)
        if not any(action is legal for legal in view.legal_actions):
            raise ValueError("graph policy must return an original legal action")
        record = {
            "seat": view.player_id,
            "input_version": view.context.version,
            "action_kind": action.kind,
            "plan_reason": None,
            "search_stop": None,
            "nodes": 0,
            "geometry_visits": 0,
            "selection_latency_ms": (time.perf_counter() - start) * 1000,
            "wrapper_diagnostics": None,
            "search_diagnostics": None,
            "wrapper_changed": False,
        }
        self.records.append(record)
        self._last_record = record
        self._last_action = action
        return action, next_memory


def _search_diagnostic_metrics(records):
    """Aggregate only attempted selections that supplied planner diagnostics."""
    total_fields = (
        "attempted_transitions", *_SEARCH_DIAGNOSTIC_COUNT_FIELDS,
        "selected_plan_depth_sum", "max_feasible_depth_sum",
    )
    totals = dict.fromkeys(total_fields, 0)
    selected_depths = Counter()
    max_depths = Counter()
    available = 0
    for record in records:
        diagnostics = record["search_diagnostics"]
        if diagnostics is None:
            continue
        available += 1
        totals["attempted_transitions"] += record["nodes"]
        for field in _SEARCH_DIAGNOSTIC_COUNT_FIELDS:
            totals[field] += diagnostics[field]
        selected_depth = diagnostics["selected_plan_depth"]
        max_depth = diagnostics["max_feasible_depth"]
        totals["selected_plan_depth_sum"] += selected_depth
        totals["max_feasible_depth_sum"] += max_depth
        selected_depths[selected_depth] += 1
        max_depths[max_depth] += 1
    return {
        "search_diagnostics_scope": "attempted_policy_selections_with_search_diagnostics",
        "search_diagnostics_record_count": available,
        "search_diagnostics_unavailable_count": len(records) - available,
        "search_diagnostics_totals": totals,
        "search_diagnostics_depth_histograms": {
            "selected_plan_depth": {
                str(depth): count for depth, count in sorted(selected_depths.items())
            },
            "max_feasible_depth": {
                str(depth): count for depth, count in sorted(max_depths.items())
            },
        },
    }


def _metrics(rows, records, partial_rows=()):
    actions = Counter()
    for row in rows:
        actions.update(row.get("action_kind_counts", {}))
    partial_actions = Counter()
    for row in partial_rows:
        partial_actions.update(row.get("action_kind_counts", {}))
    wrapper_diagnostics = [record["wrapper_diagnostics"] for record in records
                           if isinstance(record["wrapper_diagnostics"], dict)]
    statuses = Counter(
        diagnostics["status"] for diagnostics in wrapper_diagnostics
        if diagnostics.get("status") is not None
    )
    searched_records = [
        record for record in records
        if record["search_stop"] not in (None, "not_searched")
    ]
    return {
        "game_metrics_scope": "complete_four_seat_blocks",
        "action_kind_counts_scope": "complete_four_seat_blocks",
        "partial_action_kind_counts_scope": "finished_games_in_incomplete_blocks",
        "planning_metrics_scope": "attempted_policy_selections",
        "wins": sum(row.get("won") is True for row in rows),
        "resolved_games": sum(type(row.get("won")) is bool for row in rows),
        "truncated_games": sum(bool(row.get("truncated")) for row in rows),
        "action_kind_counts": dict(actions),
        "partial_action_kind_counts": dict(partial_actions),
        "attempted_action_kind_counts": dict(Counter(record["action_kind"]
                                                     for record in records)),
        "plan_reason_counts": dict(Counter(record["plan_reason"] for record in records
                                           if record["plan_reason"] is not None)),
        "search_stop_counts": dict(Counter(record["search_stop"] for record in records
                                           if record["search_stop"] is not None)),
        "searched_selection_latency_scope": (
            "attempted_planner_selections_with_search_stop_other_than_not_searched"
        ),
        "searched_plan_reason_counts": dict(Counter(
            record["plan_reason"] for record in searched_records
            if record["plan_reason"] is not None
        )),
        "wrapper_status_counts": dict(statuses),
        "wrapper_changed_count": sum(record["wrapper_changed"] for record in records),
        "plan_nodes": sum(record["nodes"] for record in records),
        "geometry_visits": sum(record["geometry_visits"] for record in records),
        "wrapper_nodes": sum(diagnostics.get("nodes") or 0
                             for diagnostics in wrapper_diagnostics),
        "wrapper_geometry_visits": sum(diagnostics.get("geometry_visits") or 0
                                       for diagnostics in wrapper_diagnostics),
        **_search_diagnostic_metrics(records),
        "selection_latency_ms": _latency_summary(records),
        "searched_selection_latency_ms": _latency_summary(searched_records),
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
        **_metrics(rows, adapter.records, partial),
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
        **_metrics([], []),
        "changed_decision_details": [],
        "changed_detail_omitted_count": 0,
    }


def _paired_summary(boards, factories, seeds, paired, resolved):
    """Keep paired comparisons separate from descriptive per-arm totals."""
    matched = set(paired)
    complete_panel = len(paired) == len(seeds)
    paired_resolved = complete_panel and len(resolved) == len(seeds)
    per_arm = {}
    for name, _ in factories:
        rows = [row for board in boards[name] if board["board_seed"] in matched
                for row in board["rows"]]
        games = len(rows)
        wins = sum(row.get("won") is True for row in rows)
        per_arm[name] = {
            "games": games,
            "wins": wins,
            "resolved_games": sum(type(row.get("won")) is bool for row in rows),
            "truncated_games": sum(bool(row.get("truncated")) for row in rows),
            "win_rate": wins / games if paired_resolved else None,
            "mean_victory_points": (
                sum(row["victory_points"] for row in rows) / games if games else None
            ),
        }
    reference = factories[0][0]
    contrasts = None
    if paired_resolved:
        contrasts = {
            name: {
                "win_rate_difference": per_arm[name]["win_rate"] - per_arm[reference]["win_rate"],
                "mean_victory_points_difference": (
                    per_arm[name]["mean_victory_points"]
                    - per_arm[reference]["mean_victory_points"]
                ),
            }
            for name, _ in factories if name != reference
        }
    return {
        "scope": "matched_four_seat_boards_only",
        "status": "complete" if complete_panel else "incomplete",
        "board_seeds": paired,
        "reference_arm": reference,
        "arms": per_arm,
        "paired_resolved": paired_resolved,
        "contrast_vs_reference": contrasts,
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
    """Run matching four-seat boards for each public policy arm.

    ``deadline`` is absolute monotonic time. Completed protocol games that
    reach the step cap remain in ``rows`` with unresolved wins. Incomplete
    blocks keep finished rows in ``partial_rows`` and never count as paired.
    Candidate-step capture is opt-in because it requests an extra public after
    observation from the evaluator for every executed candidate action. The
    callback receives ``PairedCandidateStep(arm, board_seed, step)``.
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
            policy = factory()
            if isinstance(policy, BaselineFocalSpec):
                adapter = _GraphPolicyAdapter(_SeededBaseline(policy.kind, seed), seed)
            elif callable(getattr(policy, "plan", None)):
                adapter = _PlannedPolicyAdapter(policy, seed)
            elif all(callable(getattr(policy, name, None)) for name in ("act", "modules", "eval")):
                adapter = _GraphPolicyAdapter(policy, seed)
            else:
                raise TypeError("arm factory must return a planner, graph policy, or public baseline")

            def game_end(row, *, arm=name):
                on_game_end({"arm": arm, **row})

            def candidate_step(step, *, current=adapter, arm=name, board_seed=seed):
                current.record_candidate_step(step)
                on_candidate_step(PairedCandidateStep(arm, board_seed, step))

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
            "totals_scope": "descriptive_completed_arm_boards",
            "completed_board_seeds": [board["board_seed"] for board in boards[name]
                                      if board["status"] == "complete"],
            "boards": boards[name],
            "games": len(complete_rows),
            "finished_games": sum(board["finished_games"] for board in boards[name]),
            **_metrics(complete_rows, records[name], [
                row for board in boards[name] for row in board["partial_rows"]
            ]),
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
        "paired_summary": _paired_summary(boards, factories, seeds, paired, resolved),
        "opponent_lineup": list(lineup),
        "max_game_steps": max_game_steps,
        "deadline": deadline,
        "no_offers": True,
        "changed_detail_limit_per_board": _CHANGED_DETAIL_LIMIT,
        "arms": arms,
    }
