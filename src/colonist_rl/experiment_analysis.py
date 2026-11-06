"""Read-only paired comparisons from saved public experiment artifacts.

Outcomes use only complete boards shared by each arm and the declared reference.
Trace comparisons use the captured before/action/after fields, never engine state.
Changed actions describe divergence; they do not establish better play.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from hashlib import sha256
from pathlib import Path

_BEFORE_FIELDS = (
    "version", "phase", "turn_number", "turn_owner", "hand", "known_own_vp",
)
_ACTION_FIELDS = (
    "kind", "target", "resource", "second_resource", "partner",
    "give", "receive", "offer_id",
)
_AFTER_FIELDS = (
    "version", "phase", "hand", "known_own_vp", "target_building_level",
)
_TRACE_CATEGORIES = (
    "same_trace", "same_before_state_different_action",
    "prior_public_state_divergence", "unequal_trace_length",
)
_DEFAULT_DETAIL_LIMIT = 32
_MAX_DETAIL_LIMIT = 64
_BOOTSTRAP_MIN_BOARDS = 8
_BOOTSTRAP_RESAMPLES = 20_000
_BOOTSTRAP_SEED = 20_261_106


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc


def _verify_digest(path: Path, expected):
    if type(expected) is not str:
        raise ValueError(f"{path.name} digest is missing")
    digest = sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    if digest.hexdigest() != expected:
        raise ValueError(f"{path.name} digest mismatch")


def _read_jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as source:
            rows = []
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid {path.name} line {line_number}") from exc
                if type(row) is not dict:
                    raise ValueError(f"{path.name} line {line_number} must be an object")
                rows.append(row)
            return rows
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc


def _fields(value, names, *, label):
    if type(value) is not dict or any(name not in value for name in names):
        raise ValueError(f"invalid captured public {label}: missing fields")
    return {name: value[name] for name in names}


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _resource_vector(value):
    return type(value) is list and len(value) == 5 and all(
        _nonnegative_int(quantity) for quantity in value
    )


def _require_public(condition, label):
    if not condition:
        raise ValueError(f"invalid captured public {label}")


def _public_trace_step(step):
    """Project a persisted step onto the capture writer's explicit field list."""
    if type(step) is not dict:
        raise ValueError("invalid captured public trace step")
    if type(step.get("terminated")) is not bool or type(step.get("truncated")) is not bool:
        raise ValueError("invalid captured public outcome flags")
    before = _fields(step.get("before"), _BEFORE_FIELDS, label="before")
    action = _fields(step.get("action"), _ACTION_FIELDS, label="action")
    after = _fields(step.get("after"), _AFTER_FIELDS, label="after")
    for label, state in (("before", before), ("after", after)):
        _require_public(_nonnegative_int(state["version"]), f"{label}.version")
        _require_public(
            type(state["phase"]) is str and bool(state["phase"].strip()), f"{label}.phase"
        )
        _require_public(_resource_vector(state["hand"]), f"{label}.hand")
        _require_public(_nonnegative_int(state["known_own_vp"]), f"{label}.known_own_vp")
    _require_public(_nonnegative_int(before["turn_number"]), "before.turn_number")
    _require_public(_nonnegative_int(before["turn_owner"]), "before.turn_owner")
    _require_public(
        after["target_building_level"] is None
        or _nonnegative_int(after["target_building_level"]),
        "after.target_building_level",
    )
    _require_public(
        type(action["kind"]) is str and bool(action["kind"].strip()), "action.kind"
    )
    for label in ("target", "partner"):
        _require_public(
            action[label] is None or _nonnegative_int(action[label]), f"action.{label}"
        )
    for label in ("resource", "second_resource", "offer_id"):
        _require_public(
            action[label] is None or type(action[label]) is str, f"action.{label}"
        )
    for label in ("give", "receive"):
        _require_public(_resource_vector(action[label]), f"action.{label}")
    return {
        "before": before,
        "action": action,
        "after": after,
        "terminated": step["terminated"],
        "truncated": step["truncated"],
    }


def _first_trace_divergence(reference_steps, arm_steps):
    """Return the earliest difference between captured public action traces."""
    reference_public = tuple(_public_trace_step(step) for step in reference_steps)
    arm_public = tuple(_public_trace_step(step) for step in arm_steps)
    for index, (reference, arm) in enumerate(zip(reference_public, arm_public)):
        if reference["before"] != arm["before"]:
            return {
                "category": "prior_public_state_divergence", "step_index": index,
                "stage": "before", "reference_state": reference["before"],
                "arm_state": arm["before"],
            }
        if reference["action"] != arm["action"]:
            return {
                "category": "same_before_state_different_action", "step_index": index,
                "before": reference["before"],
                "reference_action": reference["action"], "arm_action": arm["action"],
            }
        if reference["after"] != arm["after"]:
            return {
                "category": "prior_public_state_divergence", "step_index": index,
                "stage": "after", "reference_state": reference["after"],
                "arm_state": arm["after"],
            }
        reference_outcome = {key: reference[key] for key in ("terminated", "truncated")}
        arm_outcome = {key: arm[key] for key in ("terminated", "truncated")}
        if reference_outcome != arm_outcome:
            return {
                "category": "prior_public_state_divergence", "step_index": index,
                "stage": "outcome", "reference_outcome": reference_outcome,
                "arm_outcome": arm_outcome,
            }
    if len(reference_steps) != len(arm_steps):
        return {
            "category": "unequal_trace_length",
            "step_index": min(len(reference_steps), len(arm_steps)),
            "reference_length": len(reference_steps), "arm_length": len(arm_steps),
        }
    return {"category": "same_trace"}


def _keyed_rows(rows, arms, seeds, *, label):
    keyed = {}
    for row in rows:
        try:
            arm, seed, seat = row["arm"], row["board_seed"], row["seat"]
        except KeyError as exc:
            raise ValueError(f"{label} row lacks arm/board_seed/seat") from exc
        if (
            type(arm) is not str or arm not in arms
            or type(seed) is not int or seed not in seeds
            or type(seat) is not int or seat not in range(4)
        ):
            raise ValueError(f"{label} row has an invalid arm, board, or seat")
        key = (arm, seed, seat)
        if label == "game" and key in keyed:
            raise ValueError(f"duplicate game row for {key}")
        if label == "game":
            keyed[key] = row
        else:
            keyed.setdefault(key, []).append(row)
    return keyed


def _complete_seeds(arm_report, seeds):
    boards = arm_report.get("boards")
    if type(boards) is not list:
        raise ValueError("summary arm lacks board reports")
    complete, seen = set(), set()
    for board in boards:
        if type(board) is not dict or type(board.get("board_seed")) is not int:
            raise ValueError("summary has an invalid board report")
        seed = board["board_seed"]
        if seed not in seeds or seed in seen:
            raise ValueError("summary has a duplicate or unknown board seed")
        seen.add(seed)
        if board.get("status") == "complete":
            if board.get("games") != 4:
                raise ValueError("complete board report must contain four games")
            complete.add(seed)
    return complete


def _paired_games(seeds, reference_complete, arm_complete, games, reference, arm):
    matched_seeds = [seed for seed in seeds if seed in reference_complete and seed in arm_complete]
    pairs = []
    for seed in matched_seeds:
        for seat in range(4):
            reference_key, arm_key = (reference, seed, seat), (arm, seed, seat)
            if reference_key not in games or arm_key not in games:
                raise ValueError(f"complete board {seed} lacks a matched game at seat {seat}")
            pairs.append((seed, seat, games[reference_key], games[arm_key]))
    return matched_seeds, pairs


def _finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _paired_outcomes(pairs):
    wins = {"arm_only_win": 0, "reference_only_win": 0, "same_result": 0}
    win_pairs = vp_pairs = 0
    win_delta = vp_delta = 0
    for _, _, reference, arm in pairs:
        reference_won, arm_won = reference.get("won"), arm.get("won")
        if type(reference_won) is bool and type(arm_won) is bool:
            win_pairs += 1
            difference = int(arm_won) - int(reference_won)
            win_delta += difference
            wins[("arm_only_win" if difference > 0 else
                  "reference_only_win" if difference < 0 else "same_result")] += 1
        reference_vp, arm_vp = reference.get("victory_points"), arm.get("victory_points")
        if _finite_number(reference_vp) and _finite_number(arm_vp):
            vp_pairs += 1
            vp_delta += arm_vp - reference_vp
    return {
        "resolved_win_pair_count": win_pairs,
        "paired_win_rate_difference": (
            win_delta / win_pairs if win_pairs and win_pairs == len(pairs) else None
        ),
        "win_rate_difference_scope": "all_matched_game_pairs_resolved",
        "win_difference_counts": wins,
        "win_difference_counts_scope": "resolved_matched_game_pairs",
        "victory_points_pair_count": vp_pairs,
        "mean_victory_points_difference": vp_delta / vp_pairs if vp_pairs else None,
        "victory_points_difference_scope": "matched_game_pairs_with_finite_vp",
    }


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (
        position - lower
    )


def _board_bootstrap(pairs, matched_seeds):
    """Descriptive paired intervals resampling whole four-seat board blocks."""
    board_count = len(matched_seeds)
    report = {
        "scope": "matched_complete_four_seat_boards_resampled_with_replacement",
        "method": "percentile", "confidence": 0.95,
        "seed": _BOOTSTRAP_SEED, "min_boards": _BOOTSTRAP_MIN_BOARDS,
        "board_count": board_count, "resamples": 0,
        "status": "insufficient_boards" if board_count < _BOOTSTRAP_MIN_BOARDS else "unavailable",
        "paired_win_rate_difference": None,
        "mean_victory_points_difference": None,
    }
    if board_count < _BOOTSTRAP_MIN_BOARDS:
        return report
    by_board = {seed: [] for seed in matched_seeds}
    for seed, seat, reference, arm in pairs:
        by_board[seed].append((seat, reference, arm))
    if any(len(by_board[seed]) != 4 for seed in matched_seeds):
        raise ValueError("matched bootstrap board must contain four paired seats")
    win_values, vp_values = [], []
    for seed in matched_seeds:
        rows = by_board[seed]
        win_values.append(
            sum(int(arm["won"]) - int(reference["won"])
                for _, reference, arm in rows) / 4
            if all(type(reference.get("won")) is bool and type(arm.get("won")) is bool
                   for _, reference, arm in rows) else None
        )
        vp_values.append(
            sum(arm["victory_points"] - reference["victory_points"]
                for _, reference, arm in rows) / 4
            if all(_finite_number(reference.get("victory_points"))
                   and _finite_number(arm.get("victory_points"))
                   for _, reference, arm in rows) else None
        )
    have_wins = all(value is not None for value in win_values)
    have_vp = all(value is not None for value in vp_values)
    if not (have_wins or have_vp):
        return report
    report["status"] = "descriptive"
    report["resamples"] = _BOOTSTRAP_RESAMPLES
    samples = {"paired_win_rate_difference": [], "mean_victory_points_difference": []}
    rng = random.Random(_BOOTSTRAP_SEED)
    for _ in range(_BOOTSTRAP_RESAMPLES):
        picks = [rng.randrange(board_count) for _ in range(board_count)]
        if have_wins:
            samples["paired_win_rate_difference"].append(
                sum(win_values[index] for index in picks) / board_count
            )
        if have_vp:
            samples["mean_victory_points_difference"].append(
                sum(vp_values[index] for index in picks) / board_count
            )
    for name, values in samples.items():
        if values:
            report[name] = [_percentile(values, 0.025), _percentile(values, 0.975)]
    return report


def _search_work(arm_report):
    available = arm_report.get("search_diagnostics_record_count")
    unavailable = arm_report.get("search_diagnostics_unavailable_count")
    counted = (
        type(available) is int and type(unavailable) is int
        and available >= 0 and unavailable >= 0
    )
    denominator = available + unavailable if counted else 0
    return {
        "scope": arm_report.get("planning_metrics_scope"),
        "attempted_selection_count": arm_report.get("selection_latency_ms", {}).get("count"),
        "plan_nodes": arm_report.get("plan_nodes"),
        "geometry_visits": arm_report.get("geometry_visits"),
        "wrapper_nodes": arm_report.get("wrapper_nodes"),
        "wrapper_geometry_visits": arm_report.get("wrapper_geometry_visits"),
        "selection_latency_ms": arm_report.get("selection_latency_ms"),
        "searched_selection_latency_scope": arm_report.get("searched_selection_latency_scope"),
        "searched_selection_latency_ms": arm_report.get("searched_selection_latency_ms"),
        "searched_plan_reason_counts": arm_report.get("searched_plan_reason_counts"),
        "search_stop_counts": arm_report.get("search_stop_counts"),
        "diagnostics": {
            "scope": arm_report.get("search_diagnostics_scope"),
            "record_count": available,
            "unavailable_count": unavailable,
            "coverage_fraction": available / denominator if denominator else None,
            "totals": arm_report.get("search_diagnostics_totals"),
            "depth_histograms": arm_report.get("search_diagnostics_depth_histograms"),
        },
    }


def _trace_summary(pairs, traces, reference, arm, detail_limit):
    counts = dict.fromkeys(_TRACE_CATEGORIES, 0)
    details = []
    omitted = 0
    for seed, seat, _, _ in pairs:
        reference_steps = traces.get((reference, seed, seat), ())
        arm_steps = traces.get((arm, seed, seat), ())
        if not reference_steps or not arm_steps:
            raise ValueError(f"missing captured trace for matched game {seed} seat {seat}")
        divergence = _first_trace_divergence(
            reference_steps, arm_steps
        )
        category = divergence["category"]
        counts[category] += 1
        if category != "same_trace":
            if len(details) < detail_limit:
                details.append({"board_seed": seed, "seat": seat, **divergence})
            else:
                omitted += 1
    return {
        "scope": "matched_complete_board_seat_games_captured_public_fields",
        "game_count": len(pairs), "category_counts": counts,
        "details": details, "omitted_detail_count": omitted,
        "detail_limit": detail_limit,
    }


def analyze_run(run_dir, *, detail_limit=_DEFAULT_DETAIL_LIMIT):
    """Compare each arm with the declared reference using saved run data only."""
    if type(detail_limit) is not int or not 0 <= detail_limit <= _MAX_DETAIL_LIMIT:
        raise ValueError(f"detail_limit must be an integer in [0, {_MAX_DETAIL_LIMIT}]")
    directory = Path(run_dir).expanduser().resolve()
    manifest = _read_json(directory / "manifest.json")
    summary = _read_json(directory / "summary.json")
    if type(manifest) is not dict or (
        manifest.get("format") != "colonist-rl.paired-run"
        or type(manifest.get("version")) is not int or manifest["version"] != 1
    ):
        raise ValueError("unsupported saved run manifest")
    if type(summary) is not dict or (
        summary.get("format") != "colonist-rl.paired-summary"
        or type(summary.get("version")) is not int or summary["version"] != 1
    ):
        raise ValueError("unsupported saved run summary")
    _verify_digest(directory / "manifest.json", summary.get("manifest_sha256"))
    if summary.get("status") not in ("complete", "incomplete"):
        raise ValueError(f"run status {summary.get('status')!r} is not analyzable")
    validated = manifest.get("validated_manifest")
    if type(validated) is not dict or type(validated.get("arms")) is not dict:
        raise ValueError("saved run manifest lacks validated arms")
    arms = validated["arms"]
    reference = validated.get("reference_arm")
    if type(reference) is not str or reference not in arms or len(arms) < 2:
        raise ValueError("saved run manifest has no declared reference_arm")
    paired_summary = summary.get("paired_summary")
    if type(paired_summary) is not dict or paired_summary.get("reference_arm") != reference:
        raise ValueError("reference_arm mismatch between manifest and summary")
    arm_reports = summary.get("arms")
    if (type(arm_reports) is not dict or set(arm_reports) != set(arms)
            or any(type(report) is not dict for report in arm_reports.values())):
        raise ValueError("summary arms do not match the saved manifest")
    seeds = validated.get("board_seeds")
    if (type(seeds) is not list or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("saved run manifest has invalid board seeds")
    capture = validated.get("capture_candidate_steps")
    if type(capture) is not bool:
        raise ValueError("saved run manifest lacks a trace capture setting")
    if summary.get("journal_valid") is not True:
        raise ValueError("run has no validated game journal")

    _verify_digest(directory / "games.jsonl", summary.get("games_jsonl_sha256"))
    rows = _read_jsonl(directory / "games.jsonl")
    if type(summary.get("journal_games")) is int and len(rows) != summary["journal_games"]:
        raise ValueError("games.jsonl count disagrees with summary")
    games = _keyed_rows(rows, arms, set(seeds), label="game")
    traces = {}
    if capture:
        _verify_digest(
            directory / "candidate_steps.jsonl", summary.get("candidate_steps_jsonl_sha256")
        )
        steps = _read_jsonl(directory / "candidate_steps.jsonl")
        if type(summary.get("candidate_steps")) is int and len(steps) != summary["candidate_steps"]:
            raise ValueError("candidate_steps.jsonl count disagrees with summary")
        traces = _keyed_rows(steps, arms, set(seeds), label="trace")

    complete = {name: _complete_seeds(arm_reports[name], set(seeds)) for name in arms}
    comparisons = {}
    for arm in arms:
        if arm == reference:
            continue
        matched_seeds, pairs = _paired_games(
            seeds, complete[reference], complete[arm], games, reference, arm
        )
        comparison = {
            "matched_board_seeds": matched_seeds,
            "matched_game_count": len(pairs),
            **_paired_outcomes(pairs),
            "board_bootstrap_95": _board_bootstrap(pairs, matched_seeds),
        }
        if capture:
            comparison["trace_divergence"] = _trace_summary(
                pairs, traces, reference, arm, detail_limit
            )
        comparisons[arm] = comparison
    return {
        "format": "colonist-rl.paired-analysis", "version": 1,
        "run_dir": str(directory), "run_status": summary.get("status"),
        "reference_arm": reference,
        "comparison_scope": "pairwise_matched_complete_board_seat_games",
        "trace_capture": capture,
        "search_work": {name: _search_work(arm_reports[name]) for name in arms},
        "comparisons": comparisons,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a saved paired public experiment")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_run(args.run_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
