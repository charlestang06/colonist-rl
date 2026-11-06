"""Read-only paired board summaries for the prospective teacher-strength ladder.

The gate is operational teacher suitability, not a claim that raw beats a
native policy in a population. EVAL rows and native choices never become labels.
"""

from __future__ import annotations

from .experiment_analysis import _board_bootstrap, _paired_outcomes
from .native_opponents import NATIVE_IDS


def paired_panel_summary(board_seeds, arms, game_rows, *, native_identity=None):
    """Summarize exact board/seat pairs, preserving missing and unresolved rows."""
    if (not isinstance(board_seeds, (list, tuple)) or not board_seeds
            or any(type(seed) is not int for seed in board_seeds)
            or len(set(board_seeds)) != len(board_seeds)):
        raise ValueError("unique explicit board seeds are required")
    if (not isinstance(arms, (list, tuple)) or "raw" not in arms or "parent" not in arms
            or len(set(arms)) != len(arms)):
        raise ValueError("raw and parent must be paired arms")
    if native_identity is not None and native_identity not in NATIVE_IDS:
        raise ValueError("unknown native opponent identity")
    indexed = {}
    for row in game_rows:
        key = (row.get("arm"), row.get("board_seed"), row.get("seat"))
        if (key[0] not in arms or type(key[1]) is not int or key[1] not in board_seeds
                or type(key[2]) is not int or key[2] not in range(4) or key in indexed):
            raise ValueError("duplicate or unexpected arm/board/seat game row")
        indexed[key] = row
    matched = [seed for seed in board_seeds if all(
        (arm, seed, seat) in indexed for arm in arms for seat in range(4)
    )]
    selected = {arm: [indexed[(arm, seed, seat)] for seed in matched for seat in range(4)]
                for arm in arms}
    arm_summary = {}
    for arm, rows in selected.items():
        finite_vp = [row["victory_points"] for row in rows
                     if type(row.get("victory_points")) in (int, float)]
        arm_summary[arm] = {
            "games": len(rows),
            "wins": sum(row.get("won") is True for row in rows),
            "resolved": sum(type(row.get("won")) is bool for row in rows),
            "truncated": sum(row.get("truncated") is True for row in rows),
            "invalid": sum(bool(row.get("illegal") or row.get("illegal_action")) for row in rows),
            "mean_vp": sum(finite_vp) / len(finite_vp) if finite_vp else None,
        }
    contrasts = {}
    for arm in arms:
        if arm == "raw":
            continue
        pairs = [(seed, seat, indexed[(arm, seed, seat)], indexed[("raw", seed, seat)])
                 for seed in matched for seat in range(4)]
        outcomes = _paired_outcomes(pairs)
        uncertainty = _board_bootstrap(pairs, matched)
        contrasts[arm] = {
            "win_count_difference": arm_summary["raw"]["wins"] - arm_summary[arm]["wins"],
            "paired_win_rate_difference": outcomes["paired_win_rate_difference"],
            "mean_vp_difference": outcomes["mean_victory_points_difference"],
            "board_bootstrap_95": uncertainty["paired_win_rate_difference"],
            "board_bootstrap_vp_95": uncertainty["mean_victory_points_difference"],
            "board_bootstrap_status": uncertainty["status"],
        }
    native_wins = None
    if native_identity is not None:
        native_wins = 0
        for row in selected["raw"]:
            native = row.get("native_opponents")
            if type(native) is not list or len(native) != 1:
                raise ValueError("raw game must have exactly one native opponent")
            metadata = native[0].get("metadata", {})
            native_seat = (row["seat"] + 1) % 4
            if (metadata.get("identity") != native_identity
                    or metadata.get("seat") != native_seat
                    or metadata.get("full_rule_equivalence") is not False):
                raise ValueError("native opponent identity or balanced seat mapping changed")
            native_wins += row.get("winner") == native_seat
    complete = len(matched) == len(board_seeds) and all(
        report["resolved"] == report["games"] == 4 * len(board_seeds)
        and report["truncated"] == report["invalid"] == 0
        for report in arm_summary.values()
    )
    return {
        "role": "EVAL_DIAGNOSTIC_ONLY",
        "board_seeds": list(board_seeds),
        "requested_boards": len(board_seeds),
        "matched_boards": len(matched),
        "complete": complete,
        "arms": arm_summary,
        "raw_minus": contrasts,
        "native_identity": native_identity,
        "native_opponent_wins_in_raw_games": native_wins,
    }


def _required_panel(panel, *, boards, arms, native_identity=None):
    return (panel.get("complete") is True
            and panel.get("requested_boards") == panel.get("matched_boards") == boards
            and set(panel.get("arms", {})) == set(arms)
            and panel.get("native_identity") == native_identity)


def stage1_advance(h_panel, gnt_panel):
    """Apply the frozen 8+8 board public-opponent strength gate."""
    reasons = []
    for label, control, panel in (("H", "H", h_panel), ("GNT", "GNT", gnt_panel)):
        if not _required_panel(panel, boards=8, arms=("raw", "parent", control)):
            reasons.append(f"{label}:incomplete_or_unmatched")
            continue
        for arm, vp_minimum in ((control, 0.25), ("parent", 0.5)):
            comparison = panel["raw_minus"][arm]
            if comparison["win_count_difference"] < 4:
                reasons.append(f"{label}:raw_minus_{arm}_wins_below_4")
            if comparison["mean_vp_difference"] is None or (
                comparison["mean_vp_difference"] < vp_minimum
            ):
                reasons.append(f"{label}:raw_minus_{arm}_vp_below_{vp_minimum}")
            interval = comparison["board_bootstrap_95"]
            if (comparison["board_bootstrap_status"] != "descriptive"
                    or interval is None or interval[0] < 0):
                reasons.append(f"{label}:raw_minus_{arm}_board_lower_below_0")
    return {"advance": not reasons, "reasons": reasons}


def stage2_interim_stop(vf_panel, ab_panel):
    """After exactly eight boards per native lineup, stop a clear downside."""
    reasons = []
    for label, identity, panel in (("VF", NATIVE_IDS[0], vf_panel),
                                   ("AB", NATIVE_IDS[1], ab_panel)):
        if not _required_panel(panel, boards=8, arms=("raw", "parent"),
                               native_identity=identity):
            reasons.append(f"{label}:incomplete_or_unmatched")
            continue
        comparison = panel["raw_minus"]["parent"]
        if (comparison["win_count_difference"] <= -4
                and comparison["mean_vp_difference"] <= -0.5):
            reasons.append(f"{label}:raw_trails_parent_wins_and_vp")
        if (panel["native_opponent_wins_in_raw_games"]
                - panel["arms"]["raw"]["wins"] >= 8):
            reasons.append(f"{label}:native_wins_exceed_raw_by_8")
    return {"stop": bool(reasons), "reasons": reasons}


def stage2_qualification(vf_panel, ab_panel):
    """Apply the frozen 16+16 board hard-opponent teacher gate."""
    reasons = []
    for label, identity, panel in (("VF", NATIVE_IDS[0], vf_panel),
                                   ("AB", NATIVE_IDS[1], ab_panel)):
        if not _required_panel(panel, boards=16, arms=("raw", "parent"),
                               native_identity=identity):
            reasons.append(f"{label}:incomplete_or_unmatched")
            continue
        comparison = panel["raw_minus"]["parent"]
        if comparison["win_count_difference"] < 6:
            reasons.append(f"{label}:raw_minus_parent_wins_below_6")
        if (comparison["mean_vp_difference"] is None
                or comparison["mean_vp_difference"] < 0.5):
            reasons.append(f"{label}:raw_minus_parent_vp_below_0.5")
        interval = comparison["board_bootstrap_95"]
        if (comparison["board_bootstrap_status"] != "descriptive"
                or interval is None or interval[0] < 0):
            reasons.append(f"{label}:raw_minus_parent_board_lower_below_0")
        if panel["native_opponent_wins_in_raw_games"] > panel["arms"]["raw"]["wins"]:
            reasons.append(f"{label}:native_wins_exceed_raw")
    return {"qualified": not reasons, "reasons": reasons}
