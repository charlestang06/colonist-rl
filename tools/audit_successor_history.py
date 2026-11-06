"""Audit public candidate facts against executed TRAIN transitions.

Only `train-*/history.json.gz` is read. No teacher overlays, validation files,
opponent private state, or outcomes are needed for this mechanical check.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from colonist_rl.adapters.observation_io import view_from_dict
from colonist_rl.contracts import RESOURCES, Building, Road
from colonist_rl.successor_features import (
    SUCCESSOR_DIM,
    successor_feature_values,
    successor_features_for_menu,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(args.boards_root.glob("train-*/history.json.gz"))
    if not files:
        raise ValueError("no TRAIN public histories found")
    kinds: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    mismatches: list[tuple] = []
    sources = {}
    rows = known = masked = piece_checks = point_checks = winning_checks = 0
    for path in files:
        raw = path.read_bytes()
        sources[path.parent.name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(gzip.decompress(raw))
        for episode_index, episode in enumerate(data["episodes"]):
            for row_index, row in enumerate(episode["rows"]):
                if row["behavior_executed"] is not True:
                    raise ValueError("audit requires an executed transition in every row")
                before = view_from_dict(row["observation"])
                after = view_from_dict(row["transition"]["after"])
                action = before.legal_actions[row["student_action_index"]]
                label = (path.parent.name, episode_index, row_index, action.kind)
                rows += 1
                kinds[action.kind] += 1
                phases[before.context.kind] += 1
                matrix = successor_features_for_menu(before)
                if matrix.shape != (len(before.legal_actions), SUCCESSOR_DIM) or not bool(
                    torch.isfinite(matrix).all()
                ):
                    mismatches.append((*label, "menu_features"))
                values = successor_feature_values(before, action)
                if values["hand_known"]:
                    known += 1
                    predicted = tuple(
                        round(values["hand_after_" + resource] * 10)
                        for resource in RESOURCES
                    )
                    if predicted != after.hand:
                        mismatches.append((*label, "hand", predicted, after.hand))
                else:
                    masked += 1
                if action.kind != "BUY_DEVELOPMENT_CARD":
                    # Buying from a hidden deck may reveal a private VP card.
                    point_checks += 1
                    actual_gain = after.own_victory_points - before.own_victory_points
                    if values["known_point_gain"] != actual_gain:
                        mismatches.append((
                            *label, "known_point_gain", values["known_point_gain"], actual_gain
                        ))
                    winning_checks += 1
                    actual_win = int(
                        actual_gain > 0 and after.own_victory_points >= before.rules.victory_points
                    )
                    if values["wins_now"] != actual_win:
                        mismatches.append((*label, "wins_now", values["wins_now"], actual_win))
                if action.kind not in ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"):
                    continue
                own = next(player for player in after.players if player.player_id == before.player_id)
                for name, capacity, actual in (
                    ("road", 15, own.roads_left),
                    ("settlement", 5, own.settlements_left),
                    ("city", 4, own.cities_left),
                ):
                    piece_checks += 1
                    predicted = round(values[name + "_pieces_after"] * capacity)
                    if predicted != actual:
                        mismatches.append((*label, name + "_pieces", predicted, actual))
                if action.kind == "BUILD_ROAD":
                    expected = Road(action.target, before.player_id)
                    if expected not in after.roads:
                        mismatches.append((*label, "road_placement"))
                else:
                    level = 2 if action.kind == "BUILD_CITY" else 1
                    expected = Building(action.target, before.player_id, level)
                    if expected not in after.buildings:
                        mismatches.append((*label, "building_placement"))
    report = {
        "schema": "candidate-successor-train-transition-audit-v2",
        "board_count": len(files),
        "episode_count": len(files) * 4,
        "rows": rows,
        "known_hand_checks": known,
        "masked_hand_rows": masked,
        "piece_checks": piece_checks,
        "point_checks": point_checks,
        "winning_checks": winning_checks,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:30],
        "executed_action_kinds": dict(kinds),
        "phases": dict(phases),
        "history_sha256_by_board": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "history_sha256_by_board"}, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
