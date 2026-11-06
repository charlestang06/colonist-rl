"""Time complete public-menu inference from captured student observations.

Usage: python tools/benchmark_successor.py --history history.json.gz \
    --row 0:19 --row 0:78 --repeats 32 --output /tmp/successor-bench.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from colonist_rl.adapters.observation_io import view_from_dict
from colonist_rl.policy import (
    GraphPolicy,
    candidate_successor_from_consequence,
    consequence_from_legacy,
)
from colonist_rl.successor_features import successor_features_for_menu


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _times(function, repeats: int) -> dict[str, float]:
    for _ in range(4):
        function()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
    return {"p50_ms": statistics.median(elapsed), "p95_ms": _percentile(elapsed, 0.95)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--row", action="append", required=True, help="episode:row, zero based")
    parser.add_argument("--allow-nonturn", action="store_true")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError("at least five repetitions required")
    raw = args.history.read_bytes()
    data = json.loads(gzip.decompress(raw))
    torch.set_num_threads(1)
    torch.manual_seed(7)
    parent = consequence_from_legacy(GraphPolicy(64)).eval()
    child = candidate_successor_from_consequence(parent).eval()
    results = []
    with torch.no_grad():
        for coordinate in args.row:
            episode, row = map(int, coordinate.split(":"))
            observation = data["episodes"][episode]["rows"][row]["observation"]
            view = view_from_dict(observation)
            if (not args.allow_nonturn and
                    (view.context.kind not in ("PLAY_TURN", "TURN") or not view.context.has_rolled)):
                raise ValueError(f"{coordinate} is not a rolled public turn")
            baseline = parent(view)
            candidate = child(view)
            if not all(torch.equal(getattr(baseline, field), getattr(candidate, field))
                       for field in ("logits", "value", "memory")):
                raise AssertionError(f"zero-start parent parity failed for {coordinate}")
            results.append({
                "row": coordinate,
                "phase": view.context.kind,
                "menu_size": len(view.legal_actions),
                "action_kinds": sorted({action.kind for action in view.legal_actions}),
                "parent": _times(lambda v=view: parent(v), args.repeats),
                "candidate": _times(lambda v=view: child(v), args.repeats),
                "encoder_only": _times(
                    lambda v=view: successor_features_for_menu(v), args.repeats
                ),
            })
    report = {
        "schema": "candidate-successor-benchmark-v1",
        "history_sha256": hashlib.sha256(raw).hexdigest(),
        "repeats": args.repeats,
        "torch_threads": torch.get_num_threads(),
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
