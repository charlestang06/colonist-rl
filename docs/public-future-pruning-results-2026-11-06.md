# Public future pruning: paired result

The [frozen mechanism](public-future-pruning-experiment-2026-11-06.md) skips
future build or bank-trade intentions only when the projected hand or
conservative bank lower bound already fails a necessary resource check. The
current legal root menu and final transition validator remain authoritative.

## Frozen screen

The [screen manifest](../runs/swarm-2026-11-04/agent-research/public-pruning-001/screen-manifest.json)
compared the raw 2,048-node planner with the same planner plus
`prune_invalid_future=True` on eight new boards, four focal seats each, against
three fixed heuristic opponents. All 64 games resolved; the journal, source,
engine version and manifest checks passed. The source SHA was
`1ff940a63abb0b4d1d4f9232cb588e51cf0c6f3a77f7a9cd8a56e075c892108f`.
The [smoke board](../runs/swarm-2026-11-04/agent-research/public-pruning-001/smoke-manifest.json)
was excluded. The [order control](../runs/swarm-2026-11-04/agent-research/public-pruning-001/order-control-manifest.json)
repeated the same eight boards with the pruned arm first; it tests sequential
timing order, not independent playing strength.

| Measure, eight-board screen | Raw | Pruned |
|---|---:|---:|
| Resolved focal games / wins | 32 / 9 | 32 / 9 |
| Attempted transitions | 429,058 | 32,739 |
| Feasible / unique successors | 32,638 / 30,990 | 32,739 / 31,091 |
| Future candidates skipped before charge | 0 | 414,217 |
| Node-limit stops | 42 | 0 |
| Searched decisions | 1,126 | 1,126 |
| Searched selection mean / p95, ms | 3.839 / 15.144 | 3.360 / 13.804 |
| All selection mean / p95, ms | 1.946 / 10.281 | 1.803 / 8.754 |

All 32 matched seat traces and outcomes were identical. Pruning exposed 101
additional feasible and unique successors under the fixed cap, but they did
not change a selected action on this panel. The searched-decision mean was
12.5% lower and p95 was 8.8% lower. These are selection-time measurements,
not whole-game speedups; skipped intentions are still scanned, so the drop in
charged transitions is not a CPU-work ratio.

With the pruned arm run first, all 64 games again resolved with identical
traces. The same transition counts held. Searched selection mean was
3.616 ms raw versus 3.186 ms pruned; p95 was 14.397 versus 12.511 ms. The
pruned mean was lower on seven of eight boards in the original order and all
eight with reversed order. The one-board smoke had slower pruned timing, so
individual small boards remain noisy.

## Decision

Enable `prune_invalid_future` by default for subsequent public-planner work.
Pass `False` explicitly to reproduce earlier search behavior. This is a
resource-safety and throughput change; the screen provides no playing-strength
gain because selected actions and results were identical. Retain the
[machine-readable readout](../runs/swarm-2026-11-04/agent-research/public-pruning-001/pruning-readout.json),
the [full first-order run](../runs/swarm-2026-11-04/agent-research/public-pruning-001/screen-run/paired-20261106T170151Z-37a3f340/summary.json),
the [reversed-order run](../runs/swarm-2026-11-04/agent-research/public-pruning-001/order-control-run/paired-20261106T170658Z-f9bb62ea/summary.json),
and the [archive hash manifest](../runs/swarm-2026-11-04/agent-research/public-pruning-001/archive-manifest.json).
