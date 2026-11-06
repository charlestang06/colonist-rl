# Winning build search before a development-card purchase

The new optional planner path fixes one engine-confirmed missed win. It has not
established a stronger player or changed default checkpoint loading.

A TRAIN-only scan nominated 24 near-win BUY+trade observations. Nine had
conditional one-trade/build resource and placement witnesses. The first positive
was TRAIN018, seat 0, own decision 95: known VP 9 and hand
`[2,0,1,4,3]` in wood/brick/sheep/wheat/ore order. The old raw planner returned
its development-card purchase fallback without searching supported alternatives.

The keyword-only `search_build_win_before_buy=False` option preserves that default.
When enabled, only a supported standard-rule own turn with a BUY fallback and
known VP at least target minus one may search. BUY has no invented card result or
projected baseline. An alternative is returned only for a complete existing
building-win projection; an ordinary score gain cannot override BUY. Existing
root-menu validation, node/geometry caps, legal-action identity and reobservation
requirements remain.

Root observed the missing-API failure, then nine targeted checks and 43 affected
pure regressions passed. Scoped source review passed. A separate exact-fixture
regression protects the original published numeric types. These tests establish
mechanisms, not playing strength.

The first engineering attempt rejected the saved fixture before any engine action:
its earlier JSON preparation had normalized 24 integral floats to integers. Values
were equal, but canonical hashes differed. That attempt, source and old fixture
are preserved. A separately declared repair restored the exact original public
input, with no relaxed equality check or different decision information; all nine
targeted tests passed again.

Engineering attempt 002 matched all 95 original public prefix inputs/actions.
The candidate then traded two wheat for brick, reobserved, and built a settlement
at vertex 37. The actual engine terminated with winner 0 and known VP 10 after
346 total engine steps and two new planner calls. Source, runtime and input
postflight checks passed. Root session 68271 exited zero; 28 OS1 samples peaked at
236,634,112 bytes. No model, NN forward, fit, native reference or new teacher label
was used. This is an exposed TRAIN execution certificate, not held-out evidence.

Eight fresh development EVAL boards, all four seats and two matched arms
were reserved for a separate 64-game comparison: raw planner versus the optional repair,
with three persistent H opponents and no domestic offers. All 64 games completed.
Both arms won 5/32, all eight paired board differences were zero, and the frozen
bootstrap interval was [0, 0]. Only one BUY decision changed. Root session 96947
exited zero in 29.943 seconds; both guards exited zero. The 250 live OS1 samples
peaked at 255,639,552 bytes, with one accepted final ESRCH exit race. Saved-metadata
recomputation passed all 64 joins and current source/runtime hashes. Its reviewer
authored the operational writer and disclosed that limitation. The single changed
route traded four brick for wheat, then built a city; both paired arms already won
that game. The primary advancement rule failed.
The primary rule requires all games resolved, no integrity failure, positive
paired board win difference and positive 95% whole-board bootstrap lower bound.
A failed or inconclusive comparison cannot be rescued by VP or the TRAIN example.
The helper may remain experimental with its default off.

[Implementation](../src/colonist_rl/public_turn_planner.py),
[experiment ledger](../runs/swarm-2026-11-04/agent-research/buy-winning-prefix-001/Root-ledger.md),
[engine result](../runs/swarm-2026-11-04/agent-research/buy-winning-prefix-001/engine-run-002/Root-finish.json).
