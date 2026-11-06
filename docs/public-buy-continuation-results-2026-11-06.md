# Public BUY continuation guard: fresh screen and retirement

The [frozen protocol](public-buy-continuation-protocol-2026-11-06.md) tested one
narrow correction to the optional purchase comparator. When its fixed score
would replace a legal BUY with a one-trade point build, the guard preserves
the original BUY only if the known post-BUY hand and public bank can still
reach that **same kind and vertex** after at most two validated maritime
trades. Before an END_TURN override it checks for any point build. Incomplete
scans preserve BUY. The hidden development-card draw is never projected, and
the next action is chosen from a fresh Observation. The flag remains off by
default on the isolated implementation branch. This screened code is retained
on local branch `cytang/buy-continuation-guard`; it is outside the working
player on the main integration branch.

The exact archived public TRAIN Observation has hand `(4, 0, 5, 1, 5)` and
legal BUY at index zero. The old option chooses a sheep-for-wheat trade
intending CITY15; raw BUY was followed by two bank trades and CITY15 in the
earlier engine replay. The new pure regression confirms that the guarded
policy returns that **original** BUY object and detects the two-trade public
route. The implementation also retains the old decision when the BUY cost
prevents the proposed build and abstains on an incomplete node-limited scan.

## Fresh paired evidence

The excluded [smoke board](../runs/swarm-2026-11-04/agent-research/public-buy-continuation-001/smoke-manifest.json)
completed all 12 games without a guard intervention. The frozen
[development manifest](../runs/swarm-2026-11-04/agent-research/public-buy-continuation-001/screen-manifest.json)
then ran raw, old option, and guarded option on eight new boards
`138296000`–`138296007`, four focal seats each, against three trading-H
opponents with no domestic offers. All **96 games resolved**, with no
truncation. The game and 9,193 public-step journals passed saved hash and
provenance checks. Screen source SHA-256 was
`6b8cad11968c81f1527da1a6154b9345728d89b334772d3c498ed3eba2550a03`;
the pinned-engine version digest was
`90789ec1a608f8ad24e5f6a17352c4ef6349da84e89477f99875daea738b6493`.

| Arm | Wins / 32 | Mean final VP | Selection mean / p95, ms |
| --- | ---: | ---: | ---: |
| Old option | 8 | 7.0625 | 1.197 / 6.091 |
| Guarded option | 8 | 7.0000 | 1.200 / 6.010 |
| Raw planner | 11 | 7.0000 | 1.146 / 5.124 |

Guarded minus old option changed **zero paired wins** and mean final VP by
**−0.0625**. The descriptive 20,000-resample whole-board 95% interval was
`[0, 0]` for observed win differences and `[−0.1875, 0]` for mean VP. The
zero-width win interval reflects the eight observed boards, not population
equivalence. Raw minus old option won three more of 32 paired games, with a
board interval `[−0.03125, 0.21875]`; raw's mean VP was lower by 0.0625.
Fixed arm order and different later trajectories make these latency figures
descriptive rather than a controlled speed comparison.

The guarded arm reported **three** `buy_continuation_same_build` decisions,
**zero** `buy_continuation_point_before_pass`, and **zero**
`buy_continuation_incomplete`/cap decisions. All three were first-action
divergences from old option at an identical recorded public state, always
old bank trade → guarded BUY. Every guarded BUY led to a point build in that
same turn, with equal known VP at the turn end in each matched pair:

| Board / seat | Old option's same-turn path | Guarded same-turn path | Final VP old / guarded |
| --- | --- | --- | ---: |
| `138296002` / 0 | Trade → SETTLEMENT50 | BUY → trade → SETTLEMENT50 | 10 / 10 |
| `138296003` / 3 | Trade → CITY12 → trade → ROAD15 | BUY → two trades → CITY10 | 9 / 7 |
| `138296004` / 3 | Trade → SETTLEMENT52 → BUY | BUY → trade → SETTLEMENT52 | 6 / 6 |

The CITY12 case is an important limit: the guard projected a conservative
route to the **same intended target** after BUY, but actual replanning selected CITY10 and omitted
the road that old option built. The turn-end point count tied, while the
guarded game ended two VP lower; both arms lost. In the SETTLEMENT52 case,
old option bought a card later in the same turn, so moving BUY earlier did
not create an extra purchase. These recorded trajectories describe the
consequence of different decisions and later state; they do not establish
the hidden card's value or a general causal effect.

## Decision

The frozen rule retires this guarded configuration because its mean VP
difference is negative and it gained no paired wins. It is **not promoted**
to the default player. The mechanism repaired the specific archived
BUY→later-build blind spot mechanically, but preserving BUY merely because
one point route exists does not compare building site quality, other
same-turn actions, or the card's uncertain value. A future purchase policy
needs a unified continuation comparison and its own fresh protocol.

The [machine-readable readout](../runs/swarm-2026-11-04/agent-research/public-buy-continuation-001/screen-readout.json),
[full paired analysis](../runs/swarm-2026-11-04/agent-research/public-buy-continuation-001/screen-analysis.json),
[saved run](../runs/swarm-2026-11-04/agent-research/public-buy-continuation-001/screen-run/paired-20261106T175826Z-3febd9c6/summary.json),
and [archive manifest](../runs/swarm-2026-11-04/agent-research/public-buy-continuation-001/archive-manifest.json)
preserve the exact evidence. The archive has 14 payload files totaling
5,366,973 bytes, with manifest SHA-256
`6e630eb0199c441b05a74db50149ce768b854cfdd1968a524b51695e5c96a7ad`.
The optional code was screened at isolated commit `f53bcc7`. A subsequent
schema-only commit `3e7bf09` made an omitted option flag explicitly record
`false` in newly expanded manifests. The saved screen declared both flag
values explicitly; it was not rerun or reinterpreted after that change.

Affected tests passed **147/147** after the metadata correction; Ruff and
diff checks passed. These engineering checks do not override the failed
fresh-game decision gate.
