# City before card purchase: executable, but no gameplay advance

The opt-in city-funding wrapper is an executable intervention. It did not
improve the first fresh complete-game screen and stays off by default.

## What changed

When the raw public planner would fall back to buying a development card, the
wrapper can choose one legal bank trade that publicly funds a city at an owned
settlement. It ranks completed two-action projections with the existing public
score. The selected root action is the original legal menu object; the next
action is chosen from a new observation. It does not know the card that would
have been drawn or compare its expected future value with the city.

The source is [city_before_purchase.py](../src/colonist_rl/city_before_purchase.py).
The [paired runner](../src/colonist_rl/paired_gameplay.py) calls the existing
complete-game evaluator with a fresh planner per arm and board.

## Real engine route and adverse control

An exact TRAIN replay matched 55 original own actions before TRAIN001 seat0's
purchase point. The selected current legal bank action traded four Wheat for one
Ore, changing the actual hand from `[0,4,3,8,2]` to `[0,4,3,4,3]`. The next
real menu offered cities; building at vertex 46 changed the hand to
`[0,4,3,2,0]` and known own VP from 4 to 5. This proves the route executes,
not that it was the best purchase decision.

The adverse TRAIN002 seat3 replay matched 126 original own actions and its
original **buy → bank → bank → city** sequence. That baseline also built a city
and gained one known VP. The wrapper would choose a city-funding bank trade at
the purchase point on this public observation, so the card's lost future option
value is a real competing explanation. Neither case supplies an optimality label.

The two-case factcheck used pinned Catanatron
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`, original frozen
TRAIN journals, persistent three-H backgrounds, no offers and no new focal
policy, teacher, model or native query. It executed 577 engine steps in 10.33
seconds. Both source/runtime postflights and the existing guard accepted; sampled
peak RSS was 255 MB. The [replay config](../runs/swarm-2026-11-04/agent-research/city-before-purchase-001/engine-factcheck-001-config.json)
and [accepted result](../runs/swarm-2026-11-04/agent-research/city-before-purchase-001/engine-factcheck-001/Root-finish.json)
are preserved with the study.

## Fresh paired gameplay

The source at commit `8707e1a2214ea2a193c51dbc4f7b05a8670e30d1` ran on
12 fresh development boards (`138210000`–`138210011`), all four seats, raw and
wrapper arms, each against the same three persistent H opponents. Offers were
disabled; bank trading remained legal. The distinct one-board smoke seed
`138209999` was excluded. Every one of the 96 planned games resolved, with no
truncation; wall time was 57.94 seconds.

| Arm | Wins / 48 | City builds | Card buys | Selection p95 |
|---|---:|---:|---:|---:|
| Raw public planner | 13 | 61 | 208 | 7.50 ms |
| Opt-in city wrapper | 11 | 68 | 194 | 7.79 ms |

The wrapper changed 12 purchase decisions to bank trades. All 12 selected trades
executed, and all 12 intended cities were built on the next own action. This
supports the local mechanical hypothesis. Across matched seats, two boards lost
one focal win each, ten tied at the board-win count, and none gained a net board
win. The paired point difference was **−2/48 = −4.17 percentage points**. One
board had an offsetting gain and loss at different seats.

The changed raw winning seats on boards `138210009` and `138210011` lost under
the wrapper. On `138210003`, one changed seat lost a raw win and another gained a
wrapper win. The wrapper arm bought 14 fewer cards and played 11 fewer Knights;
these are descriptive downstream differences, not proof that any particular
forgone card caused the losses. The score omits card uncertainty and an army
race. The screen is small and no-offer; it neither proves the wrapper harmful
in general nor supports a strength promotion.

The [frozen exploratory protocol](../runs/swarm-2026-11-04/agent-research/city-before-purchase-001/paired-screen-001/protocol.json)
and [complete per-game results](../runs/swarm-2026-11-04/agent-research/city-before-purchase-001/paired-screen-001/results.json)
are preserved with the study. Root checked the protocol/source binding, all 96
journal-to-result joins, all 12 complete paired boards and all 12 public city
continuations. The source and wrapper/runner focused checks passed 73 tests;
the affected planner regression earlier passed 103 tests. These checks support
execution and accounting, not playing strength beyond this panel.

## Decision

Do not change the default or repeat this unchanged wrapper on a larger panel.
Keep it as an optional, diagnosed mechanism. The next economic policy question
is a unified comparison of the certain city route with a development card's
uncertain future value and a sensible pass. The [capability audit](capability-audit-2026-11-06.md)
also shows that search hit its node cap in only 103 of 4,896 searched decisions
on the earlier 128-game raw panel. Instrument feasible and unique states and
reached depth before a larger-node experiment.
