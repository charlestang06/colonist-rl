# Public purchase option: fresh development screen

The [frozen comparator](public-option-purchase-design-2026-11-06.md) tests a
public card option credit against passing or one legal bank trade followed by
a projected city or settlement. It never reads an unseen draw. The actual
first action must be a current legal root, and the player reobserves before
any intended build. This treatment remains off by default.

## Scope and result

The [excluded smoke](../runs/swarm-2026-11-04/agent-research/public-option-001/smoke-manifest.json)
completed one board and confirmed two real trade→build continuations. The
[frozen development panel](../runs/swarm-2026-11-04/agent-research/public-option-001/screen-manifest.json)
then compared raw and option policies on 12 new boards × four seats against
three fixed heuristic opponents, with no domestic offers. All 96 games
resolved, the 8,226 public candidate steps and game journal passed integrity
checks, and source and pinned-engine identities matched before and after. The
frozen source SHA was
`267120fd96107cae3bbad2d44c409607a67f23b96155fe0a291cc11bf9d31a5c`.

| Measure | Raw | Option |
|---|---:|---:|
| Resolved focal games / wins | 48 / 13 | 48 / 16 |
| Mean final victory points | 6.875 | 6.958 |
| All selection mean / p95, ms | 1.869 / 9.572 | 2.240 / 11.859 |

Four matched games were won only by option, one only by raw, and 43 had the
same win result. The paired win difference was +3/48 and mean victory-point
difference +0.083. A descriptive 20,000-resample **whole-board** bootstrap
gave 95% intervals of [−0.021, +0.146] for win rate and [−0.083, +0.250]
for mean victory points. Both include zero. This modest development screen
does not establish playing-strength superiority.

The option changed 50 recorded decisions: 26 trade→build intentions and 24
passes. The intended building was executed as the next own action in 25 of
26 trade cases. In the exception, board 138250001 seat 3 already had a legal
winning settlement after the first trade; the parent instead made two more
bank trades and won by building a city later in the same turn. An
[exact engine replay](../runs/swarm-2026-11-04/agent-research/public-option-001/failure-investigation.json)
confirmed the settlement was on the real menu. The [direct-win guard](agent-research-roadmap.md)
is a separate planner repair, not evidence for this purchase score.

## Conceptual decision audit

The [public full-turn trace audit](../runs/swarm-2026-11-04/agent-research/public-option-001/screen-trace-audit.json)
found 23 first-action divergences among 48 matched seat games: 16 raw BUY →
option bank trade, and seven raw BUY → option END_TURN. The other 25 traces
matched. All 16 chosen trade roots in those first divergences led immediately
to an actual city or settlement. At the end of the divergent own turn, option
had more known victory points in 11 pairs, raw in two, and they were equal in
ten. These are turn-level descriptions, not estimates of long-run value.

Raw BUY still led to a build later in the **same** own turn in three divergent
games, including a city. On board 138250006 seat 1, raw bought a card, made
two bank trades and built CITY52; option traded and built the same city sooner.
Raw ended that turn at seven known points versus option's six, and finished
the game at eight versus seven; both lost. On board 138250008 seat 3, raw BUY
also ended the turn a point ahead of option PASS. These are concrete adverse
cases for a score that projects a trade→build but omits a BUY→later-build
continuation. A positive counterexample on board 138250003 seat 0 had option
build a settlement after a shared city while raw bought a card, ending the
turn at five versus four known points. The trace records exact action lines
and hashes; its same-state comparison rests on matching recorded prefixes and
decision fields because it does not store a complete public board snapshot.

## Decision and next question

Keep `PublicOptionPurchasePolicy` optional. The predeclared continuation gate
missed one next build, the board-level effect is uncertain, and fresh adverse
turns show a structural value blind spot. Do not tune the fixed card credit to
these development boards or enlarge this same screen. A distinct next
hypothesis should compare a known BUY cost **and its possible later bank/build
continuations in the same turn** with trade/build/pass, while retaining a
public uncertainty credit for the unknown card. It needs its own frozen fresh
panel and an independent confirmation before default promotion.

The [machine-readable readout](../runs/swarm-2026-11-04/agent-research/public-option-001/screen-readout.json),
[saved run](../runs/swarm-2026-11-04/agent-research/public-option-001/screen-run/paired-20261106T160433Z-cea34047/summary.json),
[audit script](../runs/swarm-2026-11-04/agent-research/public-option-001/trace_audit.py),
and [archive hash manifest](../runs/swarm-2026-11-04/agent-research/public-option-001/archive-manifest.json)
retain the evidence.
