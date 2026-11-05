# Training status — November 5, 2026

The learner remains weaker than the fixed heuristic opponents. All completed
experiments and failed hypotheses are preserved; passing tests is not a strength
claim. This is an intermediate checkpoint while the next learning wave runs.
Every gameplay result below uses the legacy Catanatron3.2.1 engine. The requested
upstream migration is isolated; new-engine results will form a separate cohort.

| Experiment | Teacher agreement on non-forced choices | Heuristic-opponent outcomes | Interpretation |
|---|---|---|---|
| Original two-epoch pilot, seeds11/29 |32/83 each|0/8 wins each; meanVP3.375/3.250|Weak initial baseline|
| Separate twenty-epoch uniform diagnostic |43/83;40/83|0/8 each; meanVP3.875/3.125|More agreement without established strength|
| Controlled twenty-epoch uniform, new four-board panel |50/77 each|0/16 each; meanVP3.625/3.500|Reference for balanced supervision|
| Controlled twenty-epoch phase-balanced |49/77;48/77|0/16 each; meanVP4.125/4.500|Higher VP in this small panel, no wins|

The controlled four-arm study completed all requested epochs and3,380 updates
per arm, with identical per-seed initial weights and datasets, in559.538/900seconds.
Its fixed heuristic comparator won3/16 with meanVP6.5625. Boards shared across
seats are dependent; these are descriptive results. The new77-choice validation
set differs from the earlier83-choice set and their percentages are not a
matched improvement comparison.

The baseline public-observation audit replayed36/36 sampled choices exactly.
It found24 reasonable,11 ambiguous and one clearly bad choice. Neither the
two-epoch nor twenty-epoch uniform students selected bank trades on their known
validation boards (0/143 and0/202 opportunities respectively). No pass skipped
a directly legal purchase. Intermediate conversions remain a concrete concern.

The follow-up training audit reproduced5,000/5,000 teacher labels and983/983
action vectors. Relevant gradients reach the own-hand inputs. Teacher jitter
and exact-score ties create legitimate label variation across RNG streams.
Measured prefix replay accounts for68.1% of forward time, while graph construction
accounts for40.4%; these overlapping measurements must not be added. A tiny
full-context training probe improved overall fit but reached only2/9 bank labels
for both initializations after100epochs. This identifies neither a broken
encoding nor a proven model-capacity limit.

The user authorized the next wave: profile-guided batching/static-feature caching;
a separate public consequence-feature model; audited teacher labels on student
states with dataset aggregation; and progressively harder PPO opponents afterward.
The current PPO reward is terminal winning plus victory-point potential shaping.
Behavioral cloning does not use that reward. Reward changes require their own
measured comparison. Curriculum code is implemented and task reviewed; no sustained PPO curriculum
has run. The latest priority is controlled student-state labels, then scaling
toward100,000 non-forced decisions; PPO waits for consistently useful imitation.

The [experiment archive](../runs/swarm-2026-11-04/README.md) links raw outcomes,
checkpoints, datasets, traces, diagnostics, manifests and review records.
[Artifact locations](../runs/swarm-2026-11-04/artifact-locations.json) map historical
scratch paths to preserved bytes. The original internal work horizon elapsed
during a long tool/approval wait; catch-up work and the later explicit user
request are recorded separately rather than relabeling elapsed time as training.


## Expanded supervision and public hand consequences

Two expanded teacher collections contain10,000 observations each, including
5,862 and6,041 non-forced choices. All20,000 stored labels replay exactly.
Forced observations remain in every recurrent history. These counts are smaller
than the next proposed100,000 **non-forced** decision target.

The legacy control attempted40epochs under840seconds per seed. Both missed that
target; the greatest common completed checkpoint was30epochs. The separately
run hand-recipe model attempted30epochs under1,200seconds per seed and also
missed its primary target. Its predeclared completion-only fallback selected
**15 full epochs** across all four control/feature arms, before any game results.
Partial-last16-epoch feature weights were retained and excluded from this match.

| Checkpoint cohort, seeds11 /29 | Wins per64 games | Mean VP | Held-out non-forced matches |
|---|---|---|---|
| Expanded legacy operational partial-last |5 /0|5.078125 /4.828125|Different work; not the matched endpoint|
| Exact full30 legacy reference |5 /2|5.1875 /4.65625|357 /343 of593|
| Matched full15 legacy control |4 /0|5.09375 /4.21875|347 /330 of593|
| Matched full15 hand-recipe features |7 /3|5.453125 /4.8125|357 /343 of593|
| Fixed heuristic reference |10|6.375|Teacher reference|

All320 games in the matched feature experiment completed without truncation.
The exact30 reference adds128 completed games. They use the same16-board,
four-seat validation panel,72010000..72010015, against three fixed heuristics.
The heuristic reference reproduced identical outcome rows. These are dependent,
board-shared descriptive samples; neither feature seed surpassed the heuristic.
Different checkpoint cohorts must not be conflated.

The feature comparison uses identical training data, original base weights,
phase weights, learning rate, sequence length, batch size, completed updates and
supervised presentations within each initialization. The added34 coefficients
start at zero; all1,024 initial held-out logits/value/memory outputs match exactly.
Only hand/recipe features are active; connectivity/piece features remain a
separate experiment. Runtime ceilings differ, so this is not equal compute.
Historical timing comparisons also cannot isolate feature-branch overhead.

On the same68 held-out maritime targets, control→feature exact agreement is
6→15 for seed11 and5→19 for seed29. END_TURN agreement is35→36 and36→34 of38.
Game bank selections increased103/1157→181/1145 and95/1122→175/1001 legal-menu
decisions. Resource-sufficient recipe transitions increased36→132 and16→129.
These are public hand arithmetic counts, not proof of connected legal builds,
useful trades or strategic quality. No graph arm passed over a directly legal
purchase in those games.

The separate24-state conceptual audit found student11:17reasonable,3ambiguous,
4clearly bad; student29:11reasonable,11ambiguous,2clearly bad. Teacher choices
were22reasonable and2ambiguous. The deterministic earliest-case selection spans
four teacher-state boards and is not representative of student-game distributions.
Concrete mistakes include buying redundant brick with settlement brick already
covered and no road pieces, while wheat would permit a public city upgrade.
Other teacher disagreements support reasonable alternative plans; labels are not
strategic ground truth. Passing can be sensible despite a legal bank menu.

Base4:1 trading requires no port. A saved-state audit found1,169 legal4:1
observations without an owned port. Ports improve the rate to3:1 or the relevant
2:1; executable menus also depend on phase, own hand and engine legality.

## Next controlled work

The reviewed acquisition driver compares frozen-student states with equal-query
additional teacher trajectories, using the same legacy full30 start per pair.
The planned real launch has4,000 teacher queries per arm/seed, identical original
components, full forced histories, deterministic prefix selection through equal
new choice counts, and five training epochs. Queries/choice presentations match;
updates, weighted loss mass and prefix costs can differ and are recorded.
No result from that study is claimed here before completion.

Then build a shared, diverse corpus toward100,000 non-forced choices, with complete
games across boards/seats and separate learning curves. Stream immutable components:
materializing the full corpus would consume excessive memory. Keep resource priors,
connectivity and teacher changes separate. The requested exact upstream upgrade
must retain old engine identities; new seeded evaluations do not reproduce legacy
boards. PPO remains conditional on useful imitation, with its reward unchanged.

Raw evidence: [feature report](../runs/swarm-2026-11-04/feature-study/artifacts/report.md),
[exact30 reference](../runs/swarm-2026-11-04/legacy-30-evaluation/results.json),
[public strategy audit](../runs/swarm-2026-11-04/public-trade-audit/report.md), and
[resource-prior research](resource-valuation-priors-2026-11-05.md).


## Strict plain-greedy benchmark

The frozen plain policy maximizes raw production in setup, buys legal cities then
settlements then roads then development cards, uses fixed legal mandatory/card
rules, rejects offers and never trades. No heuristic delegation or search.
See [exact rules](greedy-no-trade.md).

| Opponents,64 games per candidate | Plain greedy | Learned11 | Learned29 |
|---|---|---|---|
| Three fixed heuristics |2 wins /3.71875 VP|7 /5.453125|3 /4.8125|
| Three plain greedy |16 /5.890625|42 /8.34375|32 /7.484375|

Both learned FULL15 hand-recipe models outperform this plain reference in direct
games. Paired board-bootstrap95% win differences are+40.625 points[28.125,53.125]
and+25.0[9.375,40.625]; VP differences+2.453125[1.546875,3.34375] and
+1.59375[0.71875,2.515625]. Shared-heuristic win intervals touch/cross zero;
VP differences are positive. This compares whole policies, not trading in isolation.

All384 outcomes resolved:256new games plus128 exact verified reused outcomes,
counted once. Sixteen boards are resampling units;10,000 bootstrap draws use
seed20261105. Rules and checkpoints were fixed before outcomes. Greedy made zero
bank trades,accepts or proposals across19,906 decisions with3,010 bank menus.
Opponent-role adapter changes only return interface;all1,024 saved-view actions
matched exactly. Runtime660.519/900s,peak227MiB. See
[report](../runs/swarm-2026-11-04/greedy-benchmark/report.md).

## First student-state attempt: incomplete acquisition

Student collections missed4,000queries under120s caps:1,740 and1,240 versus4,000
teacher queries each. Timings128.584/124.450s include bounded-operation overruns,
not extended deadlines. Full raw histories and1130/758-choice prefixes remain.
All four extra five-epoch fits and320 games finished, but planned equal-query
study is PARTIAL and all pairs are suppressed. Raw teacher/student wins3/3 for
seed11,2/1 for29 per64games. No DAgger benefit established. Byte-identical
persistence optimization reduces conversions3→1/sample;128-sample save probe
0.313→0.177s is not a full-collection speed claim.
[Partial-run report](../runs/swarm-2026-11-04/dagger-study/report.md).

## Current implementation and next gates

Engine pinned3.3.0 at82aae93ab1f7c267218be0566df573ce477ec3d8. Exact provenance,
ActionRecord/native mechanics and explicit legacy weights-only bridge reviewed.
Old artifacts retain3.2.1 identity; new official-spiral games form a separate
cohort. Verified isolated legacy runtime preserves ongoing old-engine work;
main runtime transition has its own verification record.

Actual100k training is held while state-conditioned nonlinear trading residuals
are tested against matched linear heads on audited difficult TRAINING prefixes.
No held-out gradients. Tied-best/pre-jitter targets are a separate ablation;
teacher scores are imitation signals, not strategy truth. Greedy rules stay
frozen. PPO and reward shaping remain separate.


## Reviewed integration checkpoint

The source now pins Catanatron3.3.0 at commit
`82aae93ab1f7c267218be0566df573ce477ec3d8`. Its backend identity includes
the full Git revision. Legacy3.2.1 experiments keep their original identity and
run in a preserved isolated runtime. New initial policies use an explicit
weights-only bridge with zero updates and fresh optimizer/RNG state; default
cross-engine checkpoint loading rejects the old identity. No new-engine
playing-strength result is claimed.

The integrated engine, persistence optimization, frozen GreedyNoTrade baseline,
reserved final-board fix and component scaling infrastructure passed1,123tests
with one optional browser test skipped. An installed wheel loaded both new
untrained initializations, rejected implicit legacy loads, imported the scaling
driver and produced32legal greedy decisions. Actual100k collection and PPO
remain on hold while the smaller trading diagnostic is reviewed and run.

That diagnostic uses512non-forced choices with311forced history observations
from11real training episode prefixes. All selected prefixes belong to seed11,
which that parent has already seen; seed29has not. Architecture comparisons
must be within each initialization. Held-out observations contribute no
gradients, and pre-jitter teacher scores remain imitation diagnostics rather
than an independent statement of strategic quality.
