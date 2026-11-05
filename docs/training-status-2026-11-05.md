# Training status — November 5, 2026

The learner has not demonstrated that it can beat the fixed heuristic opponents. All completed
experiments and failed hypotheses are preserved; passing tests is not a strength
claim. This checkpoint records completed work and bounded incomplete attempts.
Every gameplay result below uses the legacy Catanatron3.2.1 engine. The project now pins the reviewed3.3.0 engine; its results form a separate
cohort. Legacy experiments retain their original source/runtime identities.

Latest checkpoint: the strict `plain-greedy-no-trade-v1` baseline is integrated
and passes42 focused tests on the pinned3.3 runtime. Its completed legacy-engine
benchmark remains separate: the two frozen learners won42/64 and32/64 against
three plain Greedy opponents, versus16/64 for Greedy itself. This does not isolate
the value of trading or establish heuristic-level strength.

The first targeted3.3 trading attempt collected64 learner episodes and prepared
64 training candidates plus16 separate decision cases. It stopped at the
independent label-audit deadline before any fit or evaluation. The public facts
were verified, but the complete independent judgment envelope was not persisted
in time. No labels were approved, no training updates occurred, and no improvement
is established. The [attempt record](../runs/swarm-2026-11-04/targeted-trade/attempt-report-v3-001.md)
preserves the evidence and failure. Broad PPO and100k scaling remain unrun.

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
has run. Earlier roadmapping prioritized student-state labels and100,000 non-forced
decisions. The current priority is stronger public planning and controlled
outcome-based PPO; near-perfect heuristic imitation is not a prerequisite.
Further100k collection from the same teacher remains on hold.

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


## Standard no-trade benchmark and current diagnostic

The opt-in `greedy_no_trade` name is now available as an evaluation candidate,
opponent baseline, explicit lineup entry and CLI `--baseline` choice. It reuses
`plain-greedy-no-trade-v1` exactly. Pure greedy opponent panels cannot silently
replace a seat with a historical policy. Existing heuristic/random routes and
curriculum promotion gates retain their previous behavior. Integration passed
1,189tests with one optional browser skip; the original project environment
also passed the explicit local browser fixture. Installed-wheel API and CLI
checks used bounded24-step games and are protocol evidence, not strength results.

The separately reviewed six-arm trading diagnostic started at
2026-11-05T17:46:58UTC under a1,200-second whole-run cap. It compares old34D,
context-linear and nonlinear residual scorers within each of two frozen FULL15
parents on the same512difficult training choices and311forced history steps.
Exact scalar/batched initial outputs matched on all1,847training/held-out
observations per initialization. Inherited Graph/value/recurrence are frozen;
no held-out gradients or label changes occur. Common complete work is selected
before comparative validation. The trial completed every primary100-epoch target. Its failed promotion gate
and limitations are recorded below; no playing-strength claim follows. Concurrent integration verification
overlapped its first arm, so observed wall times do not isolate architecture speed.


## Completed trading diagnostic and revised strength priority

All six arms completed100epochs:5,700updates and51,200choice presentations
each. The common FULL100 checkpoint was selected before comparative validation.
Training/scoring finished in715.79seconds, with final read-only auditing and
sealing inside the original1,200-second cap. All protected source, runtime,
input, optimizer-ownership and frozen-parameter checks passed. This entire
experiment uses legacy3.2.1 and the previously used diagnostic held-out panel.

| Initialization | Best linear held-out bank/END direction | Nonlinear direction | Best linear exact hard labels | Nonlinear exact hard labels |
|---|---:|---:|---:|---:|
|11|135/182|124/182|55/106|48/106|
|29|132/182|124/182|50/106|46/106|

Nonlinear training bank/END cross entropy (82choices) improved15.51%/13.38%,
and held-out teacher-score regret decreased, while held-out cross entropy and
directional agreement worsened. The predeclared conjunction failed in both
initializations. No larger fit or game result was substituted for that gate.
The unchanged, independently frozen11-case rubric judged all six arms identically:
nine defensible choices, two ambiguous choices and no clearly bad selections,
with no direct-road passes in four opportunity cases. These are selected training
cases, not representative strategic or generalization evidence.

[Full results](../runs/swarm-2026-11-04/trading-smallfit/report.md) preserve every
milestone, Adam state, gradient/feature audit, score and source binding.
[Independent public-case audit](../runs/swarm-2026-11-04/trading-smallfit-case-audit/comparative-report.md)
retains its rubric from before any arm outputs.

The [updated strength roadmap](strength-roadmap-2026-11-05.md) prioritizes bounded
public turn planning and controlled PPO from unchanged FULL15 BC parents.
A read-only PPO audit found two current-history gaps: collector memory survives
actor updates, and PPO chunks start from stored behavior-weight memory.
An opt-in path with exact current-weight prefixes and complete episode/outcome
records is now being implemented before critic warmup and controlled PPO.
No sustained PPO experiment has started. Rewards remain unchanged; trading or
road frequency will not be rewarded directly.


## Reviewed planning and outcome-learning foundation

Audited current-history collection/PPO, genuine complete-episode reward suffixes,
value-head-only warmup, optional isolated frozen-BC KL anchoring, and a conservative
public turn planner are implemented and independently reviewed. Combined checks:
1,400passed,3skipped. The planner uses public bank bounds and actual legal root
actions; its projection has not demonstrated stronger play. H_view is a distinct
deterministic wrapper and cannot be conflated with historical H.

The latest direct user request prioritizes a small targeted learner-state trading
study before broad PPO. The protocol pairs original-H versus independently
audited local targets on identical histories, with actual sequential-legality
checks and fresh decision/game panels. Generic100k and reward changes remain held.
This tiny audited-label trial is an explicit exception to the earlier global
planner-teacher proof gate; broader label scaling still needs strength evidence.
The targeted harness is being corrected after static review and has not launched.
Broad PPO preparation is preserved on hold, with no empirical PPO run.
