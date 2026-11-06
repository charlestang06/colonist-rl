# Training and playing-strength status

The goal is reproducible complete-game superiority to authentic VF and AB under
the [canonical roadmap](agent-research-roadmap.md). BC is a diagnostic/warm-start
tool; PPO remains deferred. No authentic VF/AB superiority or new model promotion
is established.

## Completed grounded public-value wave

The exact core is `3f0a29835d7121d0f6b66db95b9d9de45216989a`, using pinned Catanatron
3.3.0 / Git `82aae93`. Public candidate transitions, eight productive geometry features,
and bounded residual inference were independently reviewed. The old known-win
cap repair remains integrated. A later root-only road projection correction accepts
an original legal root menu's connectivity while retaining stock/cost/occupancy
checks and strict future geometry. 203 affected-path tests passed in 3.25 seconds;
the reserved old known-win fixture was not repeated.

The initial four-game smoke passed. Main attempt001 verified 96 train games, then
correctly stopped when the engine accepted a root road rejected by inherited
public geometry. It remains immutable, incomplete and unused for fitting.
[Diagnosis](../runs/swarm-2026-11-04/agent-research/grounded-value/actual-parity-diagnosis/REPORT.md),
[root correction review](../runs/swarm-2026-11-04/agent-research/grounded-value/root-legality/REVIEW.md).

Attempt002 reran the entire same development schedule under corrected frozen
source, without mixing old outcomes or claiming fresh independent boards:

| Quantity | Train | Validation |
|---|---:|---:|
| Boards / resolved games | 24 / 96 | 8 / 32 |
| Wins / losses | 22 / 74 | 5 / 27 |
| Raw public pairs | 4,093 | 1,314 |
| Eligible states before deduplication | 5,480 | 1,732 |
| Shared deduplicated states | 4,107 | 1,318 |
| Unsupported award afterstates excluded | 34 | 7 |
| Missing / unknown / truncated / interrupted | 0 | 0 |

All 1,805 eligible after transitions passed feature/score parity. The 41 unsupported
public-award afterstates retain their before states and true outcomes; no other
mismatch was masked. Normal-turn passes and card actions remain before-state data.
All 16,501 immutable journal hashes and 128 outcome joins passed independent audit.
Collection took 134.847 seconds, with guard exit 0, one sampled thread and RSS below 2 GiB.
[Dataset review](../runs/swarm-2026-11-04/agent-research/grounded-value/pilot-002/EMPIRICAL-REVIEW.md).

Both residuals completed the fixed 600 full-batch updates at learning rate 0.05,
L2 coefficient 0.1 and logit bound 0.5, with zero initialization, train-only
standardization, common states and game-balanced weights. The old calibrated H
anchor remains a prior. The fit body took 28.913 seconds, with guard exit 0;
no selection or early stopping occurred.

| Model | Validation Brier | Validation log loss |
|---|---:|---:|
| Zero residual anchor | 0.122236097 | 0.404128809 |
| Existing features | 0.116802910 | 0.386338809 |
| Existing + productive | 0.116776061 | 0.386256451 |

Productive-minus-existing validation Brier is −0.000026849, with a paired eight-board
95% t interval [−0.000743688, 0.000689990]. All validation loss-difference intervals include zero.
The Root-run independent arithmetic audit reproduced train statistics and metrics
for 16,275 row/model cases in 10.499 seconds, with guard exit 0. No refits or runtime policy
probes were performed by the reviewer. Offline improvement is not strength.
[Fit review](../runs/swarm-2026-11-04/agent-research/grounded-value/pilot-002/fit-001/FIT-REVIEW.md).

## Completed fresh 128-game comparison

Four arms × H/VF × four fresh boards × four seats, with zero reused controls. All 128 games
resolved under eight zero-exit guards in 98.093 seconds; maximum registered overlap was 2,
sampled thread count was 1 and peak RSS was 744,865,792 bytes. Independent review checked 128
callbacks, 12,053 decisions, 192 packets, all 17,978 installed-runtime hashes and
all 12 paired contrasts. The original harness HOLD and its reviewed recovery fixes
remain preserved.

| Arm | H wins / 16 | H mean VP | VF wins / 16 | VF mean VP |
|---|---:|---:|---:|---:|
| Raw handcraft | 3 | 6.5625 | 7 | 8.3125 |
| Zero residual | 3 | 6.8750 | 7 | 8.4375 |
| Existing residual | 2 | 6.8750 | 7 | 8.3750 |
| Productive residual | 4 | 6.7500 | 7 | 8.4375 |

The productive-minus-zero H win-rate difference is +6.25 percentage points [0, 18.75],
from one extra win on one board; the mean VP difference is −0.125 [−0.5, 0.25].
The productive-minus-existing H win-rate difference is +12.5 percentage points
[0, 25]. All 16 paired VF win/loss labels agree across arms; zero-width observed
bootstrap win ranges do not establish population equivalence. Equal totals do
not imply identical action sequences. H/VF share four board clusters, not eight.
These are coarse descriptive 95% whole-board bootstrap intervals, with no superiority claim.

The three probability controls share probability-millionth units, margin 10000
and nominal search limits. Raw uses handcraft units and margin 250000 and is a
practical benchmark. All 2,488 supported nonforced decisions searched;
291/12,053 decisions were marked `planned`. Broad fallback counts include
forced/unsupported/insufficient-gain cases and are not error rates. Raw's
injected-evaluator count of 0 excludes its built-in handcraft work.
Equal limits are not equal compute: productive p95 latency was 8.935/24.542 ms against H/VF,
versus existing 6.183/14.474 ms, with fixed order/I/O/contention caveats and no registered
per-decision SLO. Native planning-node/timeout work remains unknown.

Native VF uses privileged state, declared discard/trade shims and a declared
chance-semantics mismatch; `full_rule_equivalence=false`. Native-panel offers are
disabled at all seats; H opponents retain normal proposals. There is no AB arm or
focal native reference.
[Empirical review](../runs/swarm-2026-11-04/agent-research/grounded-value/gameplay-study/run-001/EMPIRICAL-REVIEW.md),
[compact analysis](../runs/swarm-2026-11-04/agent-research/grounded-value/gameplay-study/run-001/compactanalysis.json).

The public conceptual audit selected 24 cases: 18 reasonable, 6 ambiguous, 0 clearly
bad. It found concrete productive roads/cities, sensible passes, and trades whose
recipes must not be mistaken for immediately possible builds. No immediate winning
root was missed in this sample. Selection was post-run after Root saw aggregates,
20/24 cases are seat 0, and no selected root offers a settlement. It proves neither
arm quality nor elimination of previous failure cases. All 192 cached packets are
also seat-skewed. These EVAL artifacts never enter fitting or training.
[Conceptual report](../runs/swarm-2026-11-04/agent-research/grounded-value/gameplay-study/run-001/conceptual-audit-001/REPORT.md).

## Independent setup hypothesis and next decision

The separate setup audit/probe completed 32 public roots. Effective H judgments
are 17 reasonable / 14 ambiguous / 1 fallible clearly bad after independent critique.
The 64-game second-settlement-only pilot resolved all games: baseline 4/32 wins,
mean VP 6.5625; control 7/32, mean VP 6.46875. The control-minus-baseline win-rate
difference is +0.09375 [0, 0.21875]; the mean VP difference is −0.09375 [−0.9375, 0.625].
These are descriptive 95% whole-board bootstrap intervals across eight clusters.
Both arms used raw margin 10000 and source `56cb6bc`. This mixed result supports no promotion.
[Setup review](../runs/swarm-2026-11-04/agent-research/grounded-value/setup-control/run-001/EMPIRICAL-REVIEW.md).

## Completed 256-game setup transfer

The transfer study completed all 256 new games: two arms × 16 fresh shared boards
× four seats × H/VF. Both arms used corrected source `3f0a298`, common raw margin
10000 and the original frozen public second-settlement scorer. There were no
fitted models, independently queried teachers, fits, reused controls or pooled
source cohorts. Built-in raw heuristics and the frozen placement scorer remain
part of the candidate.

| Arm | H wins / 64 | H mean VP | VF wins / 64 | VF mean VP |
|---|---:|---:|---:|---:|
| Raw baseline | 20 | 7.015625 | 26 | 7.703125 |
| Second-settlement control | 19 | 7.484375 | 32 | 8.406250 |

Control-minus-baseline H win difference is −0.015625, with predeclared 95%
whole-board bootstrap interval [−0.125,0.109375]. VF difference is +0.09375
[−0.046875,0.21875]. The panels share 16 board clusters. Mean VP differences
are +0.46875 for H and +0.703125 for VF, but VP cannot rescue the failed primary
H advancement rule. Deprioritize this unchanged placement intervention and
proceed to trade/pass diagnosis; no strength promotion follows.

The independent audit joined all 256 resolved outcomes, 128 identical pre-second
public-root/prefix pairs, source/runtime, original-menu identities, guards and
durable attempts; 338,196 checks across 107 categories had zero failures. Root
wall time was 243.678 seconds, maximum job 55.306 seconds, maximum heavy overlap
2, and peak sampled RSS 765,394,944 bytes with one sampled thread. Control paths
used about 11–12% more candidate time and 23–25% more nodes than baseline, despite
equal nominal limits. Future paths differ after placement; equal budgets are
not equal measured work.

A preparation-test incident accidentally created real guard/worker processes
against a manufactured configuration. Its temporary child evidence was lost;
static binding checks support expected rejection before the runtime factory,
but per-child progress cannot be certified. No incident data is accepted.
Subsequent isolation repairs passed 12 focused tests before the separately
frozen Root-run study. The incident, missing evidence and retained failures remain
documented in the [empirical review](../runs/swarm-2026-11-04/agent-research/setup-transfer/run-001/EMPIRICAL-REVIEW.md).

## Active trade/pass diagnosis

An existing 3.3.0 TRAIN corpus contains 6,016 rows from 64 complete historical
neural-student episodes. After excluding the 192 prior targeted/background
observation hashes, public screening found 563 passes and 191 bank selections
within bank-plus-END menus. A frozen sample contains 14 apparent single-trade
recipe/site opportunities, nine other passes and nine executed bank trades.
Its 32 cases span 11 boards and 15 episodes, with 26 same-episode case pairs;
the selected screen is not a population error rate.

Public judgments and scoped current-weight model checks remain diagnostic.
Recipe/site arithmetic does not certify a future engine menu. The historical
parents and occupancy are distinct from the current raw planner; no new rollout,
training label approval, fit or added-data experiment is claimed here.
[Next learning plan](../runs/swarm-2026-11-04/agent-research/next-learning-plan.md).

## Earlier evidence retained

Targeted FULL25 BC improved selected agreement but seed 11 regressed and seed 29
mixed/regressed in complete games. Retain FULL15 parents. Initial planning (96 games)
and fixed evaluator (96 games), plus a cap-corrected 96-game paired repeat, established
no reliable advantage. The genuine outcome-value 192-game pilot had mixed offline
results. The native seam's 10 cases and eight one-board H reference profiles are engineering
or exposed-board evidence, not strength confirmation.

Background batch 001 has 64 certified mechanical cases and final judgments of
53 reasonable / 11 ambiguous; 40 targets are approved only for future limited BC,
none used. Batch 002 is a 64-case unapproved metadata draft of old states, not fresh rollouts.
Neither gates planning/value research.

Detailed earlier counts, source pins and reviews are preserved in the
[previous status](../runs/swarm-2026-11-04/research-roadmap-history/training-status-2026-11-05.md.before-grounded-study-close),
[historical results](training-history-through-checkpoint08.md), and underlying
archived studies. The user's `docs/roadmap.md` edit remains untouched.
