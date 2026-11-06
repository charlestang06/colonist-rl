# Current agent research status — 2026-11-05

The objective is a public-information Catan agent whose complete-game results
reproducibly exceed authentic, version-pinned VF and AB in declared harness
protocols. The [canonical roadmap](agent-research-roadmap.md) defines the milestone
ladder and next experiments. BC is a limited diagnostic/control; public planning
and outcome-value proceed independently. PPO and large cloning collection remain
deferred. The first fixed 96-game evaluator comparison and cap-corrected paired
repeat completed without an established advantage. No authentic VF/AB superiority
is established.

## Current 3.3 cohort

These results use Catanatron
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`, with source/runtime
identities preserved per attempt. Value collection uses its separately frozen
checkout; native gateway integration does not change that source pin.

- **Targeted recovery completed:** the separate v4 attempt completed FULL25 and
  288 resolved games after the earlier v3 audit-cutoff failure. Exact agreement
  on 48 selected training cases improved 28→35 (seed 11) and 28→38 (seed 29).
  Qualified target-or-approved-alternative agreement improved 32→42 and 35→42.
  Seed 11 wins fell 3/32→0/32 against H and 11/16→6/16 against Greedy. Seed 29
  remained 1/32 against H and changed 5/16→6/16 against Greedy, with mean VP lower
  in both panels. This supports retaining stable parents and testing retention,
  not declaring imitation impossible or recurrence defective.
  [Recovery outcomes](../runs/swarm-2026-11-04/roadmap-resume/attempt-recovery-v4-003/final-summary.json)
  and [qualified training analysis](../runs/swarm-2026-11-04/roadmap-resume/training-fit-diagnostic-001/qualified-training-analysis.json).
- **Public planner development completed:** 96 new games resolved with zero
  truncations and zero root-action identity failures. Planner/H_view wins were
  10/32 versus 8/32 against H and 15/16 versus 16/16 against Greedy. Against H,
  paired win difference was +6.25 points [0, 15.625] and VP +0.125
  [−0.21875, 0.4375]; against Greedy, win difference was −6.25 points [−18.75, 0].
  These descriptive board-bootstrap intervals establish no reliable general
  advantage. Only 137/4,317 planner decisions were marked `planned`, with 4,180
  fallbacks. Planner p95 was about 8.4–8.9 ms versus H_view 0.68–0.93 ms. The
  126-series panels are exposed development and reused controls are counted once.
  The selected-case audit is too weak to establish competence.
  [Planner results](../runs/swarm-2026-11-04/agent-research/planning-lane/run-001/summary.json).
- **Authentic native opponents integrated:** VF base defaults and AB depth=2,
  `prunning=False` preserve native source identity and privileged-state access.
  Root gateway/seam commits are `ea36614`/`d1e11bd` (isolated reviewed sources
  `a67576c`/`1fb9d32`). The guarded real-engine seam passed 10 cases with 8 authentic
  query completions and exit 0. Subsequent reference games resolved all eight
  games on exposed board 126100000: H won 0/4 versus VF and 3/4 versus AB2,
  with zero candidate legal-identity failures. Guarded wall times were 8.771/74.881
  seconds. One board cannot establish strength.
  Atomic discard and reject-trade shims declare `full_rule_equivalence=false`;
  candidate access remains public-only. Chance-model defects and AB multiplayer
  assumptions are disclosed baseline limitations. The next evidence is a larger fixed
  candidate/reference tournament. Integrated native/ordinary paths passed 186
  regression tests in 12.65 seconds.
  [Access audit](../runs/swarm-2026-11-04/agent-research/native-baselines/access-audit.md),
  [implementation review](../runs/swarm-2026-11-04/agent-research/native-opponent-implementation/REVIEW.md)
  and [real-engine seam](../runs/swarm-2026-11-04/agent-research/native-opponent-implementation/seam-001/seam-results/manifest.json).
- **True-outcome value pipeline smoke completed:** v3's 14 pure integrity tests
  preceded collection. The smoke resolved all 12 games on three disjoint boards,
  with 495 eligible positions: 133 train, 186 validation, 176 diagnostic test.
  Fixed constant, calibrated handcrafted-score and regularized linear models
  fitted; total guarded wall time was 10.895 seconds, exit 0 with no root resource
  failure. Train and validation included both outcome classes; the diagnostic
  test contained four losses and no wins. One board per split cannot establish
  generalization or strength. This is actual post-roll position value under the
  declared H continuation, not a validated projected-afterstate evaluator.
  [v3 review](../runs/swarm-2026-11-04/agent-research/value-lane/v3/REVIEW.md),
  [smoke inventory](../runs/swarm-2026-11-04/agent-research/value-lane/smoke-001/worker-complete.json)
  and [fit report](../runs/swarm-2026-11-04/agent-research/value-lane/smoke-001/fit/report.json).
- **Value pilot completed:** all 192 games resolved across 32 training,
  8 validation and 8 diagnostic-test boards with four seats each. The 8,674 public
  positions split 5,851/1,401/1,422, with 21/128, 7/32 and 9/32 wins. Both guards
  exited 0 with no resource failure in 192.657 seconds. Independent data audit
  verified all journals, sources, observations and terminal-outcome joins.
  Public-linear/calibrated-handcraft validation Brier was 0.16086/0.16094;
  diagnostic Brier was 0.15442/0.19835. Validation is essentially tied and
  eight-board uncertainty is coarse. Fixed hyperparameters were not selected
  on diagnostic results. Offline loss does not establish stronger play. This
  development split is separate from sealed final strength boards. The first
  gameplay comparison below tests fixed evaluators inside the same bounded planner.
  [Pinned manifest](../runs/swarm-2026-11-04/agent-research/value-lane/pilot-001/manifest.json)
  and [fixed fit report](../runs/swarm-2026-11-04/agent-research/value-lane/pilot-001/fit/report.json).
- **First fixed value/search comparison completed:** all 96 new games resolved,
  with no reused controls, six accepted guards exiting 0, maximum two concurrent
  workers and 170.861 seconds root wall time. There were 8,554 recorded root legal-
  identity checks with zero failures. Three frozen arms each played 16 games per
  panel: raw/calibrated/linear wins were 3/2/3 against H and 3/2/3 against native VF.
  Mean VP was 6.2500/6.3750/6.5625 against H and 7.6250/7.6875/7.5000 against VF.
  Both panels share four exposed development board clusters, not eight.
  Linear-minus-calibrated win difference was +6.25 points [0, 18.75] in each
  descriptive board-bootstrap interval; VP differences were +0.1875
  [−0.2500, +0.6875] against H and −0.1875 [−0.8750, +0.5000] against VF.
  Linear added no wins over raw; raw contrasts also differ in score units/margins.
  These results establish no evaluator superiority, model promotion or authentic
  VF/AB advantage. No AB or Greedy games belong to this study.
  [Empirical review](../runs/swarm-2026-11-04/agent-research/value-search-experiment/run-001/EMPIRICAL-REVIEW.md)
  and [compact metrics](../runs/swarm-2026-11-04/agent-research/value-search-experiment/run-001/compactanalysis.json).
- **Coverage and public conceptual audit:** supported nonforced decisions were
  only 17.5–20.2% of actions; setup and unsupported contexts retained H_view.
  Calibrated changed 0/274 supported H actions and 2/281 VF actions; linear
  changed 41/265 and 54/265. Equal depth/node/beam/geometry limits did not equalize
  compute, and fixed job order, different game paths and contention limit latency
  interpretation. The audit filled 22/24 planned slots, with two declared phase
  shortfalls: 9 reasonable, 9 ambiguous and 4 clearly bad. Its 18 linear/4 raw/0
  calibrated cases form a biased change-only sample, not competence or arm-quality
  rates. Four bad road choices exposed distance-blocked expansion or sacrificed
  legal productive settlements/ports. Reasonable choices need not beat H_view.
  Judgments used public snapshots, not future outcomes or certified projected
  menus. These EVAL packets and judgments must never be used for training.
  [Audit manifest](../runs/swarm-2026-11-04/agent-research/value-search-experiment/run-001/conceptual-audit-001/selection-manifest.json).
- **Separate known-win correctness repair accepted:** an old TRAIN case, excluded
  from the 96 games and 22 conceptual cases, showed a completed winning projection
  at evaluator call 20/182 discarded at node 449 and 100,000 geometry visits.
  The narrow cap fix at isolated `796e120` passed independent review and is
  integrated at `b778830`; root's affected-path tests passed 98 in 1.30 seconds.
  It preserves only completed own-known wins after the full supported root menu
  and H baseline are validated. Ordinary fallback, budgets, candidate order,
  models and margins remain unchanged. This establishes correctness, not strength.
  [Diagnostic trace](../runs/swarm-2026-11-04/agent-research/planner-cap-fix/diagnostic-001.json),
  [review](../runs/swarm-2026-11-04/agent-research/planner-cap-fix/REVIEW.md)
  and [integrated test evidence](../runs/swarm-2026-11-04/agent-research/planner-cap-fix/integration-tests-001/worker.log).
- **Cap follow-up completed:** the separately frozen
  [protocol](../runs/swarm-2026-11-04/agent-research/cap-followup/protocol.json)
  (`024c5251548a1ae480f73e34c1b16d3bd0b366cff61d2772046df29e408a880b`)
  completed all 96 new cap-fixed games against the reused old 96, paired within
  each arm/panel on the same four exposed boards. All six guards exited 0, with
  at most two concurrent heavy workers and 82.385 seconds root wall time. All six
  arm/panel win counts and mean VP match the original cohort, supplying no gameplay
  improvement on these boards. This does not establish broad equivalence or
  complete-trajectory identity. Reuse adds zero independent controls and does not
  double effective sample size. The shorter wall time is not a controlled speedup
  estimate. The narrow known-win fixture repair remains valid.
  [Completed run record](../runs/swarm-2026-11-04/agent-research/cap-followup/run-001/attempt-status.json).
- **Background batch 001 final review completed:** 64 old training-state cases
  have certified mechanical consequences and 64 final public strategy judgments:
  53 reasonable, 11 ambiguous. Forty targets are approved for a future limited BC
  checkpoint only, with ten differing from stored H. Thirteen reasonable card/
  development targets are withheld for unsupported effects; all 11 ambiguous
  targets are also withheld. Approval does not imply unique optimality or training
  incorporation. Use the authoritative envelope's effective amendments rather
  than globbing original case files. This background work never gates search/value.
  [Final audit](../runs/swarm-2026-11-04/roadmap-resume/background-cases/final-audit-001/README.md)
  and [final envelope](../runs/swarm-2026-11-04/roadmap-resume/background-cases/final-audit-001/final-envelope.json).
- **Background batch 002 draft prepared:** 64 unused old training states cover
  eight full-gameplay strata on 16 boards, balanced by parent and seat. This is
  metadata-only selection, not fresh rollouts, strategic approval or consequence
  certification. Its public labels remain unapproved for training.
  [Batch record](../runs/swarm-2026-11-04/roadmap-resume/background-cases/batch-002/README.md).

The next milestone is a controlled test of grounded public value. The public
transition callback, eight productive geometry features and bounded residual
runtime are integrated and independently reviewed; 124 combined focused tests
passed in 3.88 seconds. Four smoke games resolved with 203 persisted pairs and
262 eligible rows before deduplication. The main attempt verified96 training games
(22 wins, 74 losses, 5,471 eligible rows before deduplication), then stopped on a
validation original-menu road rejected by inherited public connectivity. The
collector correctly made this fatal. No residual fit has run; preserve the
incomplete attempt and require reviewed root-only correction plus a complete
128-game rerun under frozen source. Shared future geometry remains strict.
[Diagnostic report](../runs/swarm-2026-11-04/agent-research/grounded-value/actual-parity-diagnosis/REPORT.md).

A parallel public setup audit covered32 TRAIN cases. Independent critique leaves
H17 reasonable/14 ambiguous/1 fallible clearly bad; original judgments are retained.
The actual H_view probe completed32 original-menu calls and differed from the
proposal on13/16 second placements. Its separately frozen64-game setup-only test
resolved64/64 games under four accepted guards in23.783 seconds. Audited
whole-board win difference is+0.09375 [0,0.21875], VP−0.09375 [−0.9375,0.625],
eight clusters. Mixed directions and coarse uncertainty establish no promotion
or VF/AB strength. Both setup arms use raw margin10000; the grounded collector's
raw continuation uses250000, so these are distinct controls. Independent empirical
review passed all64 outcome joins and5,616 original-menu identity checks. No
native teacher, fit or final-board use follows.
[Setup protocol](../runs/swarm-2026-11-04/agent-research/grounded-value/setup-control/protocol.json).

The first grounded gameplay comparison is preregistered before any fit:128 new
games, raw plus three residual arms, H/VF, four shared fresh boards/all seats.
Report matched residual ablations separately from raw's different score units and
margin. PPO remains deferred. The [study specification](../runs/swarm-2026-11-04/agent-research/grounded-value/spec.md)
and post-pilot priorities preserve hypotheses and failure gates.

## Historical results and dormant infrastructure

The [training history through checkpoint 08](training-history-through-checkpoint08.md)
preserves the legacy 3.2.1 studies, failed nonlinear diagnostic, partial DAgger
attempt, first incomplete 3.3 targeted attempt and earlier superseded priorities.
The earlier v3 failure remains a failure; the completed v4 recovery is a separate
attempt. Engine cohorts and exposed panels must not be pooled as new independent
evidence.

Legacy strict-Greedy superiority is real: frozen learners won 42/64 and 32/64
against three plain Greedy opponents, versus 16/64 for Greedy. That whole-policy
comparison does not isolate trading and does not meet the authentic VF/AB goal.
BC scalar heads were not outcome-trained; old 64 histories lack winner labels.
The current-weight PPO history, genuine episode/outcome records, critic warmup and
optional frozen-BC KL infrastructure remain available but dormant. No sustained
PPO result is established; reward shaping has not been changed for this research.

The [experiment archive](../runs/swarm-2026-11-04/README.md) and
[artifact map](../runs/swarm-2026-11-04/artifact-locations.json) retain raw records,
negative results and original runtime identities. Active next actions come from
the canonical roadmap, including bounded resources, censored outcomes and sealed
final-board handling.
