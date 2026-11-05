# Current agent research status — 2026-11-05

The objective is a public-information Catan agent whose complete-game results
reproducibly exceed authentic, version-pinned VF and AB in declared harness
protocols. The [canonical roadmap](agent-research-roadmap.md) defines the milestone
ladder and next experiments. BC is a limited diagnostic/control; public planning
and outcome-value proceed independently. PPO and large cloning collection remain
deferred. No authentic VF/AB superiority is established.

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
  `a67576c`/`1fb9d32`). The guarded real-engine seam passed 11 cases with 8 authentic
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
  development split is separate from sealed final strength boards; the next test
  compares fixed evaluators inside the same bounded planner.
  [Pinned manifest](../runs/swarm-2026-11-04/agent-research/value-lane/pilot-001/manifest.json)
  and [fixed fit report](../runs/swarm-2026-11-04/agent-research/value-lane/pilot-001/fit/report.json).
- **Background cases:** batch 001 contains 64 selections reused from existing
  training states. All 64 deterministic consequence checks
  completed in 21.381 seconds across four guarded workers; independent final
  strategy review is in progress. No fresh rollouts or approved labels are implied. Continue immutable review in small
  batches across all gameplay; deliberate incorporation does not gate search/value.
  [Batch record](../runs/swarm-2026-11-04/roadmap-resume/background-cases/batch-001/README.md).

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
