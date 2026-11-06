# Grounded public value implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development.

**Goal:** Test whether actual after-action targets and productive geometry improve
continuation-specific value without repeating the unconstrained proxy failures.
**Architecture:** A public-only evaluator callback records immediate transitions;
independent feature/runtime modules consume Observations and ProjectedTurn. An
external bounded study joins real outcomes and fits two residual controls.
**Tech stack:** existing Python/Catanatron 3.3, NumPy, pytest; no new dependency.
**Spec:** [grounded public value specification](../../../runs/swarm-2026-11-04/agent-research/grounded-value/spec.md). Original working source: `/private/tmp/colonist-swarm-bxz1k39_/agent-research/grounded-value/spec.md`.

**Status:** Completed after the execution amendments below. The result is an audited
data and gameplay checkpoint with no model promotion. The separate 256-game setup
transfer replication selected by the canonical roadmap is planned, not launched.

## Global constraints

Copy the binding constraints and exact interfaces from the spec into every brief.
Base `b778830`, isolated branches; preserve all old source/studies/user roadmap edits.
Root owns real compute; at most 2 heavy workers, 1 thread / 2 GiB RSS each, existing v4 guards.
No private inputs, native action labels, fake Observations or EVAL training rows.

### Task 1: public transition callback

Files: evaluation.py, tests/test_public_candidate_steps.py, focused usage doc.
Produces frozen CandidatePublicStep(before,action,after,terminated,truncated) and
optional on_candidate_step. No other API or default behavior changes.

- [x] RED fake-engine tests prove after uses original observer, callback occurs
  before next actor, terminal action retained, private StepResult fields unread.
- [x] Implement the minimal callback immediately after `result=driver.step(action)`:
  `if on_candidate_step is not None and view.player_id == seat: ...`.
- [x] Verify default no-extra-observe; callback exception restores random/NumPy/
  Torch state and module modes. Keep ordinary positional API compatible.
- [x] Run targeted pure tests, lint, commit and produce full review package.
- [x] Fresh independent review, resolve concrete blockers, integrate.

### Task 2: productive features and frozen residual runtime (parallel with 1)

Files: productive_geometry.py, public_residual_value.py, focused tests and doc.
Consumes existing Observation/ProjectedTurn/encoder and emits the exact spec APIs.

- [x] RED synthetic tests: blocked road endpoints do not count as settlement
  access; productive connected sites/ports and stocks change correctly.
- [x] Implement at most 12 finite public features; document names/units before data.
- [x] RED residual tests: zero weights equal calibrated anchor; offsets stay within
  plus/minus 0.5 logits; invalid schema/columns/scales/continuation reject.
- [x] Implement strict runtime and probability factory without optimizer imports.
- [x] Verify projected/actual overlays on synthetic deterministic transitions,
  privacy-field invariance, root legal identity and unchanged search limits.
- [x] Targeted tests/lint/commit/review; integrate after compatible Task 1.

### Task 3: bounded actual-state collector and fitter

Files: external grounded-value/collector.py, journal.py, fit.py and pure tests.
Consumes CandidatePublicStep and encode_residual_features, produces immutable pair
journals, exact outcome joins/inventory and frozen public-residual-value/1 models.

- [x] RED tests reject duplicate pair/episode IDs, wrong observer/version/action,
  contradictory bool outcomes, missing schedule rows and split leakage.
- [x] Implement immediate per-pair/outcome persistence; explicit unsupported award
  exclusions and fatal unexplained projection/score mismatch gate.
- [x] RED finite-difference gradient test and game-weight/zero-init checks.
- [x] Implement fixed 600-step NumPy fits and paired-board validation reporting.
- [x] Pure stub tests/lint/fresh review; freeze source/runtime/anchor/seed manifests.
- [x] Root ratifies and runs four-game smoke; audit real joins/parity before scale.
- [x] Root completed all 128 collection games within the 600-second ceiling, then
  both fixed fits within the 30-second fit-body ceiling; incomplete attempt001 is
  preserved separately.

### Task 4: conceptual/evaluation and independent setup diagnostic

- [x] Parallel public-only setup audit of 32 independent TRAIN states; no native calls.
- [x] Review offline metrics/support/saturation and genuine public decisions.
- [x] After data review, freeze and complete the fresh 128-game four-arm H/VF
  comparison with common source and nominal search limits. Match the three residual
  controls at probability margin 10000; retain raw at handcraft margin 250000 as a
  separate practical benchmark. Record exact outcomes, board CIs/latency and an
  EVAL-only audit.
- [x] Record negative/inconclusive results; update canonical roadmap and deliver
  reviewed checkpoint while preserving original protected roadmap edits.

## Preflight interface review

| Pair | Producer/consumer | Ruling |
|---|---|---|
| 1/2 | No shared edits; both depend on existing immutable views | Parallel isolated clones. |
| 1/3 | Frozen before/action/after callback | No raw StepResult/env passed; collector sees original observer. |
| 2/3 | encode_residual_features + strict runtime | Feature list/schema frozen before real collection; no model labels as targets. |
| 3/4 | True game outcomes and fixed models | EVAL stays separate; proposal-only plans are not observed effects. |

Every task's testing agrees with its scope: implementers use synthetic/pure tests;
root schedules real smoke, collection, fits and gameplay. Public award mismatches
are counted unsupported cases; all other unexplained parity mismatches block fit.

## Execution amendments

The public callback/features/runtime were reviewed and integrated. The first main
collection stopped correctly on inherited root-road projection disagreement after
96 verified training games. A narrow original-root transition correction was
reviewed and integrated, retaining strict future geometry and old evidence. After
freezing the corrected source, Root reran all 128 development games as a separate
attempt before fitting. The rerun used the same boards, without mixing outcomes
or claiming new independent boards.

Before any residual fit, Root expanded the original planned 96-game three-arm
gameplay comparison by adding unchanged raw as a fourth benchmark: 128 new games,
four arms × H/VF × four fresh boards × four seats. Matched probability-arm
comparisons remain distinct from raw's different units/margin. The canonical
specification and SDD ledger contain Root's exact rulings.

## Completed evidence and handoff

Tasks 1–4 completed after reviewed root-only legality repair `bacafa8`, integrated
in `3f0a298`. The 203 affected-path tests passed; main attempt002 resolved all 128
games, then both 600-step fits completed. Independent data/arithmetic reviews passed. The fresh
128-game comparison resolved all games under eight guards with exit 0; independent
empirical review passed. The 24-case conceptual audit judged 18 reasonable / 6
ambiguous / 0 clearly bad, with declared selection limits. These selected EVAL
judgments are never training data and do not establish general competence. No arm
was promoted. The canonical roadmap now selects a separate 256-game setup transfer
hypothesis; that future study is outside this completed plan and has not launched.
