# Grounded public value implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development.

**Goal:** Test whether actual after-action targets and productive geometry improve
continuation-specific value without repeating the unconstrained proxy failures.
**Architecture:** A public-only evaluator callback records immediate transitions;
independent feature/runtime modules consume Observations and ProjectedTurn. An
external bounded study joins real outcomes and fits two residual controls.
**Tech stack:** existing Python/Catanatron3.3, NumPy, pytest; no new dependency.
**Spec:** [grounded public value specification](../../../runs/swarm-2026-11-04/agent-research/grounded-value/spec.md). Original working source: `/private/tmp/colonist-swarm-bxz1k39_/agent-research/grounded-value/spec.md`.

## Global constraints

Copy the binding constraints and exact interfaces from the spec into every brief.
Baseb778830 isolated branches; preserve all old source/studies/user roadmap edits.
Root owns real compute; max2 heavy workers, threads1/RSS2GiB/existing v4 guards.
No private inputs, native action labels, fake Observations or EVAL training rows.

### Task1: public transition callback

Files: evaluation.py, tests/test_public_candidate_steps.py, focused usage doc.
Produces frozen CandidatePublicStep(before,action,after,terminated,truncated) and
optional on_candidate_step. No other API or default behavior changes.

- [ ] RED fake-engine tests prove after uses original observer, callback occurs
  before next actor, terminal action retained, private StepResult fields unread.
- [ ] Implement the minimal callback immediately after `result=driver.step(action)`:
  `if on_candidate_step is not None and view.player_id == seat: ...`.
- [ ] Verify default no-extra-observe; callback exception restores random/NumPy/
  Torch state and module modes. Keep ordinary positional API compatible.
- [ ] Run targeted pure tests, lint, commit and produce full review package.
- [ ] Fresh independent review, resolve concrete blockers, integrate.

### Task2: productive features and frozen residual runtime (parallel with1)

Files: productive_geometry.py, public_residual_value.py, focused tests and doc.
Consumes existing Observation/ProjectedTurn/encoder and emits the exact spec APIs.

- [ ] RED synthetic tests: blocked road endpoints do not count as settlement
  access; productive connected sites/ports and stocks change correctly.
- [ ] Implement at most12 finite public features; document names/units before data.
- [ ] RED residual tests: zero weights equal calibrated anchor; offsets stay within
  plus/minus0.5 logits; invalid schema/columns/scales/continuation reject.
- [ ] Implement strict runtime and probability factory without optimizer imports.
- [ ] Verify projected/actual overlays on synthetic deterministic transitions,
  privacy-field invariance, root legal identity and unchanged search limits.
- [ ] Targeted tests/lint/commit/review; integrate after compatible Task1.

### Task3: bounded actual-state collector and fitter

Files: external grounded-value/collector.py, journal.py, fit.py and pure tests.
Consumes CandidatePublicStep and encode_residual_features, produces immutable pair
journals, exact outcome joins/inventory and frozen public-residual-value/1 models.

- [ ] RED tests reject duplicate pair/episode IDs, wrong observer/version/action,
  contradictory bool outcomes, missing schedule rows and split leakage.
- [ ] Implement immediate per-pair/outcome persistence; explicit unsupported award
  exclusions and fatal unexplained projection/score mismatch gate.
- [ ] RED finite-difference gradient test and game-weight/zero-init checks.
- [ ] Implement fixed600step NumPy fits and paired-board validation reporting.
- [ ] Pure stub tests/lint/fresh review; freeze source/runtime/anchor/seed manifests.
- [ ] Root ratifies and runs four-game smoke; audit real joins/parity before scale.
- [ ] Root runs128 games/600s then fits/30s, reports every incomplete artifact.

### Task4: conceptual/evaluation and independent setup diagnostic

- [ ] Parallel public-only setup audit32 independent TRAIN states; no native calls.
- [ ] Review offline metrics/support/saturation and genuine public decisions.
- [ ] If data is mechanically sound, freeze fresh96game three-arm H/VF comparison;
  same source/search/margins, exact outcomes, board CIs/latency and EVAL-only audit.
- [ ] Record negative/inconclusive results; update canonical roadmap and deliver
  reviewed checkpoint while preserving original protected roadmap edits.

## Preflight interface review

| Pair | Producer/consumer | Ruling |
|---|---|---|
|1/2|No shared edits; both depend on existing immutable views|Parallel isolated clones.|
|1/3|Frozen before/action/after callback|No raw StepResult/env passed; collector sees original observer.|
|2/3|encode_residual_features + strict runtime|Feature list/schema frozen before real collection; no model labels as targets.|
|3/4|True game outcomes and fixed models|EVAL stays separate; proposal-only plans are not observed effects.|

Every task's testing agrees with its scope: implementers use synthetic/pure tests;
root schedules real smoke, collection, fits and gameplay. Public award mismatches
are counted unsupported cases; all other unexplained parity mismatches block fit.

## Execution amendments

The public callback/features/runtime are reviewed and integrated. The first main
collection stopped correctly on inherited root-road projection disagreement after
96 verified training games. Add a narrow original-root transition correction;
keep strict future geometry and old evidence unchanged. Review and freeze the
source, then rerun all128 development games as a separate attempt before fitting.
Before any residual fit, Root added unchanged raw as fourth gameplay benchmark:
128 new games, four arms×H/VF×fourfresh boards×fourseats. Matched probability-arm
comparisons remain distinct from raw's different units/margin. The canonical
specification and SDD ledger contain Root's exact rulings.
