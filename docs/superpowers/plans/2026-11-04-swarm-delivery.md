# Coordinated Learning and Adapter Delivery

> Execution: superpowers:subagent-driven-development, with separate domain coordinators and fresh task implementers/reviewers.

**Spec:** `docs/swarm-roadmap.md` and the user's authorized learning-first swarm request.
**Baseline:** `182cc01c8670234012f96a0389cef236296a93c5` snapshots the original repository including its preexisting uncommitted work.
**Goal:** Deliver reviewed learning measurement/imitation tools and bounded local evidence, plus independent adapter evidence contracts, then integrate without overwriting concurrent user changes.

## Global constraints

- Learning is primary. Simulator training and evaluation do not depend on further browser work.
- Shared Observation/Action contracts, privacy projection, dependency versions and existing public APIs stay compatible.
- No privileged actor or teacher inputs, invented resource facts, fabricated event identities or unsupported competence claims.
- CPU-only existing environment; no paid services or network installs. Learning pilot wall budget is 900 seconds including its throughput probe.
- Implementations and reviews run in separate temporary learning/adapter repositories. Root owns integration; original files are modified only after review and conflict checks.
- Domain coordinators dispatch one implementation at a time within their domain, return while it runs, and dispatch independent reviews after reports. Disjoint domain implementations may run concurrently as explicitly requested by the user.
- Preserve prior dirty files and any changes made during this run. Do not push, publish or discard files.

## Delegated work

- [ ] Learning Task 1: complete board-block evaluation, pairing and teacher protocol. Coordinator: `/root/learning_coordinator`; detailed plan: `docs/superpowers/plans/2026-11-04-learning-baseline.md` in the learning repository.
- [ ] Learning Task 2: persisted demonstrations and current-weight recurrent imitation, preserving legacy warmstart entry point.
- [ ] Learning Task 3: reproducible experiment CLI, bounded local pilot and measured next-step diagnosis; PPO continuation requires adequate baseline evidence.
- [ ] Adapter Task 1: immutable UI records and pure exact revision/action/offer validators, without gestures.
- [ ] Adapter Task 2: conservative production/bank/dice semantic candidates, preserving unresolved evidence without ledger writes. Coordinator: `/root/adapter_coordinator`; detailed plan: `docs/superpowers/plans/2026-11-04-adapter-evidence.md` in the adapter repository.
- [ ] Planner: update `docs/swarm-roadmap.md` from actual task and experiment results; propose subsequent bounded work rather than expanding the current scope implicitly.
- [ ] Root: integrate reviewed commits, run combined tests/lint/build and an independent whole-change review, preserve experiment artifacts and apply only verified changed files to the original repository.

## Integration acceptance

Each task has a saved brief, RED/GREEN evidence, implementer report, exact diff package and independent spec/quality verdict. Fixes are reviewed before task completion. Both coordinators provide a final domain review and commit range. Root's combined checks must cover legacy imports, simulator/training behavior, new adapter and learning tests, packaging and supported local Chromium fixture checks. Any skipped test or inconclusive learning result is reported explicitly.

Before original-repository writes, compare every changed file with the initial SHA-256 manifest. A conflicting current file is retained and reconciled in the integration copy before any replacement. Preserve larger ignored datasets/checkpoints in a dedicated run directory with digest manifests; do not treat ignored experiment outputs as committed evidence.
