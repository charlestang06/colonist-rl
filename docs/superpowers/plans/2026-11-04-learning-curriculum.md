# Learning Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare longer and balanced imitation, then execute sustained fresh-PPO curriculum and independent strategic audits.

**Architecture:** Extend existing imitation loss optionally; add a supervised-study runner. Extend current collector/updater configuration and deadline controls compatibly; add curriculum orchestration. Reuse existing evaluation and the independent audit script, never another simulator loop.

**Tech Stack:** Existing Python, Torch, NumPy, Catanatron 3.2.1, pytest, ruff and standard-library artifacts.

**Spec:** `docs/superpowers/specs/2026-11-04-learning-curriculum-design.md`.

## Global Constraints

- Work only in /private/tmp/colonist-swarm-bxz1k39_/learning; original repository/environment read-only.
- Existing /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python and ruff; PYTHONPATH points at learning/src, PYTHONDONTWRITEBYTECODE=1; one CPU worker and Torch thread, no installs/network/paid services.
- Observation-only actor, teacher and audit inputs; no hidden hands, deck order, private VP or future outcomes justify decisions.
- Preserve current imports/CLI/checkpoints/schema. No simulator/adapters/contracts/board/beliefs/opponent-strategy edits without root agreement.
- Clean committed unchanged source for real runs; exact source/config/seed/model/optimizer/RNG/artifact identities. Preserve old pilot bytes and all weak/failed outcomes.
- Censored outcomes remain unknown, strict block pairing and per-game persistence remain binding. Forced-only accuracy, loss and training authorization do not establish strength or checkpoint promotion.
- Fresh implementer, independent review, reviewed fixes and broad final review; coordinator does not implement production code.

### Task 1: Controlled longer and phase-balanced imitation

**Files:** Create `src/colonist_rl/imitation_study.py`, `tests/test_imitation_study.py`, `docs/imitation-study.md`; modify optional weighting in `src/colonist_rl/imitation.py` and focused `tests/test_imitation.py`. Prefer documented Python API to an extra CLI command unless integration needs one.

**Interfaces:** In imitation_study.py the dataset/checkpoint arguments are mappings from integer training seeds to local artifact paths. Add optional `phase_weights=None` to `fit_imitation`, preserving default uniform behavior. Produce `phase_weights(dataset)->dict[str,float]` and `run_imitation_study(output_dir, *, datasets, initial_checkpoints, epochs=20, wall_seconds=900, validation_board_seeds=(62100000,62100001,62100002,62100003), validation_demo_seeds=(62110000,62110001,62110002,62110003))->dict`.

**Acceptance:** Uniform versus phase-balanced, each seed11/29 starts from saved initial_student and its same2500-example dataset; independent Adam/RNG, same ordering,20epochs,seq16,lr3e-4,hidden32. Non-forced training count weight N/(P*n_phase), weighted chunk CE sum divided by the non-forced choice count (global training mean weight is one); forced inputs retained for memory and excluded loss. Reject nonfinite/nonpositive weights; explicit missing-phase treatment must not silently drop supervision. Validation uses unweighted metrics, no gradients. Actual-play diagnostics must report maritime legal/selected counts, passes when maritime is legal, feasible accept/reject opportunities, and passes over directly legal purchases. Do not import an absolute scratch audit path from production code. Task3 reuses that external analysis script, or an explicitly injected narrow diagnostic callback, for these actual-play denominators; phase balancing alone is not expected to resolve every within-PLAY_TURN choice. Compare fixed heuristic games on identical complete blocks, per-game durability, source/dataset/checkpoint digests, actual equal-budget completion. No long run before review; Task3 owns real study.

- [ ] RED tests: train-only weights, forced exclusion, invalid/missing weights, default compatibility, independent identical initial states, disjoint partitions and no validation updates. Example invariant:

```python
weights = phase_weights(dataset)
assert sum(counts[p] * weights[p] for p in counts) == pytest.approx(sum(counts.values()))
```

- [ ] Implement minimal weighting and runner reusing fit/metrics/checkpoint/evaluation interfaces. Preserve correct current-weight recurrent prefix replay.
- [ ] GREEN focused tests, full suite once, ruff; commit and report exact RED/GREEN commands/output, interfaces and limitations. No feature/architecture/reward changes.

### Task 2: Deadline-bounded fresh-PPO curriculum

**Files:** Create `src/colonist_rl/curriculum.py`, `tests/test_curriculum.py`, `docs/curriculum-protocol.md`; modify relevant collector/updater controls in `src/colonist_rl/training.py` and focused integration tests.

**Interfaces:** Preserve existing constructor and collect(count) defaults. Add keyword-only lineup configuration and explicit unique episode-board seeds; `collect(count, *, deadline=None)` returns fresh Transition list. Expose actual boards, stop reason and dropped partial-interval count. Optional `deadline=None` in ppo_update preserves old numeric metrics and adds completed-update/deadline accounting. Produce `CurriculumConfig`, `promotion_gate(stage, stage_result, fixed_result, initial_fixed_result)->dict`, `select_checkpoint(candidates)->dict`, `run_curriculum(output_dir, *, initial_checkpoints, config)->dict`.

**Acceptance:** All random -> one heuristic/two random -> all heuristic; optional two heuristics/one immutable frozen policy as diversity. Existing mixed default currently uses a different mixture: preserve it, explicitly specify new lineup rather than silently changing meaning. Changes only at episode boundaries. Independent seed models/optimizers/RNG; immutable frozen parameters with own recurrent memory. Fresh rollout records only, never teacher demonstrations. Guard deadline in proposal/learner/opponent/update boundaries; drop/count interrupted incomplete learner intervals, no new calls after expiry, no invented returns. Explicit board exhaustion cannot loop forever. Preserve old rollout/resume/checkpoint behavior.

Gate: >=2 complete uncensored blocks, current-lineup win>=0.25 and VP>=4.0, <=1VP fixed-heuristic regression versus initial. Failure keeps stage and later allocation there. Frozen policies are not asserted stronger. Selection maximizes fixed heuristic VP then win then earliest, rejecting censored/>1VP-regressed results. Training boards for seed s are `[8000000+100000*s+i for i in range(4096)]`, seeds11/29; fixed validation62000000..62000003; reserved final71001000..71001007 never enters selection. Default PPO budget1800seconds total, <=65536choices/seed, rollout128,epochs2,seq16,lr3e-4,entropy0.01,value0.5,grad0.5. Evaluate/checkpoint each4096choices or stage end; equal training budgets.

- [ ] RED real-simulator tests for lineups, explicit schedules/exhaustion, proposal/actor expiry and dropped intervals, frozen immutability and old API. Add pure gate/selection tests for missing/censored/mismatched evidence and ties:

```python
assert not promotion_gate("random", truncated, fixed, initial)["promote"]
assert select_checkpoint([earlier_tie, later_tie])["id"] == earlier_tie["id"]
```

- [ ] Implement smallest compatible controls, reusing authoritative driver/evaluator and per-game persistence. No duplicated simulator/offer loop.
- [ ] GREEN focused suites, full suite once, ruff; commit and report. No sustained run until independent review.

### Task 3: Sustained study and before/after strategic assessment

**Files:** Create `docs/learning-curriculum-2026-11-04.md` and compact evidence under `docs/experiments/learning-curriculum-2026-11-04/`; large artifacts in ignored `runs/learning-curriculum-2026-11-04/`. No production changes during runs. Consume the completed independent audit (36 choices:24 reasonable,11 ambiguous,1 clearly bad; no passed directly legal purchases) and opportunity replay script under `/private/tmp/colonist-swarm-bxz1k39_/learning-audits`.

**Interfaces:** Reviewed supervised/curriculum APIs; exact pilot datasets/initial checkpoints; root session-horizon.json. Clean committed source, manifest before work. Effective catch-up study total <=3300seconds: supervised<=900, PPO<=1800, reserve600 for validation/final/audit; absolute training/evaluation cutoff09:00UTC on2026-11-05. Original4500second/06:29UTC schedule was missed and is retained in the ledger; this is a separately declared reduced catch-up allocation. Profile<=120seconds and512fresh decisions plus one update inside PPO budget; retain progress. Explicitly shorten equal budgets if horizon requires it.

**Acceptance:** Both seeds/configurations and partial failures retained;20epoch supervised target. Select initialization only with validation and retain original imitation reference. New user authorization permits exploratory PPO despite old failed gate, but checkpoint promotion uses new fixed-panel gates. Target substantive training up to65536choices/seed within measured time. Report actual reached/unreached lineups, all failures, entropy/KL/clip/value/gradient/throughput and complete-game gates. Freeze checkpoint selection before final71001000..71001007; compare initial/selected against fixed heuristics over identical32games per variant/seed if reserve fits. No post-test tuning; partial blocks stay unknown/unpaired. Keep selected and last checkpoints with digests.

- [ ] Verify source, artifact digests, actual/candidate split disjointness against prior pilot, and effective absolute deadline; persist manifest.
- [ ] Run one bounded study with atomic checkpoints/per-game evidence and honest failures; no silent restarts or budget expansion.
- [ ] Export actual post-training choices on diagnostic63000000/63000001 with the same deterministic24–48choice stratification. Coordinator obtains independent clearly_bad/reasonable/ambiguous audit with public rationale and legal alternatives; teacher disagreement alone is not bad.
- [ ] After source-end check, commit compact reports/evidence. Report actual compute, wins/VP/truncations, paired board differences and dependence caveat, strategic pattern changes, remaining gaps or no improvement honestly.

## Final review

Task reviews require explicit spec and quality verdicts. Root may dispatch workers directly if coordinator's harness thread limit blocks dispatch; the learning coordinator remains sole plan/ledger authority. Final broad review uses most capable model and all rulings/audits. Root owns combined integration, with catch-up reserve09:00–09:15UTC; the original missed schedule is preserved in the ledger. Strategic judgment does not substitute for code review.

## Scheduling amendment, 2026-11-05 07:47 UTC

The original internal03:54–06:54UTC horizon was missed after prolonged tool/approval elapsed waits; no training time is fabricated and the history remains in progress.md. Root ruled that the user’s continuing autonomous-work authorization permits finishing existing Tasks1–3 in a separate07:47–09:15UTC catch-up window, with integration09:00–09:15. Effective study caps are supervised900seconds, PPO1800seconds equally divided between seeds, evaluation/audit600seconds; profiling counts inside these caps. All computation stops by09:00UTC, even if intended decision counts are not reached. Root runs reviewed Task1 in an isolated frozen source copy while Task2 is implemented. Task3 consumes those supervised artifacts and must not duplicate them. No new architecture, features or paid/network work.
