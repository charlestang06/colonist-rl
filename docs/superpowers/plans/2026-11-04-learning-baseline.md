# Learning Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver reproducible teacher benchmarking, current-parameter recurrent imitation and bounded complete-game evidence before spending more on PPO.

**Architecture:** Keep the current simulator and observation contracts. Add focused evaluation, imitation and experiments modules; preserve legacy training entry points. The learning coordinator owns all CLI integration and dispatches one fresh implementer and independent reviewer per task.

**Tech Stack:** Existing Python, PyTorch, NumPy, Catanatron 3.2.1, pytest and ruff; standard-library JSON/gzip/hashlib only for artifacts.

**Spec:** `docs/superpowers/specs/2026-11-04-learning-baseline-design.md`

## Global Constraints

- Work only in `/private/tmp/colonist-swarm-bxz1k39_/learning`; never modify the original repository or its environment.
- Use `/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python` and the existing ruff executable, with `PYTHONPATH=/private/tmp/colonist-swarm-bxz1k39_/learning/src` and `PYTHONDONTWRITEBYTECODE=1`.
- CPU only, one Torch thread, one simulation/training worker, no paid services, no network installs, and no dependency changes.
- Actors and teachers receive only immutable `Observation`; simulator-private state is permitted only for outcome diagnostics after a game, never labels, features, masks, or action selection.
- Preserve existing CLI commands, imports, checkpoints and action/observation schema. Own learning code, learning tests and learning docs; do not modify adapters, contracts, board, beliefs, simulator or opponents without root agreement.
- Never report forced-only accuracy, finite losses, passing tests, incomplete games, or manual browser victories as evidence of learned competence.
- Training, model-selection validation and reserved final-test board seeds must be disjoint and persisted. All variants compare the same complete four-seat blocks. Report every truncation and failure without converting either into an automatic loss.
- Fresh implementer per task, independent spec-and-quality task review, reviewed fixes, and a final broad review are required. Coordinators do not implement production code.

## File ownership and sequence

`evaluation.py` owns experiment-grade complete board blocks and pairing. `imitation.py` owns datasets and supervised recurrent fitting. `experiments.py` owns manifests, orchestration and gates. `training.py` keeps backwards-compatible wrappers and PPO. `cli.py` gains one experiment command. Tasks are sequential because Task 3 consumes Tasks 1/2 and Task 2 may modify the same training module as Task 1. Tests are task-local; shared code remains read-only.

### Task 1: Complete board-block evaluation and teacher benchmark protocol

**Files:** Create `src/colonist_rl/evaluation.py`, `tests/test_evaluation.py`; modify `src/colonist_rl/training.py` only around evaluate; add learning protocol documentation in `docs/learning-protocol.md`.

**Interfaces:**
- Consume `GraphPolicy`, `HeuristicPolicy`, `RandomPolicy`, `FourPlayerGame` and `_MatchDriver` without changing their semantics.
- Produce `evaluate_blocks(policy=None, *, board_seeds, candidate="heuristic", baseline="heuristic", max_game_steps=20000, deterministic=True, deadline=None) -> dict` in `evaluation.py`.
- Produce `paired_block_summary(reference: dict, candidate: dict) -> dict` and `validate_seed_splits(training, validation, final_test) -> None` there.
- Keep `training.evaluate` legacy positional/keyword compatibility; add optional keyword-only seed/candidate/deterministic/deadline support as needed, with one authoritative match loop. A complete protocol call must never silently use a partial four-seat block.

**Acceptance:** Explicit unique integer seed lists; four seats per board; fresh heuristic/random candidate instance and recurrent state per game; exact reproducible outcomes excluding runtime measurements; candidate identity and action mode; outcome `terminated` separate from `truncated`; learner decision and non-forced counts, phase/action-kind counts and latency. Return partial status on a deadline without calling partial blocks complete. Preserve external Torch/Python/NumPy RNG and policy mode in `finally`. Pair exact unique board/seat keys and aggregate differences at board level; reject missing/duplicate pairs. No statistical significance assertion from a small sample. JSON-safe outputs only.

- [ ] Write failing protocol tests, including this seed-split contract:

```python
@pytest.mark.parametrize("splits", [([1], [1], [3]), ([1], [2], [2])])
def test_split_overlap_is_rejected(splits):
    with pytest.raises(ValueError, match="disjoint"):
        validate_seed_splits(*splits)
```

Add a real short-game integration case with `board_seeds=[701, 702]`, `max_game_steps=24`, asserting the eight `(board_seed, seat)` keys and eight explicit truncation outcomes, then repeat and compare outcomes excluding timings. Use synthetic outcomes with two boards to verify paired per-board win and VP differences, duplicate/missing rejection, invalid seed types, and deadline partial accounting. Inject a raising policy to verify RNG/mode restoration; avoid only mocking the logic under test.

- [ ] Run the new focused tests before implementation and record the expected missing-interface failure.
- [ ] Implement the smallest compatible protocol and document precise fields and seed derivation. No policy/heuristic tuning belongs in this task.
- [ ] Run focused new tests plus training integration and CLI tests; then the entire existing suite once and ruff on touched files. Commands use the interpreter and environment in Global Constraints and `-p no:cacheprovider` for pytest.
- [ ] Commit code/tests/docs; write the report with RED/GREEN output, file list, interface details and concerns. Maximum experiment computation for this task is 60 seconds; tiny truncated integration tests are correctness checks only.

### Task 2: Persisted demonstrations and correct recurrent imitation

**Files:** Create `src/colonist_rl/imitation.py`, `tests/test_imitation.py`; modify the `imitation_warmstart` function in `src/colonist_rl/training.py`; extend `docs/learning-protocol.md`.

**Interfaces:**
- Consume existing read-only observation codecs `observation_to_dict` and `view_from_dict` from `adapters.observation_io`, `_MatchDriver`, `HeuristicPolicy`, `GraphPolicy`, backend/action schema constants.
- Produce `collect_demonstrations(*, board_seeds, max_decisions, max_game_steps=20000, baseline="mixed", deadline=None) -> dict` where returned primitive metadata plus episode sequences contain immutable observations and legal action indices in memory. For each board index i the learner seat is `i % 4`; seed/seat are explicit in every episode.
- Produce `save_demonstrations(path, dataset) -> str` (SHA-256 of persisted bytes), `load_demonstrations(path) -> dict`, `fit_imitation(policy, dataset, *, epochs=2, sequence_length=16, learning_rate=3e-4, deadline=None) -> dict`, and `imitation_metrics(policy, dataset) -> dict`.
- Preserve `imitation_warmstart(policy, decisions=256, seed=0, epochs=2, learning_rate=3e-4)` and return keys `loss`, `decisions`, `epochs`; add diagnostic fields without removing existing ones. Its board-seed range may be generated lazily to collect the requested cap.

**Acceptance:** Full ordered episodes unless step/decision/deadline caps intervene; caps are hard and incomplete episodes explicitly flagged. Teacher sees only Observation and chosen index is always legal. Persist schema/teacher/source/actual seeds/coverage/timing and observations in gzip JSON with validation; reject incompatible backend/action schema and out-of-range indices. Do not serialize engine objects, hidden states or private diagnostics into actor labels. Validation never mutates model weights/mode or RNG. Train non-forced choices only while replaying all observations; no cached hidden states across updates. Recompute each chunk prefix under current weights with no grad, detach at chunk start, backpropagate ordered chunk, reset at episode boundary, skip forced-only chunks. Report zero supervised count and null accuracy for all-forced datasets, plus per-phase and action-kind counts, agreement and CE with finite/null JSON values. Legacy tiny warmstart remains operational.

- [ ] Add failing tests for legal teacher indices and actual board provenance, gzip roundtrip and schema rejection, cap-caused incomplete episode, model-parameter invariance during validation, and all-forced metrics. Use real simulator observations for persisted roundtrip coverage.
- [ ] Add a state-sensitive test policy whose recurrent forward calls record prefix/episode replay. After multiple updates assert prefix recomputation uses current weights and no memory tensors are dataset fields. Verify forced observations advance memory without entering the CE denominator. Test this expected contract:

```python
metrics = imitation_metrics(policy, forced_dataset)
assert metrics["non_forced"]["count"] == 0
assert metrics["non_forced"]["accuracy"] is None
```

- [ ] Run the new tests to record expected missing interfaces, then implement the bounded collector/persistence/trainer and replace only legacy warmstart internals.
- [ ] Run focused imitation and existing warmstart/rollout/checkpoint tests, full suite once, and ruff. A non-forced small-dataset overfit check may establish optimization behavior but must not be called playing competence. Maximum task experiment compute is 120 seconds.
- [ ] Commit code/tests/docs and report RED/GREEN evidence, exact metadata shape and any uncovered decision kinds.

### Task 3: Reproducible experiment CLI and bounded local evidence

**Files:** Create `src/colonist_rl/experiments.py`, `tests/test_experiments.py`, `docs/learning-pilot-2026-11-04.md`; modify `src/colonist_rl/cli.py`, `tests/test_cli.py`, `docs/learning-protocol.md`. Save small raw JSON summaries/manifests under `docs/experiments/learning-pilot-2026-11-04/`; large datasets/checkpoints stay in ignored local `runs/learning-pilot-2026-11-04/` with digests and paths recorded.

**Interfaces:**
- Consume Task 1 evaluation/seed validation/pairing and Task 2 dataset/fitting/metrics interfaces, plus existing save/load checkpoint and fresh PPO collector/updater.
- Produce `run_experiment(output_dir, *, training_seeds=(11,29), decisions=2500, epochs=2, hidden_size=32, sequence_length=16, validation_board_seeds=(90000,90001), validation_demo_seeds=(91000,91001), final_test_board_seeds=(100000,100001), max_game_steps=20000, wall_seconds=900) -> dict`.
- Add `colonist-rl experiment --output PATH --decisions N --epochs N --hidden-size N --wall-seconds N` with defaults above; optional explicit training/evaluation seed arguments may be added if required for reproducibility. Keep current CLI behavior and monkeypatch seams.

**Acceptance:** Create manifest before computation, all actual seed splits disjoint, train board candidates exactly `[1000*s+i for i in range(64)]`, source HEAD+dirty status, dependency/action schema/model configs, RNG seed derivation, dataset and checkpoint digests, runtime/thread limits, statuses and failure records. Abort before simulation if split validation fails. Preserve raw per-game outcomes and per-training-seed results, not only the best. Evaluation variants: heuristic/random candidates against random/heuristic opponents; untrained and imitated students for both training seeds against heuristic opponents on identical complete validation blocks. Validation demonstrations are only for diagnostics and never gradients; final test is reserved and never consumed in this pilot. Record stochastic/argmax choice (default argmax).

The 60-second probe runs at most one board block; keep its raw evidence separate. Decide whether the 900-second total budget can fit two validation blocks using measured throughput; if not, run one common block and mark inconclusive. Target 5000 total training demonstrations (2500 per seed); use measured throughput to reduce both caps equally before training if the wall budget requires it, recording the change and shortfall. Persist progress at phase/game boundaries and check the total deadline on every simulator decision/update. Deadline/exception returns a partial/failed status with evidence, never successful competence. If fewer than all required variant blocks finish, do not calculate an apparently paired comparison over unequal sets.

PPO gate is false unless all conditions hold: at least two training seeds; eight complete board blocks per variant/seed; zero truncations; at least 100 held-out non-forced choices and agreement >=0.60; each seed's imitated VP >= teacher VP-1.0 and win rate >= teacher win rate-0.10 on identical blocks. This pilot cannot pass the sample-size gate. Record `ppo_status="deferred_insufficient_baseline_evidence"` for it. Expose a pure gate function and document the subsequent bounded fresh-data PPO continuation (1024 decisions/seed, two epochs) without launching it on inadequate evidence. Do not implement critic pretraining or an imitation anchor in this wave.

- [ ] Add failing tests for split collision before expensive work, tiny CLI run artifacts using bounded injected stage functions where appropriate, manifest digest reproducibility, deadline partial status, phase-specific non-forced gate rejection, truncated/incomplete block rejection, and exact paired-block matching. Test gate boundaries with synthetic statistical inputs, not expensive training.

```python
assert not baseline_gate(summary_with_four_games)["eligible"]
assert not baseline_gate(summary_with_forced_only_accuracy)["eligible"]
assert not baseline_gate(summary_with_truncation)["eligible"]
```

- [ ] Run new tests for expected missing interfaces, implement orchestration and CLI, and verify full old CLI behavior and suite once plus ruff.
- [ ] Run the declared runtime probe then the bounded local pilot with one CPU worker. Keep actual outcomes even when there are zero student wins or the deadline interrupts. Save raw small results and report exact commands, limits, artifacts/digests, teacher comparisons, student comparisons per training seed, non-forced metrics, uncovered phases/actions, truncations and remaining gaps.
- [ ] Commit implementation/tests/docs and small raw evidence; do not commit large datasets or checkpoints. Report measured gate decision honestly. This task owns at most 900 seconds of pilot compute including the probe, separate from short correctness tests.

## Final integration and review

The coordinator generates task-specific review packages from each saved BASE and HEAD using skill scripts. Independent reviewers must return spec compliance and code quality verdicts. Fix rounds resume original implementers for rounds 1-3; review only fix diffs. After all tasks pass, run a broad learning diff review on gpt-6-astra, hand root all commits and preserved report/ledger paths, and let root perform combined integration verification with the adapter stream. No direct original-repository writes, merge, push or new user chats.
