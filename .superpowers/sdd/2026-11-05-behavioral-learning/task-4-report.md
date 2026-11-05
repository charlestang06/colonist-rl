# Task 4: minimal audited acquisition comparison

Status: implementation complete; final covering verification recorded below;
independent review remains with root/learning lead. Base
`f555ee482e6005840aaef9ff70e17a9509604ff4`, branch
`cytang/behavioral-learning`. Initial implementation commit
`647385b5eca028c5bb1b5d1da46dca22199b579a` is retained because it identifies the
clean source used by the sole actual integration smoke. A narrow follow-up commit
adds the teacher-writer hard-link fix and this report; delivery HEAD identifies it.

## Ownership and behavior

Only these assigned files and this report changed:

- `src/colonist_rl/behavioral_study.py`: exact-protocol acquisition driver and pure
  `select_choice_prefix` function.
- `tests/test_behavioral_study.py`: 42 focused cases, including parameterized
  protocol/target/start-metadata adversarial cases.
- `docs/behavioral-study.md`: API, exact schema, resource semantics, provenance,
  interruption accounting, root's intended subsequent launch and limitations.
- `.superpowers/sdd/2026-11-05-behavioral-learning/task-4-report.md`.

No simulator, teacher, collector, corpus, shared fitter, checkpoint format,
evaluator, policy, rewards, gates, CLI, curriculum, roadmap or original checkout
was modified. No helpers/reviewers/subagents, network, installations, paid services,
control/feature refits, sustained acquisition study or PPO run were used.

The driver validates the exact seed/arm/input matrices, canonical absolute input
paths and hashes, clean source revision, disjoint five-key board partitions,
nontraining/held-out/final leakage, acquisition reuse, original component identities,
known source metadata, finite legacy starts and any declared starting seed/full
fit-call milestone evidence. It loads one frozen reference per seed and makes two
independent copies. Fresh independent Adam objects persist across fit segments;
original components/order and training-only balanced phase weights are identical
within a pair. Every selected choice phase must be covered.

It delegates collection to the accepted teacher/student APIs, saves raw immutable
components, uses student `progress_path`, and records teacher queries, paired and
executed labels/choices, unmatched calls and unexecuted pairs separately. Each
seed's deterministic full-history views retain exactly the minimum new choice
count, with forced prefixes, ordered endpoints, unused suffix locations/counts,
visible board digests, original board seeds, component/source and derivation
identities. Raw flags and sources are never rewritten or merged into a fake v1.
All selections and expected work are fixed before any fitting.

Expected and actual updates, presentations, prefix/chunk forwards, batch chunks,
weighted loss mass and runtime are separate. Partial last segments retain their
actual updates beyond completed epochs. Segment checkpoints contain optimizer
state; exceptions retain changed parameters and explicitly mark unobservable work
unknown. Held-out scoring uses the accepted no-gradient unweighted scorer, and
preserved checkpoint-specific scores cannot be reused as a later model's score.

Evaluation uses only `evaluate_blocks` and its authoritative per-game callback.
Each returned game is saved atomically. Exact declared full panels and matching
journals are required; censored/missing/mismatched panels and incomplete work
suppress every comparison, without subset salvage. Final source/input/raw-component
identities are verified before paired summaries. Handoff is only checkpoint,
selection and frozen-reference data for existing curriculum inspection.

Root clarified during implementation that `evaluation_seconds` is one global
reserve shared by every variant. `collection_seconds` and `fit_seconds` are per
arm/seed; all ceilings are clipped by the absolute whole-run deadline. The docs
record the intended later 4,000-query, legacy30, five-epoch run with 120-second
collection, 600-second per-arm fit, shared 600-second evaluation and 3,600-second
whole-run caps. That study was not started.

## RED/GREEN evidence

All commands used the existing interpreter
`/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python`, clone
`PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`;
neural fixtures explicitly set Torch threads to one. No runtime was installed.

1. Initial `python -m pytest tests/test_behavioral_study.py -q --tb=line`:
   **27 failed in 22.85s**, explicitly asserting the module was missing.
2. First implementation run: **26 passed, 1 failed in 27.36s**. The failure
   revealed a fixture assumption: the fifth real collection observation was
   forced. The fixture was replaced with explicit legal choice/forced menus and
   ordered versions, rather than changing correct prefix behavior.
3. Added wrong-board and final raw-byte-drift regressions:
   **2 failed, 31 passed in 38.43s**, both originally failed to raise. The driver
   now checks returned acquisition settings/source and ordered board prefixes,
   plus final raw identities. A subsequent run exposed that corrupt gzip raised
   `BadGzipFile` before the expected digest diagnosis; digest comparison now
   precedes decoding, preserving a clear failed integrity journal.
4. Starting seed/partial-fit evidence and weighted-mass regressions:
   `-k 'starting_checkpoint or actual_weight_mass or unmatched_teacher or global_ceiling'`
   yielded **4 failed, 2 passed, 33 deselected in 29.09s**. The first three
   accepted mismatched/partial starting metadata; the fourth lacked actual mass.
   Starting evidence audits and schedule-derived actual mass fixed them.
5. Focused covering run after those fixes: **40 passed in 17.23s**. It includes
   persistent independent optimizers, immutable inputs, full forced-history
   selection, unused illegal labels, exact matrices, expiry, partial fits,
   fit/game exceptions, unmatched teacher calls, uncovered phases and global
   evaluation deadlines.
6. Additional stale-score regression: `-k partial_last_segment`:
   **1 failed, 40 deselected in 5.55s**, because a partial later checkpoint still
   exposed an earlier checkpoint's score. Removing the stale latest score while
   retaining checkpoint-specific score history passed **1 test, 40 deselected in
   2.94s**.
7. Initial full suite: **501 passed, 1 skipped in 76.52s**.
8. Final self-review found the unchanged legacy demonstration saver has a fixed
   `.tmp` name. A disposable-input hard-link regression produced **1 failed,
   41 deselected in 2.90s**, with actual input-byte corruption. The driver now
   runs that accepted saver inside an exclusive same-directory staging folder,
   then atomically publishes the output, as already done for checkpoints.
   The same test passed **1 test, 41 deselected in 4.59s** and verifies original
   bytes, retained hard link/inode and completed study evidence. Neither accepted
   saver was changed. This new production change justified a final covering
   full-suite run; the actual empirical smoke was not repeated.

Final full-suite result is recorded in the verification addendum below.

Ruff check and `ruff format --check` on both owned Python files passed;
`git diff --check` passed. The installed-venv import smoke ran from `/private/tmp`
with this clone's absolute `src` on `PYTHONPATH`, imported the new driver, accepted
student corpus and fitter, checked both callable APIs and reported installed
`colonist-rl 0.1.0`, this clone's module path and Torch threads 1. This verifies the
existing installed dependency runtime against the implementation; it performs no
new package installation.

## Sole bounded actual integration smoke

Executed once from clean source
`647385b5eca028c5bb1b5d1da46dca22199b579a`, with no provenance overrides, using
`.superpowers/sdd/2026-11-05-behavioral-learning/task-4-smoke.py` and artifacts in
`.superpowers/sdd/2026-11-05-behavioral-learning/task-4-smoke/`.
The script installed a 30-second process alarm before Torch imports, and gave the
whole tiny invocation a 20-second monotonic deadline. `/usr/bin/time -p` reported
**12.06s process wall time**; measured post-import fixture/study work was
**7.6175s**. It was neither restarted nor extended.

This used one seed (11), a fresh legacy hidden8 fixture, four original training
choices, four held-out choices, four acquisition queries per arm, one epoch,
sequence length 2, batch size 1 and 24 game steps. Per-arm collection/fit ceilings
were 2 seconds; evaluation had one shared 5-second reserve. All boards were tiny
fixture boards 93210000..93210005, disjoint from reserved finals and root's actual
validation panel. The fixture evidence receipt explicitly states zero control
refits; this is an integration check, not a new empirical comparison.

Each arm recorded exactly **4 teacher queries, 4 paired labels, 4 non-forced labels,
4 executed decisions/choices, 0 unmatched calls and 0 unexecuted pairs**. Both
selected `Q=4`. Each fitted **1 completed epoch, 4 updates, 8 supervised
presentations, 4 prefix forwards, 8 chunk forwards, 4 batch chunks and weighted
mass 8.0**. Teacher/student collection took 0.1228/0.1063s and recorded fit work
0.1104/0.1075s respectively.

The heuristic reference and both arms each returned four authoritative games:
**12 finished, 12 step-truncated, 0 resolved**. Final study status was correctly
**partial**, with **no comparisons**. Source and every input hash verified; raw
components, selection manifests, checkpoints and all 12 game rows remain. No
playing-competence claim follows from setup-heavy tiny data or censored games.

The only subsequent production change is isolation of the accepted legacy teacher
saver's temporary path, covered by the hard-link regression and final full suite.
The historical clean smoke revision remains in commit history; its source identity
has not been relabeled as the follow-up revision.

## Self-review and remaining limitations

Self-review checked the complete owned implementation and final staging diff,
exact ownership against base, accepted API delegation, strict typed identities,
checkpoint/optimizer independence, deterministic complete-prefix selection,
training-only weights, precomputed schedule/mass accounting, exception journals,
source/input/raw drift, global evaluation deadline, authoritative game rows and
all-or-nothing paired comparisons. No general study infrastructure was added.

Known accepted-interface limitations are explicit: the legacy teacher collector
has no progress callback and cannot return partial histories after an internal
exception; a returned unexecuted teacher endpoint is retained raw but rejected by
the accepted corpus before fitting. A held-out scoring call and filesystem writes
are bounded only between operations, so deadlines may be exceeded by one bounded
operation; actual time is retained. Unknown work after an exception is never
inferred from completed epoch counters. Query/choice matching does not equalize
updates, weighted mass, forced-prefix work or runtime, and supports no pure
state-distribution causal attribution.

Independent review and any separately authorized real acquisition study remain
with root/learning lead. No other implementation concerns are known.

## Final verification addendum

After the isolated teacher-writer fix, the full existing suite plus all **42** new
focused cases passed **502 tests, 1 skipped in 65.91s**. Exact command:

```text
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q --tb=short
```

Final Ruff check and format check passed on both Python files, and
`git diff --check` passed. The same installed-venv import smoke was rerun against
the final source from outside the checkout. Only the three assigned files and
this report were staged for the final local commit. No actual smoke or study was
repeated after the staging fix.
