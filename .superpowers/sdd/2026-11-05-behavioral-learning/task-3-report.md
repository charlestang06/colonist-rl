# Task 3: student-state labels and honest component corpora

Status: implemented and verified; ready for independent review. Branch
`cytang/behavioral-learning`, base `f6f796309dbef48befc03bd3dfeadbd462bd244b`.
No subagents, installs, network calls, paid services, sustained studies,
validation/final-board gameplay, or original-checkout writes were performed.

## Implementation and ownership

Only the three assigned source/test/documentation files were created:

- `src/colonist_rl/student_labels.py`: `collect_student_labels`,
  `save_student_labels`, `load_student_labels`, `write_corpus`, `load_corpus`,
  and `corpus_sequences`.
- `tests/test_student_labels.py`: 27 focused cases, with parameterized adversarial
  metadata/deadline/corpus variants.
- `docs/student-labels.md`: exact versioned metadata schema, verified checkpoint
  identity, deadline ordering, progress/status rules, partition protection and
  caller examples.

This report is the sole extra committed file. The accepted driver/fitter,
legacy demonstration validation, curriculum, simulator, policy, adapters, CLI,
rewards, old datasets and checkpoint schemas were not changed.

The collector uses a verified frozen copy of the supplied policy. Checkpoint
identity is exactly `{path, sha256, architecture}`. It checks original bytes,
loads through accepted `load_checkpoint`, checks accepted `policy_architecture`
and every state tensor, then collects in eval/no-grad mode without altering the
caller's modes or parameters. Python/NumPy/CPU Torch RNG state is restored.
Only actual student actions drive the existing `_MatchDriver`. Every own
observation, forced choice and prefix is retained; memory resets only at episode
boundaries and is never serialized.

Student-label v1 samples have exactly Observation, teacher index, behavior index
and execution flag. Per-episode metadata records original board/teacher seeds,
seat schedule, sampling seed, checkpoint/architecture digests, full teacher call
ordinals/versions/indices, ordered history digest, steps and honest boundaries.
Serialization is atomic finite canonical gzip JSON. No reward, hidden state,
opponent input or actor tensor is persisted. Unknown observation-envelope fields
fail instead of being silently discarded by the shared decoder.

Corpus v1 manifests retain absolute input paths, original bytes/digests,
format/version, each original collection source, actual boards and order. The
derivation source remains separate. All actual boards must be declared training
boards in five exact, disjoint partitions. Reserved legacy/curriculum finals are
protected even when omitted from the caller's final list. Missing/changed inputs,
duplicate digests, altered order/identity, split leakage, inconsistent sample
flags/history, and known-zero supervision fail closed. Neither the manifest
output nor its staging path may overwrite a referenced component, including a
staging symlink. Projection validates and reloads components, then returns the
accepted `EpisodeSequence` tuples without fabricating merged legacy data.

## RED/GREEN evidence

Interpreter throughout:
`/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python`.
Commands used `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src`; Torch threads were one in neural fixtures and smoke.

1. An initial fixture typo referenced absent `action_head`; it was corrected to
   real child `node_in` before measuring the feature RED run.
2. Initial RED: `python -m pytest tests/test_student_labels.py -q --tb=line`
   produced **20 failed in 2.15s**, all explicit assertions that the student-label
   and corpus module was missing. This covered executed student disagreements,
   full forced histories, teacher replay, immutable sources/bytes/order, illegal
   labels and provenance, separate indices/flags, and deadline boundaries.
3. Initial GREEN: the same focused file passed **20 tests in 4.20s** after the
   implementation.
4. Additional RED privacy/typing regressions produced **2 failed, 22 passed in
   7.19s**: private fields in the observation envelope were silently dropped,
   and boolean coverage compared equal to integer counts. Strict envelope and
   canonical typed-count checks fixed both: **24 passed in 6.16s**.
5. Additional RED immutable-output and missing-component regressions produced
   **2 failed, 24 passed in 5.37s**. Output collision rejection and normalized
   missing-component validation fixed both: **26 passed in 4.48s**.
6. A staging-symlink variant then failed **1 failed, 26 passed in 4.33s**, proving
   the temporary output could alias a component. Resolving both actual output
   paths fixed it; final full-suite verification includes all **27** focused
   cases.

The scripted boundary tests use real immutable Observation structures and assert
the actual actions passed to engine mutation, rather than merely asserting mock
existence. Teacher/student disagreement remains explicit. Multi-episode tests
check recurrent reset, original teacher seed/call ordering, atomic intermediate
progress and final status. Real collection verifies original heuristic RNG replay.

## Verification

- Focused integration command:
  `python -m pytest tests/test_student_labels.py tests/test_imitation.py
  tests/test_supervised_fit.py tests/test_training.py tests/test_curriculum.py -q
  --tb=short`: **127 passed in 20.04s** at the 24-case stage.
- Final full suite, once after all implementation changes:
  `python -m pytest -q --tb=short`: **455 passed, 1 skipped in 24.68s**.
- `/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check
  src/colonist_rl/student_labels.py tests/test_student_labels.py`: passed.
- `ruff format --check` for the same files: both already formatted.
- `git diff --check`: passed (repeated after staging to include new files).
- Self-review checked the three-file boundary, all mutation guards, no private
  diagnostic reads, source/checkpoint checks, complete histories and unchanged
  legal-menu order. Independent review is supplied by the coordinator.

## One actual bounded two-episode smoke

Artifacts are retained locally in
`.superpowers/sdd/2026-11-05-behavioral-learning/task-3-smoke/`:
`run.py`, predeclared `manifest.json`, initial checkpoint, `labels.json.gz`,
`corpus.json`, and `result.json`. They are intentionally not added as source or
as a new study. Existing saved manifests/protocols were searched first; fresh
training boards `73031000` and `73031001` had no occurrence. The held-out/audit
lists were declared, but no games or labels were collected from them.

The smoke used hidden size 8, seed 731, three heuristic opponents, stochastic
student actions, maximum 32 paired decisions, and maximum 16 game steps per
episode. Simulation and fitting shared **one 30-second absolute deadline**.
This was the sole actual smoke; it was not restarted or extended.

Actual wall duration was **1.4071859168 seconds**. Collection took
**0.1947723341 seconds**, exhausted both requested seeds, and retained **two
step-capped episodes at 16 steps**, seats 0/1. Both episodes have
`complete=false`, `terminated=false`, `truncated=true`,
`incomplete_reason="step_cap"`; neither is represented as a terminal game.
Collection status is complete only in the sense of exhausting the requested
board schedule.

There were **8 paired observations, 8 non-forced labels, 8 student actions
executed and 8 disagreements**, with 4 teacher calls per episode. The gzip file
roundtripped exactly. Component digest:
`0adaf845214dbf9e4fc680229199bd0fc342ceb3a12ca5f18701775ff5267677`.
Checkpoint digest:
`ff1a27acaf89837db3e65b28b4cf29c4a7f9f01f7b8873c630a45a2f93276a34`.

The genuine component corpus projected ordered episode lengths `[4, 4]` and ran
the accepted `fit_sequences` with one epoch, sequence length 4 and batch size 1.
Fit status was **complete**, with **2 optimizer updates, 8 supervised
presentations, loss 2.34969162940979**, no deadline interruption. This confirms
the adapter/fitter path; setup-only step-capped smoke data establishes no playing
competence or complete-game result. Collection source is honestly the base Git
revision with `dirty=true`, because this is a pre-commit implementation smoke.

### Actual independent source-identity audit (read only)

Beyond source-fixture unit tests, a one-time read-only integration audit referenced
the unchanged existing
`/private/tmp/colonist-swarm-bxz1k39_/expanded-teacher-data/seed-11-demonstrations.json.gz`
and the new smoke student component in a separate `actual-source-corpus.json`.
It wrote/loaded/projected a real corpus with **116 episodes / 10,008 decisions**,
preserving both exact component digests and original source identities:

- Original teacher source: revision
  `8b8974a7b31bce442a9b6950655f9013def95f2c`, `dirty=false`; original bytes digest
  `b03f7b6e2b206c80c83ab6578123e81dff0002b41d81d87cdd0130cc2f51f384`.
- New student source: revision
  `f6f796309dbef48befc03bd3dfeadbd462bd244b`, `dirty=true`; digest as above.

It explicitly verified original bytes before/after, preserved entry order and
both source dictionaries, and completed in **51.56594675 seconds**. Artifacts
`actual-source-corpus.json` and `actual-source-audit.json` remain beside the smoke.
**No fitting, sampling, simulator calls or gameplay occurred in this audit**;
the existing 10,000 labels were not recollected or retrained.

## Concerns and interface notes

No blocking defect was found in the accepted driver or fitter. Teacher-first
ordering is deliberate: legality is checked before student selection. If the
teacher call itself uses up the deadline, its final call ordinal/version/index
is retained, no student call starts, and no behavior index is invented. If the
student returns at expiry, the final paired sample is retained unexecuted.
Consumers should count actual teacher queries from `teacher_calls`; coverage
`decisions` counts paired student observations. This distinction is documented
and tested, and was sent to the coordinator/root for Task 4 integration.

Corpus digests detect changed bytes relative to the manifest; they do not
authenticate a wholly rewritten manifest or independently prove a terminal
outcome by replaying a game. Full-game competence remains outside this task.

## Review fix round 1 — I1 and M1

Fix base: `f3de317658bde2db56e22f54f7fe5647f2beb50b`.
Read the complete `task-3-review.md` before editing and reproduced both findings
through the public APIs on disposable pytest fixtures. This round changes only
the original three owned module/test/documentation files plus this report.
No changes to policy, training, curriculum, scorers, legacy loaders or datasets;
no new study, smoke run, fitting experiment, subagent, network or installation.
Original artifacts, the other frozen training clone and the user's original
roadmap edit were left untouched.

### I1 — existing staging hard links could overwrite immutable input bytes

Root cause confirmed: `Path.write_bytes()` opened a predictable existing `.tmp`
path and truncated its inode; resolved path strings cannot identify hard links.

Added two targeted regressions covering corpus and student-label serialization.
Each creates a staging hard link to a disposable component, calls the public
writer, then checks exact original bytes, the retained staging link and inode,
independent output inode/digest, valid load/projection, and temporary cleanup.
For the student writer the aliased input is a distinct teacher component, so
the old behavior demonstrably changes its bytes rather than merely re-saving
the same payload.

Exact RED command:

```text
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest \
  tests/test_student_labels.py -q -k staging_hardlink --tb=short
```

Result before production changes: **2 failed, 27 deselected in 1.98s**; both
failed on the referenced component bytes changing. An earlier draft of the
student regression reused identical bytes and failed on the staging link being
consumed; it was corrected to the distinct teacher bytes before this RED run.

Fix: `_atomic()` now uses `tempfile.mkstemp()` to create a fresh exclusive file
in the output directory, writes through that returned descriptor, closes it,
then replaces the destination. A `finally` removes unused fresh staging files.
Existing paths are never opened for writing. The direct-output and resolved
symlink collision checks in `write_corpus()` remain unchanged.

GREEN command used `-k 'staging_hardlink or corpus_output'`: **4 passed,
25 deselected in 2.68s**, covering both new hard-link regressions and both
existing direct/symlink rejection cases.

### M1 — Python equality hid differences in metadata JSON types

Root cause confirmed: dictionary equality equates equal-valued booleans,
integers and floats. Entry identity checks therefore accepted malformed metadata
even when the independently recomputed order digest was internally valid.

Added three targeted cases: component `version=true`, `source.dirty` replaced
by its equal-valued integer, and an actual board seed replaced by its
equal-valued float. Each independently recomputes the canonical order digest
and calls public `load_corpus()` and `corpus_sequences()`. The cases also verify
both original component hashes and source dictionaries remain unchanged.

Exact RED command used the same environment/interpreter and
`python -m pytest tests/test_student_labels.py -q -k corpus_metadata_types
--tb=short`: **3 failed, 29 deselected in 2.38s**, all because malformed corpus
loading did not raise.

Fix: compare the full loaded component identity to its manifest identity with
canonical typed JSON, as already used for coverage; keep the separately typed
order check. No collection source is rewritten, and unchanged legacy component
validation/semantics are preserved. The same command then passed **3 tests,
29 deselected in 2.63s**. Documentation now describes both fixes.

### Covering verification and self-review for this round

```text
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest \
  tests/test_student_labels.py tests/test_imitation.py tests/test_supervised_fit.py \
  -q --tb=short
```

Result: **71 passed in 17.12s**, including all **32** student-label cases,
existing legacy demonstration/imitation tests and shared-fitter tests.
The original full-suite result above remains historical evidence for the
original implementation; this narrow fix round did not rerun unrelated game,
policy or curriculum suites and did not repeat the empirical smoke/audit.

`ruff check src/colonist_rl/student_labels.py tests/test_student_labels.py`
passed; `ruff format --check` reported both files already formatted;
`git diff --check` passed. Self-review read the complete round diff, verified
exclusive same-directory staging, closed-descriptor replacement and cleanup,
preservation of collision guards, typed comparison before board membership,
unchanged actual component sources/bytes, and the four-file ownership boundary.
Both review findings are addressed; independent re-review remains with root/lead.
