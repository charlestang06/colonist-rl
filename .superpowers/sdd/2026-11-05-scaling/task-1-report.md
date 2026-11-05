# Scaling task 1 report

Status: DONE_WITH_CONCERNS — implementation and bounded development validation
complete; independent review and root's finalized launch manifest remain required
before sustained use. No sustained collection, 100k training, real evaluation
panels, bridge conversion, original-runtime changes or artifact rewriting occurred.

## Ownership, source and commits

- Checkout: `/private/tmp/colonist-swarm-bxz1k39_/scaling/repo`, branch
  `cytang/scaling-learning`; verified initially clean at planning BASE
  `0873ba02472f47f600aa8e39884bea64f79ad300`.
- Implementation commit: `47e2a0adcb454d56a1b2bab90e83a52426e92f6a`.
- Owned production/API file: `src/colonist_rl/scaling_study.py`; tests:
  `tests/test_scaling_study.py`; public schema/launch/limits documentation:
  `docs/scaling-study.md`; this report only. A subsequent test/report commit
  isolates the forced-history negative control's optimizer state. Production code
  is unchanged from the full-suite and smoke commit.
- Read task-1-brief.md, dispatch.md and the approved scaling design first. Used TDD
  and verification-before-completion; no helpers, reviewers or subagents spawned.
- Consumes existing demonstration persistence/collection, fitter/scorer, policy
  architecture, checkpoint and evaluation APIs. Does not import the migration
  owner's absent module or modify any existing production file.

## Implemented behavior

The three specified entry points accept strict documented version-one launch
protocol dictionaries, persist them before work, verify clean imported-source Git
identity and exact installed backend/runtime, reject unknown fields/wrong types,
check absolute deadlines/RSS ceilings and all expected input paths/hashes. The
four arm identifiers remain exactly10000/11,10000/29,100000/11,100000/29.

Canonical initial metadata follows root's explicit ruling: reviewed parent and
bridge artifact refs, migration source backend3.2.1 / target pinned new backend,
source steps0 and matching seed, zero_additive_consequence upgrade linked to the
bridged Graph checkpoint SHA, purpose untrained_initialization. Starts must be
finite hidden32 hand_recipe, steps0, optimizerNone and additive coefficients zero.
The driver verifies supplied lineage bytes/metadata; root verifies actual tensor
ancestry. Legacy parent checkpoints are never loaded by this driver.

Collection uses ordered four-board calls and learner seats0/1/2/3, retains all
forced/earlier own histories, saves every returned raw component and stops on an
incomplete/truncated batch without replacement. Only complete prefixes count.
Milestones10k/25k/50k/100k are first crossing complete component counts with honest
overshoot. Validation-label components remain separate. Finalization requires100k
and exact declared supervised phase coverage; both sizes/seeds share full-training
N/(P*n_phase) weights. Stopping at10k remains partial and cannot fit primary arms.

Before use, each raw component is hashed, checked for ignored envelope/hidden
fields, loaded through the existing loader, validated for exact typed provenance,
board/seat/order/coverage and complete boundaries, then replayed with the original
simulator/driver/teacher call order and board+100 teacher RNG. Replay compares
every own view, legal menu/index and natural terminal boundary without a second
observation dataset. Startup revalidates/recounts sequentially; only primitive
refs and summaries survive component release. No whole-corpus helpers are called.

Fitting calls existing batch-one/sequence16 fitter once per component per global
epoch with the SAME Adam, fixed LR0.0003, shared weights and profiling. It preserves
current-weight prefix semantics. Returned losses aggregate by presentation mass.
Scoring streams count/correct/CE sums and phase/kind denominators at0/1/3/5.
Boundary checkpoints precede immutable SHA-bound cursors; only successful cursor
publication commits epoch/component/counters. Partial-last snapshots and attempted
work are separate. Explicit resume restores optimizer/RNG from a verified durable
boundary, recovers pending scoring and replays the next whole component without
renewing deadlines or duplicating committed updates. Caller RNG/thread settings
are restored on normal/exceptional exits; existing fitter/scorer preserve modes.

Evaluation validates all four result/cursor/checkpoint identities and exact
completed counters before choosing largest common1/3/5, solely by completion.
It persists selection before gameplay and retains primary-five failure on fallback.
Existing evaluate_blocks runs the fixed validation-only sixteen-board/four-seat
panels with per-game immutable events and authoritative journals. Missing,
mismatched, censored or failed journals suppress paired comparisons. Existing
paired_block_summary supplies board-level teacher/size comparisons. No final-test
or audit board is played by the driver.

## Exact commands and test evidence

All Python commands ran in the scaling checkout using ONLY this upgraded runtime:

```
/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python
```

The common final command environment was:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src
```

Earlier RED/GREEN commands used the same interpreter with
`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src`;
Torch was also explicitly set to one thread. No bytecode, installation, network
operation or original interpreter import was used.

All tests used `python -m pytest -q -p no:cacheprovider`. The following are the
exact test arguments and observed sequence:

| Stage | Arguments after pytest flags | Result |
|---|---|---|
| Initial RED | `tests/test_scaling_study.py` | 4 failed, module-missing assertion,3.54s |
| Initial GREEN | same | 4 passed,15.00s |
| Collection/loader RED | `tests/test_scaling_study.py -k 'collection or component_loader or protocol_failures'` | 2 failed (missing collection/loader behavior),8 passed,4.52s |
| Collection GREEN | `tests/test_scaling_study.py -k 'collection or incomplete_component or component_loader or protocol_failures'` | 11 passed,3.60s |
| Streaming/cursor RED | `tests/test_scaling_study.py -k 'streamed or interruption or rss_guard or corpus_manifest'` | 6 failed,4.17s; one test-fixture import typo corrected from nonexistent dataset_corpus to student_labels |
| Streaming/cursor GREEN | same | 6 passed,9.51s |
| Evaluation RED | `tests/test_scaling_study.py -k 'evaluation_frozen'` | 3 failed, absent entry point,10.55s |
| Evaluation GREEN | same | 3 passed,10.42s |
| Intermediate focused | `tests/test_scaling_study.py` | 25 passed,16.43s |
| Pending-score RED | `tests/test_scaling_study.py -k 'real_expiry or pending_score or required_phase'` | 1 failed (missing recovered epoch1 score),2 passed,5.55s |
| Pending-score GREEN | `tests/test_scaling_study.py -k 'real_expiry or pending_score or required_phase or interruption'` | 6 passed,8.59s |
| Expanded focused | `tests/test_scaling_study.py` | 31 passed,18.17s |
| Hidden-envelope RED | `tests/test_scaling_study.py -k 'hidden_observation'` | 1 failed: ignored hidden_memory field accepted,2.45s |
| Hidden-envelope GREEN | `tests/test_scaling_study.py -k 'hidden_observation or component_loader'` | 2 passed,2.70s |
| Final focused before combined run | `tests/test_scaling_study.py` | 33 passed,21.58s |
| Full upgraded combined suite, ONCE | no test arguments | 972 passed,1 skipped,52.84s |
| Self-review test strengthening | `tests/test_scaling_study.py -k 'bitwise_adam'` | 1 passed,32 deselected,4.06s; removed-forced-history control now has its own persistent Adam, isolating that control from optimizer reset |

Final focused rerun after the test-only strengthening:
`tests/test_scaling_study.py` —33 passed in23.07s. The full combined suite was not repeated: production code was
unchanged and no regression concern justified another broad run.

Lint/format/diff commands (sibling upgraded Ruff):

```
/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/ruff check src tests
/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/ruff format --check src/colonist_rl/scaling_study.py tests/test_scaling_study.py
git diff --check
git diff --cached --check
```

All passed. The earlier formatting pass fixed import sorting and explicitly bound
an immediately executed save lambda's component; final source is formatted.

## Parity, controls and failure tests

Two >16-observation components with forced observations, two global epochs and
real GraphPolicy-derived hand_recipe compare concatenated batch1 fitting to
canonical component calls. Every model tensor and recursively every Adam tensor,
step and group field is bitwise equal. Updates/presentations/prefix/chunk/batch
counters match exactly; loss numerator agrees at1e-12 relative/absolute tolerance.
Both deliberately reset Adam and removed-forced-history with independent
persistent Adam change final parameters, preventing vacuous parity claims.

Tests also cover: singleton collector seat0 vs four-board rotation; whole-batch
threshold overshoot and two-component nested refs; preserved forced history;
missing phase; incomplete stop without replacement; digest/source/backend/typed
metadata/teacher label/missing-history rejection; overlapping and reserved boards;
unknown protocol keys/caps/runtime rejection before collection; changed corpus
summaries/weights/boards and partial corpus rejection; RSS refusal before load;
weakref-observed release before every following component load; forbidden
write_corpus/load_corpus/corpus_sequences patched to fail; JSON/checkpoint fixed.tmp
hard-link alias preservation and exclusive publication; real fitter expiry after
one Adam update; exception after updates; cursor publication failure after checkpoint;
changed resume identity; pending score recovery; exact resumed final model/Adam
and no duplicate committed updates; exact matrix/common milestone selection;
primary-five fallback retained; missing/censored journal suppression; validation
boards only. Inputs remain byte-bound; mutated input bytes fail before fitting.

## One development smoke

Exactly one development smoke invocation:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python /private/tmp/scaling-development-smoke.py
```

The clean source before/after was implementation commit47e2a0a. Verified installed
provenance: catanatron3.3.0, Git82aae93ab1f7c267218be0566df573ce477ec3d8,
source URL https://github.com/bcollazo/catanatron.git, official_spiral, seed offset1;
Torch2.15.0, one thread. No legacy runtime was imported.

- Output/evidence: `/private/tmp/scaling-development-smoke-kqk9ruie/smoke.json`.
- Natural complete four-game component boards119900000..119900003, seats0/1/2/3,
  terminal steps333/514/331/482, no truncations.
- Raw395 own observations,221 non-forced choices (all forced history retained).
  All six expected current-engine supervised phase names appeared in raw coverage.
- Collection2.347s; save/load plus exact teacher replay3.179s for the load/replay
  portion. Raw artifact SHA
  `bec5799572a9d34bddc36a79f0527b22cde51039f65e232f04fe2ef1b7a7272d`.
- Fit diagnostic used first21 own observations from each of two natural terminal
  histories,42 observations including20 forced; these prefixes are explicitly
  diagnostic, not eligible complete training components or primary arms.
- Two-epoch full versus boundary/resume fitting:8 updates,44 supervised
  presentations; bitwise tensors/Adam/counters,1e-12 reported-loss numerator and
  exact continuation Torch RNG. Raw artifact bytes unchanged.
- Smoke body elapsed8.2018s after imports, with a58-second alarm and earlier
  collection/replay/fit deadlines. Imports were not separately timed; the run
  completed well within the60s development allowance, with no retry/second smoke.
- Peak process RSS373571584 bytes (about356MiB). This small component does not
  establish100k fitting cost or permission for two fit processes.

## Self-review, guarantees and remaining limits

Self-review traced each public entry point, protocol binding, data release,
component order, atomic commit point and cursor failure/resume path against the
brief. It caught the pending-score recovery bug and ignored observation-envelope
extension, both RED/GREEN tested before the combined run. A final test-only review
isolated the removed-forced-history negative control from optimizer reset.

The implementation retains primitive component summaries rather than observations
between loads. The raw validation representation is released before official
loading; replay holds one simulator and the current dataset, not a second dataset.
Tests verify object lifetimes and actual resumed optimizer state, not flaky RSS
deltas or metadata-only success.

Cooperative RSS/deadline checks occur at component and persistence boundaries;
replay checks between actors. They cannot preempt JSON decoding, a Torch operation,
a scoring call or checkpoint allocation. Root must retain a hard external monitor.
The process-lifetime peak RSS check refuses an already over-budget process even
if memory was later freed. One component may itself exceed the cap; the first10k
gate and actual host measurements govern any sustained launch/second process.
Startup sequential validation is charged against the same absolute deadline but
is additional wall time beyond the returned arm's fit timing totals.

Exceptions without a fitter return expose Adam updates but not exact attempted
phase/presentation counts. Those unknowns are not fabricated or committed; prior
failed immutable results and partial-last snapshots retain the evidence. A failed
snapshot save can leave only the prior durable boundary and recorded error.
Publication crashes can leave orphan checkpoints, never an advanced unsaved cursor.
The driver does not garbage-collect these evidence files.

Root must supply/review actual migrated starts and tensor ancestry, all historical
reservation exports, final required phase list/ranges/deadlines and launch caps.
An omitted historical reservation cannot be discovered automatically from an
export. Existing active/frozen sources, old artifacts, roadmap and legacy DAgger
are untouched. Equal epochs remain a data-plus-training confound;10k uses100k
training phase weights. No cross-engine score-equivalence or playing-competence
claim is supported by the smoke. No blocker requiring an existing-interface edit
was found. Independent review has not been dispatched by this implementer.

## Scoped review fixes — startup persistence and honest attempt accounting

Review input: `.superpowers/sdd/2026-11-05-scaling/task-1-review.md`, two Important
findings against HEAD `3d499b70c94cb93b4d947b73c3e7a0f2200e5bee`. Verified both
against the original ordering: fitting and evaluation performed the full replay
before output publication, and the cost was missing from reported timing totals.
The earlier report's candid statement that startup was outside arm timing totals
is superseded by this fix. Used receiving-code-review, TDD and verification skills.
No new smoke, full-suite rerun, sustained collection or real panel was launched.
The user's expensive100k HOLD remains in effect; strict GreedyNoTrade benchmark and
stronger trading diagnostics belong to root's other work, not this driver fix.

Changes remain in the four owned files. The two public fit/evaluation entry points
now do cheap strict preflight, publish the exact protocol and immutable running
`attempt-start.json`, then begin the guarded corpus replay/checkpoint startup.
Well-formed already-expired invocations are recorded. Startup expiry, later
component validation failure and interruption publish an immutable failed-startup
`attempt.json` before re-raising the original exception. They do not create a
training cursor, partial checkpoint or completed-work result. Cheap malformed
schema/path/hash inputs can still be rejected before creating output. A hard kill
or filesystem failure can only preserve the already published evidence; final
failure publication cannot be guaranteed in those cases.

Normal and partial fit/evaluation results now contain an `attempt` path/SHA
reference to the final per-invocation record. That version-one record binds the
persisted protocol, stage, status/startup status/error and three wall-time fields:

- `startup_validation_seconds`: guard, full training/validation replay and initial
  checkpoint validation/restore/setup, measured on success and failure.
- `fit_seconds`: raw fitter calls made by this invocation, including uncommitted
  work and calls that raise; zero for evaluation.
- `total_elapsed_seconds`: starts before cheap preflight and includes attempted
  startup, fitting, loading/scoring and checkpoint/evaluation work. It is sampled
  before final accounting/result metadata writes; those final self-publication
  writes and caller-state restoration are explicitly the measurement cutoff.

The new fields are outside durable optimizer/counter state. Existing cursor
positions, accumulated committed counters, committed timings, boundary/checkpoint
publication order, partial-last snapshots and original absolute deadlines are
unchanged. Every resumed invocation records its repeated scan and subsequent
attempted fit work separately; it does not copy prior attempt elapsed time into a
new record or remove the previous immutable record. Distinct attempt totals can
be summed; committed timing/counter state must not be added to those totals again.
Evaluation verifies the new result attempt reference alongside its other inputs.
Documentation now specifies all attempt fields, failure behavior and this cutoff.

Meaningful RED/GREEN evidence, using the same exact upgraded interpreter and final
thread/bytecode environment shown earlier:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_scaling_study.py -k 'startup_'
```

- RED:10 failed,33 deselected in8.69s. Both fit/evaluation component-load callbacks
  observed no protocol; expired invocations had no attempt record; success/resume
  results had no accounting reference. These were the intended missing behaviors.
- A text-edit script asserted before writing production changes; the subsequently
  issued unchanged RED command ran again:10 failed,33 deselected in8.09s. No source
  mutation occurred on that failed edit. The indentation match was corrected.
- GREEN:10 passed,33 deselected in8.34s after implementing the startup boundary and
  per-attempt measurements.
- Covering module command without `-k`:43 passed in25.03s. The final rerun after
  strengthening the preflight-inclusive total assertion is recorded below.

The six failure cases interrupt after one real fixture component has been loaded
and replay-validated, before the second component completes: TimeoutError,
ValueError and KeyboardInterrupt for each entry point. They prove the protocol is
already present at both loads, preserve7.0s of injected startup work and0.0s fitter
work, retain the original exception, and leave no checkpoint/cursor/result claim.
Two additional tests use an already expired original global deadline and prove
no scan starts or deadline renewal occurs while recording the attempt.

A deterministic clock drives a two-component real-fitter/resume case. One second
of cheap preflight, five seconds of startup and two seconds per fitter call are
accounted separately. The first invocation commits only component0 and then fails
after updates in component1: completed_epochs0, next_component1,4 committed
updates,4.0s attempted raw fitting,5.0s startup and10.0s total. Explicit resume
performs its own5.0s scan and18.0s of remaining fitting,24.0s total including its
own preflight, reaching40 updates/80 presentations. An uninterrupted control has
5.0s startup,20.0s fitting and26.0s total. Resumed model/Adam trees match bitwise;
prior attempt evidence and all original limits remain unchanged. This also covers
the review's Minor request for a nonzero component cursor without any extra
production behavior change. A successful evaluation-startup test records6.0s even
when no common milestone permits gameplay; evaluate_blocks is forbidden there.
The earlier input-byte failure test now expects persisted failed-startup evidence,
while retaining its original ValueError behavior and zero fitter work.

Self-review inspected the whitespace-insensitive production diff to confirm the
large textual move only extracts the existing fit/evaluation bodies behind the
new invocation boundary. Optimization calls, boundary/cursor metadata, scoring,
partial work, completed counters and evaluation selection/journal behavior are
otherwise unchanged. The original covering parity and failure tests remain in
the focused module. No concrete new concern justified another full-suite run;
the prior972 passed/1 skipped combined evidence remains applicable to unchanged
consumers. No outside production file or runtime was modified.

Final scoped verification after all fixes and assertions:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_scaling_study.py
```

Result:43 passed in26.63s. This includes the preflight-inclusive exact10/24/26s
accounting assertions and the bitwise nonzero-component resume control.

```
/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/ruff check src tests
/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/ruff format --check src/colonist_rl/scaling_study.py tests/test_scaling_study.py
git diff --check
git diff --cached --check
```

Results: all Ruff checks passed; two owned Python files already formatted; both
diff checks passed. Full suite and smoke intentionally not repeated. Ready for
root's scoped rereview of the two Important findings and covering tests.
