# Task 1 implementation report

Status: DONE_WITH_CONCERNS (opt-in batching has a small short-run benefit and
floating-point Adam trajectory sensitivity; default compatibility is exact).

Workspace: `/private/tmp/colonist-swarm-bxz1k39_/behavioral/repo`
Branch: `cytang/behavioral-learning`
Base: `6d7a65a3b8da00550610d814a70c685cb72e148b`

## Audit findings and scope

Read the assigned brief and the actual external report at
`/private/tmp/colonist-swarm-bxz1k39_/behavior-diagnostics/report.md` before code.
The report replayed **5,000/5,000 original labels**, checked **983/983 actual
canonical candidate encodings**, **154/154 own-hand graph slices**, and
**120/120 actual maritime port/resource legality checks**. All **28 real
maritime/end targets across two fitted checkpoints** had connected finite
nonzero gradient paths. Teacher RNG jitter is legitimate: alternate-seed replay
differed at 387 decisions, 342 on exact base-score ties, all within the jitter
range. There was no confirmed label/encoding/disconnected-gradient defect to
reproduce or fix, and no evidence proving a capacity limit.

The audit measured 336 current-weight prefix replays plus 154 supervised forwards.
Prefix replay was 68.1% of forward time; nested graph reconstruction was 40.4%.
These percentages are not additive. The implementation targets this measured
replay/reconstruction work, without changing labels, legal ordering, architecture,
rewards, consequences, DAgger, evaluation, checkpoints or curriculum.

Exact-index CE and agreement remain the primary score. The teacher-score/tie
figures above are separately reported audit diagnostics, never replacement
labels or altered student metrics.

## Files and interfaces

- Added `src/colonist_rl/supervised_fit.py`: `SupervisedSample`,
  `EpisodeSequence`, `fit_sequences`, `score_sequences`. The shared scorer
  retains all/non_forced/by_phase/by_action_kind/coverage/note results; scoring
  restores Python, NumPy and CPU Torch RNG and individual module modes.
- `fit_sequences` moves the original ordered truncated-BPTT algorithm with the
  same Adam defaults, weighted non-forced CE denominator, forced-memory
  propagation, clipping at 0.5, default episode/chunk ordering and return keys.
  Supplied optimizers must contain every parameter of this policy exactly once;
  foreign, missing, duplicate and non-optimizer inputs are rejected. Exact
  coverage avoids stale excluded-parameter gradients affecting global clipping.
- Optional `batch_size > 1` schedules at most one chunk per episode in each
  minibatch. Prefixes replay with current weights immediately before each
  update. Custom policies without batching/recurrent APIs retain ordinary
  forward memories and a serial frozen-weight sum, including per-episode
  prefix-then-chunk call order. A partial expired minibatch changes no parameter
  or optimizer state.
- Added `src/colonist_rl/graph_inputs.py`: a 32-entry full
  board/schema/device/dtype/player-order LRU of parameter-free static inputs.
  Feature/link buffers are cloned before dynamic state is added. Robber,
  ownership links, public state, hand and beliefs are reconstructed; dynamic
  degrees are computed from complete current links, never cached. Entries built
  inside inference mode remain safe to use in later training.
- `policy.py`: original public `graph_features` remains imported there;
  `GraphPolicy.forward` retains its numerical behavior and state keys. Added
  disconnected-graph `forward_many`, `recurrent_step`, `recurrent_many` with
  per-graph pooling, independent ordered GRU rows and separate variable menus.
  Recurrent-only calls skip action/value heads and preserve downstream gradients.
- `imitation.py`: validated v1 fit/metric delegation only; appended keyword-only
  `batch_size=1, profile=False`. All provenance validation remains intact.
- Added `tests/test_supervised_fit.py`, `tests/test_graph_inputs.py`; added focused
  policy tests. `docs/supervised-fit.md` documents interfaces, semantics, cache,
  profiler counts and numerical/measurement limits.
- No changes to curriculum, training, evaluation, dispatch, CLI or state-dict
  schemas. No new dependencies, network, installations, paid services, gameplay
  studies or reserved boards. Original repository/environment and saved inputs
  were read only. No subagents or reviewers were spawned.

## TDD evidence

Working directory for all commands: `/private/tmp/colonist-swarm-bxz1k39_/behavioral/repo`.

Initial RED, before production code:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_supervised_fit.py tests/test_graph_inputs.py
```

Output: `11 failed in 1.64s`, expected assertion failures for the missing shared
fitter/cache modules. Complete output: `/private/tmp/behavior-task1-red.log`.

Initial GREEN after implementation:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_supervised_fit.py tests/test_imitation.py tests/test_policy.py tests/test_graph_inputs.py
```

Output: `44 passed in 4.45s` (`/private/tmp/behavior-task1-green-initial.log`).

A self-review case then reproduced a real new-path issue: custom-policy fallback
was replaying all independent prefixes before all chunks rather than keeping
ordinary per-episode prefix-then-chunk call order. A test with sequence length 1
and batch size 2 failed before the correction:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_supervised_fit.py::test_serial_fallback_preserves_episode_prefix_then_chunk_order tests/test_supervised_fit.py::test_real_fitter_minibatch_matches_independent_serial_sum tests/test_policy.py::test_forward_many_isolates_mixed_graph_sizes_memories_and_variable_legal_menus tests/test_policy.py::test_recurrent_only_routes_never_invoke_unused_actor_or_value_heads
```

Output: `1 failed, 3 passed in 1.86s`; call order mismatch at position 1
(`/private/tmp/behavior-task1-red-followup.log`). After fixing the fallback,
the full focused command passed `48 passed in 4.47s`
(`/private/tmp/behavior-task1-green-followup.log`).

Optimizer ownership tightening was separately RED:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_supervised_fit.py::test_caller_optimizer_rejects_missing_policy_parameters
```

Output: `1 failed in 1.66s`, `DID NOT RAISE ValueError`
(`/private/tmp/behavior-task1-red-optimizer.log`). After requiring exact optimizer
parameter coverage, the focused command passed `49 passed in 4.44s`
(`/private/tmp/behavior-task1-green-final.log`).

The tests compare real serial and batched CE and **every participating
parameter's gradient** under the same frozen weights, including forced memory
steps and unequal episode lengths. They also compare a real fitter minibatch
against independently accumulated serial CE with the same SGD step; test batch
size 1 exact order, counts, loss and current-weight prefixes; deadline parameter
and optimizer-state atomicity; recurrence/downstream-gradient parity after an
update; mixed standard/extended graphs, variable/empty/trade/robber menus;
head-skipping; provenance and sequence validation; static-cache keys, capacity,
dynamic isolation and inference-mode safety. Existing imitation tests continue
to cover RNG/mode restoration, forced-only chunks and legacy warmstart.

## Bounded real-data compatibility/profile

Script: `/private/tmp/behavior-task1-profile.py`
Output: `/private/tmp/behavior-task1-profile.log`
Structured results: `/private/tmp/behavior-task1-profile.json`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python /private/tmp/behavior-task1-profile.py
```

The first attempt stopped before any fitting with `KeyError: 'policy'` because
the saved checkpoint uses `policy_state`; the script then used that actual key.
The successful single comparison completed in **2.470538 seconds**, below the
120-second fit/profile budget. Torch 2.15.0, CPU, one intra/interop thread.

Inputs were the unchanged audited 64- and 90-observation complete episodes and
saved seed-11 initialization:

- Dataset SHA-256:
  `9cda913ed159759fcdbe2a47ab83fb48a27dd009d981d78ca15455bba814ec7f`
- Initial checkpoint SHA-256:
  `8a54d76a9d35ef02240836a866b5fba04996c013a9bad15f87a783ff6f5db579`

All 154 cached graph feature tensors, link order and offsets matched the actual
baseline implementation exactly. All sequential logits, memories and values
matched at zero tolerance; state-dict keys and complete exact-index metrics
matched. One epoch of optimized serial fitting matched the original default
fit result and **every final parameter tensor bitwise**. Both state digests
(using the script's key-and-value digest convention) were
`504eb218264dc283258d8a448b0ce520b5419e96d07107fc48bc6a110b958d6b`.

| Route | Seconds | Updates / presentations | Prefix / chunk rows | Chunks | Presentations/s | Mean fit loss |
|---|---:|---:|---:|---:|---:|---:|
| Original serial | 0.349926 | 10 / 101 | 336 / 154 | 10 | 288.63 | 1.832529143532 |
| Cached + memory-only serial | 0.219921 | 10 / 101 | 336 / 154 | 10 | 459.26 | 1.832529143532 |
| Serial frozen-weight sum, batch 2 | 0.218206 | 6 / 101 | 336 / 154 | 10 | 462.87 | 1.832981086013 |
| Disconnected graph batch 2 | 0.210873 | 6 / 101 | 336 / 154 | 10 | 478.96 | 1.832981102537 |

Optimized serial inclusive scope seconds: encoding `0.062951`, prefix `0.103520`,
chunk `0.074963`, backward `0.037044`, step `0.002762`. Graph encoding is nested
within prefix/chunk. Batch-2 disconnected scopes: encoding `0.062706`, prefix
`0.099085`, chunk `0.074992`, backward `0.033726`, step `0.001793`.

At frozen weights over all real sequences, serial and batched CE both equaled
`1.8345478773117065`; the maximum gradient difference across all **32**
participating parameter tensors was **3.725290298461914e-9**. None/participating
status also matched for every parameter. This compares serial SUM/mean at the
same frozen weights, not sequential Adam updates.

A single tiny real optimizer update under the installed CPU Torch profiler
confirmed ranges: `supervised_fit.prefix:1`, `.chunk:1`, `.encoding:2`,
`.backward:1`, `.step:1`. No large trace was written.

## Verification and self-review

Full suite, run once before commit:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/colonist-swarm-bxz1k39_/behavioral/repo/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q
```

Output: **`346 passed, 1 skipped in 14.90s`**
(`/private/tmp/behavior-task1-full.log`). The skip is the existing optional
browser test; the fresh starting baseline was 330 passed/1 skipped.

```sh
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/supervised_fit.py src/colonist_rl/graph_inputs.py src/colonist_rl/policy.py src/colonist_rl/imitation.py tests/test_supervised_fit.py tests/test_graph_inputs.py tests/test_policy.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src/colonist_rl/supervised_fit.py src/colonist_rl/graph_inputs.py src/colonist_rl/policy.py src/colonist_rl/imitation.py tests/test_supervised_fit.py tests/test_graph_inputs.py tests/test_policy.py
git diff --check
```

Output: `All checks passed!`, `7 files already formatted`, no whitespace errors.

Reviewed the diff against the assignment: original v1 collection/persistence and
provenance validation remain intact; unchanged model modules preserve checkpoint
keys; all cache entries are static and bounded; no learned memories/embeddings
cross optimizer updates; minibatch losses use one frozen parameter state;
partial minibatches never step. All edited/added source and tests are owned by
Task 1. Independent review is left to the parent coordinator as instructed.

Concerns/limits:

1. The matched batch-2 benefit is only about **3.5% in one short run**, insufficient
   to promote batching. The original/default optimization was about 1.59x here,
   but this is not a stable throughput guarantee or a capacity/generalization
   claim. Optional profile instrumentation is included in optimized timings.
2. After six batch-2 Adam updates, the largest final parameter difference between
   serial frozen-sum and disconnected batching was **0.0002062768**, while mean
   fit loss differed by **1.6524e-8**. Frozen-weight losses/gradients match within
   strict tolerances, but batched roundoff and near-zero gradients can lead to
   different Adam trajectories. Do not promise bitwise equivalence for opt-in
   batching. The default serial path was bitwise exact in the real-data check.
3. Caller optimizer groups must cover **every policy parameter exactly once**.
   Intentional subset optimization is not supported by this interface; this
   avoids stale omitted gradients participating in the original global clipping.
4. CPU is the measured/validated runtime. The cache key includes device and dtype,
   but no GPU throughput claim is made. No longer training study was run.

Profile script SHA-256: `b29c0df6512008b36194a0855c249eda7ca5c309230b88c43875efad3ee64490`.
