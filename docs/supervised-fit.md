# Observation-sequence supervised fitting

`colonist_rl.supervised_fit` accepts ordered immutable observations and their
original legal-action indices, without collection metadata:

```python
from colonist_rl.supervised_fit import fit_sequences, score_sequences

# episodes: tuple[tuple[tuple[Observation, int], ...], ...]
result = fit_sequences(policy, episodes, epochs=2, sequence_length=16)
scores = score_sequences(policy, episodes)
```

Each episode must have strictly increasing decision versions, schema-v1 own-player
decisions, and legal indices into its unchanged action menus. The fitter creates
no dataset provenance. Use `fit_imitation` and `imitation_metrics` for persisted v1
demonstrations: these wrappers still validate the complete collection provenance
before projecting the same observations and targets into the shared algorithm.
Their existing default options and result keys are unchanged. `batch_size=1` and
`profile=False` are appended keyword-only options to `fit_imitation`; CLI and
checkpoint formats are unchanged.

The default algorithm keeps the original episode-major chunk order, Adam learning
rate `3e-4`, mean non-forced CE, and gradient clipping at `0.5`. Optional finite,
positive `phase_weights` must cover every phase with a choice. Weighted CE sums
are divided by the number of non-forced choices, never by the sum of weights.
Forced observations still advance recurrent memory, including through their
within-chunk gradients, but contribute no CE or supervised-presentation count.
Chunks with no choices are skipped; later chunks replay those observations in
their prefix. Both fitting and scoring restore every module's prior train/eval
mode; scoring also restores Python, NumPy, and CPU Torch RNG state.

Every optimizer update first replays that chunk's entire prefix with current
parameters under `no_grad`. It never reuses learned embeddings or hidden states
from an earlier optimizer step. An expired partial minibatch never calls
`optimizer.step`, so it changes neither parameters nor optimizer state. The
shared API also accepts a caller-owned Torch `optimizer`; its parameter groups
must contain every parameter of this policy exactly once, with no foreign or
duplicate parameters. This ensures clipping never includes stale gradients from
parameters omitted by the optimizer.

## Optional independent sequence minibatches

`batch_size > 1` schedules round-robin chunks from distinct episodes, with at
most one chunk from any episode in an update. Variable episode lengths and legal
menus are retained. The update uses the mean weighted CE across all non-forced
choices in the minibatch. This changes the update grouping and count; it is a
separate opt-in training setting rather than a replacement for serial Adam
steps.

`GraphPolicy.forward_many(views, memories)` encodes disconnected graph components,
pools each component separately, and advances ordered independent GRU rows. Each
row retains its own legal-menu logits. `recurrent_step` and `recurrent_many`
compute the same memories while skipping unused action/value heads. The fitter
uses these routes for prefixes and forced observations. A custom policy without
these APIs uses ordinary forward memories and a serial frozen-weight CE sum.

Batched matrix operations and summation order can introduce floating-point
roundoff; frozen-weight CE and gradients match within Torch tolerances. Repeated
Adam steps can amplify differences in very small gradients. The default serial
path preserves its original numerical behavior and state-dict keys.

## Static graph-input cache

`graph_inputs` caches at most 32 immutable full-board/schema/device/dtype/player-
order combinations. Entries contain only constant feature/topology tensors with
no parameters, gradients, learned embeddings, or memories. Each call clones its
feature buffer, then rebuilds robber, ownership, hand, beliefs, public player
state and dynamic ownership links. Degrees are computed from each observation's
complete links. Board numbers, terrain, ports and topology participate in the
full board key. Building a cache entry in `inference_mode` is safe for subsequent
training.

## Profiling and measured limits

`profile=True` adds a `profile` field containing inclusive `scope_seconds` for
`encoding`, `prefix`, `chunk`, `backward`, and `step`; prefix/chunk forward counts
count observation rows, including forced rows. `presentations` counts non-forced
CE terms in completed updates. `batches` counts completed optimizer updates;
`batch_chunks` counts their independent chunks. `updates`, requested `batch_size`,
and `wall_seconds` are included. Encoding is nested within prefix/chunk scopes,
so these timings must not be added together. Installed Torch CPU profiler ranges
use the names `supervised_fit.encoding`, `.prefix`, `.chunk`, `.backward`, `.step`.

A one-thread CPU check on the previously audited 154-observation, two-episode
training subset used its saved initial checkpoint and original labels:

| Route | Updates / presentations | Prefix / chunk rows | Seconds | Presentations/s |
|---|---:|---:|---:|---:|
| Original serial implementation | 10 / 101 | 336 / 154 | 0.350 | 289 |
| Cached inputs + memory-only serial | 10 / 101 | 336 / 154 | 0.220 | 459 |
| Serial frozen-weight sum, batch size 2 | 6 / 101 | 336 / 154 | 0.218 | 463 |
| Disconnected graphs, batch size 2 | 6 / 101 | 336 / 154 | 0.211 | 479 |

The optimized default had identical inputs, logits, memories, values, metrics,
losses and final parameters to the original implementation in this check. At
frozen weights the batched CE was identical and the largest gradient difference
across all 32 participating parameter tensors was `3.73e-9`. After six batch-size-2
Adam updates the largest parameter difference from the serial frozen-weight sum
was `2.06e-4`, despite aggregate loss differing by `1.65e-8`. The two batch-size-2
routes have matching work and update grouping; comparing either directly with
ten sequential Adam updates does not establish training equivalence.

These are single short measurements, including optional timing instrumentation
on optimized routes, rather than a throughput guarantee. The matched batching
benefit was about 3.5% here, which is too small and noisy to promote batching as a
default. The full parity/profile check completed in 2.47 seconds, below its
120-second limit. No speed assertion is included in tests.

Exact-index agreement remains the scoring target. The external label audit
replayed all 5,000 original labels exactly, confirmed all 983 captured canonical
candidate encodings and 120 maritime legality checks, and found connected finite
nonzero maritime/end gradients at all 28 audited positions across two fitted
checkpoints. Its separate teacher-score/tie diagnostics found 387 alternate-seed
choice disagreements, 342 on exact base-score top ties, all within the teacher's
jitter range. These diagnostics describe genuine teacher stochasticity; they do
not relabel supervision, establish an optimizer defect, or prove a model capacity
limit. The measured original prefix and graph-construction forward-time shares
were 68.1% and 40.4%; graph construction is nested within forwards, so those
percentages are not additive.
