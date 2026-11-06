# Sparse supervision with approved alternatives

`colonist_rl.supervised_fit` supports an optional sparse target path alongside
its existing dense `(Observation, action_index)` interface. It learns from
explicitly approved alternatives while retaining every real observation needed
for recurrence. Default dense callers retain their behavior.

## Targets and ownership

`ApprovedTarget(accepted_indices, *, role, train_approved=False,
val_approved=False, legacy=False)` is immutable. Indices must be a sorted,
duplicate-free tuple of exact integers forming a nonempty proper subset of the
current legal menu. Whole-menu accepted sets may describe a reasonable case,
but cannot provide informative supervised loss.

New TRAIN targets require `role="TRAIN"`, `train_approved=True` and
`val_approved=False`. Validation targets require `role="VAL"`,
`val_approved=True` and `train_approved=False`; they are scoring-only. Flags are
strict booleans. Other roles are rejected.

The separate legacy path uses `role="TRAIN"`, `legacy=True` and both approval
flags false. Its accepted set must be the original tuple target's singleton.
This preserves fallible historical labels without claiming independent approval.

The caller owns dataset provenance, independent judgments, role assignment and
approval/hash verification. Constructing an `ApprovedTarget` does not certify
that a judgment occurred. Evaluation data cannot become approved training data.

## Fitting

`fit_sequences` accepts paired optional `approved_targets` and `anchor_schedule`
arguments. Supply both or neither. Targets are tuples parallel to every episode
and saved row, containing an `ApprovedTarget` or `None`. The schedule is a
nonempty ordered tuple of `(episode_index, row_index)` anchors. Repetition and
caller order are preserved; every anchor must reference a nonforced TRAIN target.
All sidecars and options validate before any forward, including inactive rows.

Sparse fitting requires `epochs=1`, `batch_size=1`, `sequence_length` from 1 to
16 and no phase weights. The caller freezes and supplies its own sampling
schedule. There is no implicit sampler or dense-loss fallback.

Each anchor restores its complete real prefix under current weights with no
buildable gradient graph, then differentiably replays at most the last 16
chronological observations ending at that anchor. Forced and unapproved rows
remain in the history. Only the scheduled anchor contributes loss, even if other
approved rows occur in its suffix. Recurrence through those rows can receive the
anchor's gradient. Memory resets at episode start and is recomputed after each
update; no future rows or stale learned memories enter the anchor.

`accepted_set_cross_entropy` computes negative log total accepted probability
using selected `log_softmax` values and `logsumexp`. This avoids cancellation
under large common logit offsets. A singleton uses ordinary `cross_entropy`
exactly. Each completed schedule slot contributes one loss presentation,
backward pass and clipped optimizer step. `supervised_decisions` counts distinct
scheduled anchors; `supervised_presentations` counts completed slots. Existing
optimizer ownership, deadline, mode, RNG and profile handling remain in use.

## Validation

`score_sequences(..., approved_targets=...)` replays complete histories under
`no_grad` and scores only explicitly approved VAL anchors. TRAIN and EVAL target
records reject before forwards. `None` rows provide history without labels.
Agreement means the chosen legal index belongs to the accepted set; loss is the
same accepted-set loss used in fitting.

The existing result schema is retained. Action-kind grouping uses the original
tuple target's kind as descriptive metadata; it does not assert that the approved
set has one unique kind. Empty approved denominators report count/correct zero
and accuracy/cross-entropy `None`. Scoring preserves caller modes, RNG and weights.

## Verification scope

The focused suite covers accepted-set loss/gradients, exact singleton behavior,
large-offset stability, sparse anchor/order/history rules, approval failures,
validation isolation and dense compatibility. The existing fitter suite covers
serial/minibatch updates, wrappers, parameter gradients and recurrence. Passing
these mechanics does not establish audited-data benefit or stronger gameplay.
