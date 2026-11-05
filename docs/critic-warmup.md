# Critic-head warmup

`colonist_rl.critic_warmup.warm_value_head` fits the independent value head
against actual complete-terminal, gamma-1 reward suffixes. It does not change
reward shaping or interpret the critic as a win probability.

```python
from colonist_rl.critic_warmup import warm_value_head

metrics, value_optimizer = warm_value_head(
    policy,
    episode_traces,  # committed Mapping[str, EpisodeTrace] from audited collection
    epochs=2,
    batch_size=64,
    learning_rate=3e-4,
    deadline=None,  # optional absolute time.monotonic() deadline
)
```

`epochs` must be a positive integer, `batch_size` an integer from 1 through 64,
and `learning_rate` a finite positive scalar. Booleans are not numeric arguments.
The caller owns checkpoint persistence. Every invocation creates a fresh Adam
owning exactly `policy.value.parameters()`, including no-data/expired calls.
There is no optimizer argument, resume, or reuse of BC/PPO moments. Subsequent
PPO must construct its own fresh full-policy optimizer.

Supported policies are ordinary, unmodified instances of `GraphPolicy`,
`ConsequenceGraphPolicy`, `ContextLinearConsequencePolicy`, and
`NonlinearConsequencePolicy`, with their independent scalar `nn.Linear` value
head. Subclasses, replacement actor implementations, custom hooks with side
effects, and concurrent mutation are outside the supported contract. Value
parameter/actor/buffer storage sharing is rejected before work begins. The
function can be called under `torch.no_grad()`, but rejects
`torch.inference_mode()` before mutation.

Every supplied trace is structurally validated before optimizer/model work,
including excluded entries. Mapping keys must equal stable trace identities;
crossed, gapped, duplicated or malformed observation/transition bindings fail.
False boolean `began_at_reset`/`history_complete` flags mark incomplete history;
they do not bypass the remaining structural validation. Supposedly complete
terminal traces are passed directly to `resolved_returns`; corrupt provenance
is an error, rather than an exclusion. Nonfinite suffix totals are errors.

Pending, truncated, dropped-interval, incomplete-history, nonterminal and
unknown-outcome traces are explicitly excluded, with episode keys, reasons and
counts. Exclusion reasons may overlap. Empty genuine terminal traces count as
eligible outcomes but provide no states. With zero eligible states the status
is `no_data`, all work counters are zero, and the returned Adam state is empty.
This takes precedence over an expired deadline when both apply.

For each eligible observation ordinal, the target is the actual sum of its
committed transition reward and every later transition reward, including all
intervening opponent rewards. `initial_reward` precedes the first own decision
and is excluded. Targets retain their signs and magnitudes; they are neither
clipped nor normalized, and winner labels never replace rewards.

Replay begins at `memory=None` for each episode and epoch, includes forced
observations, and runs under `no_grad` in evaluation mode. Only detached
recurrent outputs reach the value head. Batches follow mapping insertion order,
then observation order, span episode boundaries, and flush their final short
batch at each epoch end. Ordinary mean squared error weights states equally;
long episodes intentionally contribute more states. Features are retained only
for the current batch, with no persistent feature cache or trunk gradients.

Every non-value parameter and every buffer is compared bitwise against its
entry snapshot, including on exception/deadline exits. Because the supported
architectures' logits and recurrence do not consume the independent value head,
this certifies actor logits and carried memory without an additional replay
after expiry. Focused tests also compare actual logits, memory, deterministic
actions and identically seeded sampled actions. Invariance failure raises a
`RuntimeError` naming changed tensors; changed model tensors remain available
for diagnosis. The function does not claim to roll back model corruption.

Caller Python, NumPy and Torch RNG states (CPU, initialized CUDA, and MPS when
used), individual module modes, parameter `requires_grad` flags, and all entry
`.grad` objects and contents are restored on success, exception and deadline
exits. This includes prior value gradients; returned Adam moments reflect only
this invocation. Committed value updates survive a later deadline or exception.

Deadlines use the existing cooperative monotonic convention: check before every
recurrent step, backward pass and optimizer step. An operation already underway
is not preempted. No new optimizer step starts after observed expiry. Pending
batch samples and gradients are discarded; completed updates are retained. A
partial epoch is never reported complete. Snapshot validation/restoration and
bitwise certification still run on an expired call, without recurrent work.

The returned dict contains only JSON-compatible primitives:

| Metric | Meaning |
| --- | --- |
| `status` | `complete`, `deadline`, or `no_data` |
| `epochs_started`, `epochs_completed` | Entered epochs and fully committed epochs |
| `states_attempted`, `states_replayed` | Recurrent calls begun and returned successfully; includes discarded work |
| `samples_optimized`, `optimizer_steps` | Samples and batches with a completed Adam step |
| `squared_error_sum`, `squared_error_count` | Sum of squared errors and sample count for committed batches at their pre-update weights |
| `samples_discarded`, `batches_discarded` | Pending batch work abandoned on expiry |
| `target_min`, `target_max` | Range across eligible targets, or `None` with no data |
| `eligible_episodes`, `eligible_samples`, `empty_episodes` | Unique eligible outcomes/states, counted once regardless of epochs |
| `excluded_episodes`, `excluded_samples`, `excluded_observations` | Unique excluded traces, committed transitions and observations; a dropped observation has no committed sample |
| `excluded`, `exclusion_counts` | Per-key reason lists and per-reason episode counts, including original `incomplete_reason` |

The error numerator/denominator describe the training batches actually
committed, not a final-model validation loss. This module is an opt-in consumer;
no collector, curriculum, policy architecture, reward or checkpoint defaults
are changed.
