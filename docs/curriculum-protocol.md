# Bounded fresh-PPO curriculum

`colonist_rl.curriculum.run_curriculum(output_dir, initial_checkpoints={11: path,
29: path}, config=CurriculumConfig(...))` starts independent policies, new Adam
optimizers and Python/NumPy/CPU Torch RNG streams from supplied checkpoints.
Checkpoint metadata is preserved as input evidence; a legacy checkpoint with
unknown provenance is not relabelled. Execution requires a clean committed source
and an empty output directory. Input artifacts are read only. No teacher samples,
oracle labels, model changes or reward changes are used.

## Budgets and stages

The default **total PPO compute allowance is 1800 seconds**, split equally among
supplied seeds (900 each for 11 and 29). Each seed has at most 65536 applied learner
actions (including actions whose incomplete interval is dropped), rollout 128, PPO epochs 2, sequence length 16, learning rate 3e-4,
entropy 0.01, value coefficient 0.5 and gradient norm cap 0.5. The first real
rollout/update supplies a profile and counts inside compute. Constructor opponent
advancement also counts. Evaluation, journaling and checkpoint I/O are outside PPO
compute but inside the default 3600-second overall wall allowance. Set
`config.deadline` to an absolute `time.monotonic()` cutoff supplied by the caller;
the earlier wall/absolute deadline controls calls. An in-flight call may finish,
so elapsed compute can exceed an allowance by that call. No later allocation
starts after the seed's total compute allowance is spent.

Three equal compute/decision allocations start at these lineups, expressed in
relative seats learner+1, learner+2, learner+3:

| Stage | Opponents |
| --- | --- |
| `random` | random, random, random |
| `one_heuristic` | heuristic, random, random |
| `heuristic` | heuristic, heuristic, heuristic |

A failed gate leaves the next allocation at the same stage. Passing a gate at an
allocation end advances the pending stage, with the next collector starting a new
episode. `exposed_stages` records only stages with completed learner intervals;
`stage` is the current/pending stage and does not establish harder exposure.
Optional `historical_checkpoints` add a fourth allocation/stage with two
heuristics and one frozen graph opponent. This provides diversity; the frozen
opponent is not assumed stronger. Frozen policies are private copies in eval
mode, with disabled parameter gradients and separate per-seat recurrent memory.

These are equal **allowances**, not a claim of equal completed work. Overall
deadline interruption, board exhaustion and decision caps can leave unequal
realized decisions/updates. `decisions` counts applied learner actions;
`completed_intervals` counts usable rollout records. `status`, allocation rows, actual compute, completed
updates and dropped intervals retain those limitations.

## Seeds, gates and selection

For training seed `s`, the unique candidate board schedule is
`[8000000 + 100000*s + i for i in range(4096)]`. The actual boards consumed are
recorded separately. Finite schedules stop cleanly and never wrap. The fixed
validation boards default to 62000000 through 62000003. The current final panel
71001000 through 71001007 and legacy final seeds 70000017 and 70000019 remain
reserved. Training candidates and configurable validation panels must exclude
every reserved seed; promotion and selection reject evidence containing any of
them. For example, initialization seed 620 is rejected because its candidate
schedule contains both legacy finals. All partitions are validated before input
loading, output creation or collector/evaluator calls. The manifest retains the
current panel in `seed_splits.final_test_reserved` and records the full legacy and
current reservation in `seed_splits.all_final_test_reserved`.

The evaluator runs the same four-seat blocks against the exact current lineup
and against the fixed all-heuristic panel. Gates require at least two complete,
uncensored blocks; current-lineup win rate at least 0.25; current-lineup mean VP
at least 4.0; and fixed-panel mean VP no more than 1.0 below the initial supplied
checkpoint on matching boards/seats. Missing, interrupted, truncated, unresolved,
duplicate or mismatched evidence fails the gate. Each curriculum outcome must
explicitly contain `terminated=True`, `truncated=False`, an integer winner seat
0..3 and a boolean win flag consistent with that winner. The legacy paired-result
validator retains its older permissive compatibility. Metrics are recomputed from game
rows; optimistic summary fields cannot override censored rows. This is an
operational allocation gate, not a claim of competence or statistical significance.

The initial checkpoint and checkpoints every 4096 learner decisions or allocation
end are selection candidates. Selection uses only complete fixed all-heuristic
validation, rejecting more than 1VP regression versus the initial checkpoint. It
maximizes mean VP, then win rate, then the earliest checkpoint. The earliest
eligible checkpoint by declared order establishes the common board/seat panel
and initial-reference winner/win/VP outcomes for the selection call; candidates
with a different panel or initial outcomes are rejected. Row order and diagnostic
latency metadata do not alter that reference. The last checkpoint
is retained independently even when selection is unavailable. Validation is
outside the evolving opponent pool. Every completed evaluation game is atomically
journalled via `evaluate_blocks(on_game_end=...)`; completed raw rows survive a
later interruption, while strict summaries still use complete blocks only.

## Collector/updater compatibility and deadlines

Existing `RolloutCollector(policy, seed, max_game_steps, historical_policies)` and
`collect(count)` calls retain their defaults and list return. The old mixed
lineup remains **two heuristics and one random**. New keyword-only
`opponent_lineup`, `board_seeds` and constructor `deadline` make curriculum
configuration explicit. Setting `opponent_lineup` or `historical_policies` affects
new episodes only. `actual_board_seeds`, `stop_reason` and
`dropped_partial_intervals` expose collection accounting.

`collect(count, deadline=...)` checks before/after actors and proposals and before
later engine mutations. Observation construction is followed by another deadline
check immediately before proposals, actors and bootstrap calls. Expiry while
building a bootstrap observation drops and seals the incomplete interval without
launching the critic. Expiry before collection starts does not mutate the game.
An interval interrupted after the learner acts is dropped, counted and seals that
collector; it cannot later resume an interval with unknown reward. It launches no
critic/bootstrap call after expiry. Completed records retain their genuine next
value and terminal/truncation/episode boundaries. A critic call already in flight
may finish and supply the valid bootstrap. The next allocation can start a fresh
episode with the next unused board.

`ppo_update(..., deadline=None)` retains existing numeric loss metrics and adds
numeric `completed_updates`, `requested_updates` and `deadline_reached`. A completed
update means one actual optimizer step. Guards before replay forwards and before
optimizer steps prevent a deadline-expired chunk from changing parameters; policy
mode is restored. Existing checkpoint, resume and CLI interfaces remain unchanged.
`evaluate_blocks` adds optional lineup/frozen-pool passthrough to the same match
driver and returns the explicit lineup in its evidence. There is no second match
loop.

## Opt-in audited recurrent history and resolved outcomes

The default collector/updater route remains the legacy **stored-state
approximation**: PPO starts each gradient chunk from its rollout-time stored
memory, and a collector carries memory between calls. Those memories can become
stale when recurrent weights change. Existing curriculum wiring still uses this
route. A controlled driver must explicitly select both corrected interfaces:

```python
collector = RolloutCollector(policy, audited_history=True, run_id="arm-A-seed-11")
records = collector.collect(count, deadline=deadline)
metrics = ppo_update(
    policy, optimizer, records, config, deadline=deadline,
    episode_traces=collector.episode_traces, require_current_history=True,
)
```

Both additions are keyword-only. `audited_history` and
`require_current_history` must be booleans. An explicit `run_id` requires audited
mode and is a printable, nonempty string of at most 256 characters without
leading/trailing whitespace. It is a **caller-owned unique collector namespace**:
choose a distinct reproducible ID for every arm/seed/collector; reusing one for
independent collectors is invalid caller use. Omission generates a local UUID
without consuming Python, NumPy or Torch actor RNG. Episode keys length-prefix
the namespace and append the collector-local episode ordinal. They remain stable
through count cutoffs, collection calls and snapshot versions; independent default
collectors have distinct keys even when each local episode is zero.

At every audited `collect` entry, the collector starts at `None` and replays all
past committed own actor observations with the current weights under `no_grad`.
It uses `recurrent_step` when available, otherwise `forward(...).memory`. A replay
interrupted by a deadline starts no learner action and can resume if no interval
was lost. The caller must keep weights fixed during each collect call. Forced
single-action observations remain in history. A current observation is added only
after its action commits; bootstrap probes and future/opponent observations are
never past actor history. An acted-on observation whose interval is subsequently
dropped is retained once in the sealed, censored trace.

Required-current PPO validates every supplied record and referenced trace before
any optimizer mutation: key, episode, contiguous own-decision ordinal, exact public
observation/action/behavior-statistic binding, complete reset prefix, and contiguous
engine-version intervals. Missing, crossed or gapped provenance raises
`ValueError`. Within each episode the supplied rollout records must be contiguous
and ordered; a later call's suffix may begin above ordinal zero when its trace
retains the whole prefix. An episode cannot reappear after another episode in the
same update. Stable episode keys separate both GAE recursion and gradient chunks.
Immediately before **every** optimizer chunk, including later chunks and epochs,
PPO replays exactly `trace.observations[:start_ordinal]` from `None` under current
weights with `no_grad`, then detaches and differentiates only the finite chunk.
Stored memory is ignored. Stored behavior log probabilities/values/rewards and
real nonterminal bootstraps retain their PPO roles. Gamma remains 1, default GAE
lambda remains .95, and chunk boundaries do not zero continuation targets. Merely
passing `episode_traces` without `require_current_history=True` retains the legacy
route.

`Transition` preserves its original eleven positional fields and adds four
optional fields at the end, each defaulting to `None`: `episode_key`,
`observation_index` (zero-based own-decision ordinal), `step_start`, `step_end`.
The version interval starts at the acted-on observation and ends after all
intervening opponents, through the next learner decision or actual episode end.

`collector.episode_traces` returns a read-only mapping of immutable `EpisodeTrace`
snapshots. Older mappings and traces never grow. Each trace contains:

| Fields | Contract |
| --- | --- |
| `episode_key`, `run_id`, `episode_index`, `board_seed`, `learner_seat` | Stable collector/episode provenance and actual board/seat. |
| `backend_version`, `reward_identity` | Pinned engine identity and unchanged `terminal-win+0.2-vp-potential-v1` reward. |
| `began_at_reset`, `history_complete` | Evidence that own recurrence starts at the true reset with no omitted actor observation. |
| `observations`, `transitions` | Ordered tuples spanning all collect calls. Transition copies share immutable observations and set `memory=None`, avoiding mutable stored tensors and per-record prefix copies. |
| `initial_reward`, `initial_step_start`, `initial_step_end` | Separate reward/version interval before the first learner decision, including resumable initial opponent advancement. |
| `final_version`, `terminated`, `truncated`, `winner`, `finalized` | Actual latest engine version/outcome; a pending cutoff has no inferred winner. |
| `incomplete_reason`, `dropped_intervals` | Pending `count_cutoff`/`deadline`, final `truncation`, or sealed `deadline_dropped_interval` and its missing interval count. A genuine resolved terminal trace has no incomplete reason. |

Resolved, zero-decision and censored traces are retained before reset. Pending
traces survive count cutoffs and resumable deadlines. An interrupted acted-on
interval seals the collector as before and records the dropped interval; known
terminal truth is retained independently of stop reason. Metadata and later
observations are audit/training data, never actor inputs. This is in-memory
retention for a later durable outcome-data consumer, with no crash-durability or
resume/serialization promise. A future driver's manifest must identify the behavior
checkpoint and opponent lineup alongside these traces.

`resolved_returns(trace)` returns a tuple of actual undiscounted reward suffixes,
one per committed own decision, only for a genuine complete terminal trace with
known winner, full reset history, contiguous interval coverage, no truncation or
dropped interval, and terminal bootstrap zero. It rejects pending count cutoffs,
unresolved deadlines and false/incomplete provenance while leaving raw traces
available. An empty genuine terminal episode remains an outcome record and returns
`()`. Initial pre-decision reward is retained separately and excluded from these
suffixes. Targets sum the recorded learner interval rewards, including off-turn
rewards; they are never manufactured from winner or VP. With the existing shaping,
a losing suffix can be negative. No critic warmup, anchor, planning, curriculum
schedule change or playing-strength claim is part of this foundation.
