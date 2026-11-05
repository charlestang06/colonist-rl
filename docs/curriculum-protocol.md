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
validation boards are 62000000 through 62000003. Final seeds 71001000 through
71001007 remain reserved; the runner never evaluates them and selection rejects
evidence containing them. All partitions are validated before output creation.

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
