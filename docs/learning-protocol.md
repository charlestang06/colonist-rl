# Learning benchmark protocol

Use `colonist_rl.evaluation.evaluate_blocks` for teacher and learned-policy
comparisons. Supply a nonempty explicit list (or tuple) of unique Python integer
`board_seeds`; booleans, duplicate seeds, strings, sets, and empty lists are
rejected. Each board runs four independent games, with the candidate in seats
0, 1, 2, and 3 in that order. The same board seed is reused across those games.
The default candidate and opponent baseline are both `heuristic`. The other
candidate baseline is `random`; opponent baselines also support the existing
`mixed` driver behavior. No heuristic parameters are tuned by this protocol.

```python
from colonist_rl.evaluation import (
    evaluate_blocks, paired_block_summary, validate_seed_splits,
)

training_seeds, validation_seeds, final_test_seeds = [11, 12], [701, 702], [901, 902]
validate_seed_splits(training_seeds, validation_seeds, final_test_seeds)
teacher = evaluate_blocks(board_seeds=validation_seeds, max_game_steps=20000)
learned = evaluate_blocks(policy, board_seeds=validation_seeds, max_game_steps=20000)
comparison = paired_block_summary(teacher, learned)
```

These small lists illustrate the interface, not a sufficient benchmark sample.
Keep training, validation, and final-test lists disjoint. Use validation for
selection; reserve the final-test list until policy and experiment settings are
fixed. `validate_seed_splits` checks explicit, nonempty, internally unique integer
lists and every pair of partitions for overlap.

## Actors and reproducibility

A supplied `GraphPolicy` takes precedence over the candidate baseline name and
is identified as `candidate="graph"`. Graph actions default to
`deterministic=True` (argmax); `deterministic=False` samples seeded stochastic
actions. Heuristic and random actors retain their existing seeded semantics,
identified by `action_mode="seeded_heuristic"` or `"seeded_random"`. The graph
modes are `"deterministic"` and `"stochastic"`. The `deterministic` field records
the requested graph-action option; it does not alter heuristic or random logic.

For board seed `b` and candidate seat `s`:

- The simulator receives `reset(seed=b)`; its existing backend applies `b + 1`
  when constructing the Catanatron game.
- The graph/global action seed is `(4 * b + s) % 2**63`. Torch and Python receive
  that seed; NumPy receives it modulo `2**32`.
- Each heuristic/random candidate is newly constructed with seed `b + 31 + s`.
  Recurrent graph memory starts at `None` in every game.
- The unmodified `_MatchDriver` receives `seed=b`; each opponent is newly
  constructed with seed `b + opponent_seat`. A mixed baseline uses its existing
  seat-relative random-opponent selection.

Every actor and teacher receives only an immutable `Observation`. Actual victory
points are read from simulator diagnostics only after a game terminates or
truncates. Python, NumPy, and CPU Torch RNG states, plus every supplied policy
module's original training/evaluation mode, are restored in `finally`, including
when an actor raises. Runs with identical policy weights, seeds, and settings
have reproducible non-timing outcomes in the same runtime/backend. Use CPU and
one Torch thread for the local baseline. Latency is measured with
`time.perf_counter()` and is intentionally excluded from exact comparisons.

## Result fields and incomplete blocks

The result contains only JSON-safe primitives and these fields:

| Fields | Meaning |
| --- | --- |
| `status`, `incomplete_reason` | `"complete"` and `None`, or `"incomplete"` and `"deadline"`. Completeness describes execution of the requested schedule, not natural game termination. |
| `requested_board_seeds`, `completed_board_seeds` | Requested order and boards with four finished seat games. |
| `requested_games`, `games`, `finished_games` | Scheduled games, games entering summary statistics, and all finished games including those in a partial block. |
| `aggregation_scope` | `"complete_blocks"` for the strict protocol. |
| `candidate`, `action_mode`, `deterministic`, `baseline`, `reject_trades`, `max_game_steps` | Actor identity and execution settings; strict calls do not use the legacy rejection ablation. |
| `outcomes`, `partial_outcomes`, `interrupted_game` | Complete-block game rows; finished game rows from an unfinished block; and in-progress accounting or `None`. |
| `wins`, `win_rate`, `mean_victory_points`, `truncations`, `terminations`, `trade_responses`, `offers` | Summaries over `outcomes` only. |
| `learner_decisions`, `non_forced_decisions`, `phase_counts`, `action_kind_counts` | Candidate decision totals and counts, summed over `outcomes` only. A non-forced decision has more than one legal action. Phase keys are observation context kinds; action keys are chosen action kinds. |
| `decision_latency_ms_mean`, `decision_latency_ms_p95` | Mean and 95th percentile over individual candidate decisions in `outcomes`, in milliseconds. No opponent latency is included. |
| `win_rate_ci95_nominal_wilson`, `confidence_interval_note` | Legacy descriptive Wilson interval and dependence warning. This interval does not model within-board dependence and must not support a significance claim. |

Each outcome row includes `game` (zero-based schedule index), `board_seed`,
`seat`, `won`, `winner`, `victory_points`, `terminated`, `truncated`, `steps`,
`trade_responses`, `offers`, and the same decision counts, count dictionaries,
and latency summaries. `steps` is the simulator version, including offers.
`won` is true only when `winner` equals the candidate seat. Natural completion
sets `terminated`; the simulator step limit instead sets `truncated`. A
step-limited game is a finished protocol game and can belong to a complete
block. It is not evidence of playing competence.

`deadline` is an optional finite absolute `time.monotonic()` timestamp. The loop
checks it before each game and before every decision step, including forced
phases. A running step or policy call is not preempted. If it expires, execution
stops and `status` explicitly becomes `"incomplete"`. Entirely finished blocks
remain in `outcomes`; any finished games from the interrupted board move to
`partial_outcomes`. An interrupted game's record contains only `board_seed`,
`seat`, `steps`, decision counts, count dictionaries, and latency, with no
fictional terminal outcome or private victory-point read. If expiration occurs
between games, `interrupted_game` is `None`. With no complete blocks, summary
counts/rates/means/latencies are zero and the nominal Wilson interval is `[0, 1]`;
these are empty-result placeholders, not performance estimates.

## Paired comparisons and compatibility

`paired_block_summary(reference, candidate)` requires the exact same unique
`(board_seed, seat)` keys in both results and all four seats for every board.
It rejects missing/duplicate pairs and explicitly incomplete results. Synthetic
results may omit `status`; they must still contain complete outcome rows with
integer keys, boolean wins, and finite victory points. Differences are
candidate minus reference. The function first averages win and VP differences
across a board's four seats, returning `boards` with `board_seed`,
`win_rate_difference`, and `mean_victory_points_difference`. It then averages
those board differences, returning the same two aggregate metrics with
`paired_boards`, `paired_games`, and a descriptive `comparison_note`. It makes
no significance assertion.

`training.evaluate` retains its six existing positional arguments and default
stochastic graph actions. Its new keyword-only arguments are `board_seeds`,
`candidate`, `deterministic`, and `deadline`. Explicit `board_seeds` select the
strict block behavior and determine the schedule. Without them, legacy `games`
may end with a partial four-seat block and all finished games enter its summary
(`aggregation_scope="finished_games"`). Both public evaluators use one
authoritative match loop, preserving CLI, checkpoint, and import behavior.
