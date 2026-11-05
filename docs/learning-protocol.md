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
| `wins`, `resolved_games`, `unresolved_games` | Counts over `outcomes` only. Wins count resolved candidate wins; unresolved games are neither wins nor losses. |
| `win_rate`, `win_rate_denominator`, `win_rate_reporting` | Strict mode is `"strict_all_games_resolved"`: win rate and denominator are `null` if any selected game is unresolved or no games are selected; otherwise rate is wins divided by all selected games. |
| `mean_victory_points`, `victory_points_note`, `truncations`, `terminations`, `trade_responses`, `offers` | Summaries over `outcomes` only. VP is the descriptive observed endpoint, including truncated games, not final-game competence. |
| `learner_decisions`, `non_forced_decisions`, `phase_counts`, `action_kind_counts` | Candidate decision totals and counts, summed over `outcomes` only. A non-forced decision has more than one legal action. Phase keys are observation context kinds; action keys are chosen action kinds. |
| `decision_latency_ms_mean`, `decision_latency_ms_p95` | Mean and 95th percentile over individual candidate decisions in `outcomes`, in milliseconds. No opponent latency is included. |
| `win_rate_ci95_nominal_wilson`, `confidence_interval_note` | Descriptive Wilson interval and dependence warning. Strict results use `null` whenever win rate is undefined. An available interval does not model within-board dependence and must not support a significance claim. |

Each outcome row includes `game` (zero-based schedule index), `board_seed`,
`seat`, `won`, `winner`, `victory_points`, `terminated`, `truncated`, `steps`,
`trade_responses`, `offers`, and the same decision counts, count dictionaries,
and latency summaries. `steps` is the simulator version, including offers.
`won` is `true` for a resolved candidate win and `false` for a resolved loss.
It is `null` for unresolved outcomes, including truncations with no winner;
this uncertainty is retained even in legacy raw outcome rows. Natural completion
sets `terminated`; the simulator step limit instead sets `truncated`. A
step-limited game is a finished protocol game and can belong to a complete
block, but is neither an automatic loss nor evidence of playing competence.
Strict `win_rate`, its denominator, and its nominal interval are all `null`
when even one selected game is unresolved. The protocol does not estimate a
win rate from only the resolved subset. The resolved/unresolved counts describe
selected `outcomes`; partial-block games remain separately accounted for in
`partial_outcomes`.

`deadline` is an optional finite absolute `time.monotonic()` timestamp. The loop
checks it before each game and before every decision step, including forced
phases. A running step or policy call is not preempted. If it expires, execution
stops and `status` explicitly becomes `"incomplete"`. Entirely finished blocks
remain in `outcomes`; any finished games from the interrupted board move to
`partial_outcomes`. An interrupted game's record contains only `board_seed`,
`seat`, `steps`, decision counts, count dictionaries, and latency, with no
fictional terminal outcome or private victory-point read. If expiration occurs
between games, `interrupted_game` is `None`. With no complete blocks, summary
counts, observed VP mean, and latencies are zero placeholders, while strict win
rate, win denominator, and nominal Wilson interval are `null`. None of these
empty-result values is a performance estimate.

## Paired comparisons and compatibility

`paired_block_summary(reference, candidate)` requires the exact same unique
`(board_seed, seat)` keys in both results and all four seats for every board.
It rejects missing/duplicate pairs, explicitly incomplete results, and any
truncated or unresolved outcome in either input. This includes legacy-shaped
rows with `won=false` and `truncated=true`; a numeric legacy rate does not make
them eligible for paired performance comparisons. Explicit `terminated=false`,
`winner=null`, or `won=null` also indicates an unresolved row and is rejected.
Synthetic results may omit `status` and termination metadata; their complete
outcome rows must assert resolved boolean wins, integer keys, and finite victory
points. Differences are
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
(`aggregation_scope="finished_games"`). For compatibility, this mode retains
numeric `win_rate` (wins divided by all finished games) and its nominal Wilson
interval, explicitly labeled `win_rate_reporting="legacy_truncations_as_losses"`
with `win_rate_denominator=games`. At zero legacy games it retains `0.0` and
`[0, 1]`. These compatibility metrics treat unresolved games as losses in the
aggregate and must not be used as strict benchmark estimates; raw `won` remains
`null` for unresolved rows and resolved/unresolved counts remain explicit. Both
public evaluators use one authoritative match loop, preserving CLI, checkpoint,
and import signatures.

## Persisted observation demonstrations and imitation

Use `colonist_rl.imitation.collect_demonstrations` to collect teacher sequences,
`save_demonstrations` / `load_demonstrations` for gzip JSON, `fit_imitation` to
train, and `imitation_metrics` for validation. Collection accepts an ordered
iterable of unique integer `board_seeds`, including a lazy iterator. Board index
`i` uses learner seat `i % 4`; this collects one seat per board, unlike the
four-seat evaluation protocol. The simulator receives `board_seed` (and the
backend uses `board_seed + 1`), the observation-only `HeuristicPolicy` teacher
uses `board_seed + 100`, and `_MatchDriver` uses `board_seed`. Default opponents
are `mixed`; `heuristic` and `random` are also accepted.

```python
from colonist_rl.imitation import (
    collect_demonstrations, save_demonstrations, load_demonstrations,
    fit_imitation, imitation_metrics,
)

training_data = collect_demonstrations(
    board_seeds=training_seeds, max_decisions=256, max_game_steps=20000,
)
sha256 = save_demonstrations("training-demonstrations.json.gz", training_data)
training_data = load_demonstrations("training-demonstrations.json.gz")
fit = fit_imitation(policy, training_data, epochs=2, sequence_length=16,
                    learning_rate=3e-4)
# Collect validation_data separately from disjoint validation_seeds.
validation = imitation_metrics(policy, validation_data)
```

The version-1 dataset has the following exact top-level fields:

| Field | Contents |
| --- | --- |
| `format`, `version` | `"colonist-rl.demonstrations"`, `1`. |
| `backend_version`, `action_kinds` | The simulator backend version and complete ordered policy action vocabulary; incompatible values are rejected. |
| `teacher` | `name="HeuristicPolicy"`, `seed_rule="board_seed + 100"`, `input="immutable Observation"`, `labels="legal action index"`. |
| `source` | Actual collection checkout `revision` and boolean `dirty`, read from Git. Collection refuses a changed revision/dirty status at the end. Do not commit or edit the source during collection. A dirty flag does not preserve a source snapshot. |
| `collection` | `baseline`, `max_decisions`, `max_game_steps`, `deadline`, `actual_board_seeds` in consumed order, `stop_reason`, `started_unix_seconds`, `elapsed_seconds`. `stop_reason` is `seeds_exhausted`, `decision_cap`, or `deadline`. |
| `coverage` | `decisions`, `non_forced_decisions`, `phase_counts`, `action_kind_counts`; counts describe actual retained learner observations and chosen teacher kinds. |
| `episodes` | Ordered episode records, including any empty episode whose step cap precedes its learner's first decision. |

Each episode contains `board_seed`, `backend_seed`, `seat`, `teacher_seed`,
`complete`, `terminated`, `truncated`, `incomplete_reason`, `steps`, and ordered
`samples`. Each sample contains exactly `observation` and `action_index`; in
memory observations are immutable `Observation` values, while persisted values
use the existing versioned observation codec. There are no engine objects,
recurrent tensors, rewards, private diagnostics, or return labels. Validation
rejects extra sample fields, incompatible schemas, nonfinite JSON, inconsistent
coverage/provenance, and indices outside the observation's legal action tuple.
The saver writes atomically and returns SHA-256 of the exact compressed bytes;
gzip headers are deterministic for an unchanged dataset.

Collection keeps full ordered episodes until a cap intervenes. The global
learner-decision cap is exact. The per-game version cap includes opponent moves
and trade offers, and is checked after an offer before any additional action.
A naturally terminated episode alone has `complete=true`, `terminated=true`,
and no incomplete reason. Step-limited episodes have `truncated=true` and
`incomplete_reason="step_cap"`; decision/deadline cutoffs use their corresponding
reasons and remain incomplete. Such sequences may train imitation but cannot
supply terminal-return evidence. A dataset with exhausted seeds may contain
step-limited episodes; collection exhaustion does not assert terminal games.
The optional finite absolute monotonic `deadline` is checked between actor
calls, including opponent decisions; running calls are not preempted. Python,
NumPy, and CPU Torch RNG states are restored after collection.

`fit_imitation` trains only observations with more than one legal action.
Forced observations still advance recurrent memory inside a supervised chunk
and in every replayed prefix. Forced-only chunks produce no optimizer update.
For every supervised chunk, the entire earlier episode prefix is recomputed
under the **current parameters**, with no gradient, starting at `None`; that
memory is detached before ordered chunk backpropagation. Memory resets at each
episode boundary, and no hidden state is cached in the data. Prefix replay costs
increase with episode length. Adam uses the requested learning rate; gradient
norm is clipped to `0.5`. An expired deadline discards the unfinished chunk.
Original module modes are restored after training.

The fit result keeps `loss`, `decisions`, and `epochs` and adds
`requested_epochs`, `updates`, `supervised_decisions`,
`supervised_presentations`, `status`, and `incomplete_reason`. `epochs` counts
fully completed passes; `loss` averages CE over non-forced presentations across
successful updates and is `null` when there were none. The legacy
`training.imitation_warmstart(policy, decisions=256, seed=0, epochs=2,
learning_rate=3e-4)` lazily generates consecutive board seeds and uses the same
collector and trainer. For compatibility its empty-supervision loss is `0.0`;
its diagnostic `supervised_decisions=0` and `updates=0` identify that case.

`imitation_metrics` replays whole ordered sequences from fresh episode memory
in evaluation mode and without gradients. It restores every module's prior mode
and Python, NumPy, and CPU Torch RNG states, and never updates parameters. The
`all` and `non_forced` summaries each contain `count`, `correct`, `accuracy`, and
`cross_entropy`; `by_phase` and `by_action_kind` provide the same two summaries
per observed context kind / chosen teacher action kind. It also returns
`coverage` and a descriptive `note`. Empty groups have zero count/correct and
`null` accuracy/CE. In particular, an all-forced dataset has
`non_forced.count=0`, `non_forced.accuracy=null`, and no supervised competence
estimate. All outputs are finite or JSON `null`. Teacher agreement and tiny
non-forced overfit checks measure supervised optimization only; playing
competence requires the separate strict, uncensored match evaluation protocol.

## Deadline mutation boundary

Collection and evaluation pass their optional deadline to the shared
`_MatchDriver.maybe_offer` hook. It checks before proposal selection and again
before applying a returned offer. Both callers also check after a proposal or
action actor returns, before a subsequent simulator action. An actor already
running at expiration may finish; its returned choice does not authorize another
engine mutation. Evaluation retains the interrupted game's actual simulator
version and decision accounting. A learner choice returned during that final
call remains in its decision/latency counts even when no engine step follows.
No terminal outcome is invented for a deadline interruption. Calls without a
deadline retain the existing proposal behavior used by rollouts.

Offers increment simulator version. If an offer consumes the final allowed game
step, evaluation records that exact endpoint as a truncated, unresolved game and
performs no following action. Its raw `won` is `null`, and strict win-rate
summaries remain undefined under the existing unresolved-game rules.

## Bounded experiment runner

Run a single reproducible pilot from a **clean committed checkout**:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m colonist_rl experiment \
  --output runs/learning-pilot-2026-11-04 --decisions 2500 --epochs 2 \
  --hidden-size 32 --wall-seconds 900
```

The Python entry point is `experiments.run_experiment(output_dir, *,
training_seeds=(11,29), decisions=2500, epochs=2, hidden_size=32,
sequence_length=16, validation_board_seeds=(90000,90001),
validation_demo_seeds=(91000,91001), final_test_board_seeds=(70000017,70000019),
max_game_steps=20000, wall_seconds=900)`. A directory with an existing manifest
is refused; deadline or failure preserves its first attempt instead of restarting.
CLI output reports experiment execution status; inspect `status`, `stages`, and
`gate`, since a zero CLI exit is not a claim of competence.

Before simulation, the runner checks all partitions and writes `manifest.json`.
Each training seed `s` has exactly 64 board candidates `[1000*s+i for i in
range(64)]`. Candidate lists and **actual consumed** boards are kept separately.
Validation demonstration boards are diagnostic only and receive no gradients.
The final-test namespace is reserved and never consumed by this pilot. Probe
board 92000 is separate from every learning/evaluation partition. The manifest
records dependency versions, backend/action schema, model/fit configuration,
initialization and actor seed rules, CPU/one-thread/one-worker limits, source
revision and `git:<revision>` content identity. Dirty execution is refused.
Source identity is checked again before the final result is written. The
configuration SHA-256 excludes start/deadline timestamps; the results also
record the digest of the exact manifest bytes. Datasets and checkpoints have
exact byte digests and local paths. `results.json` is atomically refreshed at
stage boundaries, and raw evaluation results remain separate JSON files.

The probe has at most 60 seconds and one four-seat board block, inside the
single total deadline. Its raw outcomes are separate from validation. Its
observed finished-game time and teacher decisions per second determine equal
per-seed demonstration caps **before training**. The budget reserves up to 55%
for comparisons, assumes a twofold graph-game slowdown, and conservatively
budgets sixteen teacher-decision time units per demonstration (minimum 50 ms)
for collection, recurrent prefix replay, fitting and diagnostics. Forty percent
of the learning allocation is diagnostic/reserve time. These are scheduling
estimates, not measured training speed. If two validation blocks do not fit
the estimate, every variant requests the same first block and the result is
explicitly inconclusive. Reductions, requested/actual coverage and shortfalls
are retained. Each seed receives an equal share of remaining learning time;
collection, fitting and diagnostics also receive subdeadlines. Every collection,
evaluation and fit uses the absolute deadline supported by those interfaces.
Diagnostics install a temporary forward guard that stops before each policy
call; unknown metrics are left absent rather than fabricated.

Each training seed initializes fresh Python/NumPy/Torch RNG state and a fresh
model, and its untrained and imitation candidates copy identical initial
weights. `fit_imitation` creates its own optimizer per seed. All seeds are
reported separately. Comparisons cover heuristic and random candidates against
both random and heuristic opponents, plus untrained and imitated candidates for
each training seed against heuristic opponents. All graph actions use argmax.
Evaluation variants receive equal shares of the remaining wall time. Deadline
or exceptions yield partial/failed stage records and retain raw finished games,
truncations, partial blocks and interrupted-game counts. A running bounded actor
or filesystem write may finish just after the deadline. No unresolved `won=null`
or `win_rate=null` value is converted to zero. If any variant is incomplete,
unresolved or has different board/seat keys, paired teacher comparisons are
suppressed for the schedule. Descriptive endpoint VP remains available.

`experiments.baseline_gate(summary)` is pure. It requires at least two distinct
training seeds, eight complete resolved board blocks in every variant and each
student stage, no truncations, exact matching teacher/student board-seat keys,
and at least 100 held-out non-forced decisions with agreement >=0.60 per seed.
Observed phases with non-forced decisions must also have agreement >=0.60;
forced-only phase scores cannot supply that evidence. Each imitated seed must
have paired mean VP difference >=-1.0 and win-rate difference >=-0.10 versus the
heuristic teacher. Missing, malformed or nonfinite scores reject eligibility.
The tiny pilot cannot satisfy the eight-block sample gate and records
`ppo_status="deferred_insufficient_baseline_evidence"`. Finite imitation loss is
not permission to train PPO. Missing phase/action coverage remains a limitation.

After a larger, uncensored baseline passes every gate, a subsequent explicitly
bounded continuation can load each imitated checkpoint and collect **fresh**
1024-decision rollouts per seed with the existing `RolloutCollector`, then run
`ppo_update` for two epochs with a fresh optimizer. It must set a new wall
budget, check that deadline at rollout/decision and update boundaries, preserve
all seeds, and rerun held-out evaluation. The existing PPO APIs do not expose a
per-decision deadline, so that continuation needs a bounded collector interface
before it can enforce the same deadline contract. This wave does not launch
PPO, pretrain a critic, or add an imitation anchor.
