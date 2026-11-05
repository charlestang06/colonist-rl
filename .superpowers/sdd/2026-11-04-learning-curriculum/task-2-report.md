# Task 2 implementation report

Implemented bounded configurable fresh-PPO curriculum in the isolated learning
workspace, based on `4bdb0ac294a1606d02f98a381466e336bf1e439b`. This is implementation
and tests only. No sustained training, final-test games, external services,
network calls, package installations or writes to the original repository occurred.

## Owned changes and interfaces

- New `src/colonist_rl/curriculum.py`: `CurriculumConfig`, pure conservative
  `promotion_gate`, pure `select_checkpoint`, `run_curriculum` orchestration.
- New `tests/test_curriculum.py`: real simulator deadline/lineup/schedule/frozen
  regressions; pure evidence-gate tests; tiny runner and controlled-clock
  orchestration tests.
- New `docs/curriculum-protocol.md`: budget semantics, seed schedule, gates,
  checkpoints, deadline contracts and compatibility.
- `src/colonist_rl/training.py`: `_MatchDriver` optional explicit lineups and
  guarded opponent actor; collector keyword-only lineups, unique finite boards,
  constructor/collect deadlines, actual-board/stop/drop accounting; PPO optional
  deadline and numeric completed/requested-update accounting. Existing CLI,
  checkpoint loader/save format, resume behavior, default `collect(count)` list
  API and default PPO coefficients/epochs are preserved. Curriculum supplies
  epochs 2 explicitly; the old PPOConfig default remains epochs 4.
- Minimal necessary `src/colonist_rl/evaluation.py` extension: keyword-only
  lineup/frozen-pool passthrough into the existing driver; evidence contains
  explicit lineup. This was reported to the coordinator as an interface gap:
  old random/mixed/heuristic baselines cannot express one heuristic/two random.
  The authoritative match loop and complete-block/censored aggregation remain.

No policy/model/reward, simulator, contracts, adapters, board, belief, opponent
strategy or teacher implementation was changed. Existing mixed still means two
heuristics/one random. All actors receive their existing immutable observations.
Frozen opponents are copied, eval/no-grad, and retain separate seat memories.
Pool and lineup setters affect new episodes, not an existing driver snapshot.

## Runner behavior

Default compute is 1800 seconds TOTAL, equally divided across supplied training
seeds (intended 11/29), <=65536 applied learner actions per seed, rollout 128,
epochs 2, sequence 16, lr3e-4, entropy .01, value .5, grad .5. Constructor
advancement and the first real rollout/update profile count inside compute.
Three equal allocation allowances use random -> one heuristic/two random -> all
heuristic, with optional fourth two-heuristic/one-frozen diversity allocation.
Gate failure reuses later allocations at the failed stage. No inherent difficulty
claim is made for historical policies. `exposed_stages` only records stages with
completed learner intervals, distinct from pending `stage`.

Training candidates are the exact 4096 unique board formula per seed; actual
consumption is recorded. Fixed boards default 62000000..62000003. Final reserved
71001000..71001007 are partition-validated, never evaluated, and rejected by
selection. Models, optimizers and RNG starts are independent; inputs and legacy
unknown metadata are preserved. Outer wall allowance defaults 3600 seconds and
accepts a caller absolute monotonic deadline. It bounds evaluations as well as
compute. Equal allowances do not imply equal realized completed work if interrupted.

Gates inspect complete resolved rows, not optimistic summary metrics. They need
>=2 uncensored matching four-seat blocks, exact current lineup win>=.25 and
VP>=4, and fixed-heuristic VP regression<=1 versus the initial checkpoint.
Selection considers initial and periodic/allocation-end snapshots, fixed-heuristic
VP then win then earliest; censored/regressed evidence cannot select. Last weights
are saved independently. `evaluate_blocks(on_game_end=...)` journals every
completed raw game durably before continuing; aggregate evidence stays strict.

Deadline checks cover initial advancement, actors/proposals, later engine actions,
critic start, replay forwards and optimizer steps. An in-flight call may finish.
Incomplete learner intervals are dropped/count/seal the collector; they cannot
resume with invented return/bootstrap. A completed in-flight bootstrap retains
its genuine record and marks deadline. Applied learner actions whose interval is
dropped count toward the decision cap; completed usable records are separately
`completed_intervals`. Numeric PPO metrics include actual optimizer steps and
requested steps. A seed's total spent allowance blocks later allocations even if
one call overruns an allocation.

## RED evidence

All commands run in `/private/tmp/colonist-swarm-bxz1k39_/learning`, interpreter
`/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python`, with
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`. No clock-query-count tests are used;
controlled clocks advance on actual forward/step/optimizer work.

1. Initial command:
   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py -q`
   Output `/private/tmp/task2-red.log`: `10 failed in 1.45s`. Missing curriculum
   asserted explicitly; collector/updater keyword controls were absent.
2. Same command after adding real frozen/update/runner tests:
   `/private/tmp/task2-red2.log`: `1 failed, 13 passed in 2.99s`.
   Slow actor exhausted total allowance, but three allocations started instead of
   one. Fixed by checking total spent allowance before later allocation.
3. Same command with duplicated-requested-board evidence:
   `/private/tmp/task2-red3.log`: `1 failed, 14 passed in 2.87s`.
   Duplicate requested board metadata incorrectly passed a set comparison;
   explicit seed-list validation now rejects it.
4. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py::test_inflight_bootstrap_is_retained_and_reports_deadline -q`
   `/private/tmp/task2-red4.log`: `1 failed in 0.64s`, valid record retained but
   stop_reason was None instead of deadline. Fixed final completed-record guard.
5. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py::test_runner_decision_cap_counts_applied_actions_even_when_interval_is_dropped -q`
   `/private/tmp/task2-red5.log`: `1 failed in 1.22s`, three actual learner actions
   could occur against cap1 because dropped intervals were excluded from choice
   accounting. Fixed applied-action versus completed-interval accounting.

One intermediate fixture assertion used `isinstance(RandomPolicy)` without
accounting for HeuristicPolicy inheritance; corrected to exact random type.
An intermediate tiny runner failed because progress directory was not created;
the runner now creates it before journalling. Both were caught during focused
iteration, before completion claims.

## GREEN and static checks

Focused final command:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py tests/test_training.py tests/test_training_integration.py tests/test_evaluation.py -q`

Output `/private/tmp/task2-green6.log`: `70 passed in 5.05s`.
This includes constructor expiry without board reset, learner/opponent/proposal
expiry before engine mutation, partial interval drop/sealing, valid bootstrap
retention, finite schedule exhaustion, old default API/mixed behavior,
frozen parameter digest and separate memory under real PPO, no optimizer step
on expiry, actual finished optimizer-step accounting, strict one-heuristic
lineup evaluation, cap enforcement after dropped action, and external Torch RNG/
input-byte/unknown metadata preservation.

First broad command:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q`
Output `/private/tmp/task2-full.log`: `345 passed, 1 skipped in 18.05s`.
A final broad rerun was justified by the two subsequent boundary fixes, rather
than stopping with the earlier broad evidence. Final output is recorded below.

Static command:
`/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/curriculum.py src/colonist_rl/training.py src/colonist_rl/evaluation.py tests/test_curriculum.py`
Output: `All checks passed!`. `git diff --check`: no output, exit0.

## Self-review and limits

Reviewed owned diff against the brief and legacy constructor/collect/PPO/checkpoint
and strict evaluator interfaces. No alternate simulator loop exists. Checked
that explicit stage lineups preserve the legacy mixture and that a failed gate
cannot imply later-stage exposure. Complete records and dropped actions have
separate accounting. Final-test boards are untouched. No other agents/reviewers
were used, per instruction; independent review remains the next gate before any
sustained curriculum run.

Real tiny tests deliberately use short simulator caps, yielding censored evidence
and failed promotion; they establish mechanics, not learned competence. Pure gate
fixtures and isolated runner orchestration fixtures supply resolved evidence only
for control-flow testing. Optional historical diversity is implemented but was
not trained or benchmarked. In-flight calls can finish beyond deadlines, and
wall interruption can leave unequal realized training; results retain partial
statuses and actual counts. No new CLI command or curriculum resume protocol is
introduced. Existing CLI/checkpoint resume remains intact.

Final broad output `/private/tmp/task2-full-final.log`: `347 passed, 1 skipped in 17.13s`. Final Ruff and diff checks passed after that run.
