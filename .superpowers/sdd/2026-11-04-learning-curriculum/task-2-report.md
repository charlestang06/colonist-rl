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

## Fix round 1 — independent review findings I1/I2/I3

Fix base: `d5ec5f3742b8d7b8ec41a0023d719822fe98c5fa`. Read the complete
`task-2-review.md` and inspected all three code paths before editing. Applied the
receiving-code-review skill plus the existing TDD and verification workflow. No
subagents, external reviewers, network/install activity, sustained PPO, policy,
reward, checkpoint-format or original-repository changes occurred in this fix.

### I1: explicit strict curriculum resolution evidence

`_resolved` now requires every row to positively contain exact boolean
`terminated=True` and `truncated=False`, an exact integer winner in seats 0..3,
and a boolean win flag consistent with that winner and learner seat. Existing
`_paired_rows` validation still checks row pairing, boolean `won`, finite VP,
complete blocks and censored results. The stricter checks are confined to the
curriculum boundary: the older evaluator/paired-comparison validator remains
permissive for legacy minimal rows. A compatibility regression exercises that
old minimal-row API separately.

Added parameterized negative gate AND selector coverage across stage, fixed and
initial results for missing, null and incorrectly typed terminal/truncation/
winner fields, plus winner range and winner/win inconsistency. No positive
terminal resolution can be inferred from absent legacy fields.

### I2: one panel and initial reference per selection call

Selection processes candidates by declared chronological order (stable original
index for ties). The earliest fully eligible candidate establishes a canonical
mapping of board/seat to initial winner, win flag and VP. Every other eligible
candidate must match that exact mapping; candidate fixed/initial rows must also
pair exactly through the existing pairing checks. This establishes both one
exact panel and one common initial-reference outcome set. A different panel or
initial endpoint cannot influence the winner or weaken its regression criterion.

Regressions cover different boards, different initial VP, and different initial
winner/win outcomes on identical boards, with both input-list orderings. Reordered
outcome rows and irrelevant diagnostic latency metadata still represent the
same reference and remain accepted. This does not introduce a new selection API.

### I3: observation-to-call deadline boundaries

`_MatchDriver` now checks immediately after observation construction and again
at the proposal/opponent call boundary. `RolloutCollector` rechecks after
constructing the learner's bootstrap observation. If that work crosses expiry,
it increments the partial-interval drop count, seals collection and stops without
calling the critic. Its existing pre-expiry in-flight-bootstrap behavior remains:
a valid completed critic call may finish and supply the real next value.

The authoritative evaluator had the same observation-to-learner-call gap. Added
its matching observation and actor-call guards, including its legacy trade-reject
critic path. No match loop or permissive evidence-validation API was replaced.

Five new controlled-clock regression cases progress time ONLY inside actual
simulator observation construction, never by clock-query counts: baseline and
graph opponent act, heuristic proposal, collector bootstrap, evaluator learner
actor. They verify no post-expiry actor entry or engine mutation; the bootstrap
case additionally verifies no record, exactly one drop, sealed continuation and
no subsequent forward. Existing in-flight-bootstrap and actor-expiry cases
remain in the focused suite.

### Exact RED/GREEN evidence

All commands use the same isolated workspace, existing interpreter and env as
above. Full command strings and output summaries:

1. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py -q`
   RED log `/private/tmp/task2-fix1-evidence-red.log`:
   `34 failed, 20 passed in 2.91s`. The failures reproduce missing/typed resolution
   acceptance, inconsistent winner acceptance, and cross-panel/reference selection.
   Some explicit-null cases already rejected correctly; those remain regressions.
2. Same command after the I1/I2 production fix:
   GREEN log `/private/tmp/task2-fix1-evidence-green.log`:
   `54 passed in 3.18s`.
3. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py -k 'observation_expiry' -q`
   RED log `/private/tmp/task2-fix1-deadline-red.log`:
   `5 failed, 54 deselected in 1.01s`. Actor/proposal/critic entry was observed
   at time 10 after observation crossed deadline 5, exactly reproducing the fault.
4. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_curriculum.py tests/test_training.py tests/test_training_integration.py tests/test_evaluation.py -q`
   Final focused GREEN log `/private/tmp/task2-fix1-focused-green.log`:
   `112 passed in 5.07s`.
5. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q`
   Final full-suite output is recorded below after completion.
6. `/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/curriculum.py src/colonist_rl/training.py src/colonist_rl/evaluation.py tests/test_curriculum.py`
   Final output is recorded below. `git diff --check` also runs after the suite.

### Fix self-review and remaining limits

The owned changes are strict curriculum validation, canonical selection checks,
small deadline guards in training/evaluation, their tests and protocol/report
clarifications. Reviewed against every Important finding, the old API tests,
and the existing in-flight-call contract. No simulator, architecture, opponent
strategy, teacher, reward, policy or checkpoint behavior changes were made.
Selection cannot establish authenticity of supplied persisted evidence; it now
requires consistent explicit resolved outcomes and one shared reference/panel.
No competence or throughput claim follows from these mechanics tests. The new
BC wave remains operationally prioritized; this fix only prepares curriculum
controls for a later independently reviewed PPO run.

Fix round 1 final full GREEN `/private/tmp/task2-fix1-full-green.log`: `389 passed, 1 skipped in 15.92s`. Final Ruff: `All checks passed!`; final `git diff --check`: no output, exit 0.
