# Behavioral Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose and improve heuristic cloning, independently test public consequence features and student-state labels, then evaluate a selected BC policy before existing fresh-PPO curriculum.

**Architecture:** Reuse one ordered sequence fitter and the accepted match driver/evaluator. Introduce an opt-in consequence-policy identity and separate provenance-preserving student-label components. A new controlled supervised runner hands selected checkpoints to the existing curriculum; it does not reimplement PPO.

**Tech Stack:** Existing Python 3.13, PyTorch, NumPy, Catanatron 3.2.1, pytest, Ruff and standard-library JSON/gzip/hash utilities; CPU only.

**Spec:** `docs/superpowers/specs/2026-11-05-behavioral-learning-design.md`.

## Global Constraints

- Preserve existing Observation/Action schemas, legal-action order, old datasets, old checkpoints, CLI defaults, simulator rules and reward behavior.
- Teacher labels use only immutable Observation; no hidden hands, deck order, opponent private VP, future outcomes or engine objects enter actor features.
- CPU, one worker/thread, existing dependencies only; no paid services/install.
- Freeze source for each real run and retain component/config/checkpoint digests, all failed/interrupted arms and actual compute. No silent budget extensions.
- Preserve complete ordered student decision histories, including forced choices; teacher labels and executed behavior are different fields.
- Recompute recurrent prefixes under current weights before each optimizer update. Never cache learned embeddings or hidden states across updates.
- Keep training, held-out label validation, game validation and reserved test partitions disjoint. Validation never supplies gradients or DAgger labels.
- No new PPO collector or curriculum. Consume the existing reviewed interfaces.

Work in a root-assigned writable clone; original repository and environment stay
read-only. Paths below are relative to that execution clone. Root/learning lead
dispatches and reviews; this planner is not an additional coordinator. No edits
to `training.py`, `evaluation.py` or `curriculum.py` until the active curriculum
worker finishes and its reviewed commit is integrated. Rebase this plan's small
checkpoint edit onto that accepted state. Source inspected: accepted `8b8974a`,
documentation HEAD `4bdb0ac`, plus explicitly unaccepted live curriculum interfaces.

Runtime commands use the existing interpreter:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_supervised_fit.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src tests
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src tests
```

Each coding task has its own RED/GREEN cycle, commit and independent spec/quality
review. A task may be rejected independently. No long run during implementation.

## Ownership and integration map

| Task | Owns | Depends on / avoids |
| --- | --- | --- |
| 1 | `supervised_fit.py`, `graph_inputs.py`, policy batching seam, imitation fitter/metric delegation, focused tests/docs | Diagnostic report first; no training/curriculum edits |
| 2 | `consequences.py`, `ConsequenceGraphPolicy` in policy, checkpoint architecture dispatch in training, focused tests/docs | Task 1 policy changes and active curriculum must be reviewed first |
| 3 | `student_labels.py`, corpus loader/projection, tests/docs | Task 1 sequence APIs; accepted `_MatchDriver`; no training edits |
| 4 | `behavioral_study.py`, runner tests/docs | Tasks 1–3; consumes reviewed curriculum configuration only |
| 5 | Reports/manifests/datasets/checkpoints and independent audit | All code reviewed and clean; no code changes during runs |

### Task 1: Audited, profile-driven supervised fitting

**Files:** Create `src/colonist_rl/supervised_fit.py`,
`tests/test_supervised_fit.py`, `docs/supervised-fit.md`; modify
`src/colonist_rl/imitation.py` only for shared fitter/metric delegation and optional
batch controls, plus focused `tests/test_imitation.py`. Own additions to
`src/colonist_rl/policy.py` for `forward_many`. Create
`src/colonist_rl/graph_inputs.py` and `tests/test_graph_inputs.py` for the measured
static input-rebuilding bottleneck; do not split unrelated policy code.

**Interfaces:**

```python
SupervisedSample = tuple[Observation, int]
EpisodeSequence = tuple[SupervisedSample, ...]

def fit_sequences(policy, episodes: tuple[EpisodeSequence, ...], *,
                  epochs=2, sequence_length=16, learning_rate=3e-4,
                  deadline=None, phase_weights=None, batch_size=1,
                  optimizer=None, profile=False) -> dict: ...

def score_sequences(policy, episodes: tuple[EpisodeSequence, ...]) -> dict: ...

# Added to GraphPolicy; same state dict and default forward behavior.
def forward_many(self, views: tuple[Observation, ...],
                 memories: tuple[torch.Tensor | None, ...]) -> tuple[PolicyOutput, ...]: ...

def recurrent_step(self, view: Observation, memory=None) -> torch.Tensor: ...
def recurrent_many(self, views: tuple[Observation, ...],
                   memories: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor, ...]: ...
```

`fit_imitation` remains the validated v1 wrapper with existing defaults/results;
optional `batch_size=1, profile=False` are appended keyword-only. `imitation_metrics`
remains its v1 wrapper with the same result keys. Sequence APIs validate legal
indices, monotone decision versions within each episode and finite positive
weights; no provenance is invented by this in-memory projection. `score_sequences`
returns the existing all/non_forced/by_phase/by_action_kind/coverage structure.
Caller-supplied optimizer must reference this policy's parameters; otherwise
construct current Adam. Existing fit return keys remain; optional profile adds
scope seconds, prefix/chunk forwards, presentations, updates and batch counts.

- [ ] Read `/private/tmp/colonist-swarm-bxz1k39_/behavior-diagnostics/report.md` and record its actual label/encoding,
  gradient and profile findings in the task report. Reproduce only a confirmed
  defect with a regression before fixing it. Do not treat teacher RNG jitter,
  forced-sample zero loss or BC's unused value-head gradient as bugs. The completed
  report found no label replay/canonical-input error or disconnected maritime/end
  gradient. It measured 336 prefix replays plus 154 supervised forwards, with
  68.1% forward time in prefix replay and 40.4% in nested graph construction.
  Keep exact-label metrics and separately report teacher-score/tie-aware diagnostics
  using the existing external audit; do not silently change supervision targets.
- [ ] Write failing compatibility and minibatch tests. Build small ordered
  sequences from existing `collect_demonstrations`, preserving every observation.
  For one frozen-weight minibatch, compare real serial CE/gradients to batched
  CE/gradients, including forced steps and distinct episode lengths:

```python
batched = policy.forward_many((view_a, view_b), (None, memory_b))
serial = (policy(view_a, None), policy(view_b, memory_b))
for got, expected in zip(batched, serial):
    torch.testing.assert_close(got.logits, expected.logits)
    torch.testing.assert_close(got.memory, expected.memory)
    torch.testing.assert_close(got.value, expected.value)
loss = sum(torch.nn.functional.cross_entropy(o.logits[None], t[None])
           for o, t in zip(batched, targets)) / len(targets)
loss.backward()
assert torch.isfinite(policy.gru.weight_ih.grad).all()
```

  Compare every participating parameter's gradient against the same serial sum,
  not merely its finiteness. Verify batch size one preserves original update
  ordering/counts; a state-sensitive policy observes changed weights during the
  next prefix replay. An expired minibatch changes no parameter/optimizer state.
- [ ] Run the new tests RED using the command above, then implement the shared
  fitter by moving the existing algorithm without changing its default weighting,
  clipping (`0.5`), optimizer or forced-step behavior. Batch chunks from distinct
  episodes; recompute prefixes immediately before each minibatch update. Models
  without `forward_many` use the serial sum under the same frozen parameters.
  Prefixes may call `recurrent_step`/`recurrent_many` to skip unused heads, with
  fallback to ordinary forward memory for custom policies. Test identical
  recurrent values and downstream gradients under both routes after an update.
- [ ] Implement the profile-supported optimization. Use CPU `record_function`
  scopes supported by installed Torch for encoding/prefix/chunk/backward/step.
  Static cached data must have no `grad_fn`/parameters; changing robber, owner,
  hand, board numbers or player order must update the relevant inputs. Cache keys
  include full board/schema/device/dtype, and capacity is bounded at 32 boards.
  `forward_many` constructs disconnected graph batches, per-graph pooled states,
  ordered GRU rows and separate legal-menu logits. No cross-episode graph edges.
- [ ] GREEN `tests/test_supervised_fit.py tests/test_imitation.py tests/test_policy.py`
  and cache tests if added; compare default fit/metrics and checkpoint state keys.
  Run one tiny matched serial/batched profile within 120 seconds, report actual
  work and throughput rather than impose a flaky speed assertion. Keep opt-in
  behavior unpromoted if slower. Ruff, commit, independent review.

### Task 2: Public consequence scorer and explicit checkpoint identity

**Files:** Create `src/colonist_rl/consequences.py`,
`tests/test_consequences.py`, `docs/consequence-policy.md`; modify
`src/colonist_rl/policy.py` and only checkpoint save/load dispatch in
`src/colonist_rl/training.py`; add focused checkpoint cases to
`tests/test_training.py`. Serialize ownership after the accepted curriculum task.

**Interfaces:**

```python
CONSEQUENCE_SCHEMA = "public-consequence-v1"
CONSEQUENCE_DIM = 34
def consequence_features(view: Observation, action: Action, *, device="cpu") -> torch.Tensor: ...

class ConsequenceGraphPolicy(GraphPolicy):
    architecture_kind = "graph_consequence_v1"
    # __init__(hidden_size=64, *, feature_set="hand_recipe")
    # Base initialization is unchanged. This branch starts with all-zero weights.
    # self.consequence = nn.Linear(CONSEQUENCE_DIM, 1, bias=False)

def consequence_from_legacy(policy: GraphPolicy, *, feature_set="hand_recipe") -> ConsequenceGraphPolicy: ...
def policy_architecture(policy: GraphPolicy) -> dict: ...
def policy_from_architecture(architecture: dict) -> GraphPolicy: ...
```

The spec defines every feature slot and recipe. Apply the additive scorer to
both scalar and batched paths after base logits, including trade responses;
never change value/memory output. Supported features derive post-hand deficits
for legal maritime/accept/reject/end actions only. Unsupported kinds are zeros.
Reject a supplied action absent from the legal tuple. Public connected sites
respect distance, own roads, piece limits and ordinary post-roll phase.

- [ ] RED tests for exact hand arithmetic/offer perspective, recipe-only flags,
  blocked/occupied/disconnected settlement sites, city piece limits, unknown bank,
  unsupported actions, and source Observation immutability. Construct a legal
  synthetic exchange from `(5,1,1,0,0)` giving four wood for one wheat; expected
  post-hand is `(1,1,1,1,0)`, settlement deficits are zero, but connected-affordable
  stays false until an own road reaches a distance-legal vacant vertex.
- [ ] RED zero-branch compatibility and gradient tests:

```python
legacy = GraphPolicy(hidden_size=32)
variant = consequence_from_legacy(legacy)
before, after = legacy(exchange_view), variant(exchange_view)
assert torch.equal(before.logits, after.logits)
assert torch.equal(before.memory, after.memory)
assert torch.equal(before.value, after.value)
torch.nn.functional.cross_entropy(after.logits[None], teacher_target[None]).backward()
assert variant.consequence.weight.grad.abs().sum().item() > 0
assert policy_architecture(legacy) == {"hidden_size": 32}
assert policy_architecture(variant) == {
    "kind": "graph_consequence_v1", "hidden_size": 32,
    "feature_schema": "public-consequence-v1",
  "feature_set": "hand_recipe",
}
```

  Compare named base parameters exactly, verify helper construction preserves
  caller RNG, and reject a second upgrade or unknown architecture/feature schema.
- [ ] Implement a single zeroed bias-free projection, not a detached feature
  path or doubly zeroed MLP. Extend checkpoint loader with exact two-shape
  dispatch and strict state loading. Existing save/load schema version remains 1;
  legacy architecture metadata/output and all existing files remain unchanged.
  `hand_recipe` masks feature slots 30–33; `hand_recipe_and_connectivity` enables
  them for a separately declared comparison. Test that enabling the four slots
  with zero new coefficients preserves logits and that each metadata identity
  survives save/load. Do not silently upgrade an existing variant in place.
- [ ] GREEN policy/consequence/checkpoint suites, plus curriculum load/frozen-copy
  regression on one zero-step opt-in checkpoint after its interface is accepted.
  Confirm installed-wheel loading still needs no Git. Ruff, commit, independent
  review; no new training study during this task.

### Task 3: Student-state labeling and honest component corpora

**Files:** Create `src/colonist_rl/student_labels.py`,
`tests/test_student_labels.py`, `docs/student-labels.md`. Consume Tasks 1–2;
do not alter legacy demonstration validation or active curriculum/driver code.

**Interfaces:**

```python
def collect_student_labels(policy, *, checkpoint_identity: dict, board_seeds,
                           max_decisions, max_game_steps=20000,
                           baseline="heuristic", seed=0, deterministic=False,
                           deadline=None, progress_path=None) -> dict: ...
def save_student_labels(path, component: dict) -> str: ...
def load_student_labels(path) -> dict: ...
def write_corpus(path, *, components: tuple[str, ...], partitions: dict) -> str: ...
def load_corpus(path) -> dict: ...
def corpus_sequences(corpus: dict) -> tuple[EpisodeSequence, ...]: ...
```

`partitions` has exact keys `training`, `validation_labels`, `validation_games`,
`audit`, `final_test`, each an explicit unique integer list. Pairwise overlap is
rejected, including reserved legacy final seeds. Corpus components are paths to
unchanged v1 teacher datasets or new student-label files; manifest entries record
path, SHA-256, format/version, original source and actual boards. Corpus derivation
source is separate from every collection source. Duplicate component digests,
altered bytes, nontraining boards or contradictory episode/sample flags fail.

New component samples have exactly `observation`, `teacher_action_index`,
`behavior_action_index`, `behavior_executed`; format is
`colonist-rl.student-labels`, version 1. Source, teacher configuration/seed rule,
student checkpoint/architecture/configuration, sampling seed, consumed boards,
coverage, episode steps/flags and deadline reasons are explicit primitive metadata.
Each episode records every learner decision in order; label-call ordinals include
forced steps. No tensors/rewards/hidden backend state are serialized.

- [ ] RED scripted-driver tests where student and teacher disagree. Confirm the
  actual executed action is the student's, both legal indices are retained, and
  all earlier/forced observations survive corpus projection. Test label replay
  with the original teacher RNG call order; a deliberately illegal teacher action
  raises before a game mutation. Test deadline after label/student return, before
  input application, keeping the final sample unexecuted and forbidding retries.
- [ ] RED provenance/corpus tests using two different real component source
  identities; neither is overwritten by derivation source:

```python
corpus = load_corpus(manifest_path)
assert [c["sha256"] for c in corpus["components"]] == original_digests
sequences = corpus_sequences(corpus)
assert sequences[student_episode][forced_step][0] == forced_observation
assert sequences[student_episode][choice_step][1] == saved_teacher_index
assert saved_behavior_index != saved_teacher_index
```

  Reject missing/changed component files, reordering inconsistent with digest
  identity, illegal indices, missing history, mixed actors, board leakage and fake
  terminal flags. Existing v1 save/load and imitation tests remain unchanged.
- [ ] Implement the supervised collector around accepted `_MatchDriver` and
  `FourPlayerGame`, with no duplicate rule/proposal implementation. Freeze the
  student through a round, restore external RNG/module modes, reset memory only
  at episode boundaries, and update memory on the actual student sequence. Query
  teacher on the same immutable view. Check deadline before actor calls and
  immediately before engine mutations. Persist completed episodes and final
  interrupted prefixes atomically; retain cap versus terminal distinctions.
- [ ] Implement versioned gzip roundtrip and component corpus projection. Fit
  through `fit_sequences`, never by forging a merged legacy dataset. Recompute
  memory under the fitting policy; stored behavior memory is not a model input.
  Validate a tiny actual two-episode smoke and all focused privacy/RNG/legacy
  tests. Ruff, commit, independent review.

### Task 4: Controlled cloning/feature/DAgger runner and curriculum handoff

**Files:** Create `src/colonist_rl/behavioral_study.py`,
`tests/test_behavioral_study.py`, `docs/behavioral-study.md`. No new CLI is needed;
use documented Python API. Do not change curriculum gates or rewards.

**Interfaces:**

```python
def run_behavioral_study(output_dir, *, initial_checkpoints: dict[int, str],
                         training_components: dict[int, tuple[str, ...]],
                         protocol: dict) -> dict: ...
```

`protocol` is an exact validated JSON object with `version=1`, `stage` in
`baseline_bc|consequence|student_labels`, explicit partition lists, source and
teacher identities, seeds, per-arm epochs/presentations/wall ceilings, sequence
length, batch size, learning rate, weighting recipe, feature_set, label-query budget, student
behavior mode and absolute deadline. Unknown fields/stages and mismatched seed
keys fail before compute. No run silently chooses a different stage or budget.
Results preserve requested/actual work, source/input checks, metrics, per-game
journals, checkpoints, comparison eligibility and failures. Save optimizer state
through shared checkpoint helpers when the caller owns the fitter optimizer.

- [ ] RED fail-closed protocol tests for absent arms/seeds, unequal starts, split
  leakage, final-panel consumption, source drift, partial fits, mismatched legal
  architecture and missing evaluation variants. Do not let a fake subset of the
  expected evidence matrix satisfy completion, as in the earlier Task 3 defect.
- [ ] Implement stages by consuming existing APIs. `baseline_bc` expands fitting
  of the unchanged architecture with the accepted recipe; `consequence` starts
  legacy and additive variants from identical base weights on identical data;
  declare `feature_set` explicitly and first isolate `hand_recipe`. A later
  connectivity comparison keeps base/recipe weights identical and zeroes only
  the four newly enabled coefficients. It is a separately budgeted stage.
  `student_labels` freezes one architecture and compares equal-query teacher-state
  versus student-state acquisition while retaining the same original components
  and fitting budgets. Both seeds get independent optimizer/RNG state. Do not
  combine feature and DAgger changes in the same causal comparison.
- [ ] Make every real stage require a clean fixed source and manifest before
  compute; retain all inputs by digest. Validate unweighted held-out label metrics
  without gradients. Evaluate 16 fixed boards × four seats per main arm/seed with
  existing authoritative `evaluate_blocks` and per-game persistence, plus the
  heuristic reference. Reject censored/mismatched comparisons with existing
  `paired_block_summary`. Keep partial artifacts rather than restart automatically.
- [ ] Produce a checkpoint handoff manifest for existing `run_curriculum`:

```python
config = CurriculumConfig(validation_board_seeds=tuple(protocol["validation_games"]),
                          deadline=declared_deadline,
                          compute_seconds=declared_compute_seconds,
                          wall_seconds=declared_wall_seconds)
# A later empirical task invokes the reviewed API; no duplicated PPO loop here.
run_curriculum(output_dir, initial_checkpoints=selected_paths, config=config)
```

  Record the chosen BC checkpoint/digest and frozen reference beside it. The old
  four-board curriculum experiment is historical; its scores do not become
  evidence on the new 16-board panel. Gate/selection behavior stays owned by the
  existing curriculum implementation.
- [ ] GREEN runner tests using bounded injected operations plus one minimal
  integration smoke; assert no default final-test execution, independent arm state,
  exact input preservation and installed package imports. Run combined suite,
  Ruff/format, wheel contents/import checks, commit and independent broad review.

### Task 5: Actual training, complete-game benchmarks and decision audit

**Files:** New report `docs/behavioral-learning-2026-11-05.md`, compact raw
manifests/results under `docs/experiments/behavioral-learning-2026-11-05/`, large
artifacts under ignored `runs/behavioral-learning-2026-11-05/`. No source edits.

- [ ] Read the diagnostic report and final code reviews. Root records fresh
  absolute stop/integration times and per-arm compute/label/evaluation budgets;
  this plan grants no extension of an earlier horizon. Run a ≤120-second CPU
  profile within that allocation; size equal work targets using measured cost.
  Check new partition candidates in the spec against every prior manifest.
- [ ] Consume the accepted uniform/balanced study as a control, not duplicate it.
  It completed 559.538/900 seconds, all four fits at 20 epochs/3,380 updates;
  balanced VP 4.125/4.500 versus uniform 3.625/3.500, all students 0/16, teacher
  3/16 and VP 6.5625. Choice agreement was 49/77,48/77 versus 50/77,50/77.
  These are descriptive control results, not evidence of competence.
  Push BC on training data with saved curves/checkpoints and full train/held-out
  choice-only metrics. Preserve rare setup, maritime, discard and feasible reject
  counts. Longer training is judged by learning curves and full-game play, not by
  assuming 100% fit is guaranteed or by silently resetting Adam at each checkpoint.
- [ ] Execute the separate consequence and one-round DAgger comparisons only
  with their own explicit equal starts/data-query/work budgets. Keep every weak,
  interrupted and regressed arm. A second DAgger round is a recorded conditional
  decision, not an automatic extension. Preserve original pilot files and every
  component's source identity.
- [ ] Evaluate all main variants against fixed heuristics on the common 64-game
  validation panel per seed. Report raw board/seat outcomes, wins/VP, censoring,
  teacher-relative differences and board-dependence uncertainty. The goal is to
  approach/beat the heuristic in actual games; make that claim only if both seed
  evidence and uncertainty support it. Otherwise report weak/inconclusive results.
- [ ] Run the reviewed random → one-heuristic/two-random → heuristic curriculum
  last, from a validation-selected BC checkpoint, with frozen BC reference and
  unchanged reward. Use the same new fixed panel. Keep training trajectories
  fresh; teacher datasets never enter PPO as rollout records. Record reached and
  unreached stages, entropy/KL/clip/value/gradient/throughput and regressions.
- [ ] Freeze checkpoint/settings selection before the retained final panel and
  do not tune afterward. If time cannot complete the panel, label final evidence
  incomplete. Repeat the deterministic stratified actual-choice audit with public
  observations, legal alternatives and opportunity denominators, including useful
  maritime conversions and feasible accept/reject decisions. Teacher disagreement
  is not an error label; near-uniform logits alone are not random play.
- [ ] Finish source/input digest checks, archive durable artifacts, commit compact
  reports and update root with actual budgets/results. A negative result is a
  valid experiment outcome. Reward redesign or larger models remain separately
  justified future work, not an automatic response to failure.

## Planner self-review and handoff

Four coding tasks plus one empirical task cover the approved mechanisms. Shared
interfaces above match each consumer; component formats and legacy checkpoint
identity are explicit. Task 2 waits for the active curriculum worker's ownership
to clear. Root/learning lead chooses dispatch order under existing authorization;
this planning task neither spawns agents nor implements or trains.

## Approved execution isolation

Task1 executes in /private/tmp/colonist-swarm-bxz1k39_/behavioral/repo, branch cytang/behavioral-learning, based on accepted4bdb0ac. Root explicitly authorizes this disjoint fitter/policy/imitation work while the original learning checkout fixes curriculum/training/evaluation. No curriculum import or Task2 checkpoint changes until that work is accepted and integrated. This is the user-authorized November5 behavioral-learning scope, separate from earlier missed schedules; actual computation requires a newly declared budget, never a reset of old runs.
