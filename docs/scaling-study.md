# Shared nested-corpus scaling

`colonist_rl.scaling_study` has three entry points, with no CLI or automatic
launch. The approved experiment is hand_recipe hidden32, nested 10k/100k distinct
non-forced learner observations, initializations 11/29, and five global epochs.
A process holds one four-game component, one policy and one persistent Adam.
Production 100k execution is on hold; this document does not authorize a launch.
The original teacher, simulator, observation contracts and fitter remain unchanged.
No whole-corpus adapter, learned embedding cache or hidden-state cache is used.

```python
collect_scaling_corpus(output_dir, *, protocol: dict) -> dict
fit_scaling_arm(output_dir, *, corpus_manifest: str, initial_checkpoint: str,
                endpoint: int, seed: int, protocol: dict,
                resume_cursor: str | None = None) -> dict
evaluate_scaling_study(output_dir, *, arm_results: dict[str, str],
                       corpus_manifest: str, protocol: dict) -> dict
```

All paths in artifact references are absolute canonical paths. A reference is
exactly `{"path": str, "sha256": lowercase_hex_64}`. Paths passed separately to
entry points must equal the corresponding protocol reference. Rehashing inputs,
checking imported-source Git identity and verifying the installed engine happen
before computation. Missing/unknown fields and wrong JSON types are rejected.
Integers exclude booleans. JSON must be finite. Output files are exclusive and
immutable; choose a fresh output directory for each invocation, including resume.
Every invocation persists its exact `protocol.json` before work.

## Protocol version one

The top dictionary contains exactly these fields:

| Field | Meaning and exact structure |
|---|---|
| `format`, `version` | `"colonist-rl.scaling-protocol"`, integer `1` |
| `source` | `{"revision": lowercase_hex_40, "dirty": false}`; matches the clean repository containing the imported module |
| `runtime` | `{"engine": verify_engine_installation(), "python": sys.executable, "torch": str(torch.__version__), "threads": 1}`; all values must match this process |
| `partitions` | Exactly `training`, `validation_labels`, `validation_games`, `audit`, `final_test`: nonempty unique integer lists, mutually disjoint; training/label counts are multiples of four, games has exactly sixteen boards |
| `prior_reservations` | `{"artifacts": [reference, ...], "excluded_boards": [int, ...]}`; each referenced reservation export contains exactly `{"reserved_boards": [int, ...]}`; every exported board must be excluded and no partition may overlap exclusions |
| `inputs` | Exactly `starts`, `corpus`, `results`, `resume`, described below |
| `collection` | Exactly `{"milestones": [10000,25000,50000,100000], "max_decisions": 80004, "max_game_steps": 20000, "batch_seconds": 120}` |
| `training` | Exactly `{"epochs": 5, "sequence_length": 16, "batch_size": 1, "learning_rate": 0.0003, "hidden_size": 32, "feature_set": "hand_recipe", "score_epochs": [0,1,3,5], "required_phases": [str, ...]}` |
| `evaluation` | Exactly `{"max_game_steps": 20000, "deterministic": true, "baseline": "heuristic"}` |
| `limits` | Exactly `global_deadline`, `collection_deadline`, `fit_deadlines`, `evaluation_deadline`, `rss_bytes`; deadlines are positive finite absolute `time.monotonic()` values; fit deadlines have exactly the four arm keys below; RSS is a positive integer at most `2147483648` |

`required_phases` must declare the exact non-forced phase set expected from the
full training prefix, using policy-supported context kinds. Missing or undeclared
phases fail finalization. This is explicit because policy PHASES includes legacy
aliases that the current engine never emits. No validation phase counts influence
training weights. Prior reservation exports are an explicit review input: the
driver cannot discover omitted historical manifests or prove a reservation export
covers every prior study. The launch owner must include all prior final reserves.

`inputs.starts` has exactly string keys `"11"`, `"29"`. Each value has exactly
`checkpoint`, `parent`, `bridge` (references) and `metadata`. Root supplies the
reviewed ordinary new-engine start after verifying actual tensor ancestry. The
driver checks all references without loading legacy parents. It checks the full
exact checkpoint metadata below and rejects nonzero additive coefficients,
nonfinite weights, optimizer state, nonzero steps or an architecture other than
hidden32 graph_consequence_v1 / hand_recipe:

```python
{
    "migration": {
        "kind": "weights_only_cross_engine",
        "source_sha256": parent["sha256"],
        "source_backend_version": "3.2.1",
        "target_backend_version": "3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8",
        "source_steps": 0,
        "seed": seed,
    },
    "feature_upgrade": {
        "kind": "zero_additive_consequence",
        "parent_sha256": bridge["sha256"],
        "feature_set": "hand_recipe",
    },
    "seed": seed,
    "purpose": "untrained_initialization",
}
```

Collection requires `corpus`, `results`, `resume` to be null. After collection,
set `corpus` to the returned immutable `manifest` reference. Fitting requires
`results` null. Initial fitting has `resume` null; explicit resume binds `resume`
to the prior cursor reference and passes that exact path as `resume_cursor`.
Evaluation requires `results` to map exactly `10000/11`, `10000/29`, `100000/11`,
`100000/29` to the four returned immutable `result` references, and `resume` null.
These bindings are excluded from the shared protocol identity; every other field,
including the original absolute deadlines, must remain identical across stages.
Resuming cannot renew a deadline or alter an input, source, seed or endpoint.

## Corpus and fitting guarantees

Each collector call receives four distinct ordered boards and baseline heuristic;
learner seats are 0/1/2/3. Four separate singleton calls would all select seat0.
The collector keeps every own observation, including forced choices. It finishes
the batch before checking non-forced thresholds. A raw incomplete or truncated
batch is saved with its original boundary reasons, stops collection without
replacement and contributes nothing to eligible endpoints. Complete components
before it remain visible. Validation-label components are separate and do not
count toward training. Running out of boards or stopping at 10k remains partial;
primary fitting requires a finalized shared 100k prefix.

`corpus.json` (`colonist-rl.scaling-corpus`, version1) contains the full protocol,
its shared identity, ordered training/validation/raw reference lists, honest
non-forced counts, first-crossing endpoint reference counts and final global
phase weights `N / (number_of_phases * phase_count)`. Each reference row includes
boards, eligibility, split, load/replay time and primitive coverage, supervised
phase counts, four history lengths, and expected updates/prefix work per epoch.
No raw artifact bytes are merged or relabeled. Incremental `progress-N.json`
files retain prior milestones. `first_gate` reports compressed bytes, component
count, observed history quantiles, replay time, expected prefix work, peak RSS and
a clearly labeled linear collection-cost estimate. It is a measurement gate,
not evidence that fitting or a second process will fit the host budget.

Before each load, bytes are hashed. A raw envelope pass rejects ignored hidden
fields and missing full-observation fields, then releases that representation
before the standard demonstration loader runs. Exact typed provenance, seats,
boards, boundaries, coverage, legal indices and ordered versions are checked.
A fresh simulator/driver/teacher replay compares every immutable own view,
ordered legal menu, teacher index, forced call and natural terminal boundary.
Replay uses the original board+100 teacher seed and opponent call order. It holds
no second observation dataset. Fit startup sequentially revalidates and recounts
the finalized corpus and recomputes weights and nested boundaries.

The order is global epoch, component, episode, existing sequence16 chunks. Each
component calls existing `fit_sequences(..., epochs=1, batch_size=1,
optimizer=the_same_adam, phase_weights=shared_weights, profile=True)`. Current
parameter prefixes remain wholly inside the fitter. Components and episode tuples
are released before the next load. The exact parity test compares all policy
parameters, Adam tensors/steps/groups and integer counters against concatenated
batch-one fitting for two epochs. Regrouped reported loss numerators agree within
relative/absolute `1e-12`; per-call rounded means cannot promise bitwise scalar
loss. Reset-optimizer and removed-forced-history controls change final parameters.

Statistics retain committed updates, presentations, weighted presentation mass,
loss numerator, prefix/chunk forwards, batches and fitter timing scopes. Validation
scores stream component CE sums, count/correct totals and phase/action-kind
counts; they do not average component accuracies. Scores are taken at initialization
and completed epochs1/3/5. `load_replay_seconds`, fitter scope/wall seconds,
`score_seconds` and `checkpoint_seconds` are separate cumulative committed-boundary
measurements. They remain separate from per-invocation attempt accounting below.

Every fit/evaluation invocation publishes `protocol.json` and an immutable
`attempt-start.json` before the first budget guard or full-corpus scan. Cheap
strict schema/path/hash preflight can still reject malformed inputs before output.
A valid but expired invocation is recorded. A startup exception or interruption
publishes `attempt.json` with `status="failed_startup"`, preserves the original
exception, and re-raises it; it creates no training cursor or completed-work result.
An uncatchable process kill can leave the running start record and protocol, but
cannot create a final failure record. Filesystem failure may also prevent evidence
publication; existing exclusive evidence remains unchanged.

Both normal results and partial fit results include `attempt`, a path/SHA reference
to the immutable final `attempt.json`. Its exact version-one fields are:

| Field | Meaning |
|---|---|
| `format`, `version` | `"colonist-rl.scaling-attempt"`, integer `1` |
| `stage`, `protocol` | Canonical arm identifier or `"evaluation"`, and the persisted launch protocol reference |
| `status` | Start record: `"running"`; final record: `"complete"`, `"partial"`, `"failed_startup"` or `"failed"` |
| `startup_status` | `"running"`, `"complete"` or `"failed"` |
| `error` | Exception type/message for an exception escaping the invocation, otherwise null; partial-training details remain in the arm result |
| `startup_validation_seconds` | This invocation's guard/full-corpus scan and checkpoint validation/restore/setup wall time, including unsuccessful startup work |
| `fit_seconds` | This invocation's raw fitter-call wall time, including uncommitted attempts and calls that throw; zero for evaluation |
| `total_elapsed_seconds` | Wall time from before cheap preflight through all attempted startup, fitting, scoring, checkpoint and evaluation work to final accounting publication; includes failed work |

These are per-invocation costs, not resumed optimizer/counter state. The total is
sampled immediately before publishing final accounting/result metadata, so it
excludes only those final metadata writes and caller-state restoration. Every
startup scan is included. Resume produces its own record for its repeated scan
and subsequent attempted work; it neither carries prior attempt time into the new
record nor drops the old immutable record. Add the distinct attempt records to
measure total study cost. Committed cursor counters/timings remain cumulative and
must not be added again to per-attempt totals. Every invocation uses the original
absolute stage/global deadlines; recording or resuming never renews them.

## Durable boundaries, memory and limits

Every component boundary publishes a unique checkpoint with optimizer, continuation
Python/NumPy/CPU-Torch RNG and committed counters, then publishes a unique cursor
bound to that checkpoint SHA. Exclusive link publication uses a private staging
directory, fsyncs evidence, and isolates the existing helpers' fixed `.tmp` names.
Pre-existing paths, hard links and predictable `.tmp` aliases cannot overwrite
inputs. A failure between checkpoint and cursor leaves an orphan checkpoint.

A cursor records protocol/corpus/start identities, seed/size, completed global
epochs, next whole component, counters, completed milestone references and score
summaries. An epoch advances only after its final component. Exceptions, deadlines
or cursor-write failure leave the prior cursor authoritative. A separate unique
`partial-last-*.pt` records attempted optimizer state; `attempted` reports observed
Adam updates and the fitter result if one exists. If fitting throws without a
result, exact attempted presentations/phases are unavailable; they are not added
to committed totals. Snapshot publication itself may fail; the error is retained
and the previous cursor still stands. There is no automatic retry.

Explicit resume verifies the cursor/checkpoint/counters and reloads optimizer and
continuation RNG, then replays the entire interrupted component. Pending milestone
scoring is recovered before the next component. Prior failed results remain
immutable; `resume_from` links to the last committed cursor. Suffix replay,
duplicate committed updates and invented completed epochs are prohibited. Public
entry points restore caller RNG and Torch thread settings on success and failure;
the existing fitter/scorer preserve module modes. Checkpoint RNG is saved before
caller-state restoration and remains a valid continuation.

Peak process RSS is checked before/after component loads, replay, fitting, scoring
and checkpoint boundaries. Replay also checks limits between actor steps. The
unit retention test tracks component lifetimes, avoiding unreliable RSS-delta
assertions. JSON decoding, a fitter call, a scoring call and checkpoint writing
are bounded units with cooperative checks at their boundaries; they cannot
preempt an allocation or an in-flight Torch operation. Root's hard RSS/deadline
monitor is still required. The peak RSS counter is a process lifetime high-water
mark, so an already over-budget process refuses work even after memory is freed.
One component may itself exceed the host allowance; no unmeasured 100k run or
second process is justified by the unit suite.

## Completion-only evaluation

Selection reads only validated completed checkpoints and exact update/presentation
counters. It chooses the largest common epoch in1/3/5 across the exact four-arm
matrix and publishes `selection.json` before gameplay. `primary_five_complete`
remains false if a fallback is used. No common milestone means no games or pairs.
The frozen sixteen validation boards are played at all four seats by the four
selected candidates and unchanged heuristic, using existing `evaluate_blocks`.
Only one policy is loaded at a time. Final-test and audit boards are never played
by these entry points.

Every completed game publishes an immutable event before joining the authoritative
journal. Final journals and all failures are retained. Panels must match all
journal rows and the exact requested board/seat set. Existing
`paired_block_summary` rejects censored or unresolved outcomes. Any missing,
failed or censored panel suppresses the study's paired comparisons. Valid studies
report teacher-to-arm and within-seed 10k-to-100k board-level differences.

Equal epochs deliberately imply different updates and presentations: this is a
data-plus-training scaling comparison, not a pure causal data-volume effect.
The 10k weights also use the full100k training phase distribution. Cross-engine
scores are not equivalent, and this driver makes no cross-engine performance
parity claim, promotion decision or PPO launch.

## Concrete launch preparation example

This is executable launch preparation for root-provided files at the paths below;
it is not permission to collect now. Root must first finish legacy DAgger,
complete bridge integration/review, verify all reserved ranges and actual tensor
ancestry, set the real absolute deadlines, and install a hard process monitor.
Use the upgraded isolated interpreter, `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1` and `PYTHONPATH` pointing to this checkout's `src`.

```python
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from colonist_rl.engine_identity import verify_engine_installation
from colonist_rl.scaling_study import (
    collect_scaling_corpus, fit_scaling_arm, evaluate_scaling_study,
)

def ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

root = Path("/private/tmp/colonist-swarm-bxz1k39_/scaling/repo")
inputs = Path("/private/tmp/scaling-reviewed-inputs")
starts = {}
for seed in (11, 29):
    starts[str(seed)] = {
        "checkpoint": ref(inputs / f"seed-{seed}-hand_recipe.pt"),
        "parent": ref(inputs / f"seed-{seed}-legacy-untrained.pt"),
        "bridge": ref(inputs / f"seed-{seed}-new-engine-graph.pt"),
        "metadata": json.loads((inputs / f"seed-{seed}-metadata.json").read_text()),
    }
reserved = inputs / "prior-reservations.json"
excluded = json.loads(reserved.read_text())["reserved_boards"]
start = time.monotonic()
arms = ("10000/11", "10000/29", "100000/11", "100000/29")
protocol = {
    "format": "colonist-rl.scaling-protocol", "version": 1,
    "source": {
        "revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": False,
    },
    "runtime": {"engine": verify_engine_installation(), "python": sys.executable,
                "torch": str(torch.__version__), "threads": 1},
    "partitions": {
        "training": list(range(114000000, 114016384)),
        "validation_labels": list(range(116000000, 116000032)),
        "validation_games": list(range(115000000, 115000016)),
        "audit": list(range(117000000, 117000004)),
        "final_test": list(range(118000000, 118000016)),
    },
    "prior_reservations": {"artifacts": [ref(reserved)], "excluded_boards": excluded},
    "inputs": {"starts": starts, "corpus": None, "results": None, "resume": None},
    "collection": {"milestones": [10000,25000,50000,100000], "max_decisions": 80004,
                   "max_game_steps": 20000, "batch_seconds": 120},
    "training": {"epochs": 5, "sequence_length": 16, "batch_size": 1,
                 "learning_rate": 0.0003, "hidden_size": 32, "feature_set": "hand_recipe",
                 "score_epochs": [0,1,3,5], "required_phases": ["BUILD_INITIAL_SETTLEMENT",
                 "BUILD_INITIAL_ROAD", "PLAY_TURN", "DISCARD", "MOVE_ROBBER", "TRADE_RESPONSE"]},
    "evaluation": {"max_game_steps": 20000, "deterministic": True, "baseline": "heuristic"},
    "limits": {"global_deadline": start + 14400, "collection_deadline": start + 5400,
               "fit_deadlines": {arm: start + 13200 for arm in arms},
               "evaluation_deadline": start + 14400, "rss_bytes": 2147483648},
}
# Persist/review this exact launch dictionary before calling any entry point.
Path("/private/tmp/scaling-launch-protocol.json").write_text(json.dumps(protocol, indent=2))
```

On an explicitly authorized launch, collect into a fresh external directory,
set `protocol["inputs"]["corpus"] = corpus["manifest"]`, then call each arm with
its fixed endpoint/seed and same per-seed checkpoint. Freeze each immutable result
reference into `inputs.results` before calling evaluation. Each call persists its
newly bound dictionary. On partial training, bind only `inputs.resume` and supply
that cursor path to a fresh output directory; preserve all original deadlines.
Do not infer budgets or shorten panels from observed performance.

Proposed envelopes remain collection/validation/replay<=5400s, each10k arm<=900s,
each100k arm<=7200s, evaluation<=1200s, whole<=14400s and RSS<=2GiB. They are launch
planning proposals, not guaranteed completion. Two fit processes are allowed only
after a measured host-safe gate; this module never schedules them. Even proposed
two-worker maxima sum14700s before overhead, exceeding the global horizon.
The example's common absolute fit cutoff is merely a concrete frozen deadline;
root must replace it with reviewed per-arm absolute cutoffs before any launch.
No sustained corpus collection, real panels or100k launch occurred in development.
