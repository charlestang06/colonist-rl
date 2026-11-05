# Student labels and component corpora

`colonist_rl.student_labels` collects observation-only DAgger data with a frozen
student checkpoint. The student chooses every learner action; the unchanged
heuristic supplies a legal supervised target from the **same immutable own
Observation**. Teacher agreement is a supervised diagnostic, not evidence of
playing competence. This module does not collect rewards or PPO transitions.

```python
import hashlib
import time
from pathlib import Path

from colonist_rl.policy import policy_architecture
from colonist_rl.student_labels import (
    collect_student_labels, save_student_labels, write_corpus,
    load_corpus, corpus_sequences,
)
from colonist_rl.supervised_fit import fit_sequences
from colonist_rl.training import load_checkpoint

checkpoint = Path("runs/student.pt").resolve()
policy, _ = load_checkpoint(checkpoint)
identity = {
    "path": str(checkpoint),
    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "architecture": policy_architecture(policy),
}
data = collect_student_labels(
    policy, checkpoint_identity=identity, board_seeds=[73031000, 73031001],
    max_decisions=128, max_game_steps=20000, baseline="heuristic",
    seed=11, deterministic=False, deadline=time.monotonic() + 30,
    progress_path="runs/student-labels.json.gz",
)
save_student_labels("runs/student-labels.json.gz", data)
write_corpus(
    "runs/corpus.json",
    components=("runs/original-teacher.json.gz", "runs/student-labels.json.gz"),
    partitions={
        "training": [73030000, 73031000, 73031001],
        "validation_labels": [73032000],
        "validation_games": [73033000],
        "audit": [73034000],
        "final_test": [70000017, 70000019],
    },
)
corpus = load_corpus("runs/corpus.json")
result = fit_sequences(policy, corpus_sequences(corpus), epochs=1)
```

Use project-declared disjoint boards in actual experiments; the example does not
authorize a new run. Every actual board in every referenced component must be in
`training`, including the original teacher component's boards. The fitter sees
only ordered `(Observation, teacher_index)` episode tuples. It recomputes memory
from each episode start under current weights, preserves forced prefixes and
supervises non-forced choices. A final unexecuted choice remains a valid label for
its observed prefix; it is never represented as a transition.

## Checkpoint and collection contract

`checkpoint_identity` has **exactly** `path`, `sha256`, and `architecture`.
`sha256` is 64 lowercase hexadecimal characters identifying checkpoint bytes.
The collector verifies the file digest, uses the accepted `load_checkpoint`,
compares its architecture and every state tensor to the supplied policy, then
collects using an independent `eval()` copy with gradients disabled. It checks
the checkpoint bytes again at the end. Legacy and consequence architectures use
the existing explicit `policy_architecture` identity without translation.

The caller's weights, gradient flags and every module's mode remain untouched.
Python, NumPy and CPU Torch RNG states are restored even on failure. Each episode
starts with empty student memory; all own decisions, including forced choices,
advance it. No memory tensors or opponent observations enter the data.

The collector accepts an ordered, possibly lazy board iterable. Consumed boards
must be unique integers; booleans and reserved final boards fail. Seat is
`board_index % 4`. The existing driver receives the original board seed; the
backend reset receives that seed and internally uses `board_seed + 1`. Teacher
seed is the original `board_seed + 100`. Behavior RNGs are seeded independently
per episode with `(seed + board_index) % 2**63` (NumPy modulo `2**32`).

Only `_MatchDriver` handles proposals, opponents and game mutations. The
collector checks the absolute monotonic deadline before actor calls, after
expensive observation/teacher/student calls, and immediately before applying an
input. A returned student action and its teacher label are appended with
`behavior_executed=false`; this changes to true only after the driver applies it.
An expired final choice is retained, never applied or retried.

The teacher is called first so illegal labels raise before student selection or
game mutation. If the teacher itself exhausts the deadline, the collector records
its final call ordinal/version/index and starts no student call. Thus there may
be exactly one more teacher call than paired samples, only at a deadline endpoint.
Every student observation still has a teacher label, and no missing behavior
index is invented. Teacher replay must include forced calls in the original order.

## Student-label version 1 schema

Files are deterministic gzip-compressed canonical finite JSON, with
`format="colonist-rl.student-labels"`, integer `version=1`, the current
`backend_version` and ordered `action_kinds`. Unknown fields fail validation.
Atomic writes create an exclusive fresh temporary file in the output directory
and replace the destination after closing it. Existing staging paths are never
opened for writing, so their symlinks and hard links cannot change input bytes.

The other top-level fields are:

| Field | Exact content |
| --- | --- |
| `source` | Collection Git `revision` and boolean `dirty`; unavailable provenance fails for new collection. |
| `teacher` | `name="HeuristicPolicy"`, `seed_rule="board_seed + 100"`, `input="immutable Observation"`, `labels="legal action index"`, `call_order="once before every student call, including forced decisions"`. |
| `student` | `checkpoint` as above, `behavior="student_only"`, `mode="eval"`. |
| `collection` | Settings, schedule, status and timing described below. |
| `coverage` | Recomputed counts described below. |
| `episodes` | Ordered complete episodes and final interrupted prefix. |

`collection` has exactly `baseline`, `max_decisions`, `max_game_steps`,
`deadline`, `seed`, `deterministic`, `seat_rule="board_index % 4"`,
`sampling_seed_rule="(seed + board_index) % 2**63"`, `actual_board_seeds`,
`status`, `stop_reason`, `started_unix_seconds`, and `elapsed_seconds`.
Caps are positive integers, mode is boolean, and deadline/timings are finite.
`actual_board_seeds` records the consumed environment-reset schedule without
predicting unused lazy seeds. Git revision/dirty status is checked during progress
persistence and at return; a change raises instead of relabeling provenance.

Final status is `complete` only for `stop_reason="seeds_exhausted"`.
`decision_cap` and `deadline` are `incomplete`. Collection completion does not
imply that its games terminated: a collection can exhaust its board schedule
with honestly step-capped episodes. Atomic progress snapshots after finished
episodes have `status=stop_reason="collecting"`; such active snapshots cannot
enter a corpus. The final interrupted prefix is atomically persisted before
return. Saving performs validation and I/O even after the simulation deadline.

Every episode has exactly `board_seed`, `backend_seed`, `seat`, `teacher_seed`,
`sampling_seed`, `student_sha256`, `architecture_sha256`, `complete`,
`terminated`, `truncated`, `incomplete_reason`, `steps`, `teacher_calls`,
`history_sha256`, and `samples`.

The student/checkpoint digest is repeated per episode. Architecture digest is
SHA-256 of canonical architecture JSON. `teacher_calls` is an ordered list of
`{ordinal, version, action_index}`; ordinals start at zero and include forced
calls. Each paired call must match the corresponding sample's context version
and teacher index. `history_sha256` covers the complete ordered serialized sample
list, including observations, both indices, and execution flags. Missing or
reordered history is rejected rather than silently projected.

Each sample has **exactly** `observation`, `teacher_action_index`,
`behavior_action_index`, and `behavior_executed`. The observation uses the
existing versioned observation export, rebuilt as immutable records on load.
Extra observation-envelope fields are rejected. Both indices must be integer
indices into that observation's unchanged ordered legal menu; booleans fail.
Player and context actor must match the episode seat. Context versions strictly
increase. Executed samples precede the endpoint version; the sole unexecuted
sample may be the last sample of a deadline prefix at its endpoint version.

`complete` equals `terminated`; only terminal episodes have no incomplete reason.
`truncated` is true exactly for `incomplete_reason="step_cap"` at the configured
step cap. Decision/deadline prefixes are neither terminal nor truncated, occur
only at the end, and agree with the collection stop reason. Terminal games and
step caps are never inferred from decision counts.

`coverage` has exactly `decisions`, `non_forced_decisions`, `behavior_executed`,
`disagreements`, `phase_counts`, `teacher_action_kind_counts`, and
`behavior_action_kind_counts`. These typed integer counts describe paired samples;
unmatched final teacher calls remain visible separately in `teacher_calls`.
All fields must agree with the actual histories. Metadata contains finite JSON
primitives only. Samples contain no rewards, tensor caches, hidden engine state,
or private opponent input.

## Corpus version 1 schema

`write_corpus` atomically writes canonical JSON with
`format="colonist-rl.supervision-corpus"`, integer `version=1`, `source`,
`partitions`, `components`, and `component_order_sha256`. Its source records the
**derivation** revision/dirty status, separately from every collection source.

`partitions` has exactly `training`, `validation_labels`, `validation_games`,
`audit`, and `final_test`, each an explicit unique integer list (empty lists are
allowed). All five are pairwise disjoint. Reserved legacy final seeds 70000017,
70000019 and curriculum final seeds 71001000–71001007 may only appear in
`final_test`, even when a caller omits them from its declared final list.

Each component entry has exactly `order` (zero based), absolute resolved `path`,
`sha256` of the original persisted bytes, `format`, `version`, original `source`,
and `actual_board_seeds`. Accepted inputs are unchanged version-1
`colonist-rl.demonstrations` or finalized student-label files. Legacy provenance,
including explicit unavailable source identity, remains unchanged and is checked
by its existing loader; legacy datasets are never rewritten or forged into a
shared collection source. `component_order_sha256` covers the full ordered entry
list, including identities and positions.

Writing, loading, and projection reject duplicate component digests, missing or
changed bytes, inconsistent order/source/board metadata, boards outside training,
split overlap, malformed observations or histories, contradictory endpoint flags,
and corpora with no non-forced supervision. Component identities are compared as
canonical typed JSON, so `true`, `1`, and `1.0` cannot substitute for each other,
even with a recomputed order digest. Projection reloads and verifies the
original files every time. Keep them available at their recorded paths; the
manifest does not contain a mutable cached copy. It does not claim to authenticate
arbitrarily rewritten manifests or prove terminal resolution by replaying games.
