# Audited acquisition comparison

`colonist_rl.behavioral_study` implements only teacher-state versus student-state
acquisition and a pure full-history prefix selector. It consumes existing BC and
feature evidence by exact byte identity. It never fits those controls, starts PPO,
changes rewards, chooses an architecture from outcomes, or executes final boards.

```python
from colonist_rl.behavioral_study import run_behavioral_study, select_choice_prefix

result = run_behavioral_study(
    output_dir,
    initial_checkpoints={11: legacy_11_path, 29: legacy_29_path},
    training_components={11: original_11_paths, 29: original_29_paths},
    protocol=protocol,
)
```

Use canonical absolute string paths; each component list is an ordered tuple.
The output directory must be empty and outside the source tree or ignored by Git,
so writing evidence preserves a clean execution revision. There is no CLI, resume, automatic retry,
endpoint fallback, registry or scheduler. A failed invocation raises and preserves
its `results.json`, manifest, raw data and any completed checkpoints/games.

## Exact protocol

The protocol is finite JSON with exactly these keys; extra or missing keys fail
before collection. Integer fields reject booleans, floats and nonpositive values.

| Key | Contract |
| --- | --- |
| `version`, `stage` | Exact integer `1`, string `student_labels` |
| `seeds` | Nonempty ordered unique integer list, matching both input mappings |
| `source_revision` | Expected clean Git HEAD; checked before work and at completion |
| `partitions` | Exactly `training`, `validation_labels`, `validation_games`, `audit`, `final_test`; disjoint explicit integer lists, with Task 3 reserved-final protection |
| `acquisition_boards` | Exact string seed keys; each has exactly `teacher_state` and `student_state` ordered nonempty integer lists |
| `input_digests` | Exact union of input checkpoint, original component, validation-label and consumed-evidence paths mapped to lowercase SHA-256 |
| `consumed_evidence` | Ordered nonempty unique artifact paths; recorded without retraining or selecting from metrics |
| `validation_labels_path` | Immutable legacy demonstration file covering the full declared label panel |
| `query_budget`, `max_game_steps` | Positive exact integers, identical for each acquisition arm |
| `baseline` | `heuristic` |
| `deterministic` | Exact boolean for student behavior and policy evaluation |
| `epochs`, `segment_epochs`, `sequence_length`, `batch_size` | Positive exact integers; same schedule for both arms |
| `learning_rate` | Positive finite number, excluding booleans |
| `weighting` | `original_phase_balanced` |
| `collection_seconds`, `fit_seconds` | Positive finite ceilings **per arm and seed** |
| `evaluation_seconds` | One positive finite **global evaluation ceiling**, shared by the heuristic reference and every arm/seed |
| `deadline` | Finite absolute `time.monotonic()` deadline for the whole invocation |

Every local ceiling is clipped by `deadline`. Collection and fitting do not borrow
unused time from another arm. Evaluation receives the same absolute deadline for
all variants. The accepted routines check deadlines at bounded operation
boundaries; serialization and a single held-out scoring call cannot be preempted.
Actual elapsed time includes those calls and is reported. Expiry does not reduce
the requested panel or silently restart a phase.

Root predeclared the intended subsequent launch: seeds 11/29; 4,000 all-decision
queries per arm/seed; equal 120-second collection ceilings; the same completed
legacy epoch-30 checkpoint within each pair; fresh independent Adam optimizers;
five epochs in one five-epoch segment; sequence length 16, batch size 1, learning
rate 0.0003; 600 seconds of fitting per arm; 600 seconds shared evaluation time;
and 3,600 seconds for the whole run, with its monotonic deadline computed at
launch. The fixed evaluation panel is 72010000..72010015 × all four seats: 64
games per arm/seed and one shared heuristic reference. These documented settings
do not authorize or start a study. The caller must supply all exact input hashes,
disjoint acquisition boards, maximum game steps and behavior mode before launch.

The driver loads one frozen finite legacy `GraphPolicy` per seed, then copies it
independently for both arms. Any declared checkpoint seed must match. When a
checkpoint includes fit-call evidence, every call must be complete and its full
actual epoch, update and presentation totals must match the checkpoint counters.
A historical `partial=true` marker alone does not disqualify an exact completed
milestone; an incomplete fit call does. Starting metadata and counters are kept in
the manifest, while the accepted collector identity remains exactly
`{path, sha256, architecture}`. Both arms discard inherited optimizer state and
create fresh independent Adam objects, retaining each object across fit segments.

## Raw data and matched training views

Teacher-state acquisition calls `collect_demonstrations`; student-state acquisition
calls `collect_student_labels` with its accepted `progress_path`. Both use the
existing match driver, heuristic teacher and opponents. There is no second game
loop. Raw components retain their original source, observations, flags and bytes.
The legacy teacher collector has no per-episode progress callback: only a returned
teacher component can be saved. Student progress is retained through the accepted
collector even if a later operation raises. The driver does not invent a teacher
callback or claim to retain data the legacy collector never returned.

Counters distinguish teacher queries, paired labels, non-forced labels, executed
decisions and choices, unexecuted paired labels, and unmatched teacher calls.
Teacher-first expiry can leave a real unmatched call. A returned student choice
can remain explicitly unexecuted. Neither becomes a fabricated executed sample.
Legacy teacher execution is inferred conservatively from the observation version
and executed episode endpoint; a deadline label at the endpoint is unexecuted.
The accepted corpus rejects that legacy endpoint, so the driver retains its raw
artifact and fails before fitting instead of changing the label or raw flags.

Before fitting anything, for each seed set `Q_s` to the minimum newly acquired
non-forced-label count across arms. Zero choices fail closed. The pure
`select_choice_prefix(episodes, choices=Q_s)` validates every label, including the
unused suffix, then keeps complete earlier episode histories and the final prefix
through the `Q_s`-th choice. Forced observations before the cutoff remain. Episode
memory starts remain independent; no isolated choices are spliced together.

Selection manifests use zero-based episode indices and exclusive sample stops.
They record retained endpoints, all/forced/choice counts, unused suffix endpoints
and counts, and visible board digests. Because `EpisodeSequence` has no seed field,
the pure selector identifies its actual visible `BoardSpec` by canonical digest;
the driver adds each raw episode's original board seed, component byte/source
identity, derivation source and derivation digest. A clipped view is never marked
as a completed raw episode. Raw components are loaded through the accepted corpus
and projected directly to fitter sequences; there is no invented merged v1 file.

Both arms concatenate the same original component order with their selected new
view. One frozen mapping `N/(P*n_phase)` is derived solely from each seed's original
non-forced training counts and used for both arms. Every selected non-forced phase
must have a positive explicit weight. Held-out labels never affect those weights.

## Work, interruption and evidence

Before fitting, the driver records expected choice presentations, non-forced chunk
updates, prefix/chunk forwards, batch chunks, forced observations and weighted
supervision mass using the accepted fitter schedule. Actual completed epochs,
updates, presentations, prefix/chunk forwards, weighted mass and elapsed seconds
are reported separately. A partial final segment can contain updates beyond its
completed epoch count. Every segment saves its policy, persistent optimizer and
work counters, including a partial last segment. If fitting raises after unknown
internal work, the checkpoint is retained and the unknown work is explicitly
identified rather than guessed. Unweighted held-out metrics use `score_sequences`
without gradients and belong only to the latest scored model state.

Equal queries and selected new non-forced choices match acquisition counts and
new choice presentations per epoch. They do not match optimizer updates, weighted
loss mass, forced-prefix replay or runtime. This supports neither an equal-compute
claim nor a pure state-distribution causal claim.

Evaluation uses `evaluate_blocks` on the full declared board/seat panel. Its
`on_game_end` callback atomically saves each authoritative finished row before the
next game; censored rows remain censored. Returned rows must match that journal
and the complete requested panel. The exact variant matrix is the two arms × all
seeds plus one heuristic reference. Any missing/censored/mismatched panel or
incomplete acquisition, fitting or held-out score suppresses **all** paired
comparisons; surviving subsets are never salvaged as a completed study.
`paired_block_summary` supplies the descriptive comparisons only after those
checks. Source/input and acquired raw-component identities are verified again
before comparisons are emitted. Hashes do not replace the source provenance.

`handoff` contains saved checkpoint identities, their frozen starting references,
fit status and selection digest for the existing curriculum to inspect. This
module never invokes curriculum training or changes its gates. A partial handoff
is labeled partial and cannot establish completion.
