# Controlled longer imitation study

`colonist_rl.imitation_study` compares uniform and phase-balanced supervision
without changing the policy, observations, teacher, action features or rewards.
This is a Python API; sustained study compute requires independent review first.

```python
from pathlib import Path
from colonist_rl.imitation_study import run_imitation_study

pilot = Path("runs/learning-pilot-2026-11-04")
result = run_imitation_study(
    "runs/imitation-study-2026-11-05",
    datasets={s: pilot / f"seed-{s}-demonstrations.json.gz" for s in (11, 29)},
    initial_checkpoints={s: pilot / f"seed-{s}-initial_student.pt" for s in (11, 29)},
    epochs=20,
    wall_seconds=900,
    validation_board_seeds=(62100000, 62100001, 62100002, 62100003),
    validation_demo_seeds=(62110000, 62110001, 62110002, 62110003),
)
```

The input arguments are mappings from integer initialization seeds to local
artifact paths, with the same keys. The real comparison uses seeds 11 and 29,
their saved initial students and their respective 2500-example pilot datasets.
The API also accepts smaller datasets and epoch counts for bounded verification;
all actual counts and requested settings are recorded. Initial checkpoints must
have hidden size 32, zero training steps and no optimizer state. They are loaded
separately for each arm. Each fit creates fresh Adam, resets Python/NumPy/CPU Torch
to the seed, and follows the same saved episode and sample order. Sequence length
is 16, learning rate is 3e-4, CPU Torch uses one thread, and there is one worker.
External RNG and Torch thread settings are restored after the run.

`phase_weights(dataset)` counts only non-forced choices in the supplied training
dataset. For N such choices, P represented choice phases and n_phase choices in a
phase, it returns N/(P*n_phase). Consequently the training-choice mean weight is
one. Forced-only phases are absent and an all-forced dataset returns `{}`.
Validation samples never determine weights. `fit_imitation(...,
phase_weights=None)` retains uniform training; a mapping must cover every
non-forced training phase and contain only finite, positive numeric values.
Missing phases raise an error rather than dropping supervision. Extra entries
are checked too. Weighted chunk CE is summed then divided by the number of
non-forced choices, **not by the sum of weights**. This matters for single-phase
chunks. Forced observations retain their recurrent role without contributing CE.
Every update still reconstructs its prefix with the current parameters.

Validation collects at most 128 fresh teacher decisions on the explicit held-out
demonstration candidates. All metrics use the existing unweighted, no-gradient
`imitation_metrics` replay. State digests before and after validation must match.
Training boards, validation demonstration candidates, evaluation boards and the
reserved final-test boards `(70000017, 70000019)` must be disjoint. Duplicate
training boards across seeds are rejected. No reserved final-test game is run.

The wall budget starts after input preflight. Sixty percent is divided equally
among every seed/arm fit (135 seconds per fit for two seeds and 900 seconds).
Every fit has the same requested epochs and fixed wall allowance; the global
absolute deadline additionally bounds collection, fitting, metrics and games.
An actor step, filesystem operation or optimizer operation already in progress
may finish after a deadline. No dataset, epoch target or board schedule is
silently reduced. The default wall allowance is a cap, not a guarantee that the
study completes on every machine. Training completion requires every arm to
finish all requested epochs and all expected non-forced presentations.
`equal_training_budget_complete` is false on any interrupted/failed fit or
provenance failure. Comparisons are suppressed in those cases.

All variants play deterministic games against the fixed heuristic on identical
four-seat board blocks using the authoritative `evaluate_blocks` loop. The
heuristic reference also uses that schedule. `paired_block_summary` produces only
resolved, non-truncated complete matched blocks. A protocol-complete but
step-truncated study can have `status="complete"` and no comparisons: inspect
`comparison_unavailable`, unresolved outcomes and training completion separately.
Teacher agreement and descriptive board differences do not establish competence
or statistical significance.

An empty output directory is required. `manifest.json` records source revision,
input file SHA-256/byte counts, original dataset/checkpoint provenance, initial
parameter digests, settings, partitions and a configuration digest. Each arm
saves its own final checkpoint and held-out metrics. `results.json` is replaced
atomically after stage transitions, and `progress/*.json` is replaced after each
completed game, including step-truncated games. Journals are recovery evidence;
only complete blocks enter evaluator summaries. Failures retain earlier evidence
and never trigger automatic retries. Dirty/unknown execution sources are refused.
Source identity and immutable input bytes are checked again at the end.

## Actual-play opportunity audit

The runner records `actual_play_diagnostics.status="external_analysis_required"`.
Task3 supplies an observation-only audit alongside the study results. Production
code has no dependency on an absolute temporary audit path and does not invent
opportunity denominators from selected-action counts. The audit must report:

- Maritime-trade legal opportunities and selected maritime trades.
- Passes when maritime trade was legal.
- Feasible accept and reject opportunities, with the corresponding selections.
- Passes over directly legal purchases (settlement, city, road or development).

Use the same complete evaluation boards/seats, learner observations and legal
menus, and retain the denominators. Non-forced PLAY_TURN passes can coexist with
maritime choices without passing a directly legal purchase. Maritime trades and
END_TURN share PLAY_TURN, so phase balancing alone may leave that choice
unchanged. This study adds no action-family weighting or oracle labels, and
teacher disagreement is not a correctness label. The separate earlier uniform
20-epoch diagnostic informs runtime only; it does not replace this shared-input
uniform control.
