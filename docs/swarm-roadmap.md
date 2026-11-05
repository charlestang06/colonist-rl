# Swarm roadmap: evidence before expansion

Planning snapshot: November 4, 2026. This document extends the priorities in
[the product roadmap](roadmap.md) with bounded work packages and decision gates.
It records the starting evidence and the outcomes required from the current
implementation wave; it does not claim that those new outcomes have passed.
The learning coordinator owns learning and CLI work, the adapter coordinator
owns adapter work, and the root coordinator integrates and verifies changes.

## What the first completed wave established

- Shared, Colonist, and physical adapter boundaries are implemented. Legacy
  imports, observation JSON and optional browser isolation were preserved.
- Ordered rendered-log normalization and an offline historian are implemented.
  The two durable log fixtures contain no stable row IDs. Unique text overlap
  remains a hypothesis; the historian deliberately certifies no complete history.
- The combined baseline recorded **205 passing tests, no skips**, clean lint and
  formatting, a successful wheel build, and clean independent reviews. See
  [verification](validation.md). These are historical baseline results; every
  subsequent integration needs fresh verification.
- A manually controlled guest game reached 10 points in 67 turns. This supports
  the [controls evidence](colonist-controls.md), not autonomous execution or
  learned policy strength.
- The existing smoke-trained policy won zero of four held-out games. Four games
  cannot establish strength or diagnose a learning failure. The current
  `imitation_warmstart` and seat-rotated `evaluate` functions provide a starting
  path, not evidence of competent imitation.

The useful next outcome is an auditable teacher-to-student comparison. More UI
features cannot answer whether the learning system works. The adapter track can
deliver small independent components while the learning track answers that
question.

## Current wave and order

The first learning work package is implemented but **not accepted**. Independent
review found that no-winner truncations entered strict win-rate and paired
comparison summaries as losses. The original implementer is correcting this;
acceptance requires focused regression evidence and a fresh independent review
of the correction. The adapter UI contracts are written and await independent
review. Neither status establishes a completed learning or deployment milestone.

| Order | Bounded deliverable | Required evidence | Stop or defer when |
| --- | --- | --- | --- |
| 1: L1 + A9 | Reproducible teacher and imitation pilot | Frozen configuration, source identity, disjoint seed splits, complete seat blocks, raw outcomes, coverage and timing | Results cannot be reproduced, observations leak hidden state, or most episodes truncate |
| 2: L1 | A student that learns useful teacher decisions | Held-out agreement split by phase and forced versus choice decisions; complete-game comparison against the same opponents | Agreement is driven by forced moves, teacher is weak, or student loses competence in full games |
| 3: L2 | A bounded PPO comparison from the same initialization | Fresh learner trajectories; paired seeds and budgets; seed-level outcome, KL, entropy, critic and throughput metrics | Initialization differs between variants, trajectories are stale, or evaluation differences are inconclusive |
| Secondary: A4 | Immutable staged UI records and pure validators | Local stale-revision, exact-action, pending/committed and trade-row tests | The component requires live gestures or modifies shared policy observations |
| Secondary: A3 | Conservative semantic projection of supplied rendered rows | Production and completed bank-trade fixtures, explicit player mapping, unknown/unsupported diagnostics | Parsing would turn an offer, ambiguous row or guessed identity into a transfer |

The secondary tasks may finish before the learning pilot. Their completion does
not reorder the primary learning gates. A6/A7 gesture execution and A5 ledger
integration remain separate follow-up decisions after the relevant contracts and
evidence pass review. A new parser result alone does not authorize ledger writes.

## Learning experiment contract

Use the current simulator and information-limited heuristic first. If an upstream
teacher is evaluated, pin its source separately and audit its full-game access,
four-player assumptions, rule semantics and action conversion before collecting
training labels. Preserve the working learner dependency environment.

Each run should save source revision plus dirty-source identity, engine and
action schema, configuration, opponent versions, all seeds, checkpoint/dataset
digests, actual decisions, wall time, completion status and raw per-game records.
Distinct board-seed namespaces are required for training, model selection and
final testing; merely changing policy sampling seeds does not create new boards.
Do not consume the final test split repeatedly while selecting experiments.
Reserve fresh final-test board seeds **70000017** and **70000019**; do not use
them for throughput probes, training or model selection. The former `100000`
range may already have been used in historical smoke evaluation and must not
be presented as an untouched final test split. These two reserved boards are an
initial reservation, not a sufficient sample for a strength claim.

Start with a short local throughput probe and a bounded demonstration pilot of
at most 10,000 learner decisions. Treat that as a ceiling for the initial probe,
not a required expense. Record incomplete games and decision coverage rather
than filling missing outcomes. Use complete four-seat blocks for comparison;
tiny samples remain diagnostic. Set explicit time, decision and concurrency caps
in the actual run manifest before launch. No paid compute or services are needed.

Measure teacher performance against random and fixed heuristic opponents before
calling it competent. Report setup, normal, forced and trade-response coverage,
including how many decisions had more than one legal choice. Save ordered
sequences; recompute recurrent state under the current student during training.
Count completed and truncated games separately and retain every failed run.

Compare untrained, imitation-only and imitation-plus-PPO checkpoints against the
same fixed opponents and board blocks. Keep teacher results beside the student
results. For a later PPO anchor experiment, use two configurations and three
paired training seeds with identical initial checkpoint digests and independent
optimizer/RNG state. Profile before choosing concurrency; parallel jobs should
not reduce useful throughput through CPU or memory contention.

Use paired board-block differences for evaluation. Games on the same board are
dependent, so a game-level nominal Wilson interval must not become a claim of
independent-sample certainty. Predeclare the promotion criterion and required
sample size before a larger run; the initial pilot only estimates feasibility,
variance and cost. An improvement claim needs held-out play evidence, not a
lower supervised loss, finite PPO updates or a winning training episode.

The first review exposed a concrete distinction: a scheduled four-seat block
is not a finished four-seat block. A game stopped without a winner has a
censored outcome, not a loss. Strict win-rate summaries must distinguish known
wins/losses from censored games and expose their denominators. Strict paired
comparisons require the same four seats to finish on both sides; retain and
report incomplete blocks instead of filling missing results with zeros. If no
eligible finished block remains, report the comparison as unavailable. A
completed-game-only rate can also be biased when completion depends on policy,
so material or unequal censoring prevents promotion even if that rate improves.

## Decision gates after results

| Finding | Next bounded response | Evidence needed before expansion |
| --- | --- | --- |
| Review finds no-winner truncations scored as losses | Correct outcome aggregation before interpreting pilot results | Regression cases for one-sided censoring, partially finished seat blocks and all-censored samples; independent re-review; no invented loss or paired difference |
| Teacher does not beat simpler opponents reliably | Diagnose teacher decision traces and benchmark one audited alternative | Comparable four-player board blocks and latency; no hidden-information advantage |
| Teacher plays well, imitation does not | Check target/action alignment, legal candidates, phase imbalance and recurrent sequence handling | A small deterministic learning probe plus phase-level held-out error and complete games |
| Imitation agreement is high but play is weak | Inspect non-forced strategic choices and student-induced state distribution | Choice-only metrics and targeted failure positions; consider a bounded student-state relabeling pilot only after this diagnosis |
| PPO destroys imitation competence | Audit trajectory freshness, KL, entropy and value targets; compare a decaying anchor | Equal initial checkpoint and equal budgets, with all paired-seed outcomes retained |
| PPO makes no credible improvement | Separate high variance from optimization and representation failures | More adequately sampled validation blocks or a focused diagnostic; no architecture expansion by default |
| Simulator or graph processing dominates runtime | Profile one complete collection/update cycle, then optimize the dominant measured path | Before/after decisions per second at unchanged behavior and evaluation configuration |
| A4 and semantic parsing pass | Add a small offline staged replay using explicitly synthetic or observed fixtures | A gesture acknowledgement remains separate from a committed action; unsupported paths remain explicit |
| A history gap prevents accounting | Preserve the valid ledger and incomplete provenance; design a verified resynchronization epoch | Transactional recovery tests and established event identity; content overlap is insufficient |

Student-state relabeling is a possible later experiment, not a current
implementation commitment. The same applies to richer event encodings, larger
networks, deeper graph processing and search-assisted teachers. Select one only
when a measured failure points to it, and compare with an unchanged baseline.

## Integration risks and safeguards

- **Source drift:** independent branches may pass while their combined state
  fails. Integrate one reviewed work package at a time, preserve ownership, then
  run the combined suite, lint/format and packaging checks on the final source.
  Attribute any optional browser skip explicitly.
- **Misleading evaluation:** shared boards, seat imbalance, truncations and tiny
  trade-response counts can dominate apparent gains. Retain raw board/seat data
  and label inconclusive comparisons; do not silently score a truncation as loss.
- **Reward interpretation:** the current critic predicts shaped return, not win
  probability. Teacher-continuation returns are not learner-policy values.
  Calibration claims require completed frozen-policy games and a declared target.
- **Hidden-information leakage:** teachers and simulator helpers can access
  more than the deployed actor. Audit the teacher boundary and keep privileged
  diagnostics separate from deployable-policy results.
- **Recurrent and RNG contamination:** old hidden states, shared optimizers or
  evaluation that perturbs training randomness can invalidate comparisons.
  Recompute sequence state and isolate each run's mutable state.
- **UI identity errors:** a changed revision, same action kind on another target,
  same trade terms on another row or another partner must invalidate confirmation.
  Pending willingness must never update resources or be reported as commitment.
- **Parser and history confusion:** a correctly parsed transfer is still only
  a candidate until event identity and coverage are established. Preserve source
  rows, unknown identities and ambiguity through each layer.
- **Scope drift:** six-player simulation, unrestricted negotiation, general photo
  recognition and full autonomous browser games are not part of this wave. They
  remain useful future directions after the current learning and replay gates.

## Wave completion and next planning update

A wave is ready to close when the root coordinator has reviewed the combined
diff, recorded fresh checks and stored bounded experiment artifacts or a concrete
failure diagnosis. A failed learning hypothesis can close an experiment; it
cannot mark the competent-imitation or PPO-improvement milestone complete.

For each delivered work package, update the primary roadmap with its actual
scope, evidence path, remaining limitations and the next gate. Keep previous
verification dates distinct from new results. Choose the next experiment from
the measured bottleneck table above, preserve the losing/failed run records, and
allocate additional local compute only after the probe supplies a cost estimate.
