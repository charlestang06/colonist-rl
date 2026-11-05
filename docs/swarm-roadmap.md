# Swarm roadmap: evidence before expansion

Planning snapshot: November 4, 2026. This document extends the priorities in
[the product roadmap](roadmap.md) with bounded work packages and decision gates.
It records the starting evidence, accepted implementation results and the
outcomes still required from the current wave. Implementation acceptance does
not establish teacher competence, student strength or PPO improvement.
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

The first learning work package is **accepted after scoped re-review**. Review
initially found that no-winner truncations entered strict win-rate and paired
comparison summaries as losses; the implementer corrected censoring and the
correction passed re-review. The adapter UI contracts and production/bank/dice
semantic candidates passed the final broad independent review with no
outstanding findings. Both minor documentation findings are resolved. UI records
and pure validators issue no gestures; semantic candidates establish neither
event identity nor ledger updates. Year of Plenty and Road Building descriptors
do not establish support for those browser flows.

Learning Task 2, including persisted demonstrations, current-weight recurrent
imitation and the shared deadline-mutation guard correction, passed independent
re-review. Learning Task 3 subsequently passed scoped re-review after its
evidence-matrix and per-game persistence fixes, and the root integrated it.
The latest **intermediate** combined check found **437 tests passed, 1 optional
browser skip in 8.96 seconds**, with clean Ruff lint/format across **49 source/test
files**. This supersedes the 394-pass intermediate count and is not final-source
verification. Broad learning review remains pending and has identified an
installed-wheel regression: legacy warm start incorrectly requires Git. Its fix
and re-review remain required; passing checkout tests does not establish installed
package behavior.

The first adapter wave remains accepted. The follow-on staged executor Task 1
is **not accepted**: its implementer is correcting two Important findings about
terminal evidence and the review lock after input, plus a minor nested-argument
exception boundary. Drag work remains gated. No live execution is claimed.

## Results now informing the next work

The [accepted pilot evidence](learning-pilot-2026-11-04.md) completed in
**92.301 seconds** at clean source `1350de2`, with 5,000 demonstrations, 64 natural
validation-game terminations and four probe games. No evaluation game truncated.
The teacher won 8/8 against random opponents and 3/8 against heuristics with
6.750 mean VP. Both imitation students won 0/8 against heuristics; mean VP was
3.375 and 3.250. Both matched 32/83 non-forced held-out teacher choices (38.55%).
Only two validation boards and 83 held-out choices were available, and several
phase metrics were weak. The declared baseline gate failed; **no PPO ran in that
pilot**. These measurements establish a weak student baseline, not competent
imitation or an architecture-capacity diagnosis.

A separate [uniform-imitation diagnostic](../runs/swarm-2026-11-04/uniform-diagnostic/report.md)
completed 20 epochs and 3,380 updates for each seed in 270.541/300 seconds using
the same 2,500 observations per seed and frozen source `35b1045`. Original pilot
bytes were preserved, and same-source original outcomes reproduced exactly.

| Seed | Choice agreement: 2 epochs → 20 epochs | Wins after 20 epochs | Mean VP: 2 epochs → 20 epochs |
| --- | --- | --- | --- |
| 11 | 32/83 → 43/83 | 0/8 | 3.375 → 3.875 |
| 29 | 32/83 → 40/83 | 0/8 | 3.250 → 3.125 |

All diagnostic games terminated naturally on the same known boards 90000/90001.
Agreement improved, especially on robber and ordinary-turn choices; discard
agreement remained 0/6 and setup agreement was sparse. More uniform optimization
has measurable effect but has not established stronger play. These reused tiny
validation samples do not support checkpoint promotion or a model-capacity claim.

The [baseline strategic audit](../runs/swarm-2026-11-04/baseline-audit/baseline-strategic-audit.md)
classified 36 actual non-forced decisions as 24 reasonable, 11 ambiguous and one
clearly bad under its public-observation rubric. This is qualitative evidence,
not 36 optimal-action labels or a strength estimate. Across both policies there
were **zero maritime trades in 143 legal opportunities** and all **29 feasible
incoming offers were accepted**. There were **zero end turns passing over a
directly legal build or development purchase**; all 113 non-forced passes faced
maritime alternatives. The clear error missed a conversion enabling an immediate
productive settlement. This supports investigating intermediate conversions and
offer selectivity, without asserting every trade opportunity should be taken.

Prioritize the planned uniform-versus-balanced imitation study and bounded fresh
PPO curriculum against measured stronger opposition. Keep architecture fixed for
these comparisons, retain original checkpoints and fixed validation opponents,
and report choice counts, learning curves, actual decisions and complete-game
outcomes. The newly authorized exploratory PPO study is separate from the failed
pilot promotion gate; neither it nor balanced imitation has a result here yet.

## Authorized three-hour work horizon

The user authorized autonomous work without questions while away, from
**03:54 to 06:54 UTC on November 5, 2026**, emphasizing conceptual judgment of
actual policy decisions and substantial training against harder opponents.
Reserve approximately **06:29–06:54 UTC** for integration, verification, result
review and a clear handoff. Use the earlier window for evidence-driven training
and diagnosis; do not start a run that consumes the integration reserve.

The baseline decision audit is complete, and the learning coordinator's follow-on
plan covers balanced imitation and a configurable fresh-PPO opponent curriculum.
These extend the existing learning priority, not the adapter feature backlog.
Allocate the remaining local budget using the completed pilot and diagnostic
results above. Prefer competent fixed opponents and frozen policy checkpoints over
spending the whole window against random opponents. Measure opponent difficulty
under common validation blocks; the word “harder” is not established by a class
name or by the learner losing a few games. Retain fixed benchmark opponents
outside the changing training mixture.

Useful ordering within the horizon is: inspect pilot artifacts and failure
modes; audit actual non-forced choices; run bounded longer training with the
reviewed curriculum; compare checkpoints on validation blocks; integrate and
verify. These are priorities, not new mandatory feature packages. A learning
strength gate governs checkpoint promotion; it need not prevent an explicitly
budgeted exploratory PPO run once trajectory/data correctness and the baseline
are established. Keep its evidence separate if imitation competence remains
unproven. Preserve all failed/collapsed outcomes and the untouched final-test
seeds, and reserve evaluation time rather than spending every second training.

### Strategic decision audit

Use only the acting player's immutable observation and legal actions, with the
actual chosen action and enough public context to explain it. Sample setup,
resource bottlenecks, expansion, bank trades, development cards, robber choices,
incoming trades and ending the turn where the pilot actually supplies examples.
Report missing categories as coverage gaps. Do not use hidden opponent hands,
future dice or privileged outcomes to retroactively declare a decision wrong.

Classify an examined action as **clearly dominated**, **reasonable**, or
**ambiguous/insufficient evidence**, with a short public-state rationale and a
specific legal alternative when relevant. Reserve “dominated” for an alternative
whose superiority follows from the visible state and stated assumptions;
strategic tradeoffs and uncertain future rewards are usually not proofs.
Disagreement with the heuristic teacher is not automatically a mistake.

Look for resource/production balance and number diversity in setup, feasible
expansion routes and contested sites, whether spending advances a coherent
near-term build, affordable builds passed over without a visible reason,
avoidable bank-exchange costs, development-card timing, public road/army races,
and trade value to both parties given public scores and production. Robber and
steal reasoning must preserve hidden-hand uncertainty. Note legitimate saving,
blocking and race tradeoffs instead of treating every immediate VP alternative
as optimal. Summarize failure patterns and their sampled denominators; selected
examples are explanatory evidence, not a substitute for full-game evaluation.

If repeated mistakes concern information already present, first test the current
training/coverage diagnosis. If different reasonable strategies merely disagree,
use complete-game validation instead of adding rules to imitate the reviewer.
If the review needs missing information, record that uncertainty before proposing
one focused representation experiment. No audit finding alone authorizes broad
architecture expansion.

| Order | Bounded deliverable | Required evidence | Stop or defer when |
| --- | --- | --- | --- |
| 1: L1 + A9, pilot completed | Reproducible teacher and imitation comparison | Clean-source pilot and scoped review complete; broad installed-package correction pending; competence gate failed | Do not relabel operational completion as competent imitation |
| 2: L1 | A student that learns useful teacher decisions | Held-out agreement split by phase and forced versus choice decisions; complete-game comparison against the same opponents | Agreement is driven by forced moves, teacher is weak, or student loses competence in full games |
| 3: L2 | A bounded PPO comparison from the same initialization | Fresh learner trajectories; paired seeds and budgets; seed-level outcome, KL, entropy, critic and throughput metrics | Initialization differs between variants, trajectories are stale, or evaluation differences are inconclusive |
| Secondary: A4, accepted | Immutable staged UI records and pure validators | Independent review and fresh combined checks passed; live execution remains separate | The component requires live gestures or modifies shared policy observations |
| Secondary: A3, bounded scope accepted | Production, completed bank-trade and two-die semantic candidates | Final broad adapter review clean; no established event IDs or ledger integration | Parsing would turn an offer, ambiguous row or guessed identity into a transfer |
| Secondary: A6/A7, follow-on pending review | Bounded staged execution | Post-input terminal-evidence and lock fixes must pass re-review before acceptance | No live or autonomous-play claim from offline implementation |

Secondary task completion does not reorder the primary learning gates. A6/A7
staged execution review and A5 ledger
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

The first review exposed three distinct states: a scheduled four-seat block,
a fully executed four-seat block, and a block with four resolved game outcomes.
A step-limited game can finish its scheduled execution while still having a
censored outcome, not a loss. Strict win rate and its denominator are null if
any selected outcome is unresolved. Strict paired comparisons reject incomplete,
truncated or unresolved inputs and require the same four seats on both sides;
retain partial and interrupted records without filling missing outcomes with
zeros or silently selecting a favorable subset. A resolved-only rate can also
be biased when completion depends on policy. Thus execution completeness alone
does not authorize a strength comparison.

## Interpreting follow-on imitation and PPO diagnostics

Use the already planned study outputs to diagnose the next limitation; these
checks do not add a new implementation work package.

- Compare persisted demonstration counts, completion markers and decision-phase
  coverage with the collector report. Missing or inconsistent records indicate
  a data-path problem before they indicate an optimization problem.
- Read held-out agreement alongside the number of non-forced choices in each
  phase. Strong aggregate agreement with weak choice-only agreement suggests
  phase imbalance or a forced-move majority; absent rare phases are a coverage
  limitation, not evidence that the student mastered them.
- If training improves but held-out errors or complete-game results do not,
  inspect target/action alignment and ordered sequence boundaries first. Current
  weights must produce current recurrent state; persisted teacher labels must
  not make cached student hidden states authoritative.
- If few supervised updates complete inside the budget, distinguish data scarcity
  from runtime cost using retained non-forced counts, completed epochs and update
  counts. The accepted implementation recomputes the whole episode prefix for
  each supervised chunk, so long episodes may make replay expensive. Profile
  that cost before choosing a later efficiency experiment; do not trade away
  current-weight recurrent correctness to make a pilot finish.
- Compare the teacher and student on the same validation board blocks, reporting
  censoring and finished paired-block count before interpreting win differences.
  If neither completes enough games, the result is inconclusive. If the teacher
  is weak too, improve or reassess the teacher before scaling student training.

If the planned balanced-imitation/fresh-PPO study still fails, choose at most
one or two focused diagnostics using existing study/audit machinery:

1. **Optimization versus coverage:** inspect train-versus-validation choice-only
   CE/ranking by phase and a small diverse fixed subset's fit. Failure to fit that
   subset points first to targets, sequence handling or optimization; fitting it
   without transfer points toward coverage or generalization. Match update and
   data budgets when comparing uniform and balanced runs. Neither result alone
   proves a need for a larger model.
2. **Strategic transfer:** repeat the fixed public-observation sampling rule on
   additional validation boards, preserving opportunity denominators for useful
   maritime conversions and feasible accept/reject decisions. Compare random and
   heuristic opponent evaluation separately to detect curriculum-specific gains
   or forgetting. Do not manufacture a preference label from teacher disagreement.

These are result-contingent diagnostics, not additional mandatory implementation
scope. Reserve final-test seeds and the 06:29 UTC integration boundary. Promote
only on credible held-out playing evidence; exploratory PPO remains separately
budgeted and honestly labeled while the imitation competence gate is unmet.

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
