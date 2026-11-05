# Behavioral cloning, public consequences and student-state labels

Date: November 5, 2026. Architectural design for mechanisms explicitly approved
by the user; no further user approval gate is required. This approved design is executed in the separate behavioral clone. Root schedules execution; the existing learning
coordinator remains the sole learning coordinator.

## Evidence and objective

Push behavioral cloning far enough to test whether this policy can reproduce the
heuristic's useful behavior, then test whether it can exceed that benchmark.
Neither outcome is promised. Uniform 20-epoch fits improved held-out choice
agreement from 32/83 to 43/83 and 40/83 without wins on the two known boards.
The training-only tiny-fit probe reached 76/101 and 77/101 correct after 100
epochs, but only 2/9 maritime labels and weak setup ranking. This is partial
optimization success, not a capacity diagnosis or a held-out performance result.

Sources: `/private/tmp/colonist-swarm-bxz1k39_/uniform-diagnostic/artifacts/report.md`,
`/private/tmp/colonist-swarm-bxz1k39_/uniform-diagnostic/artifacts/tiny-fit-probe/report.md`, baseline strategic
audit, and `/private/tmp/colonist-swarm-bxz1k39_/behavior-diagnostics/report.md`. The completed diagnostic on frozen
`8b8974a` found 5,000/5,000 labels replay exactly, 983/983 audited candidate
encodings match, and all 28 maritime/end targets have finite nonzero connected
gradients. Alternate valid RNG changes 387/5,000 teacher choices, including 342
exact ties. Report tie-aware diagnostics beside exact-index agreement without
rewriting stored labels. These findings do not establish an optimizer bug or a
capacity limit. Prefix replay consumed 68.1% of forward time; graph construction
was 40.4% of combined forward time (nested, not additive). Task 1 therefore targets
replay/input rebuilding first. Its matched profile/control fits were identical;
the tiny-prefix timing ratio is not an exact full-dataset speed prediction.

The accepted-source four-arm uniform/phase-balanced study subsequently completed
559.538/900 seconds with all 20 epochs/3,380 updates per arm. On its new four-board
panel, uniform seeds 11/29 won 0/16 with VP 3.625/3.500, balanced won 0/16 with VP
4.125/4.500, and the teacher won 3/16 with VP 6.5625. New held-out choice agreement
was 50/77 for both uniform arms versus 49/77 and 48/77 balanced. Thus balancing
modestly improved descriptive VP without wins or improved exact-label agreement.
Retain this control; do not rerun it or claim competence from it.

Inspection found learning HEAD `4bdb0ac` (documentation after accepted `8b8974a`)
with active curriculum changes in `training.py`, `evaluation.py`, and new
`curriculum.py`. Do not edit those files concurrently. The current worker owns
their completion/review; checkpoint changes below wait for its accepted commit.

## Chosen approach and alternatives

Use supervised correctness/throughput first, a separately switchable public
consequence scorer second, and explicitly versioned student-state labeling third.
Run controlled comparisons before feeding a selected supervised checkpoint to
the already planned curriculum. This separates optimization, representation and
state-distribution effects.

Two alternatives are deferred: simply increasing model size would not diagnose
the observed maritime/setup failures; starting with reward redesign would alter
PPO while the present failures already appear in reward-free supervised learning.
The current PPO reward is terminal winner reward plus `0.2` times the configured
VP/target potential difference, with terminal potential zero. It has no bank or
hand penalty. Preserve it throughout these ablations; later readiness shaping
requires a separate measured credit-assignment diagnosis.

## Non-negotiable boundaries

- Preserve existing Observation/Action schemas, legal-action order, old datasets,
  old checkpoints, CLI defaults, simulator rules and reward behavior.
- Teacher labels use only immutable Observation; no hidden hands, deck order,
  opponent private VP, future outcomes or engine objects enter actor features.
- CPU, one worker/thread, existing dependencies only; no paid services/install.
- Freeze source for each real run and retain component/config/checkpoint digests,
  all failed/interrupted arms and actual compute. No silent budget extensions.
- Preserve complete ordered student decision histories, including forced choices;
  teacher labels and executed behavior are different fields.
- Recompute recurrent prefixes under current weights before each optimizer update.
  Never cache learned embeddings or hidden states across updates.
- Keep training, held-out label validation, game validation and reserved test
  partitions disjoint. Validation never supplies gradients or DAgger labels.
- No new PPO collector or curriculum. Consume the existing reviewed interfaces.

## 1. Audited supervised training and measured throughput

First consume the read-only diagnostic report. Replaying the teacher requires its
original episode seed and call order: `HeuristicPolicy` has RNG jitter. A different
label from a freshly seeded teacher is not automatically corruption. Check exact
legal action-index mapping, serialization, actual candidate encoding/target/partner
distinctions and gradients by supervised head. The value head receives no BC
loss; forced samples update memory but contribute no CE. Near-zero gradients and
equal inputs need measured explanations, not assumptions about model capacity.

Factor the existing ordered fit into a reusable sequence fitter without changing
its batch-size-one default. An opt-in minibatch contains at most one supervised
chunk from each independent episode. Recompute every prefix with current weights,
detach at chunk entry, retain forced observations, sum weighted non-forced CE and
divide by the actual non-forced choice count. One minibatch yields one update.
An interrupted minibatch yields no update. Compare batch gradient/loss to a serial
sum over the identical minibatch, not to several sequential optimizer steps.

The measured profile supports caching immutable board topology/base input tensors in a
bounded cache keyed by full BoardSpec, player ordering where relevant, feature
schema, device and dtype. Clone mutable feature buffers. Add ownership/robber/hand
features and owner edges each call; degrees depending on dynamic links cannot be
cached as static. Batch independent graphs as disconnected components and keep
variable legal menus separate. Cached versus uncached output/gradient parity is
required. A memory-only prefix method may skip unused action/value heads while
recomputing the graph and GRU under current weights; compare its memory and
subsequent gradients against full forward. Preserve default numerical behavior;
batching is opt-in and measured before promotion. Profile CPU scopes for input
encoding, prefix replay, chunk forward,
backward and optimizer, reporting actual forwards/presentations/updates/seconds.

## 2. Opt-in public consequence architecture

Add a separate `ConsequenceGraphPolicy(GraphPolicy)` with architecture identity
`graph_consequence_v1` and feature schema `public-consequence-v1`. Copy all legacy
parameters exactly and add one **zero-initialized bias-free linear 34→1 branch**
to candidate logits, including trade-response logits. Values and recurrence remain
legacy-identical at initialization. Do not zero both layers of a multi-layer
branch, which could prevent useful gradients. Legacy `GraphPolicy` defaults and
checkpoint identity `{hidden_size: ...}` remain unchanged.

The feature function sees only `(Observation, legal Action)`. Supported hand
transitions initially are MARITIME_TRADE, ACCEPT_TRADE (offer perspective),
REJECT_TRADE and END_TURN; unsupported actions receive an all-zero vector, not a
fabricated post-state. A supported vector contains:

| Slots | Meaning |
| --- | --- |
| 0 | Known deterministic own-hand transition mask |
| 1–5 | Post-action own hand divided by 10 |
| 6–25 | Five deficits for each road/settlement/city/development recipe, divided by 4 |
| 26–29 | Resource-sufficient flags for those four recipes; not full legality |
| 30 | Public connected vacant settlement-site count, clipped at 8 and divided by 8 |
| 31 | Post-hand funds a settlement and a connected legal site/piece exists in normal play |
| 32 | At least one own level-one building and available city piece |
| 33 | Post-hand funds that city upgrade in normal play |

First test `feature_set="hand_recipe"`, which masks slots 30–33 to zero. Test
`feature_set="hand_recipe_and_connectivity"` separately only after the first
comparison; its additional four coefficients start at zero with identical other
weights. This isolates connected readiness from resource consequences. Both
feature-set identities are explicit in architecture metadata; no silent change
of an existing checkpoint's feature meaning is allowed.

Recipe order is road `(1,1,0,0,0)`, settlement `(1,1,1,1,0)`, city `(0,0,0,2,3)`,
development `(0,0,1,1,1)` in the shared resource order. Public site computation
uses occupied vertices, adjacent-building exclusion and an incident own road;
trades do not change geometry. Piece availability and ordinary post-roll build
phase are explicit. Readiness is a public resource/topology calculation under
normal-build assumptions, not permission to build after END_TURN or an incoming
response; it must not be reported as a newly verified legal action. The
recipe-only development flag does not infer deck stock.
Never infer bank inventory when it is unknown. Existing legal exchange actions
establish the first exchange's feasibility; no hypothetical simulator executes.

Checkpoint loader accepts the exact legacy architecture shape and the exact new
tagged shape `{kind, hidden_size, feature_schema, feature_set}`, rejecting unknown identities
or extra keys. Strict parameter loading prevents dropping an additive branch.
Old files stay byte-identical; new opt-in files explicitly identify the branch.
The accepted curriculum must load and clone either architecture through these
shared helpers, with no new curriculum implementation.

## 3. DAgger components with full student histories

Freeze a student checkpoint per collection round. On training boards, execute
student choices and query the audited heuristic on the same immutable observations.
Initial bounded variant uses student-only behavior, not an unrecorded expert mix.
Retain all ordered observations seen by the recurrent student, including forced
decisions and preceding observations before rare maritime/trade labels. Query
labels for every retained decision; supervise only non-forced ones. Do not splice
isolated disagreement rows into unrelated histories.

Use a separate `colonist-rl.student-labels` version-1 format, leaving the existing
`colonist-rl.demonstrations` version 1 untouched. Each sample stores Observation,
teacher_action_index, behavior_action_index, and behavior_executed. Both indices
must name legal actions in that same ordered menu. The last returned behavior may
be unexecuted after a deadline; record it honestly and perform no further mutation.
No rewards or PPO transitions are fabricated from these labels.

Episode provenance includes board/seat, behavior checkpoint/architecture digest,
behavior sampling seed/mode, teacher seed/configuration/call order and completion
reason. Store full histories, not cached student memory. Use the existing
FourPlayerGame/_MatchDriver for rules, proposals and opponents; collection only
orchestrates supervised labeling around that driver. Preserve source checks,
deadline boundaries, RNG/mode restoration and incremental atomic persistence.

A corpus manifest references immutable original-demo and student-label components
by format, SHA-256, source metadata and ordering. Each component retains its own
collection provenance. Do not concatenate episodes into a fake v1 dataset with one
invented source revision or rewritten seat schedule. The fitter projects validated
components to ordered supervision sequences and reports all component identities.
Known zero supervision, missing components, illegal labels or split overlap fail
closed. DAgger theory motivates data aggregation; it does not guarantee success
for this recurrent partial-information neural policy.

## 4. Controlled study and existing curriculum handoff

Use seeds 11 and 29, independent optimizers/RNG, and matched initial parameter
digests. First push the existing architecture's BC with the accepted weighting
recipe; report full train/validation phase and action-kind learning curves and
rare-action counts. Then compare legacy versus zero-additive consequence policy
on identical data, starts and presentations. Separately compare one student-state
aggregation round with equal-budget additional teacher-state labels, using the
same starting checkpoint, original-data mixture and teacher-query count. Keep
architecture fixed during that comparison. A second DAgger round is conditional
on useful first-round validation; do not silently grow the experiment matrix.

Proposed fresh candidates (must be checked against all earlier manifests before
the run): game validation `72010000..72010015` (16 boards/64 seat games per main
variant and initialization), held-out labels `72020000..72020015`, audit
`72030000..72030001`; new teacher training `73000000 + 100000*seed + i`, DAgger
training `83000000 + 100000*seed + i`, and teacher-state acquisition control
`93000000 + 100000*seed + i`, `0 <= i < 4096`. The explicit training partition also
retains all actual training boards in original components; component reuse must
not erase those seeds from leakage checks. Retain old fixed panels as historical
controls, not as extra independent evidence.

Use the same new game-validation panel in the later curriculum's configurable
`validation_board_seeds`. Main benchmark opponents are three fixed heuristics;
retain a random panel to diagnose progression/forgetting. Run random → one
heuristic/two random → all heuristic PPO **last**, via reviewed `run_curriculum`,
with fresh trajectories only. Do not change the old meaning of `baseline="mixed"`.
Keep reward and opponent policies fixed. A frozen selected BC reference remains
beside every PPO comparison.

Reserve final panel `71001000..71001007` until all checkpoint/settings selection is
frozen; verify it has not already been consumed. Legacy reserved seeds 70000017
and 70000019 remain untouched. If that panel was already consumed, root records
a replacement reservation before any new selection; never relabel used boards.
No post-test tuning. Complete uncensored matched blocks, both initialization
results, failed arms and actual budgets remain mandatory evidence.

Goals are high non-forced held-out agreement across adequately sampled phases,
better conversion/selectivity in public-state audits, and teacher-relative
complete-game wins/VP without regressions. Raw agreement is not strategy. A
64-game panel is a diagnostic target, not guaranteed statistical power; claim
cloning or beating the teacher only if board-level uncertainty and seed consistency
support it. Promotion never follows from a lower loss or a lone winning game.

Root declares a fresh explicit time/decision/presentation budget and integration
reserve before actual computation. This design does not extend any prior horizon.
Per-arm equal-work targets and equal wall ceilings are both recorded; interrupted
unequal arms remain incomplete rather than supporting a controlled comparison.
