# Catan agent: goal, milestones and research decisions

Revised 2026-11-06. This is the canonical direction for future work. It supersedes
algorithm-first training sequences in older roadmaps. Existing frozen experiments
retain their original methods, splits and advancement rules. The user's separate
[roadmap](roadmap.md), [previous direction](agent-research-roadmap-before-reset-2026-11-06.md)
and [measured results](training-status-2026-11-05.md) remain available.

## Top-level goal

Build a Catan decision maker that plays intelligently from information available
to its player and reproducibly outperforms authentic, pinned VF and AB references
in complete games under declared benchmark conditions. Deliver the same decision
maker through simulator and Colonist adapters, with physical-board input later.

The first milestone is reliable heuristic-level decisions and gameplay. The next
is stronger strategic play. A neural policy, a public-information planner, an
improved heuristic or a measured combination may achieve either milestone.
Behavioral cloning and PPO are tools, not completion criteria. The project does
not succeed merely because it trains, imitates a teacher, accumulates games or
executes legal screen actions.

Each research step must answer: **which decision gets better, why should this
change improve it, and do fresh games support that claim?** Winning is the final
objective. Decision audits explain successes and failures; loss, agreement, VP,
trade counts and value prediction are supporting measurements.

## What the evidence changes

- The fixed eligibility-feature comparison completed 384 games but failed its
  primary advancement rule. Every candidate won 2/16 against trading H. Retire
  this unchanged treatment; more updates are not an explanation for its failure.
- Selected trade/pass probes already found consistent labels, encodings and
  relevant gradients in that subset. The parent includes 30 hand/recipe consequence
  channels. Check existing coverage before proposing duplicate features or tests;
  no general disconnected-learning defect has been demonstrated.
- Extra imitation, outcome-residual and placement studies have not established
  reliable strength gains. Diagnose the mechanism before repeating or scaling them.
- The native focal pilot's raw/VF/AB2 results were 3/0/4 wins per 16 games. These
  four board clusters establish no superiority over both references.
- The explanatory audit found 17 reasonable and two ambiguous unique cases, but
  covered setup, bank/pass and robber only. It does not establish development-card,
  discard, domestic-response, race or endgame competence.
- All 48 planned student games and 4,729 own inputs were saved. Finalization failed
  on a controller temporary-file accounting race. A separate zero-query recovery
  passed all 9,458 strict before/after pairs and independent metadata review. The
  original failure remains. Fixed sampling selected 88 cases (59 TRAIN, 29 labelVAL)
  and preserved eight empty slots. Independent first passes are starting; labels
  and data benefit remain unestablished.
- The sparse accepted-target fitter and the 361-input engineering cache repair
  passed their scoped checks. These are usable mechanisms, not evidence of a
  stronger player. No candidate is promoted.
- The terminal-value collector produced 208 completed TRAIN games before block 53
  failed its guard-registration handshake, before its game worker started. The
  original 320-game cohort is rejected; no value fitting follows. Diagnose launch
  accounting/timing and declare any successor separately, retaining the fixed
  roles and original failure.

## Milestone 1: make the decisions recognizably good

Start with complete histories from the policy we actually deploy. Inspect its
legal alternatives and consequences, including decisions where a pass is correct.
Identify repeated, costly errors before choosing a training intervention. Judge
the teacher independently; closer imitation of a bad answer is regression.

Maintain a decision scorecard covering eight groups:

| Decision group | What a useful audit must examine |
|---|---|
| Initial placement | Production, complementary resources, scarcity, ports, expansion and blocking. |
| Bank trade, build and pass | Useful affordable follow-ups, trade sequences, opportunity cost, cycles and sensible passes. |
| Domestic responses | Accept/reject value, alternatives, opponent benefit and uncertain resources. |
| Robber and discard | Production denial, steal choices, hand preservation and legal discard constraints. |
| Development-card purchase | Buying versus building/trading/passing, draw uncertainty and play delay. |
| Development-card play | Knight, Monopoly, Road Building and Year of Plenty, including multi-step resolution. |
| Road and army races | Reachability, contest cost, award thresholds and the chance an opponent takes the award. |
| VP and endgame | Hidden own VP, immediate winning actions, timing and denying an imminent opponent win. |

Record coverage and disagreements explicitly. Use public before-state/history
packets, independent first-pass judgments and accepted sets of good alternatives.
Each judge context sees at most one anchor per episode; freeze both independent
first passes before revealing the student choice or teacher answer. Separate
ambiguous choices from demonstrably bad actions. EVAL observations never become
TRAIN labels. Public logs constrain resource beliefs; hidden steals and
unknown cards prevent perfect knowledge of opponents' hands.

For each failure, inspect legality, action encoding and target consistency first;
inspect relevant gradients when the failing component is learned. Then distinguish
bad supervision, student-state coverage, missing information, insufficient planning
and optimization trouble.
Choose the smallest remedy that addresses the diagnosed cause:

- Bad targets: correct or omit them; preserve acceptable alternatives.
- Distribution mismatch: aggregate audited labels on states the student reaches.
- Hard consequence arithmetic: test separate public features for missing build
  resources, connected affordable settlements and useful follow-ups after trades.
- Short-sighted choices: compare bounded action sequences or tactical rules.
- Optimization trouble: profile and repair the demonstrated training problem.

Bank trades remain legal whenever the rules permit them. Strict GreedyNoTrade
intentionally declines them; trading H is a separate baseline with persistent
state and a declared offer regime. Neither trade frequency nor avoiding trades
is the target. Resource values depend on the board, hand, ports, stage and
opponents. Scarcity priors are hypotheses to test, not universal labels or rewards.

A heuristic-competence claim requires an independent audit across all eight
groups from preselected fresh EVAL games, plus fresh all-seat gameplay against
strict GreedyNoTrade and trading H. Do not substitute label-validation or exposed
diagnostic cases for that audit; its observations and judgments stay out of TRAIN.
Before that evaluation, freeze coverage counts, reasonable/bad decision thresholds, critical-error rules, practical noninferiority margin, board count and
board-level uncertainty method. Missed immediate wins or destructive loops block
competence until resolved; absent coverage remains unknown. A 25% win rate or a
nonsignificant comparison does not establish parity.

## Milestone 2: improve planning and learn consequences

Pursue two testable mechanisms without making either a mandatory dependency of
the other. Start from diagnosed weaknesses, keep controls matched and change one
mechanism at a time.

**Planning:** first test a bounded tactical remedy from fresh TRAIN failures.
Development-card purchase is a candidate, not a presumed defect. Test sensible
non-purchases too. For cross-turn planning, verify public transition fidelity and
measure runtime before comparing it with the existing mainly post-roll planner.
Keep scorer, fallback and compute limits fixed in that comparison; do not bundle
an untested value head into a larger search treatment.

**Value learning:** complete the prepared true-terminal-outcome experiment with
256 TRAIN and 64 validation games, including its nested 128-game data-size
contrast. Compare public-feature and frozen-recurrent heads with matched work,
calibrated constant controls and held-out prediction gates. Claim a recurrent
encoder benefit only if it also beats the public-feature head. Keep engineering
boards excluded. A value estimate under one continuation policy is not optimal
value. Better prediction earns a separate gameplay deployment test; it does not
by itself establish better decisions or justify a larger model.

Search uses sampled worlds consistent with permitted observations and joint
resource beliefs, independent future randomness, and opponents' own observations.
The real hidden Game, development deck and future RNG never enter the policy,
teacher or planner. Validate branch observations before learning from them.

Profile before optimizing throughput. Batch independent sequences and cache static
board tensors when measured costs warrant it. Trainable recurrent prefixes must
remain consistent with current weights. Cross-update embedding caches are valid
only for frozen encoders. Faster work with stale memory is not progress.

## Milestone 3: exceed stronger opponents reproducibly

Maintain a fixed evaluation panel while choosing training opponents from the
current failure pattern. Random opponents are an entry-level diagnostic; strict
GreedyNoTrade, trading H, mixed opponents and frozen stronger policies supply
progressively harder challenges. Advance when a predeclared stage passes, and
check retention on earlier panels. Do not keep returning to random play once it
stops testing the weakness of interest.

Compare the strongest credible candidate directly with focal VF and focal AB;
this comparison does not require PPO. Use fresh paired boards, rotate all seats,
match background opponents and offer rules, and report whole-board uncertainty.
Before confirmation, freeze the minimum practical effect and an uncertainty rule
that supports superiority against both focal references. Require the declared
lower bounds for both candidate-minus-reference contrasts to exceed that margin,
with replication on disjoint fresh boards, before claiming reproducible superiority.
Positive point estimates alone do not pass.
Three-native stress panels answer a different question and cannot substitute for
these focal comparisons. A failed primary result cannot be rescued by VP or a
secondary opponent panel.

PPO or another policy-improvement method becomes useful when an already competent
policy has an identified limitation in longer-term strategy that targets and
bounded planning do not resolve. Start from the strongest validated policy, use
progressively harder opponents and retain the fixed panel. PPO is optional even
then. Scale data, model or training duration only after a measured benefit or a
new, falsifiable mechanism.

Keep terminal winning as the initial objective. Do not reward trade frequency,
development-card purchases or VP growth simply to make curves rise. New shaping
requires a specific diagnosed problem, an isolated comparison and checks that it
does not reward loops or displace winning. Existing experimental PPO shaping is
not automatically adopted by new studies.

## Milestone 4: transfer the decision maker reliably

Keep the shared player-observable Observation/Action contract independent of the
simulator and screen implementation. Catanatron is a backend, not permission to
expose privileged state. Separate extraction, belief updates, legal action mapping,
staleness checks and visible-effect confirmation. Keep Colonist observations and
control examples durable so future work does not need to recreate the demo.

Maintain adapter correctness alongside research; expand it when a validated policy
needs the capability. Incoming offers and maritime trades already have distinct
actions; initiating/countering domestic offers needs its own interface and tests.
Physical recognition, six-player learning and broad adapter refactoring follow
stronger four-player play. Live adapter success and simulator strength are separate
claims; test both before claiming a capable Colonist player.

## The next execution queue

Keep one primary decision-improvement experiment and one bounded independent
research experiment active. Independent source review may run alongside them.
Avoid creating new frameworks, review layers or broad rewrites where an affected-
path check and the existing harness suffice.

| Priority | Next concrete deliverable | Advance or stop decision |
|---|---|---|
| Primary: use saved student experience | Judge the 88 selected cases from the frozen 96-slot inventory in two independent panels, with isolated contexts and no choice/teacher reveal before both complete first passes. | Recovery grants no target approval. Audit targets and missing coverage before fitting. Never rerun gameplay merely to erase finalization failure. |
| Primary: test better data | Complete the frozen added-data-only comparison with the reviewed sparse fitter, unchanged parent and equal-work old-data control. | Keep its original advancement rules. Report an insufficient target/coverage gate instead of replacing cases. A null result prompts diagnosis, not automatic scale-up. |
| Independent: test longer consequences | Assess the stopped 320-game attempt and fix the demonstrated launch-timing bottleneck prospectively. Any successor must retain the fixed roles, account for all 320 games and preserve the original failure. | 208 completed TRAIN games are saved, not a complete accepted cohort. No automatic retry or fit. Prediction must pass before a separate policy deployment experiment. |
| Next decision remedy | Rank fresh TRAIN errors by recurrence, strategic cost and confidence; choose one tactical repair or planning experiment. | Audit the action change and test fresh paired games. Preserve a negative result and change the hypothesis. |
| After credible competence | Freeze the best candidate and run focal VF/AB comparisons and independent confirmation. | Promote only on the declared primary strength gates. PPO is not a prerequisite. |

Do not silently rewrite frozen protocols to match this strategy. A changed
experiment needs a separately declared successor; completed nulls, failures and
resource HOLDs remain in the record. Reuse existing data within its declared role.
Do not retune on EVAL, replace inconvenient cases or choose a lucky seed after
inspection. Final boards `123100000..123100063` and legacy finals remain untouched
until their confirmation protocol is frozen.

## Ownership, constraints and reporting

The learning coordinator owns failure diagnosis, data, teachers, planning, fits
and the strength queue. The adapter coordinator owns faithful translation and
visible execution. An independent research/review lane challenges the next
hypothesis and unsupported claims. Each task states its question, control, budget,
measurement and advance/stop rule. Report brief progress as decisions learned and
actual work completed, separating collected games, audited labels, fitted updates,
engineering checks and playing-strength evidence.

Use pinned Catanatron `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`.
Authentic native VF defaults/weights and AB depth 2 with `prunning=False` stay
privileged-state, evaluation-only references. Disclose information asymmetry,
`+atomic_discard_reject_trade_v1`, synthetic rejection, no-offer native panels,
chance-semantics mismatch and `full_rule_equivalence=false`. Their privileged
choices never become learner or teacher inputs.

Reuse existing process/deadline guards: at most two heavy workers, CPU/Torch
intra/inter-op one and 2 GiB per worker. Inference/native workers require sampled
OS1; prospective gradient fits use the separately reviewed OS-at-most-four policy
with autograd multithreading disabled. Historical strict-OS1 HOLD remains. Pin
source/runtime/models/schedules and preserve actual work and failures. No paid
compute is assumed. The next unit of work must resolve a decision-quality or
strength question, or remove a specific blocker to that measurement.
