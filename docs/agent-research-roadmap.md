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

The first deliverable is a sensible, deployable player whose trades, builds,
passes and tactical choices are at least heuristic level. The next is stronger
strategic play. A neural policy, a public-information planner, an improved
heuristic or a measured combination may achieve either milestone.
Behavioral cloning and PPO are tools, not completion criteria. The project does
not succeed merely because it trains, imitates a teacher, accumulates games or
executes legal screen actions.

The deliverable is the complete decision system: a permitted observation and
history enter; one legal action and its updated memory leave. Rules, search,
beliefs and learned estimates may cooperate inside that system. Keep the
strongest validated incumbent as the control; the existing neural parent is not
automatically the deployment candidate. A simpler policy that plays better is
progress. An improvement must fit a declared per-decision compute budget and
remain usable through the shared adapter contract.

Success has three separate claims: competent decisions, reproducible benchmark
strength, and reliable live execution. Evidence for one does not prove the others.
The final strength claim requires superiority to both pinned references with
independent confirmation; the final product claim also requires a complete live
game through the adapter. Until then, report exactly which claim has been tested.

Each research step must answer: **which decision gets better, why should this
change improve it, and do fresh games support that claim?** Winning is the final
objective. Decision audits explain successes and failures; loss, agreement, VP,
trade counts and value prediction are supporting measurements.

## What we should pursue

Prioritize an effective decision system over a particular learning algorithm.
Use the simplest credible policy as the starting point, then improve the part
that produces a demonstrated costly mistake. A small tactical rule or better
search score can be the right deliverable; it need not wait for a neural replica.

The normal research cycle is: **observe a weakness → explain its cause → change
one mechanism → inspect the changed choices → test fresh complete games**.
Each cycle ends with a decision to retain, revise or retire the intervention.
Good training loss, more labels and faster infrastructure do not close that cycle.

Work in this order:

1. Fix demonstrable tactical and economic weaknesses: blocked road expansion,
   useful trade/build sequences, sensible passes, immediate wins, and development
   cards with concrete benefits. Preserve reasonable alternatives and uncertainty.
2. Improve planning where a current choice requires looking beyond one action.
   Use public resource beliefs, board scarcity, ports, build costs and opponent
   incentives. Treat static resource valuations as priors to test.
3. Learn reusable estimates or policy choices when they improve that decision
   process. BC needs sound targets on relevant student states; value learning
   needs a plausible deployment use. Neither is a mandatory stage.
4. Test against progressively stronger opponents, then confirm superiority to
   both VF and AB. Use RL for a demonstrated remaining strategic limitation when
   it is a better experiment than another targeted repair or bounded search.

Keep a thin adapter lane running alongside the decision work. Screen control,
physical recognition and simulator integration translate the same core policy;
they do not each contain a separate strategy.

Allocate most effort to closing this decision-to-gameplay loop. Use infrastructure
work only to unblock the next concrete comparison. Do not scale a treatment
because it is available, preserve it because implementation was expensive, or
collect more examples without stating what decision they can change. Retire an
unchanged treatment after its declared failure; a successor needs a different,
testable explanation.

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
  and preserved eight empty slots. All 176 first passes and 49 adjudications are
  frozen. The informative TRAIN upper bound is six anchors across four strata,
  below the fixed 32/six gate: no fit ran. This does not establish BC failure.
  Public fact checks confirm two student roads end in enemy-blocked expansion
  leaves, while their possible award/defensive value remains uncertain. A flagged
  trade has an omitted wheat-funded city route and is not certainly bad. The
  cached teacher selected reasonable alternatives on both road cases; these cases
  do not establish a teacher failure.
- The resulting narrow road correction passed 22 geometry/wrapper tests and
  matched both teacher alternatives in exposed TRAIN diagnostics. Its frozen
  128-game study completed 114 games, with two unfinished and twelve unstarted.
  H wins were parent 3/32 versus correction 2/32, paired difference −3.125 points,
  bootstrap 95% [−9.375, 0]. Both GNT arms hit 512 own inputs in the same game;
  their whole-panel rates remain undefined. Retire this unchanged treatment.
  Plausible local choices did not establish a stronger player.
  [Full result and limits](road-continuation-experiment-2026-11-06.md).
- The sparse accepted-target fitter and the 361-input engineering cache repair
  passed their scoped checks. These are usable mechanisms, not evidence of a
  stronger player. No candidate is promoted.
- A TRAIN-only scan found 24 near-win BUY+trade observations where the raw public
  planner skipped search. Nine have conditional one-trade/build resource and
  placement witnesses; fifteen do not. The first positive could trade two wheat
  for brick and afford a connected final-point settlement. These are public
  hypotheses, not engine-certified missed wins. Test that exact mechanism before
  changing general development-card preferences.
- The original terminal-value collector stopped after 208 TRAIN games on a
  guard-registration failure before block 53's worker started. That failed attempt
  remains rejected. A separately declared successor completed the missing 48
  TRAIN and 64 validation games without replaying the retained 208. Independent
  component and mixed-index reviews passed for the honest two-attempt corpus:
  256 TRAIN, 64 validation, with nested TRAIN128. Caching and all four fits remain
  unstarted. Three cache/head source findings still require correction; neither
  dataset completion nor helper tests establish predictive or playing strength.

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

This broad competence gate is for the broad claim. A narrow road, trade or card
repair can be developed and tested first with relevant adverse cases and fresh
paired games. Do not build another general audit system as a prerequisite to
testing one well-defined improvement.

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

**Value learning:** use the completed, independently reviewed mixed320 inputs to
test a concrete planning hypothesis: a public outcome estimate may rank projected
trade/build/road leaves better than handcraft scoring. The existing public
`PositionEvaluator(root, ProjectedTurn, handcraft_micro)` seam does not need an
imagined observation or recurrent update. Correct the bounded helper findings,
verify them, then separately freeze actual cache and fit execution. The study
uses 256 TRAIN and 64 validation games, including a nested
128-game data-size contrast. Compare public-feature and frozen-recurrent heads
with matched work, calibrated constant controls and held-out prediction gates. Claim a recurrent
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
Observation semantics, beliefs, valuation, planning and policy live in the shared
core. Simulator, Colonist and future physical-board adapters provide observations,
translate actions and confirm effects through that contract.

Maintain adapter correctness alongside research; expand it when a validated policy
needs the capability. Incoming offers and maritime trades already have distinct
actions; initiating/countering domestic offers needs its own interface and tests.
Physical recognition, six-player learning and broad adapter refactoring follow
stronger four-player play. Live adapter success and simulator strength are separate
claims; test both before claiming a capable Colonist player.

## The next execution queue

Keep one primary decision-improvement experiment active. One bounded independent
research experiment may run alongside it when it answers a distinct decision
question. Independent source review may run alongside them.
Avoid creating new frameworks, review layers or broad rewrites where an affected-
path check and the existing harness suffice.

| Priority | Next concrete deliverable | Advance or stop decision |
|---|---|---|
| Primary: winning trade/build prefix | Verify the first positive of nine conditional TRAIN witnesses against actual supported transitions. Test the narrow opt-in `search_build_win_before_buy` path: when BUY is the fallback near victory, return a different legal first action only for a certified building-win sequence. Nine new external tests are prepared, not yet run. | Check public transition fidelity, budget fallback and unchanged default behavior. If the witness fails, retain BUY and record why. If confirmed, inspect changed decisions and run a separately frozen fresh paired gameplay test. No ordinary score gain may masquerade as a certified win. |
| Independent study: value for planning | The 320 inputs and mixed index passed scoped independent review. Resolve the three helper defects (published byte binding, stable saturated log loss, empty diagnostic cells), verify the changed paths, then freeze the single cache and four fixed fits. | Preserve TRAIN256/VAL64/nested128, raw/no-offers continuation identity and original failed attempt. Follow the original prediction gates without tuning. Only a justified candidate gets a later gameplay comparison with matched raw and calibrated-handcraft controls. |
| Following remedy | Use TRAIN evidence to select the next economic or tactical weakness: trade/build/pass, development cards, awards or immediate wins. | Change one mechanism; check the choices and fresh gameplay. Preserve a negative result and change the hypothesis. |
| After credible competence | Freeze the best candidate and run focal VF/AB comparisons and independent confirmation. | Promote only on the declared primary strength gates. PPO is not a prerequisite. |

The audited-data study is closed with insufficient informative coverage: all 225
judgments remain preserved, with zero optimizer updates and no BC-failure claim.
The unchanged road correction is retired. Do not reopen either by relaxing its
gate. Broad imitation, PPO, reward shaping, larger models and physical-board
recognition are deferred options, not a parallel backlog to execute automatically.

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
