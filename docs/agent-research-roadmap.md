# Catan agent: goal and research priorities

Revised 2026-11-06. This is the canonical project direction. Exact results belong
in [current status](training-status-2026-11-05.md); experiment protocols and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) retain their
original commitments. The user's separate [roadmap](roadmap.md) is preserved.

## The goal

Build a strong Catan decision maker from information available to its player.
First demonstrate consistently sensible play against strict GreedyNoTrade and
trading H. Then demonstrate reproducible complete-game superiority over both
authentic, version-pinned VF and AB under explicitly declared benchmark conditions.
Deliver the resulting decision maker through a shared Observation/Action interface
for simulator, Colonist and eventual physical-board adapters.

The research question is: **what information, training signal and planning are
needed to turn a legal action into a good Catan decision?** Judge progress by
decision quality and wins on fresh boards. Teacher agreement, lower training loss,
more games, more trades, higher VP and completed infrastructure are diagnostics.
None alone establishes playing strength.

No particular learning algorithm is the goal. Behavioral cloning can initialize
or repair decisions; audited student-state aggregation can improve its training
distribution; terminal-outcome learning can estimate longer consequences; public
search can compare plans. PPO becomes an option after heuristic competence, when
there is a concrete reason to expect policy optimization to help. It is not a
required next stage or a substitute for understanding obvious mistakes.

## What changes now

1. **Organize work around costly decision failures.** Use fresh student games to
   find a repeated mistake, establish why it is bad, and test one targeted remedy.
   Broader training budgets follow a measured benefit.
2. **Improve the teaching signal before demanding closer imitation.** A heuristic
   proposal is a candidate answer. Check legal alternatives, useful passes and
   consequences; preserve multiple acceptable actions and uncertainty.
3. **Keep learning and planning as competing, testable tools.** The separately
   controlled audited-data comparison remains required. Its result does not gate
   independent outcome-value research. Longer search earns investment through
   transition fidelity, cost and a gameplay contrast.
4. **Use an evaluation ladder.** Cheap mechanical and tactical checks come first,
   then paired development games, then independent held-out competence and native
   reference confirmation. Reuse applicable checks; keep review proportional to
   the decision and avoid making provenance work the dominant research activity.
5. **Keep the adapter thin.** Maintain reliable observation/action translation
   while learning gets most attention. Local strength work can proceed before
   live Colonist automation is complete.

## What we know, and what we do not

- Selected trade/pass probes found consistent labels, action encodings and
  relevant gradients in that subset. The parent already has 30 hand/recipe
  consequence channels. There is no demonstrated general disconnected-learning
  defect to repair by blindly increasing training.
- The fixed eligibility-feature treatment failed its primary advancement rule
  in 384 complete games. All six candidates won 2/16 against trading H. Secondary
  refit gains do not establish a feature benefit. Retire this unchanged treatment.
- Earlier extra imitation, outcome-residual and placement interventions did not
  establish reliable gameplay gains. These nulls limit those interventions;
  they do not establish that imitation, value learning or search cannot work.
- The separate focal-reference pilot scored raw/VF/AB2 3/0/4 wins per 16. Four
  board clusters do not establish superiority over both references.
- The explanatory EVAL audit judged 17 of 19 unique cases reasonable and two
  ambiguous, but covered setup, bank/pass and robber only. Development-card,
  discard and domestic-response competence remain unestablished.
- Four engineering games retained 361 complete own inputs. The cache-only repair
  passed independent engineering review after the original fingerprint failure.
  This establishes a usable history/cache path, not a trained value model.
- The accepted-alternative fitter now passes focused and legacy regressions plus
  independent source review after a reproduced numerical issue was repaired.
  Collection, independent target approval and real-data benefit remain untested.

No candidate is promoted. No VF/AB superiority or broad heuristic competence has
been demonstrated. Preserve failed attempts, resource HOLDs and null outcomes.

## First milestone: trustworthy heuristic-level decisions

Use one improvement loop: collect complete current-student histories, inspect
public decision packets, identify a costly recurring failure, choose the smallest
remedy, and compare it with the unchanged parent and an equal-work control.

Cover eight decision groups: initial placement; bank trade/build/pass; domestic
response; robber/discard; development-card purchase; development-card play;
road/army races; and held VP/endgame. Road Building and Year of Plenty require
their actual multi-step actions. Track unobserved groups explicitly rather than
letting abundant setup cases stand in for tactical coverage.

Bank trades stay legal whenever the engine permits them. Strict GreedyNoTrade
deliberately declines them and supplies a useful weak baseline. Trading H is a
different control, with persistent state and its declared offer regime. Neither
trading often nor trading rarely is a quality target. Check whether a trade
enables a legal useful build, preserves better options, wastes resources or
cycles; check passes for legal immediate wins and affordable useful actions.

Audit the teacher as well as the student. First check rule legality, action
encoding, consequences, labels and gradients. Then distinguish deficient targets,
distribution mismatch, representation limits and optimization failure. Use
accepted sets of good alternatives; ambiguous or unapproved rows provide history
without new supervised loss. New targets must be explicitly approved for their
TRAIN or validation role. EVAL judgments never become training labels.

Use scarcity and resource valuations as state-dependent hypotheses. Placement
must consider production, diversity, access, expansion and opportunity cost;
trade value depends on hand, ports, build opportunities, opponents and game stage.
Published average valuations can inform candidate features or teacher design,
but do not become universal labels or rewards without a controlled test.

The prepared current-student experiment uses unchanged-student rollouts, two
independent judges and adjudication, a sparse accepted-target loss, and identical
architecture/features/update work between old-data and added-data arms. Preserve
its fixed sampling rules and report missing coverage without replacement. It is
a narrow data-effect test; it cannot certify all eight decision groups.

Claim heuristic competence only after both a predeclared held-out decision audit
and fresh all-seat gameplay gates against GreedyNoTrade/trading H pass. Freeze
coverage, reasonable-decision thresholds, practical noninferiority margin, board
count and uncertainty rule before evaluation. A missed immediate win or destructive
loop blocks competence until resolved. Missing coverage blocks a broad claim.
Neither a 25% win rate nor a nonsignificant difference establishes parity.

## Second milestone: learn longer consequences

Pursue true terminal-outcome value learning as an independent bounded experiment.
Use one frozen public continuation policy and declared offer regime. Save every
real own observation, including forced decisions, and the eventual win. Keep
TRAIN and validation boards disjoint, weight games equally and cache frozen
recurrent embeddings once from complete histories. Do not fabricate observations
for missing prefixes or projected branches.

The prepared study compares public-feature and frozen-recurrent heads on identical
256 TRAIN/64 validation games, with a nested 128-TRAIN-game contrast and equal fit
work. The engineering cache repair makes preparation feasible; main collection
and fitting still require their reviewed source, schedule and resource bindings.
The engineering board remains excluded from both splits.

Require each learned head to improve held-out calibration/prediction over its
calibrated constant control. Claim an incremental recurrent-encoder benefit only
if it also beats the public-feature head. A successful public-feature head can
earn its own frozen deployment experiment. A value under one continuation policy
is not optimal value; prediction success does not imply better actions. Deploy
only through a validated public transition interface and require a separate paired
gameplay improvement. If prediction fails, diagnose the signal before scaling data
or model size.

Profile meaningful costs before optimizing throughput. Batch independent sequences
and cache static board tensors when measurement warrants it. Recurrent prefixes
must be recomputed with current trainable weights; only frozen encoders permit
cross-update embedding caches. Faster training with stale memory is not progress.

## Third milestone: improve planning and tactical coverage

Start with one bounded tactical intervention, selected from fresh TRAIN evidence.
Development-card purchase is a prepared candidate: compare buying against build,
trade and pass using public draw probabilities, opportunity costs and play-delay
rules. Audit sensible non-purchases too. Test owned-card tactics, robber targeting
and road/army races separately so a gain has an identifiable cause.

Then extend the current mainly post-roll planner across production and other
players' turns. Verify public transition fidelity, legal actions, privacy, resets
and runtime before a gameplay test. Profile the proposed eight sampled scenarios,
four root candidates and eight public transitions; these are provisional bounds,
not measured capacity. Keep scorer, fallback and limits fixed in the first
current-turn versus cross-turn comparison. Do not bundle a new value head into it.

Sample worlds from permitted observations/history and joint resource beliefs,
with independent future randomness. Simulated opponents receive their own
observations. Never pass the real hidden Game, actual deck or future RNG to the
policy, teacher or search. Validate branch observations before learning from them.

## Later learning and deployment

Once sensible decisions and fresh heuristic-level gameplay are established,
consider PPO or another policy-improvement method for longer-term tradeoffs that
the existing teachers and bounded planner miss. Start with an already competent
policy. Use progressively harder opponents only when the earlier stage passes,
and retain a fixed evaluation panel to detect regressions. Beating random agents
alone is insufficient motivation for a new training phase.

Keep terminal winning as the initial research objective. Existing experimental
PPO shaping is not automatically adopted by new studies. A new shaped reward
needs a demonstrated failure, an isolated comparison and checks for exploitation
or changing what winning means. More trades, purchases or VP cannot rescue a
failed win-rate comparison.

Colonist and physical adapters share the player-observable Observation/Action
contract. Catanatron is the current simulator backend, not an excuse to expose its
privileged state. Keep extraction, belief updates, action mapping, staleness and
visible-effect checks separate from policy learning. Initiating/countering domestic
offers needs its own interface; maritime trades and incoming responses already
exist. Public logs constrain resource beliefs but cannot reveal unobserved steals
or supply perfect hidden hands. Physical recognition, six-player learning and
broad adapter refactoring remain later work.

## Immediate execution order

| Order | Deliverable | Decision it enables |
|---|---|---|
| Primary data lane | Use the reviewed sparse fitter to collect/audit the prepared current-student cases, then run the frozen added-data-only contrast. | Whether better on-policy targets improve decisions and games at equal work. |
| Parallel bounded lane | Complete the role-aware terminal-history collector and reviewed 320-game schedule; then fit the frozen value/data-size contrasts. | Whether real outcomes provide useful predictive information beyond existing public features. |
| Parallel small remedy | Choose one tactical repair from fresh TRAIN failures and test it alone; it need not wait for sparse fitting or the data-only result. | Whether a concrete decision gap can be closed without broad retraining. |
| After transition feasibility | Verify and profile public cross-turn transitions; compare bounded search with the current planner. | Whether planning farther pays for its cost. |
| After competence | Compare the strongest frozen candidate directly with focal VF/AB and replicate; consider policy optimization independently when justified. | Whether the agent advances beyond heuristic play, with no PPO prerequisite. |

Implementation readiness does not mean an experiment has run. Report collected
games, fitted updates, audits and strength results separately. Archive closed
evidence durably and keep a short restart record. Learning coordination owns the
queue/data/strength tests; adapter coordination owns reliable translation; an
independent review lane challenges both the next hypothesis and the claims.

## How an intervention earns more resources

| Stage | Evidence required |
|---|---|
| Correctness and feasibility | Affected-path tests, public-input/history checks and measured bounded cost. |
| Development signal | One isolated intervention, matched controls, fresh paired all-seat games and whole-board uncertainty. |
| Heuristic competence | Independent held-out decision coverage plus predeclared GreedyNoTrade/trading-H game gates. |
| Beyond reference strength | Positive candidate-minus-focal-VF and candidate-minus-focal-AB contrasts under declared matched backgrounds; replication on fresh boards. |
| Final confirmation | Frozen model/source/rule/limits/practical effect and uncertainty rule; untouched final boards used once. |

Preserve final boards `123100000..123100063` and legacy finals. Keep TRAIN,
validation, exposed development EVAL and final confirmation roles distinct. Do not
retune on a failed EVAL cohort, replace inconvenient cases, choose a lucky seed
after inspection or rescue a primary failure with a secondary panel. An
inconclusive result remains inconclusive. Scale only when the primary result or
a clearly new diagnosed mechanism justifies the next experiment.

Use pinned Catanatron `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`.
Native VF weights/defaults and AB depth 2 with `prunning=False` remain authentic
evaluation-only privileged-state references. Disclose the information asymmetry,
`+atomic_discard_reject_trade_v1`, synthetic rejection, no-offer native panels,
chance-semantics mismatch and `full_rule_equivalence=false`. Three-native stress
panels are distinct from focal-reference comparisons. Never train on native
privileged choices or silently change the reference.

Reuse existing process/deadline guards: at most two heavy workers, CPU/Torch
intra/inter-op one and 2 GiB each. Inference/native workers require sampled OS1;
prospective gradient fits use their separately reviewed OS-at-most-four policy
with autograd multithreading disabled. Historical strict-OS1 HOLD remains. Pin
source/runtime/models/schedules, preserve actual work and failures, and reuse
applicable completed reviews. No paid compute is assumed. Spend the next unit of
work on a falsifiable decision-quality or strength question.
