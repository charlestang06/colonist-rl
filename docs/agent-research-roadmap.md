# Catan agent: goal and research priorities

Revised 2026-11-06. This is the canonical direction; detailed past results live in
[current results](training-status-2026-11-05.md) and the
[research history](../runs/swarm-2026-11-04/research-roadmap-history/).
The user's separate [roadmap](roadmap.md) is preserved.

## The goal

Build a Catan agent that makes sensible, explainable decisions from information
available to its player, generalizes across boards and seats, and reproducibly
outperforms authentic, version-pinned VF and AB in complete games. Deliver that
agent through a shared Observation/Action interface that can later serve Colonist
and physical-board adapters.

Playing strength and decision quality determine progress. Imitation accuracy,
training volume, infrastructure completion and higher VP alone do not. Heuristic
competence is the first development milestone; exceeding it is the research goal.
BC is one possible warm start or repair tool. It need not succeed before we test
outcome-value learning or public search. PPO is a later option, once decisions are
at least credibly heuristic-level and evidence identifies a reason to use it.

The policy may use its hand/cards, visible board, public events, its own complete
observation history and beliefs derived from those inputs. Opponent hands, true
deck order and future RNG cannot enter policy, teacher or search inputs. Native
VF/AB are privileged-state evaluation references; report that information asymmetry.

## What the evidence tells us

- The selected trade/pass replay found consistent labels, action encodings and
  relevant gradients. It did not expose a disconnected-learning defect in that
  subset. The parent already has 30 hand/recipe consequence features.
- Four fixed-data eligibility-feature fits completed with independent audit.
  Extra channels gave no validation exact-label gain over control; complete-game
  comparison remains necessary. There is no basis for selecting a fitted arm yet.
- More BC agreement, the earlier outcome residuals and the 256-game placement
  intervention did not establish reliable gameplay improvement. Do not repeat
  those unchanged interventions simply at larger scale.
- The fresh matched native pilot scored raw 3/16, VF 0/16 and AB2 4/16. Raw-minus-AB
  was −6.25 percentage points, with interval [−25,12.5] over four board clusters.
  This does not establish superiority over both references or nominate a finalist.
- Existing 96 TRAIN/32 validation outcome episodes record 4,072/1,359 own steps
  as metadata only, including setup, robber, discard and pre-roll. All 128 episodes
  lack full recurrent prefixes. Use genuine saved views only; collect complete
  histories for a recurrent value experiment rather than inventing missing states.

These findings narrow specific interventions. They do not show that imitation,
value learning or search cannot help. Choose the next experiment by a falsifiable
failure mechanism and the cost of learning something new.

## Priority 1: establish and measure sensible play

Keep distinct frozen controls: strict GreedyNoTrade, trading H, retained neural
parents and the current public planner. Bank trades remain legal whenever the
engine permits them; GreedyNoTrade deliberately declines them. Trading H's
proposals and persistent state must not silently become H-no-offers. Declare offer
regimes separately and retain useful passes as legitimate decisions.

Collect new TRAIN states from the actual current student, with every own decision
in chronological order, including forced actions. Independently judge public
packets across eight strata: initial placement; bank trade/build/pass; domestic
response; robber/discard; development-card purchase; development-card play
(including Road Building and Year of Plenty); road/army races; and held VP/endgame.
Judge legal alternatives, consequences and ambiguity, not exact H imitation.
Resource scarcity and trade valuations are public-state features or hypotheses to
test; published valuations are priors, not universal targets or shaped rewards.

For the next competence protocol, prepare eight cases per stratum across at least
eight TRAIN boards, report missing coverage without replacement after inspection,
and independently adjudicate disagreements. Proposed development gates are zero
unresolved clear legality/state-transition errors, at least 80% decidable cases,
at least six of eight decidable cases in every stratum, and at least 90% reasonable
decisions among decidable cases. Missing per-stratum coverage blocks a broad
competence claim. These are prospective
screening targets, not established performance or a population guarantee. A clear
missed immediate win or resource-destroying loop blocks competence until resolved
even if the aggregate passes. Check public counterfactuals for clearly wasteful trades and passes that miss a legal
immediate win; projected recipes alone do not certify a future legal build.

A case audit alone cannot certify heuristic-level play. Pair it with fresh,
all-seat whole-game comparisons against GreedyNoTrade and trading H. Before those
games, freeze board count, practical noninferiority margin to H, uncertainty rule,
completion requirement and retention controls. Preselect a separate blinded public
decision audit from those fresh EVAL games using the same coverage/judgment rules;
keep it outside training. Claim competence only if the predeclared game gates and
that held-out decision audit pass. Do not define parity from a 25%
win rate or a nonsignificant difference.

## Priority 2: learn the consequences of decisions

Develop terminal-outcome value learning as an independent research avenue. Freeze
one encoder, public continuation policy and offer regime. Save all real own
observations and terminal wins, keep boards disjoint across TRAIN/validation, and
weight games equally. Cache frozen recurrent embeddings once from complete
histories. A predictor under a fixed continuation is not an optimal value function.

Start with a prospective 256 TRAIN-game/64 validation-game design, profiling data
collection and encoding before launch. Compare existing public-feature and frozen
encoder heads on identical data and equal fit work. Separately compare nested
128 versus 256 TRAIN games to measure added-data value. Require held-out Brier or
log-loss improvement over calibrated constant controls before a deployment test;
then require better paired gameplay. Validation accuracy cannot substitute for wins.
Do not encode projected tuples as fabricated Observations. Branch-leaf value needs
a validated public transition interface.

Keep terminal wins as the initial objective. VP, resources, scarcity estimates and
build deficits diagnose behavior. Reward shaping needs a specific demonstrated
failure, an isolated comparison and protection against changing what winning means;
it is not the default next experiment.

## Priority 3: extend planning beyond the current turn

The current public planner mainly covers post-roll builds, maritime trades and
ending the turn. Test a bounded public simulator that includes production and
other players before the next own turn. First verify fidelity, privacy, action
coverage and cost on fresh TRAIN roots. An initial proposed bound is eight common
sampled scenarios, four root candidates and eight public transitions; profile it
before freezing an evaluation cap. These bounds are proposals, not measured capacity.

Construct worlds only from serialized permitted observations/history, joint belief
samples and independent future randomness. Never pass the real hidden Game, deck,
RNG or native runner. Simulated future agents must act on their own observations;
avoid omniscient decisions within each sampled world. Keep raw scorer, fallback and
resource limits fixed when comparing current-turn versus cross-turn planning.
Do not bundle a new value model into the first search contrast.

A smaller separate avenue is chance-aware development-card purchase versus build,
trade or pass, with public draw probabilities, opportunity costs and card play-delay
rules. Audit sensible non-purchases too. Robber and owned-card tactics can follow as
separate interventions. More purchases or higher VP do not establish improvement.

## Close existing comparisons without letting them dictate the research program

Finish the already prepared 384-game feature comparison: two retained parents and
their control/eligibility fits, four panels, four fresh boards and all seats. Keep
its predeclared schedules, advancement and retention rules unchanged. The package
is in preparation; freshness certification, independent harness review and Root
runtime authorization precede games. An incomplete or failed attempt remains so.
Do not infer selection from CE or silently turn this into a new search comparison.

After that comparison, independently audit new current-student TRAIN labels and
run the required added-data-only comparison, holding architecture, features and
optimizer work fixed. Accepted alternatives and ambiguity travel with labels.
This remains required even if the feature result is null. BC/DAgger is optional as
the use of those labels; the separately controlled data test must still be specified
and completed. Background case batch 001's 40 limited-BC-approved targets are unused;
batch 002 remains a metadata draft. Historical judgments are not automatically
training-approved. None of these obligations blocks independent value/search work.

## Immediate deliverables

1. Close the prepared feature-game comparison under its existing rules, without
   selecting a model early. Finish the independent harness review first.
2. Specify and collect the new complete-history current-student TRAIN cohort;
   independently audit its decisions and freeze the separate added-data test.
3. Build and profile the complete-history terminal-outcome collector and frozen
   embedding path. Freeze the value/data contrasts before fitting.
4. Test one bounded tactical coverage gap, starting with development-card buying,
   while validating the public transition seam needed by longer search.

Treat large cross-turn search as a staged investment: first fidelity and cost,
then a frozen gameplay contrast. Spend on additional scale only after a measured
benefit. Defer sustained PPO, default reward shaping, six-player training, physical
board recognition and broad adapter refactoring. Maintain existing adapter safety
checks and record UI knowledge while the strength experiments proceed.

## How an intervention earns more resources

| Stage | Evidence required |
|---|---|
| Mechanism and feasibility | Fresh TRAIN diagnosis, appropriate affected-path tests, measured cost, public-input and complete-history checks. |
| Development improvement | One isolated frozen intervention, paired fresh complete games, all seats, retained controls, whole-board uncertainty and independent public decision audit. |
| Heuristic competence | Predeclared sensible-decision gates plus fresh GreedyNoTrade/trading-H game gates, without using ambiguous judgments as mistakes. |
| Beyond reference strength | Separately positive candidate-minus-focal-VF and candidate-minus-focal-AB comparisons under declared matched backgrounds and information/rule regimes; replicate on fresh boards. |
| Confirmation | Freeze model, decision rule, source, limits, practical effect threshold and board-level uncertainty rule; use untouched final boards once. Inconclusive remains inconclusive. |

Reject or revise an intervention when its primary gate fails. Report failure,
partial outcomes and actual work; do not rescue it with VP, a secondary panel or
another seed chosen after looking. A small pilot can nominate a larger study but
cannot establish superiority. Preserve final boards `123100000..123100063` and
legacy finals until confirmation is frozen. No VF/AB superiority is established.

## Execution and adapter boundaries

Learning coordination owns the experiment queue, data splits and strength tests.
Adapter coordination owns observation/action translation, logs, legality, staleness
and visible-effect checks. A separate planning/review lane challenges hypotheses
and proposes the next bounded experiment. Review code and empirical claims
independently; separate readiness from actual authorization to run heavy work.

Keep Colonist controls and future physical-board recognition behind the same
Observation/Action contract. The simulator contract is the learning boundary;
adapter completeness does not need to precede local learning. Learner maritime
trades and domestic responses are supported. Initiating/countering domestic offers
needs a separate interface and experiment. Public chat can inform resource beliefs;
it cannot reveal unobserved steals or make hidden information perfectly known.

Use pinned Catanatron `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`.
Native VF weights/defaults and AB depth 2 with `prunning=False` remain authentic.
Declare `+atomic_discard_reject_trade_v1`, synthetic rejection, native-panel no-offer
behavior, chance-semantics mismatch and `full_rule_equivalence=false`. Changing
native behavior creates a different reference. Three-native stress panels remain
separate from focal-reference comparisons. Never train on native privileged choices,
evaluation packets or final boards, and never silently pool continuation policies.

Run at most two heavy workers, CPU/Torch intra/inter-op one and 2 GiB each, under
Root process/deadline guards. Inference/native workers require sampled OS-thread
count one; prospective gradient fits require their separately reviewed OS-at-most-four,
autograd-multithreading-disabled policy. Preserve the historical strict-OS1 HOLD. Pin runtime,
source, models and schedules; retain failed attempts and null outcomes. No paid
compute is assumed. Keep review proportional: reuse applicable completed checks
and spend the next unit of work on a falsifiable strength question.
