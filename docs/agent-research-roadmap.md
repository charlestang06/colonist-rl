# Goal: a strong, usable Catan decision maker

Revised 2026-11-06. This is the authoritative direction for new work. Older
roadmaps and execution plans are historical where they disagree. Frozen studies
retain their original methods, limits, data roles and failed gates.

## The outcome

Build one player that makes coherent economic and tactical decisions from its
own permitted information, reliably completes games within a declared compute
budget, and reproducibly outperforms authentic pinned VF and AB references.
Execute that same decision maker through simulator and Colonist adapters;
physical-board input comes later.

The product is the complete decision system, not a particular neural checkpoint.
Rules, exact consequences, search, beliefs and learned estimates may cooperate.
A stronger simple player is progress. Teacher agreement, prediction accuracy,
training volume and legal clicks are intermediate measurements, not success.

| Claim | Evidence required |
|---|---|
| Decision competence | Sound choices across covered action families, reliable completion, no unresolved verified destructive loops or missed immediate wins, and a predeclared practical comparison with trading H. Missing coverage remains unknown. |
| Playing strength | Fresh matched complete games against both authentic references under declared rules/budgets, meaningful board-level uncertainty, and independent fresh-board confirmation of the frozen candidate. |
| Deployment | A complete live game through the same policy/adapter contract, recording extraction, execution and recovery failures. |

These claims are separate and remain unmet. No-offer studies cannot establish
negotiation competence. Native references are privileged evaluation opponents;
the public planner and learner never receive their hidden inputs or labels.

## What the results tell us

The [working-player screen](working-player-card.md) completed 96 games: raw
15/32 wins, H 5/32, frozen neural parent 2/32. Its frozen rule retained H as the
provisional control and raw as the leading unpromoted challenger. The
[native comparison](native-reference-results-2026-11-06.md) completed 384 games:
raw 32/128, VF 19/128, AB 26/128; its qualification gate failed. These point
estimates do not establish superiority. Raw is nevertheless the main development
candidate; the historical fallback does not require treating H as strongest.

Targeted BC improved agreement without establishing reliable gameplay gains.
The earlier 88-case student-state study stopped before fitting for insufficient
approved coverage. The later [79-case blind audit](student-dagger-audit-results-2026-11-06.md)
found 20 TRAIN and five label-validation supported sets, but only three TRAIN
sets excluded H against its frozen minimum of four; it also stopped before
fitting. Neither result shows BC cannot work. [Fixed outcome heads](fixed-value-head-results-2026-11-06.md)
failed their representation/data-benefit gates, not every possible value-learning
hypothesis. Narrow road and winning-prefix interventions did not advance. Preserve
these results; do not repeat unchanged treatments with more updates or games.

The distinct [trainable public-history critic](shadow-critic-results-2026-11-06.md)
fit actual terminal outcomes in 9.14 seconds, but its development Brier was
0.15100 versus 0.15077 for the simpler public-63 control, and its within-board
winner ranking was worse. The predeclared nomination gate failed. This closes
that treatment without suggesting that all learned estimates are ineffective.

TRAIN diagnosis found 18 public one-bank city-funding opportunities across 15
episodes before purchase. The [opt-in wrapper screen](city-before-purchase-results-2026-11-06.md)
confirmed a real city route and all 12 fresh overrides built the intended city,
yet it won 11/48 games versus raw's 13/48. It remains off by default. The
purchase comparison needs card option value, not another unchanged wrapper run.

The [fast search-budget screen](search-budget-results-2026-11-06.md) completed
32 fresh paired games. Raising the node limit from 2,048 to 8,192 made all
16 matched seat traces and outcomes identical, while spending 3,341 more
attempts with no extra feasible or unique successor. Retire that unchanged
increase. The [future-pruning screen](public-future-pruning-results-2026-11-06.md)
then skipped 414,217 certainly invalid candidates, exposed 101 more feasible
successors under the original cap, and kept all 32 matched action traces and
outcomes identical. Searched selection was faster in both arm orders. Enable
the necessary-condition filter by default for new public-planner work.

The [public purchase-option screen](public-option-purchase-results-2026-11-06.md)
won 16/48 versus raw's 13/48 in 12 fresh matched boards, but the board-level
effect remains uncertain and full-turn review found adverse BUY→later-build
cases. Keep this fixed score optional. A narrower continuation repair and an
uncertain card comparison need new hypotheses, not more games on that screen.

## Track 1: improve the working public player

The city-funding experiment is complete as a mechanical test and adverse gameplay
screen. Keep it optional. Its changed raw winning seats and the exact adverse
buy-then-city replay make uncertain card value the next economic decision gap.

Attempted, feasible, unique, duplicate, depth, stop and latency diagnostics are
now in the shared runner. Necessary-condition pruning of **future** build and
maritime intentions passed the fresh paired and reversed-order timing screens;
it is the default, with an explicit off switch for reproducing old results.
Skipped-invalid work stays separate from attempted transitions. The 2,048-
versus-8,192-node screen found no new state or action, so do not repeat that
unchanged budget comparison. Change beam and depth separately only when
diagnostics justify them.

Move toward unified trading, building, purchasing and passing. Deterministic public
consequences should be exact. The [direct-win guard](public-turn-planner.md)
now selects a fully validated current legal city or settlement that reaches
the standard winning threshold before a longer future search, within the
planner's supported rolled-turn flow. Card
purchase needs uncertainty-aware future value, never its actual hidden draw.
The option screen shows why BUY→later-trade/build continuations must be compared
before a projected trade→build is credited as an earlier point. A better public
score alone does not establish superiority to buying or passing. Extend useful
coverage to Year of Plenty pairs, connected Road Building, then uncertain
Monopoly/robber/Knight choices and award races. Execute one legal root action,
reobserve and replan.

## Track 2: bounded learning alongside planning

Run one independent teacher-assisted learning investigation in parallel. It shares
the observation contract and evaluation path and need not wait for every planner
repair. Choose one hypothesis rather than automatically building this whole list.
The [conditional student-state DAgger protocol](student-dagger-conditional-fit-2026-11-06.md)
stopped before fitting when its H-disagreement gate failed. A new learning study
must freeze its own selection and gates. Public planner point-route overrides
on student-reached states are a candidate to audit, not approved supervision.

- **Relevant supervision:** check labels, candidate encoding, gradients and
  recurrent sequence handling before scaling BC. Audit consequential states the
  student reaches, preserving acceptable alternatives and ambiguity. Aggregate
  informative TRAIN labels with coverage suited to the experiment, rather than a
  universal quota. Phase balancing alone misses trade/pass confusion because both
  share a phase. Teachers are fallible candidate generators, not optimal oracles.
- **Consequence features:** separately test resources missing after a trade and
  connected-settlement affordability, with the same learner/data as control.
- **Shadow critic:** the bounded trainable-GRU test is now closed after failing
  its simple public-feature control. Future value-learning work needs a new,
  decision-relevant hypothesis and independent data. Teacher scores are not win
  probabilities. State prediction alone does not certify action ranking or
  values on synthetic planning leaves.
- **Policy improvement:** allow a small actor-critic-plus-imitation test against
  imitation-only and no-update controls when justified. Stochastic on-policy
  updates need policy-matched experience and recorded behavior probabilities;
  teacher and greedy trajectories are not automatically on-policy actor-critic
  data. Reduce teacher influence only when fresh gameplay supports doing so.

Extensive PPO and broad self-play are deferred until a specific strategic need
and effective comparison justify the cost. Small learning experiments may help
reach competence; PPO is neither a mandatory next stage nor a prerequisite for
progress. Use terminal winning initially in new studies. Test any reward shaping
separately against the unchanged objective, checking loops and actual wins.
Resource scarcity is contextual information to test, not a universal reward table.

## Make experiments faster and more useful

The loop is: diagnosed weakness → one mechanism → inspect changed decisions and
adverse cases → fresh paired complete games → retain, revise or retire.

Use modest exploratory panels to reject weak mechanisms cheaply. Report effect,
uncertainty, intervention frequency, completion and latency. A promising result
earns a separately frozen confirmation; it is not proof of superiority. Do not
repeat small panels under a rule incapable of resolving the meaningful effect.
Plan confirmation against a practical effect and say when the affordable design
is inconclusive. Never rescue a failed frozen gate by changing its metric or
extending its sample automatically.

Use the [manifest-driven paired runner and analyzer](experiment-quickstart.md)
for planner and graph-checkpoint comparisons. It records configs, explicit
reference arm, paired boards/seats, source and checkpoint identities, data role,
work counters, per-game journals, optional public step traces, and bounded
result summaries. Existing experiments.py is a broader training workflow;
extend only the facilities the next comparison actually needs. Bind a frozen
source/dependency snapshot once per study, not the same installation per job.
Scale review to the change and claim; routine exploration needs affected
checks, not another exhaustive archival campaign.

Bound concurrency, CPU, RSS and wall time; record actual utilization and thread
counts. Preserve old strict thread protocols. Future limits should distinguish
idle OS threads from active compute. Report hypothesis-to-first-gameplay time,
setup/review versus compute time, and candidates actually compared. Those are
iteration diagnostics, not strength metrics.

Keep TRAIN, development, validation and final roles explicit. EVAL decisions may
explain results but never become training labels. Retain failed attempts, losses,
censoring and adverse cases. Bank trades remain legal whenever rules permit;
GreedyNoTrade deliberately abstains as a separate diagnostic baseline. Public
logs constrain resource beliefs; hidden steals/cards prevent perfect knowledge.
The working player, learner, teachers and public-planning branches may not
inspect real hidden engine state, deck or future RNG. Privileged native evaluation
references retain their explicitly disclosed exception.

## Adapter scope

The core owns strategy, permitted history, beliefs and legal decision semantics.
Adapters own extraction, action translation, stale-state rejection, staged
execution and visible-effect confirmation. Catanatron compatibility helps simulator
integration but does not replace the shared information boundary. Keep Colonist
transport under adapters/colonist and physical input under adapters/physical;
do not fork strategy into a screen-specific policy.

The [bounded capability audit](capability-audit-2026-11-06.md) maps setup; bank/build/pass; purchase/cards;
robber/discard/steal; domestic offers/responses; awards; VP/endgame. Record legal
representation, exact versus uncertain consequences, execution support and decision
evidence. This is a map of gaps, not an exhaustive-state prerequisite. Extend
adapters when the working player needs the capability. Preserve [observed controls](colonist-controls.md);
manual walkthroughs are not autonomous play. Negotiation, physical recognition
and six-player learning follow reliable standard four-player play.

## Immediate work and ownership

| Order | Concrete output | Decision |
|---|---|---|
| Complete | [City replay and paired screen](city-before-purchase-results-2026-11-06.md): executable route, 12/12 follow-through, raw 13/48 versus wrapper 11/48. | Keep wrapper optional; no promotion or unchanged extension. |
| Complete | [Reusable paired commands](experiment-quickstart.md) and [broad capability audit](capability-audit-2026-11-06.md). | One manifest, source-bound game journals, fast read-only comparison for planner and graph arms. |
| Complete | [Trainable public-history critic](shadow-critic-results-2026-11-06.md) fit and development screen. | Gate failed against public-63; no policy change or unchanged extension. |
| Complete | [2,048-versus-8,192 search-budget screen](search-budget-results-2026-11-06.md): 32 resolved games, 16/16 same traces, 3,341 extra invalid attempts. | Retire unchanged budget increase. |
| Complete | [Necessary-condition future pruning](public-future-pruning-results-2026-11-06.md): 64 resolved screen games, 32/32 same traces, 414,217 invalid candidates skipped; speed gain repeated with reversed arm order. | Enable by default for new planner work; no playing-strength gain claimed. |
| Complete | [Opt-in purchase comparison](public-option-purchase-results-2026-11-06.md): 96 resolved games, 50 changes, 25/26 immediate builds, raw 13/48 versus option 16/48; fresh adverse BUY continuation. | Keep optional; revise the one-turn comparison and reserve new boards. |
| Complete | [Direct current legal-win regression](public-turn-planner.md): exact public engine view and a terminating one-step replay. | Keep legal root identity and full root validation; no gameplay strength claim. |
| Complete | [Audit student-reached trade/pass/BUY states](student-dagger-audit-results-2026-11-06.md): 20 TRAIN/five label-validation supported sets; only three TRAIN sets excluded H against the frozen minimum four. | Stop before fitting; preserve ambiguous cases and exact audit. |
| Next learning candidate | Audit public planner point-route overrides on distinct student-reached TRAIN/label-validation states under a new protocol. | Check exact H disagreement, independent strategic labels, route feasibility and retention before any fit or fresh-game screen. |
| As earned | Integrate retained improvements, audit broad competence, confirm against VF/AB and finish a live game. | Each claim requires its own evidence. |

One coordinator integrates results and owns the incumbent. Learning and adapter
leads own their boundaries; scoped agents handle independent tasks. A planning/
review agent identifies the next bottleneck from results rather than continually
expanding the backlog. Prefer a meaningful change tested in games to more process
documents.
