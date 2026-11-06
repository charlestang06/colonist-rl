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
The student-state study stopped before fitting for insufficient approved coverage;
this does not show BC cannot work. [Fixed outcome heads](fixed-value-head-results-2026-11-06.md)
failed their representation/data-benefit gates, not every possible value-learning
hypothesis. Narrow road and winning-prefix interventions did not advance. Preserve
these results; do not repeat unchanged treatments with more updates or games.

TRAIN diagnosis found 18 public one-bank city-funding opportunities across 15
episodes before purchase. The [opt-in wrapper screen](city-before-purchase-results-2026-11-06.md)
confirmed a real city route and all 12 fresh overrides built the intended city,
yet it won 11/48 games versus raw's 13/48. It remains off by default. The
purchase comparison needs card option value, not another unchanged wrapper run.

## Track 1: improve the working public player

The city-funding experiment is complete as a mechanical test and adverse gameplay
screen. Keep it optional. Its changed raw winning seats and the exact adverse
buy-then-city replay make uncertain card value the next economic decision gap.

Next diagnose search before increasing compute. Record attempted transitions,
feasible transitions, unique states, depth reached, chosen sequence, stop reason
and latency. Define uniqueness explicitly, including whether root-action identity
is part of the key. Reduce invalid and duplicate work while preserving legal
semantics, deterministic ordering and honest budget accounting.

Then compare 2,048 versus 8,192 nodes at depth 6 and beam 24, keeping scoring,
gain margin, fallbacks, geometry budget, seeds and opponents fixed. Both budget
arms use the same implementation; assess efficiency changes against the original
planner separately. The archived raw panel hit the node cap in only 103 of 4,896
searched decisions, so first verify that the larger budget reaches useful states.
Change beam and depth separately only when diagnostics justify it.

Move toward unified trading, building, purchasing and passing. Deterministic public
consequences should be exact. Card purchase needs uncertainty-aware future value,
never its actual hidden draw. A better public score alone does not establish
superiority to buying or passing. Extend useful coverage to Year of Plenty pairs,
connected Road Building, then uncertain Monopoly/robber/Knight choices and award
races. Execute one legal root action, reobserve and replan.

## Track 2: bounded learning alongside planning

Run one independent teacher-assisted learning investigation in parallel. It shares
the observation contract and evaluation path and need not wait for every planner
repair. Choose one hypothesis rather than automatically building this whole list.

- **Relevant supervision:** check labels, candidate encoding, gradients and
  recurrent sequence handling before scaling BC. Audit consequential states the
  student reaches, preserving acceptable alternatives and ambiguity. Aggregate
  informative TRAIN labels with coverage suited to the experiment, rather than a
  universal quota. Phase balancing alone misses trade/pass confusion because both
  share a phase. Teachers are fallible candidate generators, not optimal oracles.
- **Consequence features:** separately test resources missing after a trade and
  connected-settlement affordability, with the same learner/data as control.
- **Shadow critic:** predict actual returns under a declared continuation policy
  and opponent regime. Review prior failures and use a materially different
  hypothesis, such as a trainable encoder on current-student histories rather
  than the failed frozen representation. Compare held-out calibration and useful
  ranking with constant and simple public-feature controls before steering play.
  Teacher scores are not win probabilities. State prediction alone does not
  certify action ranking or values on synthetic planning leaves.
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

Reuse evaluation APIs and guards behind one lightweight runner path for planner
and learning comparisons. Standardize policy configs, paired boards/seats, source
identity, data roles, resource budgets, work counters and result summaries. Existing
experiments.py is a broader training workflow; extend only the facilities the
next comparison actually needs. Avoid another study-specific framework. Bind a
frozen source/dependency snapshot once per study, not the same installation per
job. Scale review to the change and claim; routine exploration needs affected
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
| Complete | [Reusable paired runner](../src/colonist_rl/paired_gameplay.py) and [broad capability audit](capability-audit-2026-11-06.md). | Use the same fast gameplay path across mechanisms. |
| Parallel | One distinct outcome-trained shadow critic experiment using accepted TRAIN data and a declared development screen. | No actor promotion from prediction alone. |
| Next | Search attempted/feasible/unique/depth diagnostics, then targeted efficiency and fixed-depth/beam 2,048-versus-8,192-node comparison if useful. | Measure actual changed decisions and cost before increasing search. |
| Following | Unified card purchase versus certain trade/build/pass evaluation. | Compare uncertain option value with exact public consequences. |
| As earned | Integrate retained improvements, audit broad competence, confirm against VF/AB and finish a live game. | Each claim requires its own evidence. |

One coordinator integrates results and owns the incumbent. Learning and adapter
leads own their boundaries; scoped agents handle independent tasks. A planning/
review agent identifies the next bottleneck from results rather than continually
expanding the backlog. Prefer a meaningful change tested in games to more process
documents.
