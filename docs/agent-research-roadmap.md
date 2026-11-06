# Build a strong, understandable Catan player

Revised 2026-11-06 following the user's request to reset the full project goal.
This is the canonical strategy and next-work order. It supersedes algorithm-first
sequences and the previously proposed five-arm incumbent screen. It does not
rewrite completed experiments or authorize changes to their frozen methods.
The [previous strategy](agent-research-roadmap-before-product-reset-2026-11-06.md),
[user roadmap](roadmap.md) and [measured status](training-status-2026-11-05.md)
remain available.

## The outcome we are building

Build one Catan player that makes coherent economic and tactical decisions from
its own permitted observations, completes games reliably, and becomes
reproducibly stronger than authentic VF and AB under declared benchmark conditions.
Run that same decision maker through simulator and Colonist adapters; physical
board input comes later.

The first useful product is a versioned working player with a documented action
scope, decision-time budget, complete-game results and inspectable decisions.
A trade should have an identifiable benefit or strategic purpose; a pass should
be sensible when spending is worse; a build or card play should advance a
credible plan. Its trace records actual inputs, alternatives examined, scores or
rules used, intended follow-up and uncertainty where available. These are records
of how it chose, not explanations invented afterward. A neural policy need not
have a fictitious verbal plan; report its actual scores and known limits.

Success has three separate requirements:

- **Competence:** recognizably sound decisions and reliable complete games.
- **Strength:** reproducible complete-game superiority to both pinned references.
- **Deployment:** faithful execution of the same policy in a complete live game.

Tests, labels, fitted updates, prediction accuracy and legal screen clicks are
supporting evidence. They cannot substitute for these outcomes. The long-term
VF/AB objective remains active and unmet; neither broad competence nor a new
strongest policy is established by the evidence so far.

## Strategy: improve the player we can actually use

Start from the strongest credible existing decision system. A heuristic, bounded
planner, learned policy or combination can be the product. The neural parent has
no preferred status. Keep the core observation/action contract independent of
Catanatron and the screen; adapter conformity does not imply access to privileged
engine state.

Choose methods by the weakness they address:

| Demonstrated problem | First experiment to consider |
|---|---|
| Wrong legal action, encoding or target | Correct that defect and verify its affected path. |
| Trade/build/pass arithmetic or short sequence | Explicit consequence calculation or bounded planning. |
| Bad supervision or off-distribution behavior | Audit acceptable alternatives on relevant reached states; targeted BC or student-state aggregation. |
| Poor ranking of uncertain future outcomes | Test a public value estimate or sampled planning against matched simpler scoring. |
| A trainable component fails to use sound targets | Inspect gradients, optimization and sequence handling; profile before scaling. |
| Competent local choices but weak longer-term strategy | Compare deeper public planning, policy improvement or RL; PPO is one option. |

Targeted learning can help reach heuristic competence. PPO is deferred until
competence and a specific strategic need justify it. No particular algorithm,
model size or number of updates is a required milestone.

## Stage 1: nominate one working player

The following is proposed scope, requiring a separately frozen execution protocol
before runtime. No new games ran during this strategy revision.
Make one matched development comparison of persistent public H, the unchanged
raw public planner and the exact frozen neural parent. Use eight fresh boards,
all four seats, the same three persistent H backgrounds, and no domestic offers:
**96 planned complete games**. Maritime trades stay legal. Freeze source, parent,
search budgets, board schedule, decision limits, contrasts and selection rule
before execution. Reuse the existing harness. This replaces the unexecuted
five-arm proposal; it changes no completed study.

Compare complete-game wins, critical decision failures, completion and latency.
Inspect representative public decisions from each policy without creating TRAIN
labels from these EVAL records. Respect board clustering and report uncertainty;
eight boards are a development screen, not reliable proof of a universal ranking.
If no policy has supported superiority, retain that uncertainty and name a
**provisional operational control** using predeclared reliability, latency and
simplicity criteria. Do not call the largest point estimate the strongest player.
H itself can be the working player. Do not add native arms or expand this screen
merely to seek significance; VF/AB belong in their dedicated strength comparison.

Deliver a small policy card: exact identity/configuration, supported action scope,
information boundary, actual work/results, limits, and why this is the working
control. Raw planning and neural inference can remain research alternatives.

## Stage 2: close the decision-to-gameplay loop

Study TRAIN histories from that working player. Existing histories are useful
only when they diagnose the same mechanism in that player. Prioritize errors by
frequency, likely game impact and confidence in the diagnosis. A rare provable
missed win merits correctness repair, but its existence does not make it the best
next strength experiment.

Each iteration has five outputs:

1. A concrete failure with legal alternatives, public information and a plausible
   cost. Include cases where passing or declining a trade/card is reasonable.
2. A causal diagnosis: targets, reached-state coverage, information, consequence
   calculation, search horizon or optimization. Check labels/encoding/gradients
   when relevant rather than assuming more data is the answer.
3. One minimal change and its adverse cases. Reuse the existing tests and harness.
4. A conceptual check of the actual changed choices and, where claimed, executable
   follow-ups. Separate certain facts from uncertain strategic judgments.
5. Fresh paired complete-game results against the incumbent under a frozen
   budget and advance/stop rule. Retain, revise or retire the change explicitly.

Only a retained improvement becomes the next incumbent. A correct narrow repair
may remain optional without a strength promotion. Do not promote on training
loss, agreement, VP or a secondary panel after the primary strength rule fails.
Do not repeat an unchanged failed treatment with more updates or a larger sample.

Begin with consequential trade/build/pass weaknesses if observed. Extend coverage
to placement; robber/discard; development-card purchase; Knight, Monopoly, Road
Building and Year of Plenty; longest-road/largest-army races; and VP/endgame.
Domestic offers/responses require their own evidence: no-offer games cannot
establish negotiation competence. Bank trades are legal whenever the rules permit
and must not be masked to make decisions easier. GreedyNoTrade intentionally
abstains and remains a diagnostic control, separate from trading H.

Before claiming broad heuristic competence, freeze a fresh eight-group audit
(placement, bank/build/pass, domestic responses, robber/discard, card purchase,
card play, awards, endgame), coverage counts, judgment thresholds and critical-error
rules. Keep acceptable alternatives and ambiguous cases explicit. Require reliable
completion within budget, no unresolved verified destructive loops or missed
immediate wins in that audit, and a predeclared practical noninferiority comparison
with H using board-level uncertainty; retain GNT as a diagnostic panel. Missing
coverage stays unknown. A nonsignificant difference or 25% win rate is not parity.
This broad claim gate does not block an earlier narrow repair experiment.

## Stage 3: improve strategy and establish strength

Once local decisions are credible, identify the next strategic limitation:
production balance, development-card investment, contested expansion, award races,
opponent threat, uncertainty or negotiation. Test one mechanism against the current
incumbent. Resource scarcity and trade values are contextual priors to test,
conditioned on board, hand, ports, stage and opponents; no universal fixed values
or shaping rewards are assumed.

Use progressively harder opponents where they test the current weakness. Retain a
fixed evaluation panel and check earlier competence for regressions. Stop spending
on random opponents when they cease to be informative. Keep TRAIN, development,
validation and final roles separate; EVAL decisions never become training labels.
Public logs constrain beliefs about opponents' resources; hidden steals and cards
prevent perfect knowledge. Simulated branches use permitted beliefs and independent
future randomness, never the real hidden Game, deck or future RNG.

For the strength claim, compare the frozen candidate directly with focal authentic
VF and focal authentic AB on matched fresh boards and all seats, with identical
backgrounds and declared trade rules. Freeze board count, practical superiority
margin, both-comparison uncertainty rule and censoring rules before execution.
Both candidate-minus-reference lower bounds must exceed the declared margin.
Replicate on disjoint fresh boards before claiming reproducible superiority.
A three-native stress panel answers a separate question. No-offer success supports
only that regime; broader negotiation strength needs a separate test.

PPO, broader self-play, learned values, more data and larger models each need a
specific decision hypothesis, a simpler control and a bounded initial experiment.
No standing background data collection or value-learning lane is scheduled.
Terminal winning remains the initial new-study objective. New shaping needs an
isolated test of a diagnosed problem and checks against loops or displacement of
winning. Existing experimental PPO rewards are not automatically adopted.

## Stage 4: deliver the same player through adapters

Maintain a thin adapter lane while improving the core. Separate observation
extraction, belief updates, legal action translation, stale-state rejection and
visible-effect confirmation. Preserve [control observations](colonist-controls.md)
and [walkthrough evidence](browser-walkthrough-status.md). Add a capability when
the working policy needs it; do not fork strategy into adapter-specific policies.

A complete live-game acceptance run must exercise the supported policy scope,
record extraction/action failures and confirm visible outcomes. Multi-step card
resolution, robber movement, awards and private own VP need faithful semantics.
Initiating/countering domestic offers needs its own contract and tests. Mark
unsupported actions explicitly. Simulator success alone cannot establish a capable
Colonist player. Physical recognition and six-player learning follow reliable
four-player play.

## Immediate queue and ownership

| Order | Deliverable | Boundary |
|---|---|---|
| 1 | Freeze and run the three-arm incumbent screen; publish its policy card. | 96 planned games; no fit, native expansion or final-strength claim. |
| 2 | Identify one consequential weakness in that player's TRAIN decisions. | Bounded diagnosis; no general annotation framework or arbitrary label quota. |
| 3 | Test one remedy through affected-path checks, changed-choice inspection and fresh gameplay. | Declare cost and promotion rule first; preserve a null result. |
| 4 | Broader competence, strategic improvement and direct VF/AB confirmation as evidence warrants. | No automatic jump to PPO or large data generation. |

The decision/learning coordinator owns this loop. The adapter coordinator owns
faithful transfer and only the capabilities needed next. An independent reviewer
challenges causal explanations and claims. One primary improvement experiment is
active at a time; parallel work must answer an independent, bounded question or
remove a specific blocker. Keep existing process/deadline guards, at most two
heavy workers and the declared CPU/thread/RSS limits. Avoid new governance,
reporting or benchmark infrastructure unless the next comparison needs it.

The old eligibility, road, winning-prefix and fixed value-head studies are closed
without strength advancement. Preserve their results, failures and exact gates.
The audited-target study ran no fit because coverage was insufficient; this is
not proof that BC fails. See [status](training-status-2026-11-05.md),
[road results](road-continuation-experiment-2026-11-06.md),
[winning-prefix results](winning-prefix-experiment-2026-11-06.md) and
[value results](fixed-value-head-results-2026-11-06.md). Do not reopen these unchanged
studies or treat their sunk implementation cost as a reason for continuation.

Keep pinned Catanatron `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`.
Native VF defaults and AB depth 2 with `prunning=False` remain privileged,
evaluation-only references. Disclose their information advantage, synthetic
rejection and `+atomic_discard_reject_trade_v1`, no-offer conditions, chance mismatch
and `full_rule_equivalence=false`. Their hidden choices never become learner or
teacher inputs. Final boards `123100000..123100063` and legacy finals remain
untouched until confirmation is frozen.

Report brief progress as **what decision we improved, what evidence changed our
view, and what we will test next**. Separate engineering success, decision quality,
playing strength and live execution every time.
