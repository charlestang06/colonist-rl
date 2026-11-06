# Prospective public point-affordance feature screen

**Status: frozen design; mechanical checks only. No labels, fit, or games have been run
for this study.** This is an opt-in neural representation experiment for bank
trade versus pass decisions before PPO. It does not alter the working public
planner or the existing `ConsequenceGraphPolicy` checkpoints. The earlier
four-channel settlement/city eligibility comparison failed its H gameplay gate;
this study tests a different, single nonlinear compression of hand deficits.

## One public feature

For a current legal `MARITIME_TRADE` in the observing player's own post-roll
`TURN` or `PLAY_TURN`, define the *point gap* of a hand as the minimum total
positive resource deficit over the point recipes that have a public, currently
available build site and piece:

* Settlement `(1,1,1,1,0)` qualifies only when a vacant distance-legal vertex
  touches an own road and a settlement piece remains.
* City `(0,0,0,2,3)` qualifies only when an owned level-one building and a city
  piece remain.

Resource order is `(WOOD, BRICK, SHEEP, WHEAT, ORE)`. With `g0` from the observed
hand and `g1` after the legal bank exchange, the one scalar is
`max(0, g0 - g1) / g0` when at least one recipe qualifies and `g0 > 0`, and zero
otherwise. Thus a trade taking gap 1 to 0 encodes 1; gap 2 to 1 encodes 0.5;
an unchanged or worse gap encodes 0. `END_TURN`, offer responses and every other
action encode 0. A currently affordable point build gives `g0 = 0` and no
extra trade credit. The current legal action certifies only the first exchange.
This feature is a public necessary-condition summary, not a claim about a
future engine menu, unknown bank stock, other actions, or strategic value.
`point_affordance_feature(view, action)` requires the identical Action object
from the current legal tuple and returns one scalar. The policy uses
`point_affordance_features(view)` for the full menu so public site eligibility
is computed at most once per Observation, including large bank-trade menus.

An explicit `PointAffordanceConsequencePolicy` starts from an exact
`hand_recipe` consequence parent, copies all inherited weights independently,
and adds one zero-initialized, bias-free scalar logit branch for this feature.
The inherited 34D features retain their schema and masked geometry slots.
Value, recurrence, legacy policies, and old checkpoint identities stay exact.
The new architecture has its own strict identity and checkpoint key.

## Mechanical preflight

Fixture checks must cover exact hand arithmetic, full and partial gap reduction,
no eligible site or piece, blocked/foreign geometry, before-roll and offer phases,
unchanged/worse exchanges, absent-action rejection, no observation mutation,
unknown bank handling, scalar/batched output parity, exact zero-start parent
outputs and RNG, first gradient, and strict checkpoint round trip. A separate
pre-fit check on four newly sealed TRAIN anchors must verify menu and history
hashes, finite accepted-set loss, nonzero finite trainable gradients, zero
frozen-parameter gradients, and unchanged inherited start weights. Any failure
stops before an optimizer update.

## Fresh data and independent judgments

Collect deterministic own-turn histories with the unchanged seed-11 FULL15
`ConsequenceGraphPolicy` parent whose checkpoint SHA-256 is
`b3aae32ac36dc55f8ae26ec4676cc289243a6c540b959e1f8aa9786e5ddd0258`.
Use all four seats on TRAIN boards `157310000..157310015` and separate labelVAL
boards `157320000..157320007`. Before collection, Root must check these seeds
against its complete archive inventory and stop on any collision. Collection
uses the pinned 3.3.0 engine, the parent in deterministic mode, and the same
three trading-H opponents without domestic offers. Save full ordered own
histories, including forced decisions, exact Observation/menu/source hashes,
and executed action and outcome separately from judge packets. No EVAL or
current planner-point pilot label, decision judgment, or outcome enters this
study.

Screen each role first by public current observations: own post-roll turn, legal
`END_TURN` and bank trade, at least one eligible point recipe, and at least one
bank action with a positive feature. On every surviving complete own history,
replay the unchanged parent from the first observation to get its exact current
argmax. Require its chosen index to match the recorded deterministic student
action; a mismatch stops acquisition. Before seeing any new label, split cases
into a **challenge stratum**
(parent chooses a nontrade) and a **control stratum** (parent chooses a bank
trade). The parent choice and stratum remain root-private and hidden from blind
judges. They are sampling signals, not evidence that the parent is wrong.

Freeze at most two cases per board and one per episode, with quotas of at most
16 challenge/eight control TRAIN and eight challenge/four control labelVAL
cases. Visit boards in ascending seed order. Within each board and stratum,
prefer the earliest case whose legal trade reaches gap zero, then the earliest
positive partial-progress case; tie order is `(seat, decision version,
action-menu hash)`. Choose the first challenge and first control from distinct
episodes when available, then fill an absent stratum with the next earliest
case from the other stratum within its quota. Seal selected packet hashes before
judging; no replacement occurs after a judgment. This is a targeted mechanism
sample, not a population trade-error estimate.

The first blind strategic reviewer sees only the current public Observation,
zero-based legal menu, and earlier own prefix. It sees no H/student/planner
choice, feature score, later event, outcome, or source pointer. It names every
strategically acceptable current root as a sorted proper subset or marks the
case ambiguous, considering BUY, card play, alternative trade orders and pass.
For **every** accepted root claiming an immediate point, badge or build, a
separate factual reviewer checks that root's exact resource vectors, public
site/piece conditions, bank premise and post-step legal menu on a pinned-engine
synthetic public-consistent state. A certificate for one trade order never
certifies its reverse order or another root; a Longest Road claim needs its own
award check. Unverified claims are removed or make the whole case ambiguous
before any baseline join. This factual check certifies feasibility only. An
independent final reviewer may retain, expand or null the blind set using these
facts; it remains blind to H/student and outcomes. Seal all final labels before
joining any baseline predictions.

Stop with zero fit updates unless at least 12 informative TRAIN targets span
four boards/eight episodes and at least eight informative labelVAL targets
span three boards/five episodes. Require at least four TRAIN accepted sets that
include a bank trade and exclude the parent choice across three boards. For
baseline headroom, require at least three labelVAL cases across two boards
where a legal bank trade is accepted but the unchanged parent chooses an
unaccepted nontrade. Compute this gate once over the sealed, fixed labelVAL
pool; do not select replacements, change thresholds, or train on labelVAL.
Any hash, role, label or route inconsistency stops the study.

## Matched fit and labelVAL gates

Make independent copies of the same frozen parent. Arm A remains the exact
34D `hand_recipe` control; arm B adds only the one scalar feature branch.
Both use the same approved TRAIN accepted sets, canonical anchor order, full
own-prefix replay, suffix length 16, two presentations per anchor, one anchor
per update, fresh independent persistent Adam at `3e-4`, and the same parameter
freeze. Freeze every parameter except `heads['turn'][2]` and inherited
`consequence.weight` in both arms, plus only the new scalar weight in B. Retain
the unchanged parent as a third control.
Do not change model width, label source, reward, PPO, optimizer or schedule
after labelVAL is inspected.

Score all sealed labelVAL anchors with accepted-set CE and exact accepted-menu
hits. Separately count *actual legal bank-trade argmax choices* and which are
in the accepted set. The primary behavioral count is a **trade rescue**: B
selects an accepted bank trade where both the unchanged parent and matched A
select unaccepted nontrades. Arm B advances to games only if mean accepted-set
CE improves by at least 0.02 over both controls, it makes at least two trade
rescues across two boards, and it makes no new bank-trade choice outside an
accepted set where either control chose an accepted nontrade. On all other
nonforced labelVAL own rows, report agreement with cached H by action kind and
require no more than a two percentage point absolute loss relative to the
unchanged parent. Review changed BUY/card-play and direct-win choices blind to
model identity and later outcomes. If any gate fails, stop before games; do not
tune on this labelVAL panel.

## Conditional fresh-game stop gate

Only after the labelVAL gates pass, run one excluded smoke board `157329999`,
then eight matched development boards `157330000..157330007`, all four focal
seats, for both fits and the unchanged parent versus the same three trading-H
opponents. Use exact source/checkpoint/manifest hashes, the existing paired
runner including the neural-only manifest preflight fix `d0e4e07`, no domestic
offers, 20,000-step cap, complete game journals and public
candidate-step traces. Require all 96 games to resolve; otherwise retain the
partial record and stop. Report paired win/VP differences with whole-board
uncertainty, changed-decision frequency, actual bank-trade selection, same-turn
trade-to-point-build follow-through, destructive trade/pass loops, missed
immediate wins, latency, and hypothesis-to-first-gameplay wall time. A practical
nomination requires at least four additional wins of 32 over *each* control,
no verified destructive loop or missed immediate win, and sensible changed
choices. Failure retires this treatment on this screen. A passing exploratory
screen only earns a separately frozen independent-board and authentic VF/AB
confirmation; it is not a superiority claim or permission to start PPO.
