# Prospective student-state point-route correction pilot

**Frozen before any new judgment, fit, or gameplay.** This is a separate
H-guided acquisition study. The closed `student-dagger-audit-001` remains
unchanged and supplies no target. The question is whether independently
audited public point-conversion roots on states reached by the seed-11 neural
student can correct enough decisions to justify one small matched fit. A
planner route is a hypothesis until both its mechanics and its strategic
alternatives are reviewed.

## Source and selected cohort

Use the corrected v2 screen in `planner-distillation-screen-001`, with these
exact SHA-256 bindings:

| File | SHA-256 |
| --- | --- |
| `run_screen.py` | `f7f220e98d8748abf04cfd6f0963534701bedc331c5e9a55b8f534a8353b3fa5` |
| `input-manifest.json` | `fe71d9aec4176c90ee6d7006a651c58d1e465f03b8b8ce279a727915015b02ec` |
| `results.json` | `2dd5a10b9a6bc82db903e7f180939a8f32c163e38a9e46e8a4081050dc224593` |
| root-only `candidate-index-root.json` | `c25106a070f7be2d64dff97e82bef397f42d685a4316f57bed9acb5340937a71` |

The screen pins source commit `7c36586c34db5d4739b4de9ecc8d581f93556d70`,
ten source-module hashes, 12 exact recovered history/overlay pairs, and parent
checkpoint SHA-256 `b3aae32ac36dc55f8ae26ec4676cc289243a6c540b959e1f8aa9786e5ddd0258`.
The 12 histories and their cached overlays have byte-identical durable copies under the Original
repository's `runs/swarm-2026-11-04/agent-research/current-student-data-collection-001/`.
The corrected occupancy package itself is archived in both repositories at
`runs/swarm-2026-11-04/agent-research/planner-distillation-screen-001/`,
archive manifest SHA-256 `c7bade718ae18048b7d67b71c1995e270a7766218843457786fa3b1ffff0fe17`.
v2 excludes both prior 79-case and prior 88-slot judged anchors by exact
own-input **or** Observation hash. The only chosen cases are the 22 rows with
`primary_review_pool=true`: **15 TRAIN on seven boards/ten episodes, seven
labelVAL on four boards/five episodes**. Their root kinds are 13 trade, one
settlement and one city in TRAIN; four trade, one settlement and two city in
labelVAL. The one new four-step projected TRAIN path and all other candidates
are excluded. Do not replace excluded or ambiguous cases after any judgment.
No EVAL state or outcome enters selection or labels.

Write an immutable root-private freeze manifest with selected input,
Observation, menu, ordered-prefix and original history hashes, role, board,
episode, ordinal, proposed planner root/steps, cached H and student indices,
and deterministic batch assignment. Public judge packets receive opaque IDs,
the exact current before Observation and zero-based legal-action menu, and
only earlier own-prefix rows. They receive no planner/H/student recommendation,
current executed action, future row, source pointer, outcome, or EVAL metadata.
Keep root-private joins and projected routes outside judge directories.

## Blinded audit and pre-fit gates

Use two preassigned balanced strategic batches. A judge first names every
strategically acceptable current legal root as a sorted, nonempty **proper
subset** of the menu, or marks the case ambiguous. The judge must consider
BUY development card, card-play, alternative trade orders and END_TURN where
legal. Public hand arithmetic, piece supply, road connection, production and
known points may support a choice. Unknown card value, uncertain bank supply,
or a plausible unexamined route requires ambiguity or an expanded accepted
set. Judges do not see H, student, planner route, hidden state, or outcomes.

After the first batch, compute the maximum possible eventual supported count
from untouched cases. If fewer than **12 TRAIN** or **four labelVAL** can
remain, stop the study immediately with no route replay, fit, or replacement.
Repeat this early-stop check after each later batch and factual review.
Only if the count remains attainable, reveal the sealed proposed routes to a
different factual reviewer, never H/student/outcome. For each candidate
planner root in a blind accepted set, require exact root-menu object/hash,
indexed resource vectors in `(WOOD, BRICK, SHEEP, WHEAT, ORE)` order, public
bank/piece/geometry necessary conditions, and deterministic same-turn route
steps ending at a city or settlement. For nontrivial trade/road routes, check
the claimed follow-up actions with the pinned catanatron
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8` simulator from a
synthetic public-consistent state whose core board and **ordered root menu**
match the packet. Record each post-step legal menu and final point gain. A
synthetic route certificate proves feasibility only; it is neither original
hidden-state replay nor a strategic preference. Direct legal point builds need
public VP/rules verification. Unknown BUY draws cannot be treated as dominated
by a guaranteed build without independent strategic reasoning.

An independent final strategic reviewer sees the initial blind answer and
route facts, but no H/student/outcome. It may retain, expand, or null the
accepted set using only public evidence. Seal final labels and their hashes
*before* joining cached H. A case is supported only if its final accepted set
contains the planner root, remains a proper legal subset, and its stated
point route passed the factual check. Route failure or unresolved strategic
alternatives make it ambiguous. The fitted label is this audited accepted
set, not the planner's singleton or the student's executed action.

Before any update require all of the following:

* At least **12 supported TRAIN** across at least four boards/eight episodes
  and **four supported labelVAL** across at least two boards/four episodes.
* At least eight supported TRAIN and two supported labelVAL bank-trade roots.
* At least **four TRAIN accepted sets excluding cached H**, spread across at
  least three boards, and at least six supported TRAIN planner roots differing
  from the student's actual choice. H is a control and sampling signal, never
  an automatic target.
* Exact history, current Observation, legal menu, input, ordered own-prefix,
  zero-based action-index and cached-overlay hash joins. Any mismatch stops
  the study. No EVAL outcome can supply a label.

These gates are fixed despite the small 15/7 pool. If any fails, publish
audited yield and stop with zero fit updates. Do not reconsider ambiguous
cases after the H join to make the gate pass.

## Conditional matched sparse fit and retention

If all pre-fit gates pass, notify Root before fitting so heavy jobs can be
staggered. Start two independent copies of the exact unchanged seed-11 FULL15
parent checkpoint above. Arm A uses audited TRAIN accepted sets; arm B uses
cached H singletons on the **same** supported TRAIN anchors. Keep an unchanged
parent control. Freeze graph, GRU, action encoder and first turn-head layer;
train only `heads['turn'][2]` and `consequence.weight`. Use a fresh Adam
optimizer at `3e-4` per arm, complete current-weight own-prefix replay,
suffix length 16, one anchor per update, and four deterministic presentations
of canonically ordered anchors (at most 60 updates/arm). Preserve intervening
forced own observations. No new reward, architecture, PPO, feature, or
opponent change. Before the first update verify four newly supported anchors
for exact target/menu joins, finite accepted-set loss, nonzero finite
trainable gradients and zero frozen-parameter gradients. A failed preflight
stops with no fit.

Score on supported labelVAL anchors only, using mean accepted-set cross
entropy and accepted-set mass. Arm A must improve mean accepted-set cross
entropy over **both** parent and matched H arm by at least `0.02`, and have
at least two more labelVAL roots inside the accepted set than each control,
including one bank-trade root. On every other nonforced labelVAL own row,
cached-H agreement may fall by no more than two absolute percentage points
relative to parent. Report action-kind retention, especially BUY, card play,
bank trade and END_TURN. This broad non-target check addresses the historical
FULL25 result where selected agreement rose while full-game play regressed.
If labelVAL or retention fails, stop before games; do not tune on labelVAL.

## Conditional fresh gameplay

Notify Root before the smoke or screen. The reserved engineering smoke board
is `138309999`; the eight fresh paired boards are `138310000` through
`138310007`, all four focal seats for arm A, arm B and unchanged parent
against the same three persistent trading-H opponents. Use the existing
manifest runner, no domestic offers and the 20,000-step cap. Hash all
checkpoints/manifests and require complete resolved games, legal-root checks,
public first-divergence traces, same-turn trade-to-build follow-through,
latency and board-level uncertainty. Do not convert any EVAL result to a
label. A practical nomination needs at least **four additional wins of 32**
over *each* neural control, no verified destructive trade/pass loop or missed
immediate win, and sensible changed decisions. Eight boards are a reject or
advance screen, not a superiority claim; independently frozen larger-board
and authentic VF/AB tests remain necessary for promotion.
