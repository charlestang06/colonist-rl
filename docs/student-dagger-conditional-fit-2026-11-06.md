# Conditional student-state DAgger fit

**Status: frozen before final audited labels; no fit authorized.** This is a
small test of better supervision on states the seed-11 neural student actually
reached. It changes label source and acceptable-action sets, not model size,
reward, PPO schedule or terminal-outcome training.

## Data and pre-fit gate

The technically recovered FULL15 student corpus has 32 TRAIN games on eight
boards and 16 labelVAL games on four separate boards. The deterministic
public-only selector froze 64 TRAIN and 15 labelVAL blind packets; one of 16
labelVAL slots was unfilled under the episode cap. The selection-manifest
SHA-256 is
`634994ae5f30a802e5c5cdb84218b2375f4f9167710baf9e4877a1278d85`,
and the packet-index SHA-256 is
`fdfc22cb14aedcb5d9d2369492b2f25d16dac8e045746582c04aeee6819e18cb`.
The 12 bound source histories already have byte-identical copies in local
`runs/swarm-2026-11-04/agent-research/current-student-data-collection-001/`.
No selected current action, teacher choice, later event or outcome appears in
a judge packet. The source collection remains a recovered corpus, not a
retroactive success of its original timed controller attempt.

Before fitting, require exact source, observation, menu, input and ordered
prefix hashes; legal zero-based approved indices that form a nonempty proper
subset of the current menu; and independent public reasoning that keeps
uncertain BUY options acceptable or marks ambiguity. A bank root plus a
post-trade build menu certifies a route, not its strategic preference.
Only final informative targets count. Require at least 12 TRAIN targets across
four boards and eight episodes, four labelVAL targets across two boards, and
four TRAIN targets whose accepted set excludes the original H singleton.
If any gate fails, stop without fitting. labelVAL is scoring-only; no EVAL
outcome can supply a target.

## Matched sparse fit, if authorized

Start from the exact frozen parent checkpoint
`seed-11-frozen-bc-3_3-start.pt` (SHA-256
`b3aae32ac36dc55f8ae26ec4676cc289243a6c540b959e1f8aa9786e5ddd0258`).
Make two independent copies with the same initial weights. One uses audited
accepted sets; the legacy-label control uses the cached H singleton on those
**same** approved TRAIN anchors. Keep the parent as a no-update third control.

Freeze the graph encoder, recurrent cell, action encoder and first turn-head
layer. Train only `heads['turn'][2]` and the existing 34-dimensional
`consequence.weight`. Use fresh Adam at 3e-4 per arm, full current-weight
prefix replay, suffix length 16, one anchor per update, and two identical
presentations of the canonically ordered anchors. At most 64 TRAIN anchors
means at most 128 updates per arm. Preserve all intervening own observations,
including forced actions, in sequence order. Before the first update, audit
four newly approved anchors for exact label↔action joins, finite accepted-set
loss and finite nonzero trainable gradients; frozen parameters must have zero
gradients. Do not change architecture or optimizer settings after seeing
labelVAL results.

First compare mean accepted-set cross-entropy on all approved labelVAL
anchors. If the audited arm does not improve over both the unchanged parent
and matched H-singleton fit, stop before game evaluation. Report action-kind
retention on the full labelVAL public sequence as a diagnostic, especially
card play; sparse Year of Plenty examples cannot certify card competence.

## Gameplay gate, only after labelVAL passes

Run one excluded smoke board 138279999, then freeze 138280000–138280007 as
eight fresh four-seat matched boards for the unchanged parent, H-singleton
fit and audited fit against three trading-H opponents, with no domestic offers
and the existing 20,000-step game cap. Use the manifest runner's exact
checkpoint and source hashes, game journal, public first-divergence traces and
board-level intervals. A practical nomination needs at least four more wins
out of 32 than **each** control, complete resolved games, and sound changed
decisions. This is an exploratory gate, not a superiority claim. Any nominee
needs separately frozen VF/AB and independent-board confirmation. PPO and
reward shaping remain deferred.
