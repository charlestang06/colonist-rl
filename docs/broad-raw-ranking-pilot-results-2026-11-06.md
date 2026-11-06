# Broad student-state ranking pilot: stopped before fitting

The prospectively frozen pairwise-ranking pilot collected **96 complete
student-versus-3H games** on 24 boards, yielding 9,291 own public observations
and 5,677 nonforced decisions. The public raw planner proposed 170 city or
settlement routes; 120 disagreed with the student's current legal root. The
selector chose 24 TRAIN and 12 labelVAL public cases, and pinned public-fact
replay verified all 36 named routes in public-consistent synthetic states.
That establishes feasibility under the replay premises, not strategic value.

Three fresh-context blind batches judged the public pairs using opaque IDs,
random order and no game locator, teacher label or outcome. The root-private
join happened only after all 36 judgments were sealed. The frozen gate required
at least 12 TRAIN raw-only judgments on six boards spanning **three parent
action kinds**, at least six labelVAL raw-only judgments on three boards, and
at most two parent-only judgments per role.

| Frozen prefit check | Required | Observed |
| --- | ---: | ---: |
| TRAIN raw-only / boards | ≥12 / ≥6 | 13 / 10 |
| TRAIN parent action kinds among raw-only | ≥3 | **2**: 12 `END_TURN`, 1 `MARITIME_TRADE` |
| labelVAL raw-only / boards | ≥6 / ≥3 | 9 / 6 |
| Parent-only judgments, TRAIN / labelVAL | ≤2 each | 0 / 0 |

The parent-kind requirement failed, so the status is
`STOP_NO_FIT_PARENT_ACTION_KIND_COVERAGE`. The parent recurrent-headroom check
was not run after this earlier failure. **No optimizer update, evaluation game,
or candidate checkpoint** was made for this pilot. Selected pairs contain no
BUY or card-play negative root, so they cannot support an audited improvement
claim for those choices.

The original protocol SHA-256 is
`35d252389159c7ceac42dca15a15b2085ac9ebd81f4f81fa882a216651f8a9b3`.
Collector, annotation and factual-screen commits are `a4ca2da`, `d492559`
and `d0be7d1` on `cytang/broad-action-value-fit`; the stage summary is
`5537c18`. The opaque-packet manifest SHA-256 is
`0aa05877dd1ad2b9d71950b359d6369843a490a687b60ee032d6f4ac064270ec`.
The final root-private gate receipt SHA-256 is
`eb5a8d7c0741bdac60b175d7d1a028b0ca3027d9ea6bd75ba3e8aea1a763920d`.
Complete capture, packet, judgment and gate artifacts are preserved in the
ignored [broad raw-ranking archive](../runs/swarm-2026-11-04/agent-research/broad-raw-ranking-screen-001/RESULT.md).
Its relative-path SHA manifest is
`61cd92d481ed2a97c2761def3d9ffd4445ff2b81229f2682f5e8f258f88fcc0d`;
all 216 files were byte-verified against the source and two local copies.

This is a failed **broad pairwise claim**, not evidence that student-state
supervision is useless. A separate full-actor behavioral-cloning study is being
frozen on the 6,215 TRAIN student observations and raw-planner root labels.
It will use fresh validation and game boards, measure actual changed choices
and complete games, and keep the pairwise pilot's stopped gate intact.
