# Unified public continuation: negative engineering screen

Status: **v2 stopped at its frozen reused TRAIN engineering gate**. No reserved
15751/52/53 fresh board, paired outcome game, or validation label was opened.
The opt-in implementation remains available for diagnosis; it is not nominated
as a teacher or promoted over the raw parent.

## Decision and provenance

V1 compared exact current END, bank-trade, BUY, and direct point-build roots on
one public option scale. Its archived public preflight found 0 changed roots in
128 eligible views. It passed exact-root parity and had 9.06 ms incremental
p95. The frozen v1 receipt and full per-root records are saved here. The older
archive contained only parent END choices, so it could not test BUY behavior.

V2 prospectively changed the successor graph: it searched bounded public
trade/road/build point routes, represented a later unknown-card BUY symmetrically
after non-BUY roots when the exact current BUY menu certified deck stock, and
prioritized a complete publicly certain same-turn win. Its source, design, and
protocol were frozen before this reused TRAIN screen. It did not retune the
v1 scoring constants or root margin. The frozen protocol is
`docs/unified-public-continuation-protocol-v2-2026-11-06.md` (SHA-256
`54f90b2c38cfaf39dca786f91ce404ae4bdc6fee22188214279188b60c346224`);
v2 source commit is `1f0c9dc6d50ac65547f21c245348fb1357719882`
(source-tree SHA-256
`4da0c8285d1514f775cb9eb8282daad52d153c6a1ef64eafb27b50b536c26d06`).
The prospective design and protocol were committed at
`5aa427d093b04d3c0b6f816d908ba50c21b0a555`.

The engineering input was **only** serialized public before-Observations and
their exact current legal menus from the broad lane's 16 reused TRAIN boards.
The collector did not read broad labels, actions, transitions, outcomes,
labelVAL, or review packets. It SHA-sorted eligible views and selected the
first 256 with BUY in the menu and 256 without. The unchanged raw parent
selected BUY on 208 views and END on 126; those 334 views form the declared
wrapper denominator. Source history hashes and the complete selection rule
are in `v2-reused-train-engineering.json`.

| Frozen gate / observation | Result |
| --- | ---: |
| Exact selected-root errors | 0 / 334 |
| Known current win errors | 0 |
| Incremental latency p95 (limit 25 ms) | 13.315 ms |
| Incremental latency median / maximum | 3.671 / 98.266 ms |
| Changed roots (required at least 8) | **2** |
| Source boards with changes (required at least 4) | **2** |

The intervention gate failed, despite parity and p95 latency passing. The
maximum latency also exceeded 25 ms; the preregistered latency criterion was
p95, not maximum. The 334 decisions ended in these diagnostic states:

| Status | Count |
| --- | ---: |
| `changed_root` | 2 |
| `insufficient_margin` | 19 |
| `parent_retained` | 109 |
| `uncertain_tie` | 30 |
| `uncertain_winner` | 147 |
| `unmodeled_later_buy` | 27 |

The compressed JSONL files retain every public observation, exact parent and
selected action, per-root option variants/scores, menu/view hashes, cap and
abstention reason, and latency. Their role is diagnostic: the reused TRAIN
screen is neither independent teacher-quality evidence nor a game-strength
estimate.

## The two changed choices

Both changes replaced an unknown development-card purchase with the exact
currently legal 4 sheep → 1 wheat maritime trade, then a city. Resource order
in the records is wood, brick, sheep, wheat, ore.

| Reused TRAIN board / actor / turn | Public hand before | Projected continuation | Robust comparison to BUY high-card endpoint |
| --- | --- | --- | ---: |
| 157410010 / seat 3 / turn 21 | `[1,1,4,1,3]` | trade, city at vertex 22; VP 2→3 | 334,260,301 vs 321,187,889 micro |
| 157410015 / seat 0 / turn 54 | `[0,0,4,1,4]` | trade, city at vertex 18; VP 2→3 | 321,014,306 vs 308,852,103 micro |

The first route's city score exceeds the BUY upper endpoint by 13,072,412
micro; the second by 12,162,203 micro. Those are *model scores*, not measured
win-probability changes. The public reasoning is plausible: the wheat receipt
completes a city recipe immediately, while the card identity remains unknown.

Root's public-fact replay tool independently reconstructed a public-consistent
synthetic engine state for each saved before-view. Both complete ordered
current legal menus matched; each indexed trade root was legal; each city was
in the actual post-trade legal menu and executed. Resource/piece accounting
and the +1 public point gain matched. The report has **2/2 `engine_verified`**
claims, zero contradictions, and zero unverified pairs. The checker source
SHA-256 was `6cbe0f1860efaef3c66961118c7b17e6243e2c9f07a993f8934c5d7afd5ebddc`.
These synthetic witnesses prove the stated continuations are feasible under
the public premises; they do not reconstruct the original hidden state or
establish that the trade would have won more games than BUY.

## Disposition

The frozen requirement of at least eight changed roots across four boards was
missed. V2 therefore stops before its fresh 157510000–015 TRAIN,
157520000–007 labelVAL, 157529999 smoke, and 157530000–007 paired development
boards. No threshold is changed on this reused corpus. Any later approach
should declare a new design and its evidence roles before using those seeds.
The 147 `uncertain_winner` and 27 `unmodeled_later_buy` records indicate where
the present conservative comparison abstained; they are diagnostic leads,
not a post hoc tuning set.

`evidence-index.json` records the SHA-256 and byte length of each tracked
receipt, public diagnostic stream, and fact-replay packet. The local seed
freshness audit receipt was
`/private/tmp/colonist-swarm-bxz1k39_/agent-research/large-track-seed-scan.json`
(SHA-256
`deeeaacc4adc91646d57b3ad44e3200cfbb92eebc8428b9ec4ebc7b0fecf144d`).
