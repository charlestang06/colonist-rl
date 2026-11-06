# Candidate successor representation: mechanical preflight

This is an **opt-in representation**, not a stronger-agent result. It adds a
68-column public current-action row for trade, build, card purchase, production,
port, piece and public award consequences. The current feature schema is
`public-candidate-successor-v3`, with checkpoint kind
`graph_candidate_successor_v3`. `CandidateSuccessorPolicy` copies a
`ConsequenceGraphPolicy(hand_recipe)` parent with zero new output weights. Its
initial logits, memory and value remain exact. The branch reads only the
immutable observation and its original legal actions.

`point_gain_lower_bound` is the actor's guaranteed immediate VP change;
`own_vp_after` uses that lower bound, and `wins_now` signals only a guaranteed
immediate win. `point_outcome_uncertain` marks an actor point result that could
differ because of a hidden development-card draw or the pinned engine's hidden
road-length insertion order or cached road-component membership. The encoder
does not claim an opponent private hand, a card identity, a robber steal, or a
future legal menu.

## Verification on current-engine student history

`tools/audit_successor_history.py` reads only public observations and recorded
immediate after-observations from 16 sealed TRAIN histories. It does not read
labelVAL, TEST, teacher overlays, opponent private state or outcomes. The
[v3 audit](experiments/candidate-successor-preflight-2026-11-06/train-transition-audit-v3.json)
retains each source SHA-256, identical to the v2 audit's 16 source hashes. It
covers 64 games, 6,215 decisions and 13 executed action kinds. Every legal menu
yielded finite `menu_size × 68` rows. The **4,181** known after-hand predictions,
**2,151** direct-build piece checks, **6,215** actor point lower-bound checks
and **6,215** guaranteed-win checks found **zero mismatches**. Of the point
checks, **5,915** were exact and **300** were uncertain: 298 hidden BUY draws
and two road endpoint awards. **2,034** resource outcomes were masked. Exact
winning indicators matched on all 5,915
deterministic point rows.

The initial review found a winning knight whose public Largest Army award was
missing, plus deterministic no-victim robber and one-resource Year of Plenty
options that were masked. The v2 repair also corrected road endpoint blocking
and passed the executed TRAIN point audit. Fresh review then found an unchosen
ordinary settlement that cuts exactly two opponent roads. A public synthetic
repro using the pinned Catanatron 3.3.0 topology has BLUE at 7 VP and ORANGE
holding Longest Road, length 6. BLUE's legal settlement at vertex 0 cuts
ORANGE's length to 3; BLUE's unchanged cached length is 5. The pinned engine
awards BLUE +1 for the settlement and +2 for Longest Road, reaching 10 VP and
winning immediately. A second pinned-engine repro awards Longest Road after a
cut with BLUE length only 3, because settlement recomputation has no five-road
threshold. When the post-cut maximum is tied at 3, reversing the engine's
hidden road-length insertion order changes the winner. A second legal cut
places another ORANGE route against an older BLUE building: its public trail
has length 6, but the backend caches length 5 because that enemy endpoint is
absent from its component. Reversing the hidden road insertion order then
changes whether BLUE keeps its existing award. A separate legal road placement
has a public five-edge trail ending at an older enemy building, but the pinned
backend's candidate length is 4, so an immediate +2 or win is not guaranteed.
The v3 tests check conservative actor point bounds in these cases, including a
possible net −1 when the actor already holds Longest Road. These candidate
cases are separate from the executed TRAIN audit.

Conversion tests check exact parent logits, value and recurrent memory,
including a mixed float64/float32 parent. Construction under an ambient MPS or
meta default preserves CPU and MPS RNG and yields repeatable new weights.
Checkpoint semantics moved to v3 so v1/v2 experimental checkpoints cannot
silently reuse the changed feature meanings. The full repository suite passed
**1,844 tests**, with 2 skipped and 48 subtests passed. Ruff passed on the
touched Python files. This is mechanical verification, not evidence of better
strategic decisions or every possible card sequence.

## Inference cost

`tools/benchmark_successor.py` reconstructs a public TRAIN observation, checks
exact parent parity, then times 32 warm single-thread CPU calls with
`torch.no_grad()`. The parent is a 64-hidden-unit graph consequence policy; the
candidate is its zero-start copy. The fresh [v3 late-road report](experiments/candidate-successor-preflight-2026-11-06/award-road-benchmark-v3.json)
retains the source hash and raw timing summaries.

| V3 public menu | Actions | Parent p50 / p95 | Candidate p50 / p95 | Encoder-only p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| TURN awarding Longest Road (`1:29`, board `157410002`) | 12 | 0.872 / 0.906 ms | 1.296 / 1.589 ms | 0.358 / 0.664 ms |

A separate 128-repeat single-thread timing used the one-action synthetic public
settlement repro from the tests. The encoder alone took **0.0360 / 0.0393 ms**
p50/p95 when one of the two opposing roads was absent, and
**0.0520 / 0.0596 ms** with the cut. This measures graph recomputation on that
one fixture. The [v2 TURN](experiments/candidate-successor-preflight-2026-11-06/turn-benchmark.json),
[v2 award-road](experiments/candidate-successor-preflight-2026-11-06/award-road-benchmark.json)
and [v2 non-TURN](experiments/candidate-successor-preflight-2026-11-06/nonturn-benchmark.json)
reports remain historical references. This small local timing sample does not
establish a stable latency change or a deployment service-level claim.

## Decision

The mechanics and measured latency support **crossing this representation with
a separately frozen broad learning objective after independent review**. Keep
the broad CE versus pairwise pilot untouched. A crossing must start from the
exact same parent weights and student TRAIN observations, use matched fit
budgets, reserve a fresh label validation panel, and require changed beneficial
choices plus fresh paired games before any strength claim. The representation
has no default effect and should not be selected from this preflight alone.
