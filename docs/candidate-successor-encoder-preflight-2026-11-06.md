# Candidate successor representation: mechanical preflight

This is an **opt-in representation**, not a stronger-agent result. It adds a
67-column public current-action row for trade, build, card purchase, production,
port, piece and public award consequences. The corrected row is schema
`public-candidate-successor-v2`, with checkpoint kind
`graph_candidate_successor_v2`. `CandidateSuccessorPolicy` copies a
`ConsequenceGraphPolicy(hand_recipe)` parent with zero new output weights. Its
initial logits, memory and value must remain exact. The branch does not read
opponent private resources, a development draw, a robber steal, or a future
legal menu.

## Verification on current-engine student history

`tools/audit_successor_history.py` reads only the public observations and
recorded immediate after-observations from the broad pilot's 16 sealed TRAIN
histories. It does not read labelVAL, teacher overlays, opponent private state
or outcomes. Its [saved report](experiments/candidate-successor-preflight-2026-11-06/train-transition-audit-v2.json)
records each source SHA-256 and covers 64 games, 6,215 decisions and 13
executed action kinds. Every legal menu yielded finite `menu_size × 67` rows.
The **4,181** known after-hand predictions, **2,151** direct-build piece checks,
**5,917** deterministic point-change checks and **5,917** immediate-win checks
matched the next public observation. **2,034** resource outcomes were masked;
the final audit has **zero mismatches**.

The first audit checked hands and pieces only. Fresh review found an actual
winning knight whose public Largest Army award was missing, plus deterministic
no-victim robber and one-resource Year of Plenty candidates that were masked.
Adding point checks exposed two Longest Road awards that a road graph initially
missed because an opponent settlement can end a road while blocking passage
through that vertex. These cases now have red-first regression tests, and the
full TRAIN point audit passes. The encoder uses public award holders, played
knights and the current road graph; it does not infer opponent hands.

Conversion tests check exact parent logits, value and recurrent memory,
including a mixed float64/float32 parent. Construction under an ambient MPS or
meta default preserves CPU and MPS RNG and yields repeatable new weights.
Checkpoint semantics moved to v2 so an earlier experimental v1 checkpoint
cannot silently reuse the changed feature meanings. The full repository suite
passed **1,837 tests**, with 2 skipped and 48 subtests passed. Ruff passed on
all touched Python files. This checks executed transitions and focused
unchosen candidates; it is not evidence of better strategic decisions or
every possible card sequence.

## Inference cost

`tools/benchmark_successor.py` reconstructs selected public TRAIN observations,
checks exact parent parity, then times 32 warm single-thread CPU calls with
`torch.no_grad()`. The parent is a 64-hidden-unit graph consequence policy; the
candidate is its zero-start copy. The saved [TURN](experiments/candidate-successor-preflight-2026-11-06/turn-benchmark.json),
[award road](experiments/candidate-successor-preflight-2026-11-06/award-road-benchmark.json)
and [non-TURN](experiments/candidate-successor-preflight-2026-11-06/nonturn-benchmark.json)
reports retain source hashes and the raw timings.

| Public menu | Actions | Parent p50 | Candidate p50 / p95 | Encoder-only p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| TURN road, settlement, buy, pass (`0:19`) | 10 | 0.654 ms | 0.905 / 1.056 ms | 0.209 / 0.241 ms |
| TURN road, bank trade, pass (`0:78`) | 13 | 0.907 ms | 1.336 / 1.605 ms | 0.359 / 0.637 ms |
| TURN road, settlement, bank trade, pass (`3:26`) | 14 | 0.857 ms | 1.284 / 1.616 ms | 0.360 / 0.692 ms |
| TURN awarding Longest Road (`1:29`, board `157410002`) | 12 | 0.873 ms | 1.274 / 1.518 ms | 0.329 / 0.635 ms |
| First setup settlement (`0:0`) | 54 | 1.678 ms | 2.795 / 3.685 ms | 0.967 / 1.262 ms |
| Second setup settlement (`0:2`) | 28 | 1.089 ms | 1.900 / 2.399 ms | 0.729 / 0.907 ms |
| Move robber (`0:70`) | 22 | 1.068 ms | 1.432 / 1.635 ms | 0.304 / 0.344 ms |

An initial profile found that menu encoding recomputed each action's entire
feature dictionary once per column, 67 times. A call-count regression test
failed at 268 calls for four actions; after the fix it makes four. Before the
fix, encoder p50 was 10–45 ms on TURN menus. The table above was rerun after
the award and card fixes. Timings are local measurements, not a deployment
service-level claim.

## Decision

The mechanics and latency support **crossing this representation with a
separately frozen broad learning objective after independent review**. Keep the
broad CE versus pairwise pilot untouched. A crossing must start from the exact
same parent weights and
student TRAIN observations, use matched fit budgets, reserve a fresh label
validation panel, and require changed beneficial choices plus fresh paired
games before any strength claim. The representation has no default effect and
should not be selected from this preflight alone.
