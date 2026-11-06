# Candidate successor representation: mechanical preflight

This is an **opt-in representation**, not a stronger-agent result. It adds a
67-column public current-action row for trade, build, card purchase, production,
port and piece consequences. `CandidateSuccessorPolicy` copies a
`ConsequenceGraphPolicy(hand_recipe)` parent with zero new output weights. Its
initial logits, memory and value must remain exact. The branch does not read
opponent private resources, a development draw, a robber steal, or a future
legal menu.

## Verification on current-engine student history

Source: `train-157410000/history.json.gz` from the broad raw-ranking pilot,
SHA-256 `d8c89fa156595b373aff0b675857d4bd331dc9e6ea32f3b75cffd161aefdece2`.
Only the public observation and its recorded immediate after-observation were
read. This board contains four complete student games, 390 decisions, 11
executed action kinds, and a largest menu of 54 actions. Every menu yielded
finite `menu_size × 67` features. Across all 390 rows, parent and new-policy
logits, value and recurrent memory were tensor-equal before training.

For the action the student actually executed, 265 rows had a known resulting
hand and all 265 matched the next observation. The remaining 125 rows carried
an unknown-hand mask. Road, settlement and city piece counts matched on all
141 checks across 47 direct builds. The sweep first found four mismatches:
each was a player's second initial settlement granting public adjacent
resources. A red regression test and the setup grant fix reduced them to zero.

The reproducible `tools/audit_successor_history.py` then swept all 16 sealed
TRAIN boards, without reading labelVAL histories, teacher overlays, or outcomes:
64 games, 6,215 decisions and 13 executed action kinds including Year of
Plenty and Road Building. All full-menu rows were finite. **4,181** known-hand
transitions, **2,151** direct-build piece checks and every direct placement
matched the next public observation; **2,034** resource outcomes were masked.
The saved TRAIN-only audit has zero mismatches and a SHA-256 for each history.

The focused policy, consequence, migration and successor suite passed **80
tests**. The full repository suite passed **1,832 tests**, with 2 skipped and
48 subtests passed. Ruff passed on all touched Python files. This is a
transition audit, not a proof of sound strategic choices or every possible
card sequence.

## Inference cost

`tools/benchmark_successor.py` reconstructs selected public observations from
the same captured history, checks exact parent parity, then times 32 warm
single-thread CPU calls with `torch.no_grad()`. The parent is a 64-hidden-unit
graph consequence policy; the candidate is its zero-start copy. The broad
capture was using the CPU concurrently, so p95 values include contention.

| Public menu | Actions | Parent p50 | Candidate p50 / p95 | Encoder-only p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| TURN road, settlement, buy, pass (`0:19`) | 10 | 1.80 ms | 2.46 / 2.56 ms | 0.57 / 0.59 ms |
| TURN road, bank trade, pass (`0:78`) | 13 | 2.54 ms | 3.32 / 3.89 ms | 0.66 / 0.73 ms |
| TURN Monopoly, pass (`1:45`) | 6 | 1.62 ms | 1.95 / 2.21 ms | 0.25 / 0.38 ms |
| TURN road, settlement, bank trade, pass (`3:26`) | 14 | 2.58 ms | 3.72 / 4.79 ms | 0.77 / 0.91 ms |
| First setup settlement (`0:0`) | 54 | 5.36 ms | 8.76 / 18.01 ms | 2.93 / 3.36 ms |
| Second setup settlement (`0:2`) | 28 | 2.36 ms | 5.52 / 8.62 ms | 1.64 / 1.75 ms |
| Move robber (`0:70`) | 22 | 3.63 ms | 4.44 / 5.50 ms | 0.87 / 1.10 ms |

An initial profile found that menu encoding recomputed each action's entire
feature dictionary once per column, 67 times. A call-count regression test
failed at 268 calls for four actions; after the fix it makes four. Before the
fix, encoder p50 was 10–45 ms on the TURN menus above. The table is from after
that fix. The setup grant was added after the TURN and first-setup timing runs;
the second-setup timing includes it. Raw local timing JSON lives in the ignored
`runs/candidate-successor-*-preflight-2026-11-06.json` files.

## Decision

The mechanics and latency are sufficient to **cross this representation with a
separately frozen broad learning objective**. Keep the broad CE versus pairwise
pilot untouched. A crossing must start from the exact same parent weights and
student TRAIN observations, use matched fit budgets, reserve a fresh label
validation panel, and require changed beneficial choices plus fresh paired
games before any strength claim. The representation has no default effect and
should not be selected from this preflight alone.
