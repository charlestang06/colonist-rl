# Working-player nomination implementation plan

Historical bounded execution plan. Its frozen studies retain their constraints;
new priorities and parallel learning follow the
[canonical roadmap](../../agent-research-roadmap.md). This is not the current
full-project execution order.

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to execute this plan task by task. Root coordinates runtime; source-only work can be delegated.

**Goal:** Nominate one public-information working player and publish an honest policy card before choosing the next improvement experiment.

**Architecture:** Reuse the current evaluation APIs, persistent H baseline, raw planner, frozen neural parent and existing guarded pilot worker. Change only the study configuration and required worker dispatch; preserve the shared policy and adapter interfaces.

**Tech stack:** Python, pinned Catanatron, existing Torch checkpoint/inference, existing JSON journals and process guards.

**Spec:** [Canonical strategy](../../agent-research-roadmap.md), especially Stage 1. This plan covers the first deliverable only; later repairs depend on its findings.

## Execution status, 2026-11-06

Tasks 1 and 2 complete: 96 resolved games, 24 zero-exit guards, independent saved
review and 55 frozen plus 3 posthoc diagnostic decisions. The [policy card](../../working-player-card.md)
keeps H as provisional control and raw as unpromoted challenger. The subsequent
[native comparison](../../native-reference-results-2026-11-06.md) completed 384 games
(raw 32/128, VF 19/128, AB 26/128), but failed its qualification gate. No automatic
extension or confirmation follows.

Task 3 is in progress on the raw challenger's mechanism: first-64-TRAIN diagnosis
found 18 one-bank city-funding alternatives across 15 episodes before unsupported
BUY fallbacks. Root checked eight public witnesses; no optimality targets or fit
were produced. Exact engine factcheck, intervention and fresh gameplay remain.
This follows the earlier ruling to investigate raw's strength signal before
repairing H merely because it won the fixed operational fallback.

## Global constraints

- Player-observable inputs only; no privileged native labels or hidden engine/deck/RNG inputs.
- Pinned Catanatron `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`.
- At most two heavy workers, existing CPU/thread/RSS/deadline guards; Root owns runtime.
- Eight fresh development boards, all four seats, three arms: 96 planned complete games.
- Three persistent H backgrounds per game, no domestic offers, maritime trades legal.
- No fit, new teacher queries, native arm or final-board use. Existing studies remain closed.
- Do not infer a strength ranking from a noisy point estimate. Preserve incomplete outcomes.

## Task 1: freeze the minimal successor study

**Read/reuse:** `src/colonist_rl/evaluation.py`, `src/colonist_rl/public_turn_planner.py`, `src/colonist_rl/training.py`, and the archived native H pilot worker/protocol under `runs/swarm-2026-11-04/agent-research/native-focal-pilot-001/`.

**Create:** study-specific protocol/schedule and bindings under `runs/swarm-2026-11-04/agent-research/incumbent-selection-001/` when execution starts. No new general harness.

**Consumes:** the existing persistent-H, unchanged raw and exact frozen-parent identities. The parent is `ppo-strength-preparation/seed-11-frozen-bc-3_3-start.pt` under the same run root, byte SHA256 `b3aae32ac36dc55f8ae26ec4676cc289243a6c540b959e1f8aa9786e5ddd0258`.

**Produces:** a frozen three-arm protocol and deterministic board/seat/job schedule.

- [x] Check source/config/checkpoint identities and the fresh-board domain against existing records. Record the search before reserving eight boards; do not generate boards to select favorable difficulty.
- [x] Declare 24 one-board/four-seat jobs, balanced arm order across boards, 20,000 engine-step game cap, existing guard budgets and total run budget. Register actual handles before dispatch continues.
- [x] Keep raw depth 6, nodes 2,048, beam 24, geometry 100,000, seed 11, margin 10,000 and optional winning-prefix flag false. Freeze the exact code/config actually used.
- [x] Use persistent focal H, not a fresh heuristic reconstructed on every observation. Neural memory resets per game and advances on forced inputs too; deterministic inference, no optimizer.
- [x] Freeze the three paired board-level win contrasts and simultaneous 95% uncertainty rule. Proposed practical superiority margin: five percentage points, requiring support against both other public arms to name a uniquely stronger policy. Small-sample uncertainty remains explicit.
- [x] Freeze completion/critical-failure criteria, latency budget and a reliability/latency/simplicity fallback order. Inconclusive strength evidence permits a provisional operational control, not promotion as strongest. Retain H as the simplest fallback unless it fails those criteria.

**Acceptance:** reviewer can identify every arm, all 96 planned games, all limits and the selection decision before any result exists. This roadmap is not itself a completed runtime freeze.

## Task 2: execute and publish the working-player card

**Modify only if needed:** the study-local copy of the existing pilot worker and its inert dispatch tests. Keep `evaluation.py` unchanged unless a concrete blocker is demonstrated.

**Consumes:** Task 1 frozen protocol and actual source/model bindings.

**Produces:** complete journals, guarded actual-work records, paired summaries and `docs/working-player-card.md`.

- [x] If worker dispatch changes, first add inert tests: H routes to persistent heuristic without loading the model; raw uses unchanged defaults; neural loads the exact checkpoint once and retains per-game memory. Require four seats per board and reject duplicate/missing records. Root observes the tests fail for the specific absent behavior, then implements and verifies only the needed change.
- [x] Run the 24 declared jobs under existing guards. Record terminal status, every completed/censored game and actual resource use; preserve failures rather than retrying until favorable.
- [x] Recompute joins and board-level contrasts from saved results. Inspect preselected representative EVAL decisions without converting them to labels. Distinguish factual critical failures from uncertain strategic disagreements.
- [x] Apply the frozen nomination rule. If unresolved, publish the operational fallback and uncertainty. Do not expand the same screen to chase a winner.
- [x] Publish the policy card: identity/configuration, observation boundary, action scope, actual results, completion, latency, critical issues, selection rationale and missing evidence. Update canonical status without altering old protocols.

**Acceptance:** the named working control can be reproduced and its limitations are visible. This task establishes neither broad competence nor VF/AB superiority.

## Task 3: choose the first improvement from that player

**Read/reuse:** the working-player card, permitted TRAIN histories, `consequences.py`, `trades.py`, `public_turn_planner.py`, and existing scoped tests relevant to the observed defect.

**Create:** one bounded diagnosis note and a separate successor experiment protocol only after the mechanism is identified. No general annotation framework.

**Consumes:** Task 2 policy identities, the raw challenger selected for strength work, legal alternatives and permitted TRAIN evidence.

**Produces:** one question of the form “changing this mechanism should improve these decisions because ...”, with a named control, affected cases, adverse cases, compute budget and fresh-game advance/stop rule.

- [x] Verify that existing TRAIN examples exercise the selected raw challenger's mechanism; otherwise collect only the bounded histories needed for diagnosis under a separate declared budget.
- [x] Rank observed opportunities by confidence, frequency and likely game cost. Include reasonable passes, declined trades and uncertain choices. A single certain costly mechanism can justify a narrow repair; no arbitrary minimum label quota applies to every intervention.
- [ ] Inspect legality, encoding and labels first where implicated; inspect gradients only for a learned component. Choose a targeted rule, consequence feature, bounded search or audited learning intervention that addresses the actual cause.
- [ ] Define meaningful affected-path regression and adverse cases, inspect the actual changed decisions, then run fresh paired gameplay under the successor's frozen rule.
- [ ] Retain, revise or retire explicitly. Add no automatic larger fit, value lane, PPO run or data-generation queue.

**Acceptance:** the next investment has a causal explanation and an observable gameplay consequence. Later competence audits, strategic methods, native confirmation and live-game deployment each receive their own bounded execution plan when reached.
