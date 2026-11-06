# Unified Public Continuation V2 Implementation Plan

**Goal:** Test a substantive goal-directed public successor search with
symmetrically available known-cost card-purchase options, while preserving v1
as the negative control and leaving the raw planner unchanged.

**Spec:** `docs/unified-public-continuation-redesign-v2-2026-11-06.md`
**Protocol:** `docs/unified-public-continuation-protocol-v2-2026-11-06.md`

1. Add tests first for a goal-directed successor generator: a trade that
   reduces a point-recipe deficit survives; irrelevant exchanges are pruned;
   direct point builds and roads with a ≤2-road public site path remain. Use
   the two existing historical public BUY fixtures to pin node/geometry caps
   and a conservative projected win.
2. Create `public_goal_continuation.py` with a v2 policy class that inherits
   the v1 eligibility and exact-root guard but overrides root comparison.
   Keep v1's search function and default behavior unchanged. Search at most
   four actions with shared 512 transition attempts, beam 12, geometry cap
   100,000. Retain no-card and one-card route variants per root. For a later
   BUY require a current legal BUY menu certificate, pay public cost only, and
   never inspect the card identity. Select a complete conservative projected
   winning route before uncertain scalar ranking.
3. Add tests for symmetric city→BUY and trade→BUY variants, card/deck
   ambiguity abstention, exact current first-action identity, cap fallback,
   direct-root continuation, and the first two public BUY fixtures. Add the
   opt-in `public_goal_continuation_v2` paired CLI arm and its validation tests.
4. Implement a public-only reused-corpus engineering probe with the fixed
   SHA-sorted 256+256 TRAIN observation selection. It must never read broad
   `val-*`, `student_action`, `transition`, private overlays, labels, or
   outcomes. Before running it, commit the source and v2 protocol and record
   hashes. Require exact-root parity, no missed known win, p95 ≤25 ms, and
   ≥8 changed roots across ≥4 boards, with ≥8 eligible parent BUY states.
5. If the engineering gate passes, coordinate CPU with the learning lead,
   then collect fresh 32-game raw TRAIN shadow decisions, blind-review the
   first 12 changes with factual route certificates, and only afterward run
   the excluded smoke and 96-game paired development panel. Stop at the first
   failed gate without retuning the panel. Commit a result with honest limits.
