# Unified public continuation v2: prospective gates

This protocol supersedes the v1 engineering gate only for the opt-in v2 arm.
V1's zero-intervention result and source/protocol digests remain unchanged.
V2 source and this protocol are committed and hashed before the reused corpus
gate. Fresh 15751/52/53 boards remain unopened until that gate passes.

## Fixed engineering data and roles

Use only `history.json.gz` in
`/private/tmp/colonist-swarm-bxz1k39_/broad-action-value-distill/runs/raw-ranking-pilot-2026-11-06/boards/train-*/`.
Read each row's serialized public before-Observation and exact current legal
menu. Do not read `val-*` histories, student actions, transitions, raw/H
overlays, labels, private evaluation, outcomes, or review packets. This is a
**reused engineering corpus**, never independent validation or a training
target. The selected input file hashes are saved in the gate receipt.

Select observations with standard, own, rolled `TURN`/`PLAY_TURN` context.
Partition by whether the exact current legal menu includes BUY. Within each
partition sort by SHA-256 of canonical public Observation JSON, then take at
most the first 256. This rule is fixed before looking at the observation
contents. Run the unchanged raw parent on each selected view; only parent
END/BUY decisions enter the wrapper latency/intervention denominator. Keep
the full public-view and exact-menu hashes in per-decision diagnostics.

## Ordered mechanical and teacher-quality gates

1. Tests must pin exact current Action identity, no mutation, direct legal
   win, a complete conservative projected trade→point win, BUY→later
   trade/build, direct build→later point, later bank flexibility for END and
   trade-now, card-cost-only later BUY after a public BUY menu, cap fallback,
   uncertainty abstention, and the scalar four-roll oracle. The two older
   public BUY fixtures must finish the bounded comparison without a geometry
   cap; the nine-point fixture must select its complete known winning route.
2. Run v2 on the selected reused public corpus. Require no exact-root mismatch,
   no known win missed, incremental p95 ≤25 ms over eligible parent END/BUY
   decisions, and at least eight changed first roots across four source
   boards. Report BUY and END counts separately, changed-route kinds, cap
   fallbacks, candidate variant scores, and all abort reasons. If the corpus
   has fewer than eight eligible parent BUY states, it is insufficient for a
   BUY-continuation quality claim; use the older public fixtures as mechanical
   evidence only and stop before fresh outcome games. If a changed first root
   is obviously destructive on a blinded public view, stop.
3. If the engineering gate passes, notify the learning coordinator before any
   CPU-heavy run. Collect 32 raw focal TRAIN games on boards 157510000–007,
   all four seats against three trading-H opponents with no domestic offers.
   Run v2 in shadow on raw public observations; it cannot alter those games.
   Save exact menu/actions, public diagnostics, and before/after facts.
   Require at least eight changed roots across four boards, no invalid root,
   no missed known win, and incremental p95 ≤25 ms. Independently review the
   first 12 changed public cases before joining outcomes. Certify any claimed
   current-turn build by exact synthetic public-consistent engine replay.
4. If the TRAIN gate passes, run one excluded smoke board 157529999 and the
   eight fresh development boards 157530000–007 × four seats × unchanged raw,
   v2 buy-only, and v2 unified arms. All 96 games must resolve. A practical
   nomination requires at least three more wins of 32 over both controls,
   with no verified destructive root or missed win. Report paired wins, VP,
   board uncertainty, actual route follow-through, trade loops, and latency.
   Failure retires v2 without tuning this panel; passage requires a separate
   H/VF/AB confirmation before promotion.

No PPO, hidden card identity, native privileged state, or VAL/EVAL decision
label enters a v2 policy decision. The four-roll proxy still omits opponent
spending/trading, robber movement, discards, and hidden card effects. This
bounded screen does not establish that v2 is stronger than H or its parent.
