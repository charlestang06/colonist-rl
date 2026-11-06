# Public one-round throughput v2: faster, stopped before game screen

The [frozen v2 protocol](public-one-round-throughput-v2-protocol-2026-11-06.md) tested an exact batched calculation of the same four-roll public point-affordability option. It **failed the unchanged 25 ms incremental p95 pre-game gate** at 28.670 ms. No excluded smoke or fresh development game was run, so this study establishes no playing-strength improvement.

## Source, parity and new TRAIN probe

V2 is isolated on branch cytang/public-production-throughput-v2 from the stopped v1 commit ed80fc1. The fixed v2 gameplay-source commit is e7d4a7b; its source-tree SHA-256 was 02639f3475b047562b66b96ad1a366d65f3a3a9bb509c0afe56cb83a043f2e3f before and after new TRAIN play. Protocol SHA-256 was 1c89d57bedc821970fd0973454f8f0ba9eace97ae8b315cc27f04ec046b629e3. The v1 branch/results and Original checkout were not changed. The pinned runtime recorded Python 3.13.2 and NumPy 2.5.4.

The exact v1 scalar routine remains the reference. Synthetic tests covered correlated production, robber, bank shortage, city/settlement/no-goal cases, starting affordability, and more than 16 endpoint scenarios. A [read-only archived-public-view parity check](../runs/public-round-throughput-v2-2026-11-06/v1-public-view-parity.json) compared the frozen v1 implementation with v2 on all 129 archived v1 public before views: **zero** root, score, diagnostic, search-work, or original-action identity mismatches. It read no game outcomes or judge packet.

The new v2 TRAIN cohort used boards 138360000..138360007, all four focal seats, raw versus three H opponents, no offers, and the 20,000-step cap. All 32 games resolved. It saved 139 complete public raw END/bank before views, all eligible for the static/forecast comparison. Static selected END on all 139; forecast chose 30 bank trades and 109 ENDs. The 30 forecast-versus-static changes occurred across all eight boards. Forecast statuses were 30 trade, 107 insufficient gain and two ambiguous bank-endpoint abstentions. These are intervention counts from raw TRAIN states, not game-strength outcomes.

| Frozen pre-game measure | V2 observed | Gate |
| --- | ---: | --- |
| Forecast versus static changed roots | 30 across 8 boards | at least 8 across 4: pass |
| Raw replay, original-root identity or search-work mismatches | 0 | 0: pass |
| Maximum forecast trades selected per raw own turn | 1 | at most 1: pass on observed raw views |
| Incremental eligible-decision p95 | 28.670 ms | at most 25 ms: **fail** |

The [timing breakdown](../runs/public-round-throughput-v2-2026-11-06/train-probe/timing-profile.md) shows new TRAIN median 16.849 ms. Nine of 139 eligible menus had nine original roots, requiring two batches of up to 16 endpoint scenarios; their p95 was 40.808 ms. The other 130 five-root menus had p95 26.811 ms. A separate [matched retrospective public-view timing replay](../runs/public-round-throughput-v2-2026-11-06/v1-v2-matched-public-latency.json) found a scalar-to-batch median reduction of 49.306 to 13.464 ms on the same archived 127 eligible views, with no outcomes read. That diagnostic supports a speedup but does not replace the prospective 28.670 ms gate result.

## Choice-blind diagnostic and evidence

The first 12 changed cases were rendered into a choice-blind public packet and [review cards](../runs/public-round-throughput-v2-2026-11-06/train-probe/judge-cards.md) containing only opaque IDs, public before observations, and exact legal menus. The root-only action selection stayed separate. Packet SHA-256: 396fcb3abd8d19f3d5d415b48d2491fec0f3ecb8a9ead3206c38f42f02429f2c. The experiment author wrote and hashed a [prejoin public review](../runs/public-round-throughput-v2-2026-11-06/train-probe/judge-review-prejoin.json) (915a4ac292b33cfa5928f832fc14dee15d2cf6eb81043cef5596228f69230c9f) before joining arm choices. The [hash-checked join](../runs/public-round-throughput-v2-2026-11-06/train-probe/review-join.json) found forecast choices in that preferred public set for all 12, with static END in all 12. This is a bounded, author-conducted plausibility check, not an independent review or a rescue of the latency failure. The other 18 changed cases remain in the complete saved audit and were not conceptually adjudicated.

The [evidence archive](../runs/public-round-throughput-v2-2026-11-06/train-probe/train-probe-evidence.tar.gz) contains the exact full public before views, root-only selection, packet, review, join, game journal, timings and manifests. Archive SHA-256: ad213ec57a50da968ac89a9699472372580eb48c709c8de080e82961119e1402. All five original frozen artifact hashes in [probe-summary.json](../runs/public-round-throughput-v2-2026-11-06/train-probe/probe-summary.json) were verified against the archive. Raw public before views SHA-256: f73cf98db7e1eb0378b93266b5601d1ea471c4812c79519ab7c1cfe954d57f8f. Decision audit SHA-256: 47f8e6afd59b4f673f23ac5dd075bb8ff138d3349651a6bcaabdf16e389594be.

## Verdict

This exact v2 configuration remains optional and stopped. Excluded smoke seed 138369999 and fresh development seeds 138370000..138370007 are unused. The batch implementation shortened the same public calculation and preserved decisions, but did not meet the fixed latency budget on new TRAIN views. A further throughput change requires a distinct frozen source and new disjoint TRAIN seeds; no threshold or source was retuned on this failed cohort. Neither v1 nor v2 has a fresh game screen for this intervention.
