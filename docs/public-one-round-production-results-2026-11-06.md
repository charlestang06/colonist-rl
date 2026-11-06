# Public one-round production option: stopped before game screen

The [frozen protocol](public-one-round-production-protocol-2026-11-06.md) asked whether a fixed-board, four-roll point-affordability forecast improves a bank trade versus END after the unchanged raw planner falls back to END. This isolated study **failed its pre-game latency gate**. No smoke or fresh development game was run, so it makes no playing-strength claim.

## Frozen implementation and TRAIN probe

Implementation branch: cytang/public-production-lookahead, base 7d11170, gameplay-source commit ca0870e. Source-tree SHA-256 during the complete probe was dfd89d5c0a6b190240f2889d3bce3368bb6383e5cd2a332ff580ee054d8921f4 before and after; protocol SHA-256 was c7256c3ad7c49fbd3a742c80303320cd58e4ef9e1c44b3b137654e39c10db234. Raw planner source and the Original checkout were not changed.

The fixed TRAIN cohort used boards 138340000..138340007, all four focal seats, three H opponents, no offers, and a 20,000-step cap. All 32 raw games resolved. The collector saved every executed raw END with an available bank trade and its complete public before Observation. It captured 129 views; 127 entered the wrapper comparison. Static and forecast wrappers used the same original END/bank roots, raw parent search, 10,000,000 margin, and exact root projections. Forecast alone added 50,000,000 times the public four-roll option probability, with lower/upper bank endpoint abstention.

| Frozen pre-game measure | Observed | Gate |
| --- | ---: | --- |
| Forecast versus static changed actions | 17, across 6 boards | at least 8 across 4 boards: pass |
| Original action identity mismatches | 0 | 0: pass |
| Raw replay or search-work mismatches | 0 | 0: pass |
| Maximum forecast trades selected in one raw own turn | 1 | at most 1: pass on observed raw views |
| Incremental eligible-decision p95 | 51.843 ms | at most 25 ms: **fail** |

Static selected END on all 129 captured views; forecast selected 17 bank trades and 112 ENDs. Forecast diagnostic statuses were 17 trade, 109 insufficient gain, one ambiguous bank endpoint, and two ineligible parent states. The 17 changes occurred on six boards (4, 1, 5, 1, 2, 4). These are TRAIN intervention counts, not a population failure rate or a win comparison. The raw search's empty-beam stops were not labeled bad decisions.

The saved-view [timing diagnosis](../runs/public-round-forecast-2026-11-06/train-probe/timing-profile.md) found p50 37.282 ms and p95 51.843 ms for the incremental wrapper. All 127 eligible menus contained five compared roots. A representative near-p95 cProfile replay spent nearly all profiled time in ten four-roll DP calls (five roots times two bank endpoints), while the unchanged raw parent call was small. Profile wall times include profiler overhead; the frozen unprofiled p95 is the gate result.

## Choice-blind diagnostic

After capture, a reporting-only script made a choice-blind judge packet and [public review cards](../runs/public-round-forecast-2026-11-06/train-probe/judge-cards.md) from the saved first 12 changed public before views. The packet contains opaque IDs, complete public observations, and legal menus, without arm identities, chosen actions, arm scores, probabilities, or later outcomes. Its SHA-256 is 6a73f07cf38cda6591b0aa21dad5de83e66c76298c85f04e3e760ace60d67c0a. A choice-blind self-review was written before joining choices (review SHA-256 6509448f3bc649295c2530fc708710389043352f0c026627a5854a5b9b41164c); the hash-checked join found the forecast's trade in the reviewer-preferred set for all 12, with static END in all 12. This is a bounded public plausibility check by the experiment author, not an independent game outcome or a rescue of the latency failure. The other five changed cases remain saved in the complete audit and were not conceptually adjudicated.

The [evidence archive](../runs/public-round-forecast-2026-11-06/train-probe/train-probe-evidence.tar.gz) contains every original artifact, including the full public before views, judge packet, and root-only choice join input; its SHA-256 is 7e845999416af1a66f496b76d95825249555ebaf2d9a46863327a9171101cc4d. The main original artifact hashes, recorded in [probe-summary.json](../runs/public-round-forecast-2026-11-06/train-probe/probe-summary.json), are: raw public before views 8c3dccc5fc2f474b03218c0e2779a2a7b5433cab22af99bd4e7bfcf1f68a3cd8; all decision diagnostics a49ed39e572b71f227f35ead5f94f0186ceb2217d15c476e305e0ac78212cb87; game journal 5407ffa96f949d82da43fc67d092f5c7f0faf5f6fb0bfbd944c0cdd4f532e11d; and choice join input cfc43b55c3ee6d52d64fda73bbe36a3aea27e8156d7070f5ae0ded638c3ed31a. The prejoin review and join are separately saved in the archive.

## Verdict and next hypothesis

The fixed attempt stops here. Excluded smoke board 138349999 and development boards 138350000..138350007 remain unused. The public conditional signal produced coherent candidate trades in the first 12 choice-blind cases, but its Python dynamic program exceeded the agreed local latency budget. A future distinct experiment can batch the exact four-roll endpoint calculation across roots or use a compact integer-state DP; it needs a separately frozen source and new disjoint TRAIN seeds before any new pre-game gate. No parameter tuning or additional game screen was performed on this failed attempt.
