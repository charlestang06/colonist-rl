# V2 throughput timing diagnosis (no game screen)

Prospective v2 source-tree SHA-256: 02639f3475b047562b66b96ad1a366d65f3a3a9bb509c0afe56cb83a043f2e3f. The new TRAIN decision-audit SHA-256 is 47f8e6afd59b4f673f23ac5dd075bb8ff138d3349651a6bcaabdf16e389594be. All 139 raw END/bank before views were eligible. Incremental wrapper_time_ns excludes the unchanged parent planner call and includes public root projection and batched exact four-roll scoring.

| New v2 TRAIN measure | Incremental milliseconds |
| --- | ---: |
| mean | 20.065 |
| median | 16.849 |
| nearest observed p90 | 26.983 |
| interpolated p95, frozen gate metric | 28.670 |
| maximum | 41.793 |

Root-count stratification is diagnostic only, not a different gate: 130 five-root menus had median 16.757 ms and p95 26.811 ms; nine nine-root menus had median 29.977 ms and p95 40.808 ms. Nine roots produce 18 endpoint scenarios, requiring two chunks under the frozen maximum of 16. The overall p95 remains 28.670 ms and fails the fixed 25 ms threshold.

Before the new TRAIN run, a read-only exact parity check on the archived v1 full public before views found 0 root/score/diagnostic/action mismatches across all 129 views. A separate after-fail, alternating-order matched public-view timing replay used those same archived before views but read no outcomes or judge packet: 127 eligible views in each arm, scalar v1 median 49.306 ms versus batched v2 median 13.464 ms; retrospective p95 96.977 ms versus 26.324 ms. These retrospective times are sensitive to process/caches and are not the prospective gate result. They support a real speedup without making v2 pass.

A later distinct hypothesis could batch more than 16 scenarios or replace the ordered-sequence array with a compact integer transition table, then freeze new source and disjoint TRAIN seeds. This v2 source and its failed gate were not tuned on the new TRAIN data.
