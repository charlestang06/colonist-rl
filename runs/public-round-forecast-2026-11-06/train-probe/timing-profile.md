# Incremental timing diagnosis (saved TRAIN views; no new games)

Frozen gameplay source SHA-256: dfd89d5c0a6b190240f2889d3bce3368bb6383e5cd2a332ff580ee054d8921f4.
Input decision-audit SHA-256: a49ed39e572b71f227f35ead5f94f0186ceb2217d15c476e305e0ac78212cb87.
The frozen run had 127 eligible captured raw END/bank decisions; every eligible menu had five compared roots. Incremental wrapper timing came from wrapper_time_ns around public root projection and the four-roll forecast, excluding the unchanged parent planner call.

| Statistic | Incremental milliseconds |
| --- | ---: |
| mean | 35.616 |
| median | 37.282 |
| p90 nearest observed | 49.696 |
| interpolated p95, frozen gate metric | 51.843 |
| maximum | 91.355 |

A fresh-process cProfile replay of capture index 33 (a near-p95 case), with the DP and roll-demand caches cleared, selected END. Under profiler overhead, the whole wrapper took 153 ms, with 148 ms in ten point_affordability_mass calls and 147 ms in ten _point_mass calls; the single unchanged parent plan took 6 ms. The profile counted 847,718 function calls, including 202,740 iterations of each tuple-construction generator in the dynamic program. These profiled wall times are diagnostic and must not replace the unprofiled 51.843 ms gate measurement.

This localizes the failed latency gate to repeated Python DP tuple/state work across five original roots and two bank endpoints, rather than raw search. Public roll demand is already memoized; the next distinct throughput hypothesis would evaluate root hands in one batched production distribution or use a compact integer-state representation, then freeze new disjoint TRAIN seeds and the same action/yield/latency gates before any fresh game screen. This stopped study is not retuned on its TRAIN views.
