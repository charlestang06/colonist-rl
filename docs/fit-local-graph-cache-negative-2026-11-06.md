# Fit-local graph-input cache experiment: stopped

The 60-update planner-point pilot gave a short, already sealed workload for
testing whether repeated prefix encoding is worth caching. Its fit manifest is
SHA-256 `c51fcc9121367f75c384adcb293ccef7fde398890251f1b5ee7ae2ff8f9cbd5a`.
This experiment did not use fresh point-affordance labels, change a training
target, or produce a candidate checkpoint.

An isolated branch added an opt-in 256-view LRU of raw observation graph inputs
for one `fit_sequences` call. It keys on observation identity, device and dtype,
returns fresh mutable tensor copies, and retains no learned nodes or recurrent
memory. The default fitter path is unchanged. Two tests were observed red before
implementation and then passed: exact anchored-update parity and independent
returned buffers.

One paired, single-thread run used identical parent checkpoints, accepted TRAIN
targets, 60-update schedule, optimizer and static-graph cold start. The benchmark
script is `/private/tmp/colonist-swarm-bxz1k39_/fit-throughput-benchmark.py`
(SHA-256 `85895536d1dfd36bc434cadbf93df59dbc764a8d2a15eeed2c57c12309601920`).
It imports the sealed pilot artifact read-only and makes no checkpoint files.

| Measure | Uncached | Opt-in cache |
| --- | ---: | ---: |
| Fitter wall seconds | 1.495037 | 1.589616 |
| Encoding scope seconds | 0.601537 | 0.507429 |
| Prefix scope seconds | 1.026628 | 1.056851 |
| Chunk scope seconds | 0.423223 | 0.495277 |
| Prefix / chunk forwards | 2,668 / 960 | 2,668 / 960 |
| Updates / final loss | 60 / 3.375418903430303 | 60 / 3.375418903430303 |

All final parameter tensors and Adam state tensors were bitwise equal. Encoding
time fell by 0.094 seconds, but total fitting was 0.095 seconds slower in this
pair. A single timing pair cannot establish a precise performance difference;
it does rule out a large benefit on this workload. Do not integrate or use this
cache in the next study.

The sealed earlier pilot recorded only about 1.4–1.6 seconds of fitter time per
arm, while its complete fit-and-score attempt took about 14.5 seconds. Future
throughput work should first segment the complete attempt, particularly source
validation and repeated labelVAL scoring, before changing the trainer again.
