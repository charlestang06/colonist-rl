# Local learning pilot, 2026-11-04

The single bounded pilot completed its requested schedule in **92.301 seconds**
under the 900-second total limit. It collected 5,000 demonstrations across two
independent training seeds and finished all 64 validation comparison games plus
four separate probe games. Every evaluation game terminated naturally; there
were zero truncations, unresolved outcomes, deadline interruptions or failed
stages. The imitation students each won **0/8** validation games. The operational
baseline gate is **false** and `ppo_status` is
`deferred_insufficient_baseline_evidence`. No PPO ran.

## Reproduction and provenance

Implementation/tests were committed **before** launching the real run as
`1350de2f15499d36a90181e7d25d555de2f4313a`. The working tree was clean at launch
and at the final source check, and no source edits occurred during the run.
The exact content identity in both records is
`git:1350de2f15499d36a90181e7d25d555de2f4313a`.

```sh
PYTHONPATH=/private/tmp/colonist-swarm-bxz1k39_/learning/src \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m colonist_rl experiment \
  --output runs/learning-pilot-2026-11-04 --decisions 2500 --epochs 2 \
  --hidden-size 32 --wall-seconds 900 \
  > /private/tmp/colonist-learning-pilot-2026-11-04.stdout.json
```

Exit status was 0. There was one run, no restart or extension. It used the
existing local Python 3.13.2, Torch 2.15.0, NumPy 2.5.4, Catanatron 3.2.1 and
Gymnasium 1.4.0 runtime, CPU, one Torch thread and one worker. No packages were
installed and no network or paid services were used. Full validation before the
implementation commit: **277 passed, 1 skipped in 6.70s**; Ruff passed.

The [manifest](experiments/learning-pilot-2026-11-04/manifest.json) was written
before compute. Its SHA-256 is
`7eb7af6cbf3be20e6b5d26c8d747947ee957c338082f6dc0d6ca48a06ad4d01a`.
Its timestamp-independent configuration digest is
`ad81e314a6f0cab36e6dd2f4729d7e7c860dcb3d0547bad6e65f6facd309556d`.
The [raw results](experiments/learning-pilot-2026-11-04/results.json) preserve
all per-game rows, per-seed fits/metrics, coverage, stage timing and source checks;
their SHA-256 is
`df643321c0e5222a83b9c8eba11054b4028d35be8d772c857444d57f367080d6`.
[Artifact index](experiments/learning-pilot-2026-11-04/artifact-index.json) contains
all raw-file and large-artifact paths, byte sizes and digests. Large local
files remain in ignored `runs/learning-pilot-2026-11-04/`.

Training initialization seeds were 11 and 29. Their board candidate lists were
exactly 11000–11063 and 29000–29063. Actual consumed boards were 11000–11031
and 29000–29027, respectively. The common validation boards were 90000 and
90001, diagnostic demonstrations used 91000 and 91001, and the probe used
92000. All partitions were disjoint. Final-test seeds **70000017 and 70000019
were reserved and never consumed**. Teachers and actors received only immutable
Observations; no oracle labels were added.

## Budget probe and training

The separate [probe evidence](experiments/learning-pilot-2026-11-04/probe.json)
records one four-seat heuristic-versus-heuristic board block: four natural
terminations, one candidate win, zero truncations and 341 candidate decisions
in 1.622 seconds. Measured throughput was 0.405 seconds/game and 210.25 teacher
decisions/second. The declared scheduling factors retained both validation
blocks and both 2,500-decision caps before training. There were **no reductions
or demonstration shortfalls**. The initial comparison reserve was 51.90 seconds;
unused learning time remained available to comparisons. These timing estimates
were budget heuristics, not independent graph-policy measurements.

Each seed independently initialized Python, NumPy and CPU Torch, a model with
hidden size 32, and an optimizer. Its untrained and initial-student checkpoints
contained equal initial weights. Imitation used two epochs, sequence length 16,
Adam learning rate 0.0003 and non-forced choices only; all graph evaluations
used deterministic argmax. Validation demonstrations never entered a gradient.

| Training seed | Decisions | Non-forced | Complete training games | Decision-cap episodes | Updates | Mean training CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 2500 | 1460 | 31 | 1 | 338 | 1.66746 |
| 29 | 2500 | 1543 | 27 | 1 | 338 | 1.65704 |

There were zero step-truncated training episodes. A decision-cap episode is an
incomplete sequence, not a terminal game. Training loss is an optimization
diagnostic and does not establish playing competence.

## Full-game comparison results

Every row uses the identical two validation boards and four candidate seats per
board. VP is candidate final VP; latency is candidate-only mean / p95
milliseconds. All rows have eight resolved games and zero truncations.

| Candidate | Opponents | Wins / games | Mean VP | Decision latency ms, mean / p95 |
| --- | --- | ---: | ---: | ---: |
| Heuristic | Random | 8 / 8 | 10.125 | 0.0514 / 0.2332 |
| Random | Random | 4 / 8 | 7.500 | 0.00075 / 0.00104 |
| Heuristic | Heuristic | 3 / 8 | 6.750 | 0.0275 / 0.1437 |
| Random | Heuristic | 0 / 8 | 2.625 | 0.00080 / 0.00121 |
| Untrained, seed 11 | Heuristic | 0 / 8 | 2.000 | 0.7045 / 1.1103 |
| Imitated, seed 11 | Heuristic | 0 / 8 | 3.375 | 0.6034 / 0.9455 |
| Untrained, seed 29 | Heuristic | 0 / 8 | 2.625 | 0.6061 / 1.0768 |
| Imitated, seed 29 | Heuristic | 0 / 8 | 3.250 | 0.6250 / 0.9997 |

The [paired comparisons](experiments/learning-pilot-2026-11-04/comparisons.json)
use exact matching board/seat keys and the reviewed `paired_block_summary`.
Imitation minus untrained mean VP was **+1.375** for seed 11 and **+0.625** for
seed 29, with zero win-rate change for both. Imitation minus heuristic teacher
mean VP was **−3.375** and **−3.500**, respectively, and win-rate difference
was **−0.375** for both. These two-board descriptive differences do not support
a significance claim, model selection, or a competence claim.

## Held-out agreement and coverage

The held-out diagnostic sequences contain 128 choices, including 83 non-forced
choices. Both students agree with the teacher on 77/128 overall (**60.16%**) but
only 32/83 non-forced (**38.55%**). Non-forced cross-entropy is 1.60710 for seed
11 and 1.60788 for seed 29. The forced-inclusive number would mask the shortfall.

| Phase | Held-out non-forced choices | Seed 11 correct | Seed 29 correct |
| --- | ---: | ---: | ---: |
| Initial settlement | 4 | 1 | 0 |
| Initial road | 4 | 1 | 0 |
| Discard | 6 | 0 | 0 |
| Move robber | 15 | 0 | 1 |
| Play turn | 52 | 28 | 29 |
| Trade response | 2 | 2 | 2 |

The two trade responses are too few for a competence conclusion. There were
zero held-out examples of action kinds `DISCARD`, `PLAY_YEAR_OF_PLENTY`, and
`REJECT_TRADE`. The declared phase vocabulary had no `TRADE_CONFIRM`, `ROBBER`,
`SETUP_ROAD`, `SETUP_SETTLEMENT`, or `TURN` examples. Several are compatibility
names; the actual backend emitted `MOVE_ROBBER`, `BUILD_INITIAL_ROAD`,
`BUILD_INITIAL_SETTLEMENT`, and `PLAY_TURN`, as preserved in raw coverage.
No observation of a missing phase/action implies no measured agreement for it.
Per-action and per-phase CE/accuracy/counts remain in raw results for both seeds.

## Gate and remaining gaps

`eligible=false`: only **two** complete blocks per variant/seed were collected,
below the required eight; only **83** held-out non-forced choices were present,
below 100; agreement was below 0.60 overall and in several observed phases;
and both seeds failed the teacher VP and win-rate tolerances. The zero-truncation
and exact-pairing conditions passed. No stage failed or remained unknown in this
run. Remaining competence and coverage evidence is insufficient.

Execution status `complete` describes finishing the requested schedule. It does
not assert baseline competence. `ppo_status="deferred_insufficient_baseline_evidence"`
is the operational decision. The protocol describes a subsequent bounded fresh
1024-decision/seed, two-epoch PPO continuation only after every baseline gate
passes; that continuation did not run here.

The current evaluator has no progress callback: atomic persistence occurs at
phase return, after its requested board list. Cooperative deadline returns
retain partial games, while abrupt process death could lose the in-flight
variant schedule (two blocks in this run). The metrics deadline guard stops
before each policy forward and is always removed. Scheduling estimates and
these persistence/API limits remain documented engineering gaps, not hidden
success assumptions.
