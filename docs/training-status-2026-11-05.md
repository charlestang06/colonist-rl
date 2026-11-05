# Training status — November 5, 2026

The learner remains weaker than the fixed heuristic opponents. All completed
experiments and failed hypotheses are preserved; passing tests is not a strength
claim. This is an intermediate checkpoint while the next learning wave runs.

| Experiment | Teacher agreement on non-forced choices | Heuristic-opponent outcomes | Interpretation |
|---|---|---|---|
| Original two-epoch pilot, seeds11/29 |32/83 each|0/8 wins each; meanVP3.375/3.250|Weak initial baseline|
| Separate twenty-epoch uniform diagnostic |43/83;40/83|0/8 each; meanVP3.875/3.125|More agreement without established strength|
| Controlled twenty-epoch uniform, new four-board panel |50/77 each|0/16 each; meanVP3.625/3.500|Reference for balanced supervision|
| Controlled twenty-epoch phase-balanced |49/77;48/77|0/16 each; meanVP4.125/4.500|Higher VP in this small panel, no wins|

The controlled four-arm study completed all requested epochs and3,380 updates
per arm, with identical per-seed initial weights and datasets, in559.538/900seconds.
Its fixed heuristic comparator won3/16 with meanVP6.5625. Boards shared across
seats are dependent; these are descriptive results. The new77-choice validation
set differs from the earlier83-choice set and their percentages are not a
matched improvement comparison.

The baseline public-observation audit replayed36/36 sampled choices exactly.
It found24 reasonable,11 ambiguous and one clearly bad choice. Neither the
two-epoch nor twenty-epoch uniform students selected bank trades on their known
validation boards (0/143 and0/202 opportunities respectively). No pass skipped
a directly legal purchase. Intermediate conversions remain a concrete concern.

The follow-up training audit reproduced5,000/5,000 teacher labels and983/983
action vectors. Relevant gradients reach the own-hand inputs. Teacher jitter
and exact-score ties create legitimate label variation across RNG streams.
Measured prefix replay accounts for68.1% of forward time, while graph construction
accounts for40.4%; these overlapping measurements must not be added. A tiny
full-context training probe improved overall fit but reached only2/9 bank labels
for both initializations after100epochs. This identifies neither a broken
encoding nor a proven model-capacity limit.

The user authorized the next wave: profile-guided batching/static-feature caching;
a separate public consequence-feature model; audited teacher labels on student
states with dataset aggregation; and progressively harder PPO opponents afterward.
The current PPO reward is terminal winning plus victory-point potential shaping.
Behavioral cloning does not use that reward. Reward changes require their own
measured comparison. Curriculum code is implemented and under independent review;
no sustained PPO curriculum has run at this checkpoint.

The [experiment archive](../runs/swarm-2026-11-04/README.md) links raw outcomes,
checkpoints, datasets, traces, diagnostics, manifests and review records.
[Artifact locations](../runs/swarm-2026-11-04/artifact-locations.json) map historical
scratch paths to preserved bytes. The original internal work horizon elapsed
during a long tool/approval wait; catch-up work and the later explicit user
request are recorded separately rather than relabeling elapsed time as training.
