# Outcome-trained public-history critic: learned, but did not pass the gate

The bounded recurrent critic fit completed. It improved on a matched frozen-GRU
head, but did not improve on the simpler public-feature control. It remains an
offline research result; no actor, planner score, or default player changed.

## Fixed development test

The [predeclared hypothesis](../runs/swarm-2026-11-04/agent-research/shadow-critic-001/hypothesis.md)
used accepted public raw-planner histories against three persistent trading H
opponents with domestic offers disabled. Bank trades remained legal. The 256
TRAIN games contained 71 focal wins; the 64 exposed VAL games on 16 boards
contained 17. Each 63-feature public vector was fed in chronological own-action
order, including forced inputs. A GRUCell(63, 32) and win head learned terminal
outcomes. The matched control froze the same initial GRU and trained only its
head. Both arms used four fixed epochs, four games per update, and 256 updates.
The 256-game constant, public-63 and frozen actor-recurrent-32 logistic controls
were scored from their previously saved coefficients.

Early/mid VAL loss averaged eligible roots within each game, four seats within
each board, then 16 boards equally. Ranking used one predetermined first
eligible early/mid turn-20-to-59 root per game, comparing winning and losing
focal seats within the 12 boards where both occurred. It ranks **observed
states**, not unchosen actions.

| Model | Brier ↓ | Log loss ↓ | Within-board winner ranking ↑ |
|---|---:|---:|---:|
| Constant TRAIN win rate | 0.19523 | 0.57926 | 0.5000 |
| Prior public-63 head | **0.15077** | 0.45940 | **0.8125** |
| Prior frozen actor-recurrent-32 head | 0.18186 | 0.54564 | 0.6181 |
| Matched frozen-GRU/head-only | 0.20668 | 0.60203 | 0.6736 |
| Trainable public-history GRU | 0.15100 | **0.45560** | 0.7361 |

Across 20,000 paired whole-board bootstrap resamples, trained-minus-public-63
improvement was **−0.00023 Brier** (95% interval −0.01639 to 0.01763) and
**−0.0764 ranking** (−0.1597 to 0.0000). A 0.00380 log-loss improvement had an
interval of −0.0445 to 0.0603. Against its frozen-GRU control, the trainable
model improved Brier by 0.05568 (0.01085 to 0.09462). The five fixed
probability-bin reliability cells and exact model outputs are in the
[analysis](../runs/swarm-2026-11-04/agent-research/shadow-critic-001/val-analysis-002.json).
The 1,465 scored roots are repeated labels from 64 games, not independent
outcomes.

The [root receipt](../runs/swarm-2026-11-04/agent-research/shadow-critic-001/Root-run-001.json)
records exit 0, 9.14 seconds outer wall time and 417 MB sampled peak RSS;
the [worker record](../runs/swarm-2026-11-04/agent-research/shadow-critic-001/run-001/work.json)
records 87,620 chronological TRAIN input presentations and 28,060 eligible
loss presentations per arm. Root verified the accepted cache and assembly
byte hashes before fitting, ran four inert worker tests, and saved the worker,
coefficients, 64-game prediction summaries, 1,771 eligible VAL rows, analysis
code, and hashes in the [archive](../runs/swarm-2026-11-04/agent-research/shadow-critic-001/archive-manifest.json).

## Decision

The predeclared gate required at least 0.01 Brier improvement over **both**
controls with positive paired lower bounds, no mean log-loss regression against
public-63, and at least 0.05 ranking improvement over public-63 with a positive
paired lower bound. It failed the public-63 Brier and ranking requirements.
Stop this treatment without changing epochs, model size, metric or cohort.
These VAL boards were exposed in the prior head study, so even a pass would
have been a development nomination rather than independent confirmation.
The next learning test should target a different decision-relevant mechanism;
the public-63 model remains a calibration reference, not a proven action scorer.
