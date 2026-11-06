# Fixed value-head study: execution accepted, advancement gates not passed

Four standalone value heads were fitted on the fixed raw-planner continuation
corpus. The recurrent representation did not pass its both-control gate, and
neither representation passed the 128-to-256-game data-benefit gate. No actor,
checkpoint or deployed policy was promoted.

The corpus has 256 TRAIN games and 64 validation games on 16 validation boards,
with a predeclared nested TRAIN128 subset. The immutable cache includes all 27,706
own inputs, including 11,818 forced inputs; only the 8,786 ordinary eligible rows
enter fitting/scoring. The three class gates passed: 33 wins/128 nested TRAIN,
71/256 full TRAIN, and 17/64 validation. All roles and original failed attempts
remain unchanged.

Each learned head used the same fixed 512 Adam updates of 64 presentations. Both
representations shared the persisted schedule at each size. Total work was four
fits, 2,048 updates and 131,072 presentations. Normalization was TRAIN-only and
game-weighted; the parent encoder remained frozen. No NN forward, checkpoint
load, codec, engine or game ran during fitting.

## Primary validation results

These are game-first, four-seat board means on early/mid validation observations.
Lower Brier and log loss are better. They are prediction measurements under the
raw continuation, not optimal value or playing-strength measurements.

| TRAIN games | Input/control | Brier | Log loss |
|---:|---|---:|---:|
| 128 | TRAIN constant | 0.195104 | 0.578950 |
| 128 | Public 63 features | 0.172013 | 0.514478 |
| 128 | Frozen recurrent 32 features | 0.184896 | 0.554944 |
| 256 | TRAIN constant | 0.195227 | 0.579257 |
| 256 | Public 63 features | 0.150774 | 0.459405 |
| 256 | Frozen recurrent 32 features | 0.181856 | 0.545644 |

The recurrent-256 Brier reduction versus constant was +0.013371, with 95% lower
bound −0.009267. Versus public-256 it was −0.031082, with lower bound −0.076204
and worse log loss. The required conjunction failed.

Public-256 versus public-128 improved the point estimate by 0.021239, but its lower
bound was −0.000266. Recurrent-256 versus recurrent-128 improved by 0.003040,
with lower bound −0.002432. Both data-benefit gates failed; this does not establish
that more data has no benefit. Public-256 also has a favorable point reduction
versus its constant, +0.044453, but interval [−0.010274, 0.097791] crosses zero.
None of these point estimates is a confirmed gain.

All intervals use the frozen 20,000 common whole-board PCG64 resamples, seed
202611061 and linear percentiles. No clipping, feature selection, checkpoint
selection, hyperparameter tuning or metric substitution followed validation.
Ordinary/endgame and fixed turn/VP/hindsight/calibration diagnostics retain empty
and partial-support cells. Hindsight bins never entered features or training.

## Verification and limits

Root session 61368 exited zero in 7.891 seconds, with guard zero. Its 37 live OS1
samples peaked at 149,618,688 bytes; the worker's self-reported process peak was
163,921,920 bytes, both below 2 GiB. The initial attempt stopped before data or
numerical admission because one closed metadata record was still writable. The
record was byte-verified and sealed; a separate receipt ran the same worker,
inputs and fit settings. No successful fit was repeated.

Independent saved-result audit recomputed all six predictions and all 228
subset/diagnostic cells from accepted cache vectors and saved coefficients using
stdlib math. Maximum logit error was 1.78e−15; Brier and log-loss errors were below
1.4e−15. It also recomputed TRAIN norms, constants, hashes, schedule eligibility
and equal-game exposure. Root separately regenerated the exact bootstrap index
hash and all 30 contrast endpoints within 1e−12, without fitting. That brief
bootstrap check had one early resource sample; it is not a representative peak
measurement. Actual fit work has its separate sampled and self-resource evidence.
Optimizer execution remains tied to the reviewed unchanged helper and durable
work counters; the independent auditor did not retrain the models.

Keep the result and stop the unchanged recurrent/data-size treatment. More PPO,
model capacity or updates do not follow from this experiment. The next strength
question is which existing public policy is actually the best incumbent: persistent
H, the raw planner or the frozen neural parent, compared on matched boards alongside
authentic VF/AB. The public-head planning proposal remains an unproven hypothesis
requiring a separately justified, controlled gameplay experiment.

[Accepted fixed results](../runs/swarm-2026-11-04/agent-research/complete-history-value-cache-head-001/Root-fixed-fit-results-acceptance-001.json),
[independent audit](../runs/swarm-2026-11-04/agent-research/complete-history-value-cache-head-001/fit-empirical-review-001/REVIEW.md),
[Root interval verification](../runs/swarm-2026-11-04/agent-research/complete-history-value-cache-head-001/fit-empirical-review-001/Root-bootstrap-addendum-001.md),
[fixed fitted arrays](../runs/swarm-2026-11-04/agent-research/complete-history-value-cache-head-001/fits-run-002/worker/models.json).
