# Reproducible teacher and imitation baseline

Learning is primary. This architectural first wave implements the L1/A9 measurement and imitation foundations needed to decide whether L2 PPO is justified. The user has authorized execution and subagent coordination. It does not claim that a competent student can be obtained within a tiny local pilot.

## Existing evidence and design choice

At baseline `182cc01c8670234012f96a0389cef236296a93c5`, 204 tests pass and the optional live-browser smoke is skipped. `training.imitation_warmstart` saves detached hidden states computed before parameter updates, trains forced decisions equally, and reports only aggregate loss. `training.evaluate` rotates seats but permits incomplete board blocks and records no experiment identity. The graph policy, observation-only heuristic, simulator, checkpoint and fresh PPO interfaces already exist.

Chosen approach: extend evaluation through a small protocol module, add a separate demonstration/imitation module, and add an experiment runner with a CLI entry point. Preserve existing public calls. Alternatives rejected for this wave: integrating uninstalled engine-native teachers (requires dependency/version and hidden-state auditing beyond the no-install budget); starting a large PPO sweep (baseline competence is not established). Existing `HeuristicPolicy` is the eligible observation-only teacher candidate; compare it with random and heuristic opponents and call its strength unproven until measured.

## Binding constraints

- Work only in `/private/tmp/colonist-swarm-bxz1k39_/learning`; never modify the original repository or its environment.
- Use `/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python` and the existing ruff executable, with `PYTHONPATH=/private/tmp/colonist-swarm-bxz1k39_/learning/src` and `PYTHONDONTWRITEBYTECODE=1`.
- CPU only, one Torch thread, one simulation/training worker, no paid services, no network installs, and no dependency changes.
- Actors and teachers receive only immutable `Observation`; simulator-private state is permitted only for outcome diagnostics after a game, never labels, features, masks, or action selection.
- Preserve existing CLI commands, imports, checkpoints and action/observation schema. Own learning code, learning tests and learning docs; do not modify adapters, contracts, board, beliefs, simulator or opponents without root agreement.
- Never report forced-only accuracy, finite losses, passing tests, incomplete games, or manual browser victories as evidence of learned competence.
- Training, model-selection validation and reserved final-test board seeds must be disjoint and persisted. All variants compare the same complete four-seat blocks. Report every truncation and failure without converting either into an automatic loss.
- Fresh implementer per task, independent spec-and-quality task review, reviewed fixes, and a final broad review are required. Coordinators do not implement production code.

## Evaluation protocol

An explicit nonempty unique list of board seeds produces all four learner seats per board. Candidate is the existing heuristic, random, or a supplied GraphPolicy. Opponents are existing random, heuristic or mixed policies. Reset candidate/opponent RNG and recurrent memory per game using recorded deterministic seed derivation. Save sampling/argmax mode. Preserve caller RNG states and policy train/eval state, including exceptional exits.

Keep raw outcomes (board seed, seat, candidate, winner, won, VP, terminated/truncated status, steps, decision counts, non-forced counts, trade responses/offers and latency), paired per-board candidate differences and aggregates. Pair only identical board/seat keys; reject duplicates or mismatches. Confidence intervals that ignore board dependence remain explicitly nominal. Missing and truncated results cannot be called complete evidence.

## Demonstrations and imitation

Collect ordered teacher decisions from completed simulator games with a declared hard decision cap and game-step cap. Each record has its observation, legal teacher-action index, board/seat/episode identity and decision phase. Record action-kind and phase coverage, forced/non-forced counts, completed and truncated games, wall time and exact seeds. If the decision cap ends a game, mark the episode incomplete; do not invent terminal returns. Persist JSON-compatible observation data in a gzip JSON artifact with version, teacher identity/configuration, backend/action schema, source revision and digest. Existing observation codecs may be consumed read-only.

Train sequentially without saved model hidden-state labels. For each current-parameter chunk replay its episode prefix under no-grad, then train the chunk; reset only at episode boundaries and retain forced observations in recurrent history. Optimize cross-entropy on non-forced choices only; skip chunks with no supervised choices. Preserve configurable sequence length, epochs and learning rate. Validation replays held-out episodes from zero recurrent state, reports all-choice and non-forced accuracy and cross-entropy per phase and action kind, and never updates parameters. An all-forced dataset reports no supervised accuracy (null), not perfect competence.

The legacy `imitation_warmstart` signature and its loss/decisions/epochs keys remain available, backed by current-parameter recurrent training. Explicitly report the demonstration cap and incomplete episodes. No topology-cache or model-capacity change is required without a measured encoding bottleneck.

## Experiment runner and budgets

Save a manifest before work starts, then incrementally persist raw results and checkpoints. Commit the implementation before launching the pilot and keep the code tree clean and unchanged throughout the run. Manifest identifies source HEAD and dirty state; dirty execution, if supported, must additionally identify tracked patch and relevant untracked source content with SHA-256, backend/action schema, Torch/Python configuration, model config, variant, complete seed split, teacher, decision/epoch/game-step/time budgets and checkpoint/dataset SHA-256 digests. Output JSON must contain finite numbers or null, never NaN.

Default bounded pilot: training seeds `[11, 29]`, at most 2500 demonstrations per seed, two epochs, hidden size 32, sequence length 16, validation board seeds `[90000, 90001]`, reserved final-test board seeds `[70000017, 70000019]`, max game steps 20000, total wall budget 900 seconds. Map training seed s to declared boards `[1000*s + i for i in range(64)]`; validation demonstration boards `[91000, 91001]` are also disjoint from training and reserved test. Record every actually consumed board. The target is 5000 demonstrations across both training seeds; the measured 900-second budget may require a smaller equal cap declared before training, and any reduction is explicit in the manifest. Evaluation covers heuristic and random candidates versus random and heuristic opponents, then untrained and imitation candidates for each training seed versus heuristic opponents on the same validation blocks. Pilot may stop at its deadline with a partial status and saved evidence. Never change the final test set in response to validation.

A 60-second runtime probe precedes the pilot. Start at most one complete board block per probe and record throughput; if the full budget cannot fit, use one validation block for every variant and mark the comparison inconclusive. The hard wall limit is checked at every simulator decision/update boundary; a single in-flight CPU action may finish beyond it. Declare this granularity.

PPO gate: at least two training seeds, at least eight complete validation board blocks per variant/seed, no truncated evaluation games, student non-forced held-out agreement at least 0.60 with at least 100 non-forced decisions, mean VP no more than 1.0 below teacher and win rate no more than 0.10 below teacher on the same blocks. This is an operational budget gate, not a statistical competence claim. Pilot-sized evidence is inconclusive and does not launch PPO. When sufficient evidence passes, a separately declared continuation uses fresh `RolloutCollector` data only, at most 1024 learner decisions per training seed and two PPO epochs, with checkpointed optimizer and equal held-out protocol. Critic warm start and anchor comparison remain L2 work unless separately planned and reviewed.

## Deliverables and honest status

Three reviewed implementation tasks yield evaluation protocol, recurrent imitation pipeline, and an executable persisted experiment plus bounded pilot report. The report names measured teacher/student results, coverage gaps, runtime, unfinished games, final-test usage (expected none), gate decision and next action. If competence is not reached, report that directly and use phase diagnostics to identify the next learning task. L2-L5, uninstalled teachers and full statistical strength evaluation remain explicit gaps.
