# colonist-rl

Local Catan reinforcement learning with player-visible observations, variable
hex boards, joint resource beliefs, and independent trade model plug-ins.

## Run

Requires Python 3.11+; development was verified with Python 3.13 on macOS.

```sh
uv sync --extra dev
uv run colonist-rl board --players 6
uv run colonist-rl simulate --games 4
uv run colonist-rl train --updates 10 --rollout 64 --warmstart-decisions 256
uv run colonist-rl evaluate --checkpoint checkpoints/latest.pt --games 16 --compare-rejection
uv run pytest -q
```

PPO uses a small graph encoder and GRU with separate normal/forced/setup heads
and an incoming accept/reject head. Legal candidates reference graph targets,
so the network also accepts six-player/30-tile representation fixtures. The
simulator currently rejects six-player games explicitly; larger-board rules
and full six-player training are a subsequent milestone.

## Learning guidance

The optional imitation warm start learns legal moves from a production-aware
heuristic teacher. Set `--warmstart-decisions 0` to compare training from scratch.
Teacher moves are supervised examples, never substituted into PPO trajectories.
PPO then learns from its own sampled decisions against random and heuristic
opponents. `--historical-opponents` adds up to three frozen policy snapshots;
`--opponent-checkpoint FILE` supplies existing frozen policies.

Reward is terminal win=1, loss=0 plus `0.2 * (next_potential - potential)`, with
discount=1 and potential=own-known-VP/winning-threshold. True terminal potential
is zero. Gains and losses outside the learner's turn accumulate until its next
decision. Collection limits bootstrap instead of turning into losses.

```sh
uv run colonist-rl train --updates 100 --rollout 128 --historical-opponents
uv run colonist-rl train --resume checkpoints/latest.pt --updates 10
uv run colonist-rl evaluate --checkpoint checkpoints/latest.pt --games 1000 --output runs/evaluation.json
```

Checkpoints save model, optimizer, schema, and Python/Torch RNG state. Resuming
starts a fresh match; it does not promise bitwise continuation of simulator
state. Evaluation uses held-out seeds and rotates the tested policy through
four seats on each board. Short smoke runs establish functionality, not strength.

## Information and trade boundaries

Only immutable `Observation` records reach policies and trade plug-ins. Exact
resource updates enter an event ledger; hidden steals/discards maintain joint
beliefs. Approximate particle agreement never certifies exact knowledge. Exact
bank quantities default to unknown unless the capture source establishes their
visibility. Backend objects and hidden development cards stay outside models.

The learner cannot initiate or counter player trades in v1. It can respond to
incoming offers, and bank/port trades remain ordinary actions. Scripted opponents
propose offers to produce response-training experience. Independent generator
and critic protocols support bounded candidates, deterministic validation,
timeouts, stale-observation rejection and decline/no-offer fallbacks. The trade
critic is distinct from PPO's value head. See [trade interfaces](docs/trades.md).

## Backends and deployment

The backend is pinned to published `catanatron==3.2.1`. Its published release
differs from upstream main; our adapter adds atomic domestic trading and fixes
selected rule defects, including free-road and development-card timing. The
adapter and source tests document its current parity boundary.

Offline capture ingestion and coordinate calibration share the observation
schema. Live Colonist integration requires verified extraction and an explicitly
authorized transport. No live game is accessed by installation or training.
Live adapter work is deferred; the optional [Playwright transport](docs/browser.md)
is available as a separate experimental module.
The current reader recognizes verified annotated screenshots by image hash;
it does not automatically infer an arbitrary Colonist screenshot. See
[capture format and UI transport](docs/perception.md). Export a schema example with
`uv run colonist-rl export --output runs/observation.json`; inspect a verified
capture with `uv run colonist-rl capture --annotation capture.json --screenshot board.png`.
Colonist's [terms](https://colonist.io/terms) require authorization for service
data AI development/private API use; training here uses simulator-generated data.

The upstream engine is GPL-3.0. This project is distributed under GPL-3.0;
see LICENSE and THIRD_PARTY.md.
