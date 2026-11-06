# colonist-rl

A public-information Catan player for simulator and Colonist adapters.

The goal is one usable player with coherent trades, builds, passes and tactical
choices, followed by reproducible superiority to authentic VF and AB and faithful
complete live-game execution. The [revised roadmap](docs/agent-research-roadmap.md)
separates those three claims. The first matched comparison completed 96 games:
raw 15/32 wins, persistent H 5/32, neural parent 2/32. The [working-player card](docs/working-player-card.md) keeps H
as provisional control under the fixed fallback and raw as the leading unpromoted
challenger. The separate [native comparison](docs/native-reference-results-2026-11-06.md)
completed 384 games: raw 32/128, VF 19/128, AB 26/128. Its superiority-qualification
gate failed. Next certify and test a targeted city-funding trade before purchase
from relevant TRAIN evidence; no automatic extension or training run follows.

Heuristics, bounded planning and learned components can cooperate. BC and PPO
are optional tools; PPO remains deferred until competence and an identified
strategic need justify it. No automatic training scale-up, value-learning lane or
reward-shaping campaign is scheduled. The
[immediate execution plan](docs/superpowers/plans/2026-11-06-decision-first-roadmap.md)
records the completed three-arm screen and its evidence-led handoff; no new fit is queued.

The [current results](docs/training-status-2026-11-05.md) show no established VF/AB
superiority. The 384-game feature comparison passed integrity review but failed its
primary advancement rule: every candidate won 2/16 against trading H. The stable
sparse-target fitter passed 43 focused and 11 legacy checks plus independent source review. The complete-history
cache repair passed engineering review on four saved
games/361 inputs. The original value attempt stopped after 208 TRAIN games;
a separate successor completed the missing 112. Component and mixed-index audits
passed for all 320 inputs (256 TRAIN, 64 validation). The one-pass cache now
contains all 27,706 own inputs and passed saved-record audit. Four fixed value
heads completed, but neither the recurrent-representation nor either data-benefit
gate passed. [Value results](docs/fixed-value-head-results-2026-11-06.md).
The student collector recorded all 48 planned games and 4,729 own inputs, but its
Root controller failed during final output accounting. Independent audit found
the histories intact; a separate zero-query recovery and independent metadata
audit now pass for all 48 episodes / 9,458 strict envelope pairs. Fixed sampling
selected 88 cases and retained eight empty slots. All 176 first passes and 49
adjudications are frozen. The informative TRAIN upper bound is six candidates
across four strata, below the fixed 32/six gate, so no fit ran.
This does not show BC failed. Public fact checks identify two expansion-blocked
road choices; a flagged trade remains uncertain because it has an alternative
city-funding route. An opt-in road correction completed 114 games of a frozen
128-game study. Against H it won 2/32 versus the parent's 3/32; both GNT arms
stopped at the fixed own-input cap. It did not advance. See the
[road experiment](docs/road-continuation-experiment-2026-11-06.md).
An opt-in winning-prefix search passed nine targeted and 43 affected pure checks.
An exact TRAIN replay confirmed a trade/build win. Its fresh 64-game comparison
finished 5/32 wins in each arm, so it did not advance; defaults remain unchanged.
[Winning-prefix experiment](docs/winning-prefix-experiment-2026-11-06.md).
The closed bounded value study asked whether outcome estimates could support
projected planning leaves; its failed gates authorize no deployment. No new imitation targets
are approved for training.
Earlier residual and placement studies found no reliable gain;
the latest native focal pilot scored raw/VF/AB2 3/0/4 per 16.

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

The `train` CLI is available experimental PPO infrastructure; sustained PPO is
deferred in the research roadmap. It uses a small graph encoder and GRU with
separate normal/forced/setup heads and an incoming accept/reject head. Legal
candidates reference graph targets,
so the network also accepts six-player/30-tile representation fixtures. The
simulator currently rejects six-player games explicitly; larger-board rules
and full six-player training are a subsequent milestone.

## Learning guidance

The examples below describe available experimental PPO infrastructure, not the
next research experiment. Decision diagnosis and targeted repairs come first;
new outcome-value work needs a distinct decision hypothesis. Imitation remains
available when sound relevant targets support it.

The optional imitation warm start learns legal moves from a production-aware
heuristic teacher. Set `--warmstart-decisions 0` to compare training from scratch.
Teacher moves are supervised examples, never substituted into PPO trajectories.
The existing optional PPO trainer learns from sampled decisions against random and heuristic
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

Our policies, teachers, search and trade plug-ins receive only immutable public
`Observation` records. Exact
resource updates enter an event ledger; hidden steals/discards maintain joint
beliefs. Approximate particle agreement never certifies exact knowledge. Exact
bank quantities default to unknown unless the capture source establishes their
visibility. Backend objects and hidden development cards stay outside our models.

The separately isolated, evaluation-only native VF/AB gateway is integrated and
has passed a real-engine engineering seam. Native references use privileged state
with disclosed discard/trade shims and independent planning RNG. Small exposed-board
reference profiles are recorded, while broader candidate comparisons remain
pending. Their state never enters learner or teacher inputs. This asymmetric
benchmark does not establish equal-information superiority.
See the [native benchmark milestone](docs/agent-research-roadmap.md#milestone-3-exceed-stronger-opponents-reproducibly).

The learner cannot initiate or counter player trades in v1. It can respond to
incoming offers, and bank/port trades remain ordinary actions. Scripted opponents
propose offers to produce response-training experience. Independent generator
and critic protocols support bounded candidates, deterministic validation,
timeouts, stale-observation rejection and decline/no-offer fallbacks. The trade
critic is distinct from PPO's value head. See [trade interfaces](docs/trades.md).

## Backends and deployment

The backend is pinned to Catanatron Git commit
`82aae93ab1f7c267218be0566df573ce477ec3d8` (distribution version `3.3.0`),
installed from the repository root without its gym extra. Runtime and artifact
identity is `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`, verified against
PEP 610 Git provenance. The adapter explicitly uses `official_spiral` number
placement and backend seed `user_seed + 1`; this is a new engine/data cohort,
so legacy `3.2.1` seed labels and metrics are not comparable. Native development
timing and free roads are used alongside wrapper atomic domestic trades; native
trade negotiation is unsupported. See [engine migration](docs/engine-migration.md)
for compatibility and runtime-transition details.

Offline capture ingestion and coordinate calibration share the observation
schema. Live Colonist integration requires verified extraction and an explicitly
authorized transport. No live game is accessed by installation or training.
The optional [Playwright transport](docs/browser.md) is available as a separate
experimental module. A manual guest-bot walkthrough completed a 10-point victory;
the [control reference](docs/colonist-controls.md) records tested gestures, dialogs,
log evidence, screenshots, and remaining checks. Autonomous policy-to-UI execution
still needs staged confirmations and pending-trade handling.
The current reader recognizes verified annotated screenshots by image hash;
it does not automatically infer an arbitrary Colonist screenshot. See
[capture format and UI transport](docs/perception.md). Export a schema example with
`uv run colonist-rl export --output runs/observation.json`; inspect a verified
capture with `uv run colonist-rl capture --annotation capture.json --screenshot board.png`.

Environment-specific code lives under `colonist_rl.adapters`: shared capture,
serialization, and calibration modules; `colonist` browser/execution modules; and
a `physical` photo-reader entry point. Physical photos require a supplied reader
or complete verified annotation; automatic board recognition is not implemented.
Existing `colonist_rl.browser` and `colonist_rl.perception` imports remain
compatible. See [adapter boundaries](docs/adapters.md).

The [offline Colonist log reader](docs/colonist-log-reader.md) preserves ordered
text and icon labels and reports ambiguous overlaps and missing history. It does
not yet infer resource-transfer events or update the belief ledger. The
[research roadmap](docs/agent-research-roadmap.md) defines current priorities;
earlier [roadmap notes](docs/roadmap.md) remain as historical/user notes.

Colonist's [terms](https://colonist.io/terms) require authorization for service
data AI development/private API use; training here uses simulator-generated data.

The upstream engine is GPL-3.0. This project is distributed under GPL-3.0;
see LICENSE and THIRD_PARTY.md.
