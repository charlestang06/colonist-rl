# Verification

Verified on macOS with Python 3.13 and pinned Catanatron 3.2.1.

- `python -m pytest -q`: **166 passed, 1 skipped**. The skipped test is an
  explicitly opt-in real-Chromium fixture test; ordinary browser-boundary tests
  use injected pages. The core simulator/training tests run real Catan games.
- `ruff check src tests`: passed.
- Chromium-enabled run on November 4, 2026: **167 passed, no skips**, using
  Playwright 1.64.1 and Chromium 156.0.8078.4 against a real loopback page.
  Evidence is saved under `runs/browser-tests/`.
- `ruff format --check src tests`: passed.
- `uv build --offline`: source distribution and wheel built successfully.
- CLI warm start, PPO parameter updates, checkpoint creation and resume verified.
- A 128-demonstration / 256-PPO-decision smoke run completed four episodes with
  incoming trade-response samples and finite updates. A separate short CLI run
  exercised the final sources and resumed checkpoint progress.
- Four seat-rotated heuristic baseline games and eight policy-versus-rejection
  comparison games completed without truncation. Per-game seeds/outcomes and
  nominal confidence intervals are supported by the evaluation command.

These short runs establish execution and plumbing, **not a strong Catan agent**.
The smoke-trained policy won none of its four held-out games. Meaningful strength
claims require longer training and substantially larger held-out evaluation.
Wilson intervals are explicitly nominal because games sharing a board are
dependent; raw paired-block outcomes are retained.

## Boundaries

Four-player simulation is supported. Six-player board/rules representations and
policy inference are tested; six-player simulation raises NotImplementedError.
The learner can respond to incoming trades and use bank/ports, but cannot initiate
or counter player trades. Generator/critic plug-ins are independently optional.

Resource bookkeeping is exact when observations determine it. Bounded joint
particles approximate remaining uncertainty; rejuvenation can broaden historical
correlations, and sampled agreement never certifies exact hidden hands.

The capture reader recognizes verified annotations for known screenshot hashes;
it is not a general Colonist computer-vision model. Browser transport code is
isolated and optional. A manually controlled live Colonist game was completed on
November 4, 2026; the autonomous adapter and RL policy were not connected to that
game. Verified autonomous browser gameplay is not claimed.

## November 4 adapter and log-reader verification

The latest full implementation checks, after adapter modularization and the
offline visible-log reader, completed with:

- **205 tests passed, no skips**, including the opt-in real-Chromium loopback
  fixture with browser smoke testing enabled.
- Ruff checks and `ruff format --check src tests` passed.
- The package wheel built successfully.
- Two independent final reviews found no actionable issues in the changes.

These local checks cover the shared/Colonist/physical adapter boundaries,
compatibility imports, optional browser lifecycle and transport, verified
capture handling, and conservative normalization/history of supplied visible
log rows. The offline log reader does not infer resource transfers or establish
complete event history.

Separately, the live manual guest walkthrough reached a verified 10-point victory
in 67 turns with displayed time 27:27. The final guest display was `8 (10)`, and
the postgame overview confirmed the win. The [controls guide](colonist-controls.md)
links 18 durable screenshots and distinguishes exercised controls from remaining
checks. The [walkthrough status](browser-walkthrough-status.md) records the
manual/autonomous boundary and partial log-history limitation. Neither the
manual win nor these local tests establish autonomous playing strength.
