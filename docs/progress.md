# Implementation ledger — docs/implementation-plan.md

Workspace: isolated new repository on cytang/initial-implementation.

| Producer | Consumer | Interface check |
| --- | --- | --- |
| contracts + board | trades / simulator / policy | Immutable observations; no backend objects |
| beliefs | simulator / perception | Player-visible events and totals only |
| trades | simulator / policy | Binary response now; bounded proposals later |
| simulator | PPO | Per-player reward accumulation across intervening actions |
| policy | perception transport | Typed actions with observation version |

Completed:

- Variable 19/30-tile topology and configurable 4/6-player rule profiles.
- Immutable public/private observation boundary and joint resource ledger.
- Bounded optional trade generator/critic interfaces, binary incoming response,
  validation, timeouts, fallbacks and stale-offer protection.
- Pinned four-player engine, atomic trades and audited wrapper rule corrections.
- Graph/GRU policy, recurrent PPO, heuristic imitation warm start, frozen historical
  opponents, checkpoint/resume and seat-rotated evaluation.
- Versioned observation JSON, verified annotated screenshot ingestion, geometry
  calibration and an isolated optional browser transport.
- Independent reviews; all actionable findings corrected and regression-tested.
- A complete manual Colonist guest-bot game: victory at 10 points, 67 turns;
  durable control sequences and screenshots in colonist-controls.md.
- Shared, Colonist, and physical-photo adapter modules with compatible legacy
  imports; physical input requires an explicit reader or verified annotation.
- Offline ordered DOM log normalization and conservative history overlap checks,
  including real captured fixtures; semantic events and ledger updates remain next.

Deferred by scope/user direction: automatic arbitrary screenshot recognition,
live Colonist adapter integration, complete six-player simulation/training, and
learned proactive trade proposal models. The browser transport is experimental;
it remains separate from autonomous policy execution.

The later manual walkthrough now establishes gameplay through termination;
it does not establish autonomous policy execution. The current browser executor
still assumes a committed action after one click. Confirmations, drag gestures,
and accepted-but-pending trades require a separate UI interaction contract before
connecting the learned policy. See roadmap.md and colonist-controls.md.

Latest combined validation: 205 tests passed with the real local Chromium test
enabled, no skips; Ruff lint/format and wheel build passed. Both adapter and log
implementations received independent reviews with no actionable findings.

See validation.md for final verification and measured limitations.

## November 4 learning and adapter continuation

The first learning wave now adds complete four-seat board-block evaluation,
explicit censoring and paired comparisons, persisted observation-only teacher
demonstrations, current-weight recurrent imitation, and a bounded experiment
runner with atomic per-completed-game journals and a fail-closed evidence gate.
Independent task reviews and scoped corrections passed; the broad learning
review is pending. The preserved original pilot is documented in
[the pilot report](learning-pilot-2026-11-04.md). Its students won zero of eight
heuristic-opponent games each, so useful imitation and PPO improvement remain
open milestones.

Immutable UI contracts and conservative rendered-log production/bank/dice
candidates passed broad adapter review. They establish no event identity or
ledger updates. A separate staged executor is being reviewed against synthetic
fixtures; live recognition and autonomous Colonist play remain unverified.

An intermediate combined check after the reviewed first-wave task integration
passed **437 tests, with one optional browser smoke skipped**, in 8.96 seconds.
Ruff lint passed and all 49 source/test Python files were already formatted.
These checks precede the pending follow-on work and final review; they do not
replace final-source verification. Historical verification above remains the
earlier baseline.
