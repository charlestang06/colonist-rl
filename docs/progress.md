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
