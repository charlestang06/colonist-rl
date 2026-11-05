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

Deferred by scope/user direction: automatic arbitrary screenshot recognition,
live Colonist adapter integration, complete six-player simulation/training, and
learned proactive trade proposal models. The browser transport is experimental;
opening a guest board did not establish verified gameplay or autonomous play.

See validation.md for final verification and measured limitations.
