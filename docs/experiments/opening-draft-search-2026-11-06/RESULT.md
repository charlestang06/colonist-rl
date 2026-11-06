# Public opening draft search: negative confirmation

Status: **stopped after the frozen CONFIRMATION gate failed**. The opening
search is retained as an isolated experiment, not promoted into the default
planner. No parameter was changed after the PILOT or CONFIRMATION outcomes.

## Study identity

The [protocol](PROTOCOL.md) was frozen before these boards. The evaluated
policy source is commit `de857d3c9a4804df37b7341d79df91e4b6b0582d`, Git
tree `087ebb449c06a7787edb5a6f97d39c8d2aa59292`; this result note was
committed later and is not part of that evaluated tree. The protocol SHA-256 is
`478fc6ede8b38184831a8883e56b885dba0c0d8663220a792ed8c76738770c1c`.
The exact-seed [clearance](SEED-CLEARANCE.json) was verified before each role.

`OpeningDraftPolicy` uses public board geometry, legal menus, a model of the
remaining snake draft, and shared short dice traces to choose opening
settlements and roads. Every nonsetup action delegates to the unchanged raw
public planner. Both arms faced three trading heuristic opponents in matched
four-seat Catanatron games without domestic offers. These are engine results,
not measured Colonist.io match results.

## Frozen roles and results

| Role and logical boards | Raw planner | Opening search | Frozen gate |
| --- | ---: | ---: | --- |
| DEVELOPMENT 157560000–015 | Mechanical reference | 256/256 legal own setup decisions; 86 changed settlements on 16 boards and four seats; p95 476.952 ms, max 707.253 ms | **Pass:** complete matrix, identity/replay, nonsetup parity, intervention, latency |
| PILOT 157560016–023 | 9/32 wins; 7.1875 mean VP | 17/32 wins; 8.0000 mean VP | **Pass:** +8 wins versus required +3, +0.8125 VP versus required +0.25; 64/64 resolved |
| CONFIRMATION 157560024–039 | 23/64 wins; 7.421875 mean VP | 23/64 wins; 7.703125 mean VP | **Fail:** +0 wins versus required +6, +0.28125 VP versus required +0.25, bootstrap lower −0.140625 versus required >0; 128/128 resolved |

An independent outcome-blind public review checked all 86 changed PILOT setup
roots against the saved before-observations and legal menus. It found zero
clearly destructive choices; all 86 verdicts were `neither`, meaning both
choices had a credible or uncertain public case, not equal expected value.
The review receipt SHA-256 is
`ad4f89d1f1a6e9f2aab61681122d5492ce8c01e74d0a33437a3b85afa3dc3290`.
The confirmation role required this receipt before reading any new board.

Confirmation changed 159 setup choices, including 94 settlements and 65 roads.
All 16 paired board blocks resolved, with no game truncations. The whole-board
paired bootstrap used 10,000 resamples and seed 20261106. The pilot advantage
did not reproduce as a win advantage on new boards. The +0.28125 mean VP
difference clears only one of the three frozen outcome terms and does not
justify promotion.

## Disposition and limits

**No promotion or confirmation-panel retuning.** The search's short-horizon
affordability surrogate omits later robber play, opponent construction,
domestic offers, and development-card chance. The tied wins could reflect the
surrogate's objective, sampling variation, or interaction with the shared
midgame planner; this study cannot separate those explanations. A structurally
different opening mechanism needs a new design and prospectively reserved
boards. PILOT and CONFIRMATION data remain evaluation evidence, not training
examples.

All 128 confirmation game journals and 134 indexed output files were rehashed
against the runner index. Result SHA-256:
`a468597f42a15b986e00072d0c5bfd14ce2cc7112e617035daffe79d13f02729`;
independent audit SHA-256:
`b046304fb5b5f91f68891cf335514b6ec61b32d7d1677e0b01754c5a646c898d`.
The DEVELOPMENT, PILOT, review, and CONFIRMATION artifacts are mirrored under
`runs/swarm-2026-11-04/agent-research/opening-draft-search-001/` in both the
Original and integration repositories. The sealed confirmation extension
manifest SHA-256 is
`a469941adc0193350f2c42ce6f568eeb2671598e08ea09a78fff1733c1ba429b`.
