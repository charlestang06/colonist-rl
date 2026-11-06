# Catan decision and execution coverage: bounded audit

This is a broad map of current capability, not an exhaustive enumeration of
board states. It uses the current source, the saved [128-game raw panel](native-reference-results-2026-11-06.md),
and the [manual Colonist walkthrough](colonist-controls.md). The panel used
three H opponents and **no domestic offers**. Its 128 focal games all resolved;
the action counts below are observed policy actions, not counts of legal
opportunities or judgments of quality.

| Family | Simulator evidence in 128 raw games | Current public decision method | Colonist evidence and next gap |
|---|---:|---|---|
| Initial setup | 256 settlements and 256 roads by phase | Raw planner falls back to H outside a rolled ordinary turn. | Manual snake placement and preview/confirm recorded. Staged execution exists; no autonomous complete-game acceptance. |
| Roll, bank trade, build, pass | 2,926 rolls; 865 bank trades; 1,096 roads, 479 settlements, 167 cities; 2,895 end turns | Raw search models exact public bank/build consequences and ranks build-ending sequences. Roll and unsupported contexts use H. The city-before-purchase extension is optional. | Manual controls exercised for each; ordinary city/settlement activation requires focus and drag. Bank port ratios and complete autonomous action confirmation remain unverified. |
| Development purchase and private VP | 605 purchases; private VP has no play action | Purchase is an H fallback; current search does not compare card option value with a certain city. Own known VP belongs in the observation, while opponents' private cards remain hidden. | Purchase and automatic own private VP count observed manually. No autonomous card-value policy or complete live run. |
| Development-card play | 312 Knights, 53 Monopolies, 47 Year of Plenty, 47 Road Building | These actions use H fallback; raw `_SUPPORTED` covers bank/build/pass only. No public search over resource pairs, free-road sequences or uncertain opponent holdings yet. | Own Knight and Monopoly multi-stage controls observed. Own Year of Plenty and Road Building were not drawn in the walkthrough; their controls need live confirmation. |
| Robber, discard, steal | 759 robber moves, 1,094 own resource discards | H fallback chooses these actions. Public logs can constrain beliefs but cannot reveal every hidden steal. | Manual robber placement and one-/two-victim flows observed; own discard controls were not exercised. A click must not be treated as a completed steal before visible confirmation. |
| Domestic offers and responses | No evidence: offers disabled in this panel | Shared action contract and H proposal/response route exist. No-offer gameplay cannot evaluate negotiation. | Manual outgoing offer/finalization and incoming willingness/reject observed. Multiple rows and delayed transfer require exact offer identity and finalization verification. |
| Road/army awards | Roads and Knights occurred; award counts were not extracted here | Raw public score includes building production and immediate points, but a complete contested award plan is not established. | Manual Longest Road and Largest Army gains observed; award loss and autonomous award race remain unverified. |
| VP/endgame | All 128 panel games resolved; 32 focal wins | Raw search recognizes public building-point gain; optional winning-prefix repair tied 5/32 versus raw on its separate panel and was not promoted. | Manual game ended at 10 known points including two private VP cards. No autonomous complete live-game evidence. |

Source boundaries: [`public_turn_planner.py`](../src/colonist_rl/public_turn_planner.py)
defines the raw supported action set and search; [`evaluation.py`](../src/colonist_rl/evaluation.py)
provides complete-game and public-step callbacks; [`staged_execution.py`](../src/colonist_rl/adapters/colonist/staged_execution.py)
and [`execution.py`](../src/colonist_rl/adapters/colonist/execution.py) hold
staged UI and post-action checks. The physical adapter accepts verified
annotations; it has no general image recognizer.

The next decision gaps are purchase versus certain build, search candidate
coverage, then development-card continuation and award races. Live adapter work
should follow the capabilities the working player actually uses. Missing cells
remain unknown until a targeted simulator or live execution test reaches them.

## Search budget signal from the same saved panel

The raw public-work records contain 11,345 decisions: 4,896 searched and 379
returned a planned build sequence. Among searched decisions, 4,536 stopped with
`empty_beam`, 257 with `geometry_limit`, and 103 with `node_limit` (the categories
sum to 4,896). These are stop reasons, not proof that the resulting actions were
good or bad. They make attempted/feasible/unique successor counts and reached
depth the next useful measurements before interpreting a larger node budget.
