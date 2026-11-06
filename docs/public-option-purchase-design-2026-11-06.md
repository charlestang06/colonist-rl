# Public purchase option comparator: fixed development hypothesis

This is a disabled-by-default decision experiment at an ordinary rolled-turn
`BUY_DEVELOPMENT_CARD` fallback of the unchanged public planner. It compares
the **original legal** BUY and END_TURN roots with original legal bank-trade
roots followed by one public city or settlement projection. The next build is
an intention, not a certified legal future action. Execute one original root,
observe the actual next menu, and replan. Parent fallback behavior and all
other decision phases remain unchanged.

## Frozen score contract

All scores use the existing planner's integer micro-score units. Its score
awards 100,000,000 units for one public or projected building point, with
production, connectivity, resource balance, and public rank also included.
The comparator uses the same public projection and scorer for known state
consequences:

| Alternative | Fixed score |
|---|---|
| BUY | Score of the hand after paying one Sheep, Wheat, and Ore, plus the broad card option credit below. The unknown draw is never projected. |
| END_TURN | Score of the unchanged public state, plus 35,000,000 units for each extra resource missing from the nearest currently reachable city or settlement recipe **if BUY were paid**. The reserve reflects retaining resources for a next-turn point build; it does not assert the needed resources will arrive. |
| Legal bank trade → one build | Score after the exact public trade and city/settlement costs, building and piece changes. No fabricated legal action is executed. |

The card credit uses the **initial standard-rule deck counts**, not an exact
posterior over an unseen deck. Fixed assumed option values by card kind are:
Knight 60, Year of Plenty 50, Monopoly 20, Road Building 30, and Victory Point
100 million micro-score units. The standard counts `(14, 2, 2, 2, 5)` give a
base credit of 61.6 million units. The option credit is multiplied by
`1 + clamp(victory_target - known_own_vp - 2, 0, 6) / 6` using integer
arithmetic. This gives action cards more room to pay off early and is at most
2× the base credit. The resource reserve is at most 70 million for standard
city/settlement recipes because BUY can increase either recipe's deficit by at
most two. These numbers are an explicit heuristic hypothesis from score
units, deck composition, and recipe arithmetic; they are not fitted win
probabilities or tuned to the exposed city screen.

BUY remains the choice unless an alternative exceeds its score by the
parent's fixed gain margin. A trade→build must also exceed PASS by that margin;
strict ties retain BUY. A currently legal building action that wins using
known own points is a safety exception. A **projected** winning build gets
priority after a complete conservative scan, but is still rechecked against
the actual menu after the trade. Other directly legal builds make this narrow
comparator abstain to the parent. A budget cap, unsupported context, partial
bank contract, ambiguous root, or incomplete candidate scan also abstains.

The player reads only its Observation: legal actions, own hand and development
cards, public board/players, public bank evidence and rules. It does not read
the actual draw, unseen opponent cards, engine deck state, future RNG, or
post-game outcomes. The broad initial-card prior deliberately leaves unknown
depletion unresolved.

## Evaluation boundary

First use pure tests and a real-menu continuation check. The manifest runner
accepts `"kind": "public_option_purchase"` with the same raw planner config
and gain margin fields; this constructs an enabled wrapper for that arm only.
Then a frozen fresh paired panel against the unchanged raw policy can measure
intervention count, first changed actions, actual follow-through, latency,
complete-game results, and adverse cases. Any result is exploratory until
separately confirmed; no default promotion follows from a projected score or
a small screen. The score
formula and constants above are fixed before fresh gameplay.
