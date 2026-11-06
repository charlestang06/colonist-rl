# Public BUY continuation preservation: frozen experiment

Frozen before implementation or gameplay on 2026-11-06. Base source commit:
`7c36586c34db5d4739b4de9ecc8d581f93556d70`. This is a narrow follow-up
to the [public purchase option screen](public-option-purchase-results-2026-11-06.md),
not a retune of its card credit or an extension of its exposed boards.

## Hypothesis and decision rule

The optional purchase comparator sometimes replaces a raw legal development
card BUY with a bank trade followed by a point build, or with END_TURN. Its
score does not represent the possibility that paying the known BUY cost still
leaves a same-turn bank-trade route to a point build. On a saved public TRAIN
Observation, raw BUY was followed by two trades and CITY15; the old option
instead trades once and builds CITY15 without the card.

The new `preserve_buy_build` switch is **false by default** and is valid only
for an enabled `public_option_purchase` arm. It changes no score, card prior,
margin, old action, or supported context. Immediately before an old option
trade→point-build choice, start from the public projected hand and bank after
the legal BUY cost. Enumerate zero, one, or two conservative legal maritime
trades, preserving port ratios, bank lower bounds, resources and pieces. If
the **same building kind and vertex** is then legal, return the original BUY
action and plan object. Before an old option END_TURN choice, preserve BUY if
**any** city or settlement is reachable in the same way. An incomplete scan
from node or geometry limits preserves BUY. The route is only a public proof
that BUY need not forfeit this point build; the policy never predicts the
hidden card or executes a projected future action. Every subsequent action is
chosen after a new Observation.

This conservative guard may miss a real route when public bank lower bounds
are too weak. It does not claim that BUY is always strategically preferable
when it can still build. Keeping the same target avoids a comparison between
different site values in this first test.

## Engineering gate

Use the exact public archived
`runs/swarm-2026-11-04/agent-research/city-before-purchase-001/engine-preparation/adverse-before.json`
(SHA-256 `53c72cd843d537765158de26ba7f567a7a5b1e8b0bc05f54d92b6eda0cbbd125`)
as the positive regression: original option chooses a bank trade intending
CITY15; guarded option returns the parent BUY object at original menu index
zero, and a public two-trade route to CITY15 exists after BUY. Include a
negative trade→build case where BUY cost prevents the same build, a negative
PASS case with no post-BUY point route, duplicate/legal-root identity,
malformed/unsupported context, and bounded work. A cap cannot publish a
partial negative. The old default and `preserve_buy_build=false` must retain
their prior decisions. Run affected tests and lint before gameplay.

## Excluded smoke and fresh development panel

After the engineering gate, run one excluded smoke board `138295999` for all
four focal seats. The fixed fresh development screen is eight boards
`138296000` through `138296007`, all four focal seats. Compare the unchanged
raw public planner, the old enabled option, and the new guarded option against
three trading heuristic opponents, with domestic offers disabled, standard
rules, 20,000 maximum game steps, and captured public candidate steps. All
three arms use the same `TurnPlannerConfig` defaults (including future
pruning) and `gain_margin_micro=10000`; only the guarded arm sets
`preserve_buy_build=true`. Freeze exact source/runtime hashes in the manifest
before play. Run no other heavyweight fit or gameplay screen concurrently.

Analyze matched complete games, first-divergent own-turn sequences, whether
guarded BUY really built the same point that the old option intended, final
wins and VP, and selection latency. Report unmatched/truncated games and
intervention frequency. The point-building route is a mechanical premise;
known turn-end points and game results are separate evidence. With only eight
board clusters, intervals are descriptive and no arm is promoted to default
from this screen.

If the guard changes no fresh decision, or its positive route cannot be
verified, retire the unchanged mechanism. If all games resolve and the guard
gains at least two paired wins out of 32 over old option with nonnegative mean
VP difference and no clearly bad changed turn, keep it as an independent
confirmation candidate. A loss of at least two paired wins or a negative mean
VP difference retires this configuration. Other outcomes are inconclusive and
leave it opt-in. Raw is a simultaneous practical comparator, not a source of
training labels. No additional boards or score adjustments rescue a failed
gate.
