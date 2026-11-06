# Working player: development selection, 2026-11-06

**Provisional comparison control:** persistent public H.
**Leading unpromoted strength challenger:** unchanged raw public planner.
**No unique stronger policy established; no neural model promoted.**

The [canonical roadmap](agent-research-roadmap.md) targets coherent decisions,
reproducible VF/AB superiority and faithful live execution. This card establishes
a reproducible working comparison, not completion of those goals.

## What actually ran

Eight fresh logical boards 138190000..138190007, all four focal seats, three arms,
three persistent H backgrounds, no domestic offers and legal maritime trading.
All 96 games terminated with resolved winners. All 24 guarded jobs exited zero;
Root wall time99.39 seconds. Full own before/action/after traces retain9,520 inputs.

| Policy | Wins / games | Worst game selection p95 | Largest recorded policy selection |
|---|---:|---:|---:|
| Persistent H |5/32|0.248 ms|0.332 ms|
| Raw public planner |15/32|11.324 ms|15.306 ms|
| Frozen neural parent |2/32|1.456 ms|7.147 ms|

P95 is evaluator selection timing. H/NN maximum measures inherited H/parent act;
raw maximum measures the reviewed planner adapter. Callback export is excluded
from selection timing but included in guarded job wall time. These scopes differ.

Raw-minus-H observed win difference was+31.25percentage points; raw-minus-NN was
+40.625 points. The frozen simultaneous bounded-KL intervals were[-66.86,+94.30]
and[-60.55,+96.42]points respectively. Eight board clusters and the conservative
rule do not establish a unique stronger arm. Descriptive bootstrap intervals are
not the promotion gate. Nominal interval coverage assumes independent/exchangeable
board draws; these fixed-panel observations alone do not prove that assumption.

The predeclared eligible H/raw/NN fallback selects H as the **provisional
operational control**. It does not claim H is strongest or justify ignoring raw's
large observed advantage. Raw is the next direct VF/AB challenger. The simulator,
CLI and deployment defaults were not changed by this selection.

## Exact player identities and scope

All arms used clean frozen source 1cfaa4c0c570a53b39569aa60f0079e78b1b6b00 and
Catanatron 3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8.

- H: existing HeuristicPolicy, one persistent focal instance per game, seed
  logicalboard+31+seat. Background instances use logicalboard+absolute player.
  It consumes the shared player-observable contract, including its own private
  hand/cards and public resource beliefs, and returns an original legal action.
- Raw: PublicTurnPlannerPolicy, depth 6/nodes 2048/beam 24/geometry 100000/seed 11,
  evaluator None and explicit gain margin 10000. Frozen source predates the later
  optional winning-prefix helper; existing
  single-plan adapter. It uses H fallbacks for unsupported planning decisions.
- NN: exact seed 11 frozen BC parent, hidden size 32 graph_consequence_v1/hand_recipe.
  Parent byte SHA256 b3aae32ac36dc55f8ae26ec4676cc289243a6c540b959e1f8aa9786e5ddd0258;
  tensor digest0158268566248edc6c48b5edc5858cbd7966b3c6eec5c1bd2c0e3483ed3429bc.
  Eight checkpoint loads and2,881 original deterministic act calls, including1,183
  forced inputs; memory reset per game and advanced on every own input.

The scope is standard four-player simulator play without domestic offers.
Maritime trade, builds, passes, robber/discard, development-card use and endgame
occurred. This screen does not establish domestic negotiation, six-player play,
physical recognition or complete live Colonist execution.

## What the decisions showed

The frozen diagnostic selection filled56of60 slots,55unique records. A bounded
single-review assessment found36 sensible,14 ambiguous and5specific strategic
concerns, with0 verified critical failures. These are enriched EVAL examples,
not population error rates, broad competence evidence or TRAIN labels.

Raw showed useful conversions: Year of Plenty followed by a settlement,
Monopoly resources converted through maritime trades into a settlement, and an
actual legal settlement at9 known VP that terminated with10 VP and the focal win.

A neural example traded wheat for brick and built a road when a legal ore trade
could instead fund a city at an existing settlement. Expansion could justify the
road; it is a concrete prioritization concern, not a verified blunder. H selected
brick with Monopoly, traded some for ore and ended one ore short of a city;
recorded public ore lower bounds supported an alternative ore choice sufficient
for that city. This was6 VP, not a missed immediate win. Discard complementarity
and pre-roll Knight timing also warrant further diagnosis. Actual raw per-step
scores/reasons and neural logits were not persisted; no invented rationale is used.

The frozen Knight rare-slot name was PLAY_KNIGHT instead of actual
PLAY_KNIGHT_CARD. All three rare slots remain missing; actual Knight actions were
H 93/raw 79/NN 39. Three separately labeled posthoc Knight examples were inspected
with0 verified critical failures. NN seat3's late-stage slot was also absent.
The original protocol and missing coverage remain intact.

## Evidence and next decision

[Protocol](../runs/swarm-2026-11-04/agent-research/incumbent-selection-001/freeze-001/protocol.json),
[results](../runs/swarm-2026-11-04/agent-research/incumbent-selection-001/analysis-001/summary.json),
[independent saved-record audit](../runs/swarm-2026-11-04/agent-research/incumbent-selection-001/saved-result-review-001/report.md),
[decision review](../runs/swarm-2026-11-04/agent-research/incumbent-selection-001/decision-review-001/report.md)
and [Root selection](../runs/swarm-2026-11-04/agent-research/incumbent-selection-001/Root-policy-selection-001.json)
are archived with source/config/runtime bindings. Independent review checked all 96
joins,9,520 serialized legal-menu memberships,49 source and17,978 runtime hashes,
parent bytes and884 saved OS1 samples (peak741,277,696 bytes; maximum overlap2).
Tensor/engine/live-object fidelity remains reviewed-source/producer evidence;
independent audit did not reload tensors, replay games or query a codec/policy.

The separate [32-board native comparison](native-reference-results-2026-11-06.md)
now completed 384 games: raw 32/128, VF 19/128, AB 26/128. Both primary intervals
cross zero and the qualification gate failed. No confirmation, extension or model
promotion follows. This does not change the earlier screen's frozen selection.

The [city-funding experiment](city-before-purchase-results-2026-11-06.md)
confirmed an executable trade-to-city route, but its fresh paired screen won
11/48 versus raw's 13/48; no default change follows. Future work follows the
[canonical queue](agent-research-roadmap.md#immediate-work-and-ownership):
search coverage/cost and a unified card-versus-build choice. The distinct
[trainable public-history critic](shadow-critic-results-2026-11-06.md) failed its
development gate against the public-63 baseline. This card records the historical selection,
not a second active plan. Broad competence and VF/AB superiority remain
unestablished; final boards remain untouched.
