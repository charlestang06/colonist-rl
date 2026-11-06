# Conservative public current-turn planner

`colonist_rl.public_turn_planner` provides `TurnPlannerConfig`, `ProjectedTurn`,
`PlannedStep`, `TurnPlan`, pure `plan_turn(view, config)`, `HeuristicViewPolicy`
(**H_view**), and `PublicTurnPlannerPolicy` (**H_view+planner**). There is no
standard evaluator/CLI registration. This is an unvalidated planning hypothesis;
manufactured tests and transition replays do not establish stronger play.

```python
from colonist_rl.public_turn_planner import PublicTurnPlannerPolicy, plan_turn

plan = plan_turn(observation)
assert plan.first_action is observation.legal_actions[plan.root_index]
action = PublicTurnPlannerPolicy().act(observation)
```

Execute only `first_action`; get a new authoritative observation before calling
again. `steps[1:]` are future intentions. A `ProjectedTurn` is a distinct frozen
record with own hand, public pieces, own stock, known building-point gain, and
bank bounds. It has no context, legal menu, opponent hands, deck, engine, adapter
or environment. It cannot enter the existing Observation exporter. No projected
state, planner override, teacher label or episode is written by this module.

## Supported public contract

Only the exact `RulesConfig.standard()` four-player rule profile is enabled,
using `hex-v1` public topology and schema-v1 Observations. Search requires a
normal `TURN`/`PLAY_TURN`, rolled dice, own actor/turn owner, no free roads,
pending offer or discard. Setup, special-build, development-card and robber
subphases fall back. Unknown/modified rule profiles fall back.

Extension support is deliberately deferred: the current real-engine adapter
rejects extension/special-build rules, so their geometry and costs cannot receive
the required real transition replay validation. Root approved this conservative
restriction on 2026-11-05. General geometry helpers do not imply policy support.

The current tuple is authoritative. `_root_transition` accepts only an original
menu object on the exact initial public projection. For a root road, that menu
certifies connectivity even when the pinned engine's menu differs from conservative
public geometry at an opponent building. Integer target range, edge vacancy, own
stock, exact costs, resources and bank bounds still apply. Other supported root
exchanges/builds retain their public geometry and port-ratio checks. Unexpected
metadata or a disagreement returns `public_contract_mismatch`. Unsupported root
members are ignored by the search, but if H_view selects an unsupported effect,
the whole decision returns `unsupported_heuristic_action`.

## Conservative bank evidence

With an explicit public `view.bank`, both bounds are that exact vector. Without
it, total supply `S = rules.bank_resources`, own hand `h`, and the aggregate
opponent public card count `N` imply `L[r]=max(0,S[r]-h[r]-N)` and
`U[r]=S[r]-h[r]`. Each current maritime member raises the lower bound to the
**maximum** receive count witnessed for that resource. Duplicates do not add
stock. Absence of an action is not evidence. Belief expectations do not increase
bounds.

A projected exchange requires `L >= receive`, updates hand by `-give+receive`
and both bank bounds by `+give-receive`. Build costs return to the bank and can
prove later withdrawals. Evidence from the root menu is never injected again.
The root checks component bounds and the known total bank count; updates preserve
these invariants and card conservation. Hidden stock and deck order stay unknown.

Maritime gives one resource for one card of a different resource. The ratio is
`min(4, applicable owned Port.ratio values)`, with ownership determined from
public building endpoints; projected settlement ownership changes future ratios.
Future projected roads cannot cross an opponent building or use an occupied edge;
the root-only menu certification is never reused for future intentions or geometry
features. Settlements obey the all-player distance exclusion, own-road connectivity
and piece stock. City upgrades return a settlement piece. Draws, steals, dice, opponent actions, domestic
negotiation and development cards are not projected.

The only shared feature changes expose immutable `PUBLIC_BUILD_COSTS` referring
to the existing literal recipes, and explicit geometry helpers in
`settlement_sites.py`. The old count remains a compatibility wrapper. The 34D
consequence order/values and 24D trading context remain unchanged, with literal
full-vector regression checks.

## Frozen initial search and score

Defaults are depth **6**, attempted transitions **2048**, beam **24**, geometry
work **100000**, seed **11**, and strict override margin **0.25**. Coefficients
were fixed before the synthetic smoke; no games informed them. Smaller budgets
can be supplied for contract tests; no option-B/particles/opening configuration
exists. There is no policy wall-clock cutoff.

Every supported root attempt (including rejected metadata and actual END), and
every generated future attempt (including rejected/duplicate children) consumes
one node. Root enumeration is city, settlement, road, maritime, END, canonical
within each group. Future enumeration uses that build order, then maritime
exchanges in the same canonical `_step_key` order used for roots and path ties.
Every feasible prefix can stop without creating an END action. Children
are deduplicated by complete projected state plus original root index. The best
24 endpoints form the next beam; no widening or appealing-chain completion occurs
past a budget.

Geometry work is a deterministic logical accounting model: one unit per board
edge in initial adjacency construction, one per tile-vertex yield visit, then
`3*edges + 4*roads + 4*buildings` per shape description for the shared public
geometry scans, plus one per visited accessibility vertex, inspected adjacency
edge and inspected candidate vertex. Shape cache hits are charged exactly the
cold description's logical work. Caches are local to one decision and cannot
change action or counters. Geometry work saturates at the configured cap. After
complete root validation, a breach retains a fully constructed known winning
prefix containing a build; all other geometry breaches return H_view. Node
exhaustion retains the best completed prefix, except incomplete root validation
returns H_view. This bounds logical work, not every Python instruction or H_view's
existing computations.

The position score is quantized to integer micro-units before comparison:

`100*P + 20*Y + 6*E + 4*B + 12*R`.

- `P`: root own public VP plus known building-point gain.
- `Y`: unblocked expected own production per roll, weighted by
  `(1.0,1.05,0.85,1.4,1.25)` in resource order.
- `E`: the best three independently reachable distance-eligible site yields,
  each discounted by `2**additional_roads`; a 0/1 graph search permits at most
  two new roads and respects current own stock, opponent roads and buildings.
- `B`: best geometrically possible recipe readiness with weights road `.5`,
  settlement `1.2`, city `1.3`, plus `.025*sum(min(h[r],4))`, minus
  `.03*max(0,sum(h)-discard_threshold)`.
- `R`: public opponent rank relative to own projected public VP, normalized by
  opponent count. Opponent public VP is fixed.

A known winning gate uses **own_victory_points** (including own known VP cards)
plus building-point gain against the supported rules' victory target. No new
longest-road/largest-army award is inferred, and no opponent hidden VP is guessed.
Accessibility sites are alternatives, not simultaneous guaranteed settlements.

Ties prefer known win, score, fewer exchanges, fewer steps, canonical path, then
original root index. Supported actions' resource/second-resource/partner/offer-id
fields are all null, so their omitted path-key fields are constant null sentinels;
kind/target (null `-1`)/give/receive determine the remaining canonical key.
Override requires a build and strict gain above both root and H_view's supported
one-step score, or a known winning path. Bank cycling alone cannot override.

## H_view and traces

H_view instantiates the unchanged production `HeuristicPolicy` for each call.
Its local integer seed is SHA-256 of compact sorted-key canonical JSON containing
`[config.seed, domain, dataclasses.asdict(observation)]`. Act and propose use
separate `"act"`/`"propose"` domains. It consumes no global RNG or historical policy
RNG, and duplicates no H scoring. The planner's proposal method is this same
unchanged H_view proposal behavior.

H_view is a new deterministic control, distinct from historically stateful
production H. A future matched study must identify H_view and H_view+planner
separately and include unchanged production H before stronger-than-H claims.
No benchmark, rollout, label generation or training is authorized by this module.

`TurnPlan.reason` distinguishes `planned`, `insufficient_gain`,
`unsupported_context`, `unsupported_rules`, `unsupported_heuristic_action`,
`public_contract_mismatch`, `root_validation_limit`, and `geometry_limit`.
`search_stop` independently records `not_searched`, `depth_limit`, `node_limit`,
`empty_beam`, or `geometry_limit`. `steps` is empty on every H_view fallback;
`score_after_micro` then equals root score (zero if scoring was not reached),
not an assertion that an unsupported H effect was simulated. No debug export
schema is added.

## Validation and runtime limits

Manufactured tests cover multi-exchange city and exchange→road→settlement chains,
withdrawal evidence consumption, exact bank stock, port ownership and acquisition,
resource/stock conservation, blocked/distance/piece geometry, current object
identity and reobservation, profile/effect fallbacks, ties, budgets, graph score,
cache charging, immutable inputs, SHA domains and global-RNG invariance. Tiny
standard-engine setup/transition replays check road, settlement, city and actual
specific/generic port exchanges and own known-VP/public-VP semantics. Poisoned
engine/environment properties are never queried. Hidden opponent composition, undisclosed bank
composition and deck changes preserving the same public view preserve the plan
and counters. These are not complete games or empirical strength measurements.

One bounded synthetic smoke on macOS arm64/Python 3.13.2/Catanatron 3.3.0, with
Torch/BLAS threads 1, made five calls to each of four manufactured fixtures:

| Fixture | Median ms | Max ms | Nodes | Geometry | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| two exchanges→city | .726 | 1.262 | 139 | 306 | planned |
| exchange→road→settlement | .700 | .870 | 169 | 510 | planned |
| sparse END | .082 | .091 | 1 | 54 | H_view / insufficient gain |
| standard board, rich hand | 6.583 | 7.370 | 639 | 100000 | H_view / geometry limit |

The measured decision work totaled .075 seconds; the entire smoke process was
under its external 15-second cap. No sample exceeded the predeclared external
100-ms decision deadline. These tiny fixtures do not establish runtime feasibility
on all real states: the rich fixture already exhausts conservative geometry
coverage. Later trials exceeding an external deadline must be runtime failures
or censored games; never substitute a timed fallback or ordinary loss label.
No coefficient/budget tuning followed these timings. The table records the
pre-review implementation. The later canonical future maritime ordering fix was
not retimed, so this table is not a runtime check of the corrected candidate.
