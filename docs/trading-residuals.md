# Public contextual trading residuals

These opt-in heads test whether nonlinear hand/state interactions improve supervised
trading agreement. They preserve an existing `ConsequenceGraphPolicy` with
`feature_set="hand_recipe"`, including its nonzero learned 34D weights. They do not
change the frozen plain Greedy teacher, labels, hard-label cross entropy, shared
sequence fitter, legacy Graph defaults, or checkpoint schema version 1.

## Public context schema

`trading_context_features(view, *, device="cpu")` returns a float32 tensor of width
24, with schema `public-trading-context-v1`. Its complete input is the immutable
public `Observation`; no teacher method, engine, opponent private hand, search,
utility, or actor rollout is consulted. The resource order in both channel groups
is **WOOD, BRICK, SHEEP, WHEAT, ORE**.

| Slot | Definition |
| --- | --- |
| 0 | Own roads left / 15 |
| 1 | Own settlements left / 5 |
| 2 | Own cities left / 4 |
| 3 | Own level-one building count / 5 |
| 4 | Own public development count / 8, without clipping |
| 5 | `min(own-road-connected distance-legal vacant settlement sites, 8) / 8` |
| 6 | Own level-one building exists and cities left > 0 |
| 7 | Own post-roll `TURN` or `PLAY_TURN`, with actor = turn owner = player id |
| 8 | Context kind is `TRADE_RESPONSE` |
| 9 | Teacher road gate: roads left > 0 |
| 10 | Teacher settlement gate: roads left > 0 and settlements left > 0 |
| 11 | Teacher city gate: an own level-one building exists |
| 12 | Teacher development gate: public development count < 8 |
| 13 | No gate in slots 9–12; teacher uses its development fallback |
| 14–18 | Own public production pips per resource / 30 |
| 19–23 | Fraction of own building vertices adjacent to each producing resource |

Connectivity uses endpoints of incident own roads, excludes occupied vertices and
vertices adjacent to any building, and counts the remaining sites. It does not
search for future roads. Production sums `building.level * (6 - abs(7 - number))`
over adjacent producing tiles; coverage counts a resource once per own building
vertex, with all own building vertices as its denominator. Desert and number-zero
tiles do not contribute. No own buildings gives zero production and coverage.
The robber is ignored, matching the teacher's placement production convention.
Unknown bank stock does not alter these features or invalidate an already legal
no-port 4:1 exchange. The current public legal tuple establishes feasibility.

## What the teacher actually uses

The frozen teacher's trade method is `_utility`. Its goals are settlement (weight
1.2) when both settlement and road pieces remain, city (1.3) when an own level-one
building exists, development (0.8) below public development count 8, and road (0.5)
when road pieces remain. With none of these gates it uses development at weight 1.
It maximizes weighted normalized recipe progress and adds
`0.025 * sum(min(hand_resource, 4))`.

The teacher city gate **does not check remaining city pieces**. Slot 11 records
that actual gate; slot 6 separately records real city-piece availability.
Production, coverage, connectivity, phase, and exhausted city-piece context are
additional public strategic ingredients, not terms in the current trading utility.
The teacher uses resource-weighted pips and diversity in placement scoring, and
nearby-site search for road placement. Those action scores and searches are not
reimplemented or called by the actor.

Maritime teacher scoring is `20 + 20 * utility_delta` when the delta exceeds 0.005,
otherwise -5, plus jitter. END gets jitter; ACCEPT gets +10 or -10 from whether
utility is nondecreasing; REJECT gets 0. The heads do not compute these scores.

## Matched heads and initialization

A is the unchanged trained Graph plus 34D hand-recipe scorer. B,
`ContextLinearConsequencePolicy`, adds a separate bias-free `Linear(24, 1)` context
offset, initially zero. This is algebraically the same input set as a linear58
control, but keeps the original 34D dot product and its rounding unchanged.
The context offset is common to maritime and END candidates in one state, so it
cannot add a state-conditioned relative preference between them. It can shift
supported actions against unsupported kinds. The same limitation applies to the
ACCEPT/REJECT pair.

C, `NonlinearConsequencePolicy`, adds
`Linear(58, 16, bias=True) -> tanh -> Linear(16, 1, bias=False)`.
The input concatenates the existing hand-recipe34 (slots 30–33 masked to zero)
and context24. The hidden map uses ordinary nonzero initialization and the final
weights are exactly zero. Its output weights can receive a first-step gradient;
the hidden map initially has zero gradient and can receive gradients after an
output update. Zeroing both maps would prevent this useful initialization.

For both B and C, the inherited parent output is computed first and the new
residual is added second. A separate explicit mask leaves the residual zero on
all kinds except `MARITIME_TRADE`, `ACCEPT_TRADE`, `REJECT_TRADE`, and `END_TURN`,
even after its weights change. Candidates always come from the current legal
menu. Values, memory, recurrence, and memory-only prefix routes remain inherited.

`trading_residual_from_consequence(parent, *, kind, seed=0)` accepts exactly an
unmodified hand-recipe `ConsequenceGraphPolicy`, with `kind="context_linear"` or
`kind="nonlinear"`. Repeat upgrades and other parents/kinds are rejected. It copies
all inherited tensors into independent storage, preserves dtype, device, inherited
parameter freeze flags and module modes, and seeds construction inside a preserved
CPU Torch RNG scope. It leaves caller Torch, Python and NumPy RNG state unchanged.
At creation, scalar and batched logits, values and memories equal the parent
bitwise; the parent is never repacked into a wider linear reduction.

## Checkpoint identities

B has exactly these architecture fields:

```json
{"kind":"graph_consequence_context_linear_v1","hidden_size":64,
 "feature_schema":"public-consequence-v1","feature_set":"hand_recipe",
 "context_schema":"public-trading-context-v1"}
```

C has the same fields with `kind="graph_consequence_nonlinear_v1"` and the additional
`"residual_hidden_size":16`. Hidden width is exactly integer 16, including rejection
of bools and floats. Factories reject extra/missing fields, unknown kinds/schemas,
and other feature sets. The existing loader strictly loads state keys and shapes,
including missing/extra residual tensors. Existing Graph and 34D identities retain
their exact metadata and state shapes. Saving/loading requires no Git checkout.

## Scope of later matched fitting

A later small comparison must use the same parent, examples, hard-label objective,
and freeze scope for all arms. Freeze inherited Graph, recurrence and value
parameters; train the parent34D scorer and the new residual. The unchanged shared
fitter requires Adam to own **every parameter exactly once**, including frozen
parameters. Frozen parameters must have `grad=None` and remain bitwise unchanged;
each arm needs fresh independent persistent Adam state. Freeze scope belongs in
experiment metadata, not the architecture identity. Prefixes continue to replay
under current weights before each update.

This implementation includes only bounded fixture checks: literal features,
exact parent copy/output equality, a two-state fixed-weight representation example,
gradient connection, strict checkpoint portability, and two shared-fitter updates.
The fixed-weight example changes only public goal availability with an identical
hand; it demonstrates representability, not teacher agreement or playing strength.
No 512-case fit, held-out gradient fitting, gameplay, reward/PPO change, or 100k run
is authorized by this architecture task. Independent architecture review must
precede the separate small-fit protocol.
