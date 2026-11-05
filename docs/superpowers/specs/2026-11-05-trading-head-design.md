# Conditional trading residuals

User-approved direction; implementation planning only. Plain Greedy rules frozen;
actual100k HOLD. Root supplies separate legacy3.2.1 checkout from4215d4c, copied
legacy-runtime only, later ports accepted actor code to3.3. No old source/run edits.

## Teacher audit and hypothesis

Inspected opponents.py at4215d4c. Actual method is `_utility`, not `_hand_utility`.
`_goals`: settlement if settlements_left AND roads_left; city if own level-one
building (does NOT test cities_left); development if public development_count<8;
road if roads_left; fallback development weight1. Weights: settlement1.2, city1.3,
development0.8, road0.5. Utility=max weighted normalized recipe progress plus
0.025*sum(min(hand_i,4)). Maritime is20+20*delta if delta>0.005, else-5, plus jitter;
END has jitter; ACCEPT scores±10 from nondecreasing utility, REJECT0.
Production/connectivity are NOT in trade utility. Placement `_production` uses
resource-weighted pips plus1.4 per distinct adjacent resource; road score searches
nearby sites. Public ingredients are context, not teacher-strategy truth.

Hypothesis: nonlinear hand/state interactions improve conditional trading fit.
Compare against SAME-input linear control, identical parent/base/data/objective.
No teacher score/utility, hidden engine/bank/opponent hand or rollout actor input.

## Context schema: public-trading-context-v1,24 dimensions

|Slots|Definition|
|---|---|
|0,1,2|own roads_left/15, settlements_left/5, cities_left/4|
|3|own level-one building count/5|
|4|own public development_count/8, no clipping|
|5|min(own-road-connected distance-legal vacant settlement sites,8)/8|
|6|own level-one building exists AND cities_left>0|
|7|ordinary own post-roll TURN/PLAY_TURN; actor=turn_owner=player_id and has_rolled|
|8|context.kind == TRADE_RESPONSE|
|9..12|teacher road, settlement, city, development gates above, in that order|
|13|none of those four gates (teacher fallback)|
|14..18|own public production pips per resource: sum(building.level*pips) over adjacent producing tiles, divided by30|
|19..23|per resource, fraction of own building vertices adjacent to that resource, zero for no own buildings|

Shared resource order WOOD/BRICK/SHEEP/WHEAT/ORE. Pips=6-abs(7-number), desert and
number0 ignored. Ignore robber as teacher `_production` does. Production channels
are raw production/coverage factors, not weighted teacher action scores. Connectivity
uses incident own roads, occupied vertices and adjacent-building exclusion, no
search. Teacher city gate and true piece availability intentionally differ.
Phase/connectivity/exhausted-piece context is additional public information NOT
consulted by current utility; report that honestly. Unknown bank changes nothing.

## Three heads and exact initialization

A unchanged trained ConsequenceGraphPolicy hand_recipe34. B adds a separate
zero-initialized bias-free Linear(24,1) context offset to A. C adds
Linear(58,16,bias=True) -> tanh -> Linear(16,1,bias=False) to A. Hidden ordinary
nonzero initialization, final output weights exactly zero; never both layers zero.
C input is current hand_recipe34 (slots30..33 masked) concatenated with24context.
B is algebraically the linear58 control, keeping old34 dot reduction separate to
preserve rounding. Never repack parent34 weights into a58-wide reduction.

Both residuals mask to0 on unsupported transitions; supported remains MARITIME,
ACCEPT,REJECT,END exact existing action names. Require action in current legal menu.
Parent output first, residual second; values/memory/prefix-only paths unchanged.
Copy ALL old Graph/34D tensors exactly into independent storage. At creation scalar
and batched logits/value/memory bitwise equal parent. B context offset is identical
across bank/END choices within a state and cannot change their relative preference;
it may shift supported versus unsupported kinds. This intended control is disclosed.

## Interfaces, metadata and ownership

Create trading_context.py: `TRADING_CONTEXT_SCHEMA='public-trading-context-v1'`,
`TRADING_CONTEXT_DIM=24`, `trading_context_features(view, *, device='cpu')->Tensor`.
Add in policy.py `ContextLinearConsequencePolicy(hidden_size=64)` and
`NonlinearConsequencePolicy(hidden_size=64, *, residual_hidden_size=16)`; helper
`trading_residual_from_consequence(policy, *, kind, seed=0)` accepts exactly
kind=context_linear|nonlinear and exact hand_recipe parent ConsequenceGraphPolicy.
Reject repeat/unknown upgrades. Preserve caller RNG, parent dtype/device/mode;
seed hidden initialization explicitly inside preserved RNG scope.

Exact B architecture has keys kind,hidden_size,feature_schema,feature_set,
context_schema, with kind=graph_consequence_context_linear_v1,
feature_schema=public-consequence-v1, feature_set=hand_recipe,
context_schema=public-trading-context-v1. C kind=graph_consequence_nonlinear_v1,
same other fields plus exact-int residual_hidden_size=16. Reject extras/unknowns/
bool-as-int/wrong widths/schema/feature set; strict state load. Existing legacy and
34D shapes/defaults/files/checkpoint schema1 remain unchanged. Policy factories
already feed training.load_checkpoint: no training.py edits are needed.

Own new trading_context.py, tests/test_trading_context.py, docs/trading-residuals.md;
modify policy.py only new classes/helper/factories and add focused checkpoint tests
in tests/test_training.py. No teacher, target, labels, RNG behavior, fitter,
consequences.py semantics, contracts, simulator, Greedy, roadmap or migration edits.

Test public slots/geometry/immutability/unknown-bank invariance and unsupported mask;
exact scalar/batch parent copies and initial outputs; RNG/storage/memory isolation;
old/new strict checkpoint no-Git roundtrips. C first output gradient must connect;
hidden first-step zero is expected, after an output update a suitable fixture must
produce hidden gradients. B's bank/END contrast is invariant to state-only offset
within floating tolerance. Synthetic state-conditioned preference demonstrates C's
representation, not gameplay quality. No architecture training run in this task.

Small experiment consistently freezes inherited Graph/value/recurrence, trains
parent34D and residual heads, and gives Adam ALL parameters exactly once as required
by existing fitter. Frozen gradients=None, tensors unchanged; fresh persistent Adam
per arm. Freeze scope is experiment metadata, not silently serialized architecture.
Test compatibility with two shared-fitter updates; no new fitter API or loss.
Independent architecture review precedes actual small-fit protocol.
