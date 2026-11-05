# Opt-in public consequence policy

`GraphPolicy` and its default checkpoint metadata remain unchanged. To create an
explicit comparison, call `consequence_from_legacy(policy)`; this copies every
base parameter into an independent `ConsequenceGraphPolicy`, preserves the
caller's Torch RNG and training mode, and adds one zeroed, bias-free 34→1 linear
projection. Initial logits, recurrent memory and value match the source policy
exactly. The projection receives gradients on its first supervised update.

The branch is added after the existing actor logits for both scalar and batched
calls, including binary trade responses. Recurrent-only prefix replay never
computes the branch. The branch does not change value or recurrent computation.

`consequence_features(view, action)` accepts only an action in the immutable
observation's legal tuple. It reads no simulator state and mutates no observation.
Supported transitions are maritime exchange, accepting an offer from the current
player's perspective, rejecting an offer, and ending a turn. Other legal kinds
return 34 zeros. Existing legal exchanges establish first-exchange feasibility;
an unknown bank is never inferred or replaced with the rule profile's stock.

Resource order is wood, brick, sheep, wheat, ore. Slots have these meanings:

| Slots | Encoding |
| --- | --- |
| 0 | Known deterministic own-hand transition mask |
| 1–5 | Post-action hand / 10 |
| 6–25 | Positive resource deficits / 4, five per recipe |
| 26–29 | Resource sufficiency for each recipe |
| 30 | Connected vacant distance-legal settlement vertices, capped at 8, / 8 |
| 31 | Settlement resources, connected site, available settlement piece and normal build phase |
| 32 | Own level-one building and available city piece |
| 33 | City resources, slot 32 and normal build phase |

Recipes, in order, are road `(1,1,0,0,0)`, settlement `(1,1,1,1,0)`, city
`(0,0,0,2,3)`, and development `(0,0,1,1,1)`. Resource flags do not certify
geometry, pieces, bank inventory or development-deck stock. Settlement sites
exclude occupied vertices and their neighbors and require an incident own road.
Normal build phase means `TURN` or `PLAY_TURN`, after rolling, with the observing
player as both actor and turn owner. Geometry slots describe the observed board;
readiness flags describe ordinary-build assumptions on that phase. They never
create a legal build after `END_TURN` or an incoming trade response.

The default `feature_set="hand_recipe"` masks slots 30–33 in the scorer. A separate
`feature_set="hand_recipe_and_connectivity"` enables them. Each is an explicit
identity; `feature_set` is read-only and upgrading an existing consequence policy
with `consequence_from_legacy` is rejected. To declare a later connectivity
comparison with learned recipe coefficients, construct the second architecture
explicitly, strictly copy the first policy's state, and zero the four newly enabled
coefficients before use. Do not relabel an existing checkpoint.

Checkpoint schema stays at version 1. The exact legacy architecture is
`{"hidden_size": 64}`. The opt-in shape is
`{"kind": "graph_consequence_v1", "hidden_size": 64,
"feature_schema": "public-consequence-v1", "feature_set": "hand_recipe"}`
(or the separately declared connectivity feature set). The shared checkpoint
loader rejects extra or missing metadata fields, unknown identities and schemas,
and missing or extra parameter keys. Curriculum loading uses that same dispatch,
and its frozen historical policies remain independent deep copies. Existing
checkpoint files are never rewritten by loading.
