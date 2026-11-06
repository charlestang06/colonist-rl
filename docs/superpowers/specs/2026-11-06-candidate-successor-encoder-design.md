# Public candidate successor encoder

## Purpose

The seed-11 graph policy has action kind, resource exchange, target node, and a
34-dimensional consequence branch. That branch is zero for direct builds and
development-card purchases. A single extra point-gap scalar trained on 24
audited states changed no held-out decisions. This experiment adds a richer,
explicit **current-action** representation before testing it with broad
student-reached ranking data. It remains opt-in and starts with exactly the
parent's logits.

## Public input and boundary

`successor_features_for_menu(view)` receives one immutable `Observation` and
returns one fixed-width row per original legal `Action`, in menu order. A
single-action function requires that exact object from the current legal tuple.
It never reads engine state, opponent hands, actual development draws, later
events, or another action's hypothetical future menu.

For deterministic ordinary-turn roots, encode current and after-action hand,
their delta, deficits and affordability for road, settlement, city and card
recipes, and whether a public settlement/city site and piece exist. Encode
known piece and public point changes, the target settlement/city's nominal and
currently unblocked production gain by resource, and own port ratios after a
new settlement. Road and settlement geometry after a build may be computed
from the public board. Setup placement and free roads have zero resource cost.
Current maritime trades and development purchases have exact immediate hand
arithmetic; a card purchase marks its draw as unknown. The second setup
settlement gains one resource from each adjacent producing hex when public bank
stock or the standard setup stock bound guarantees delivery. Otherwise its
after-hand is masked unknown. A robber move without a victim preserves the
known hand; a move with a victim masks the unknown steal. A one-resource Year
of Plenty option adds that named resource. Playing a knight can award Largest
Army, and building a road can award Longest Road; those awards and an immediate
win are bounded from public counts, holders, buildings and road geometry. An
ordinary settlement can also cut exactly two opponent roads and cause the
pinned engine to reassign Longest Road. Bound the cut player's recomputed road
length from public geometry, but use each unaffected player's public cached
length. The backend's cached connected components can omit an older enemy road
endpoint, so a public full-edge trail is only an upper bound; a trail starting
from guaranteed component nodes supplies the lower bound. This settlement
path has no five-road threshold. Equal post-cut maxima are decided by engine
insertion order absent from the observation. Road placement has the same
endpoint ambiguity when its candidate trail approaches an older enemy
building. The actor's point delta is therefore a guaranteed lower bound with
an explicit `point_outcome_uncertain` flag whenever the actor award can vary.
`wins_now` means a guaranteed immediate win; `own_vp_after` is the lower-bound
result when the flag is set. A development purchase also sets point
uncertainty because its hidden draw may be a VP card.
Other uncertain actions carry explicit masks rather than invented resource
gains. The encoder does not claim a future card identity, robber steal, or
later legal menu.

The feature names and order form a versioned schema. All rows are finite.
The settlement-cut and point lower-bound semantics use
`public-candidate-successor-v3` and a distinct `graph_candidate_successor_v3`
checkpoint identity, so earlier experimental checkpoints cannot load silently
with changed feature meanings.
Building and port geometry are computed once per observation where possible,
then only the affected public placement is updated for each build action.

## Policy and checkpoint

`CandidateSuccessorPolicy` extends the existing `ConsequenceGraphPolicy` with a
candidate-conditioned linear branch and a small nonlinear branch using the
current recurrent hidden state plus the candidate successor row. Both output
weights start at zero. Converting a strict `hand_recipe` consequence parent
copies every inherited tensor, device, dtype, trainability and mode without
advancing the caller's Torch RNG. At conversion, logits, recurrent memory and
value are bitwise equal to the parent on every legal menu. The new architecture
has its own strict checkpoint identity and cannot load as a legacy policy.

## Evaluation boundary

This task delivers representation and mechanical preflight only. No claim of
stronger decisions follows from an encoder or passing tests. A later frozen
factorial comparison can train the same broad student-state targets with and
without this branch and then require actual changed choices, public decision
review, and fresh paired games. The broad distillation lane owns data and loss;
this branch changes representation alone.

## Verification

Tests check exact legal-action identity; trade/build/BUY/END hand arithmetic;
site, production and port effects; free road/setup cost; no-victim robber and
one-resource Plenty; public road and army awards, including settlement cuts,
sub-five-road reassignment, hidden road endpoints and ambiguous ties;
unknown-outcome masks;
full-menu/single-action parity; immutable observations; finite rows; zero-start
parent/RNG equality across CPU, MPS and ambient meta construction; mixed dtype
parent preservation; nonzero useful gradients; and strict checkpoint roundtrip.
Time the incremental encoder on representative legal menus before proposing a
gameplay comparison.
