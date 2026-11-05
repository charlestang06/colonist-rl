# Simulator adapter audit

The simulator pins PyPI `catanatron==3.2.1`. The attempted upstream GitHub clone was unavailable, so this implementation audits the installed release directly rather than claiming current-main behavior. That release supplies four-player base-game board legality, setup, production, building, ports, robber, development cards, longest road, and largest army. It has no domestic-trading action implementation.

`FourPlayerGame` wraps the release with immutable backend-independent observations and typed legal actions. Player ids are seating positions. Hex coordinates use backend cube x/z as axial q/r; the adapter explicitly matches each named backend corner and edge to the shared integer-lattice topology. Policies never receive engine actions, state objects, other players' resource identities, development identities, or deck order.

The adapter corrects three release limitations: individual chosen discards replace random bulk discards; Road Building remains available and places two free roads when the player has no road resources; development cards bought this turn are unavailable until the following turn. Victory is checked for the turn owner. Backend randomness is isolated per environment and preserves the process RNG across reset and steps.

Domestic trades are a response-only wrapper extension. A valid current-turn, post-roll proposer may freeze one pending offer. The recipient sees an offer-id-bound accept/reject choice, with acceptance offered only when their own hand can fund it. Both hands are checked before acceptance writes; one transaction commits once, and used ids cannot replay. Rejection preserves every hand.

Public production and costs update observer-specific belief ledgers. Steal identities are supplied only to the two parties, and discard identities only to the discarder. No exact bank vector is published because comparing bank changes could reveal hidden discard identities. A legal maritime-trade mask can expose rule-level bank availability. Policies can infer resources exactly when public history warrants that inference; hidden events broaden uncertainty.

Four-player simulation is supported. Six-player board and rule data are representable, but attempting a six-player simulator raises an explicit `NotImplementedError`; special building is not simulated. The adapter does not claim online Colonist-client integration or equivalence to its proprietary rules implementation.

Validation covers immutable initial views, 19/54/72 topology, interleaved seeded reproducibility, global RNG preservation, full heuristic games and resource conservation, hidden-state view invariance, incoming trade rejection/acceptance/stale ids, hidden discards, forced actors, development-card age, and free roads. Ten additional seeded heuristic self-play games completed with winners in 184–503 decisions. This is mechanics validation, not an estimate of learned-policy strength.
