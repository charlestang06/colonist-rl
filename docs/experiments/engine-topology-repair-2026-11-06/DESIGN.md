# Local Catanatron road topology repair

The pinned Catanatron 3.3.0 engine permits roads to extend through an opponent
settlement and can later crash when a settlement cuts a corrupted road cache.
The 19-action excluded reproduction and the sealed failed TRAIN_FIT packet are
at `runs/swarm-2026-11-04/agent-research/second-settlement-fit-failed-001/`.

This change adds an opt-in `CorrectedBoard` owned by `colonist_rl`. It preserves
the pinned Catanatron installation and the public `Observation`/`Action`
contracts. `FourPlayerGame(corrected_topology=True)` installs the corrected board
at reset; the default remains the prior engine for historical artifacts. New
studies must bind the mode and source hash, and must use fresh cleared seeds.

## Road topology

Represent each player's road graph as undirected **edges**. Two owned road edges
are connected at a shared vertex only if that vertex has no opponent building.
An opponent building may be the endpoint of two road components, never an
expansion origin. Rebuild components after any road or settlement mutation.
Include each isolated own settlement as a singleton component so it can seed a
first road. Derive buildable roads from unblocked component vertices and empty
land edges. Derive normal settlement locations from owned road endpoints and
the engine's global distance-rule cache. Initial setup keeps the engine's
ordinary distance rule and selected settlement/road phases.

Compute each player's longest edge-simple trail by DFS over owned roads. A
trail may end at an opponent settlement, counting that final road, but may not
continue through it. A length below five has no award. The current holder
keeps a tied lead; a sole new leader takes the award; a tie without the current
holder leaves it unclaimed. Reconcile backend `HAS_ROAD`, visible VP, actual VP,
and per-player road lengths after road and settlement actions, including when
the award is lost to no holder.

## Boundaries and validation

The board shim reads engine state only inside the simulator. Policies, action
encoders, and inference remain `Observation` plus original legal actions. The
shim is opt-in; no old labels are reinterpreted. The failed FIT role stays
failed and archived.

Tests first reproduce through-blocker expansion, component loss, the later
settlement crash, blocked-endpoint road length, tie/loss award behavior, and a
branch/cycle road graph. Then compare a complete excluded normal game under
raw and corrected modes step by step where no blocker interaction occurs.
Finally measure a bounded excluded corrected run's wall time and peak memory
before any fresh outcome role is scheduled.
