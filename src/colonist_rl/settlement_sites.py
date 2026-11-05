"""Pure public settlement-site geometry shared by observation feature schemas."""

from __future__ import annotations

from .contracts import Observation


def connected_settlement_site_count(view: Observation) -> int:
    """Count vacant own-road endpoints outside every building's distance exclusion.

    This is an unclipped geometric count, independent of phase, piece stock,
    resource readiness or bank knowledge. Callers retain their own normalization
    and readiness conditions. It performs no road search and mutates no input.
    """
    occupied = {building.vertex for building in view.buildings}
    blocked = occupied.copy()
    for a, b in view.board.edges:
        if a in occupied:
            blocked.add(b)
        if b in occupied:
            blocked.add(a)
    connected = {
        vertex
        for road in view.roads
        if road.owner == view.player_id
        for vertex in view.board.edges[road.edge]
    }
    return len(connected - blocked)
