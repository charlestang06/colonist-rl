"""Pure public geometry shared by feature schemas and ephemeral turn projections."""

from __future__ import annotations

from .board import BoardSpec
from .contracts import Building, Observation, Road


def settlement_exclusion(board: BoardSpec, buildings: tuple[Building, ...]) -> frozenset[int]:
    """Occupied vertices and their immediate neighbors, for every owner."""
    occupied = {building.vertex for building in buildings}
    blocked = occupied.copy()
    for a, b in board.edges:
        if a in occupied:
            blocked.add(b)
        if b in occupied:
            blocked.add(a)
    return frozenset(blocked)


def connected_settlement_sites(
    board: BoardSpec, buildings: tuple[Building, ...], roads: tuple[Road, ...], player_id: int
) -> tuple[int, ...]:
    """Sorted vacant own-road endpoints outside building distance exclusion."""
    blocked = settlement_exclusion(board, buildings)
    connected = {
        vertex for road in roads if road.owner == player_id for vertex in board.edges[road.edge]
    }
    return tuple(sorted(connected - blocked))


def connectable_road_edges(
    board: BoardSpec, buildings: tuple[Building, ...], roads: tuple[Road, ...], player_id: int
) -> tuple[int, ...]:
    """Unoccupied edges reaching own buildings or unblocked own-road endpoints."""
    own_buildings = {b.vertex for b in buildings if b.owner == player_id}
    enemies = {b.vertex for b in buildings if b.owner != player_id}
    own_network = {
        vertex for road in roads if road.owner == player_id for vertex in board.edges[road.edge]
    }
    links = own_buildings | (own_network - enemies)
    occupied = {road.edge for road in roads}
    return tuple(
        index
        for index, edge in enumerate(board.edges)
        if index not in occupied and any(vertex in links for vertex in edge)
    )


def connected_settlement_site_count(view: Observation) -> int:
    """Unclipped compatibility count, independent of phase, stock and resources."""
    return len(connected_settlement_sites(view.board, view.buildings, view.roads, view.player_id))
