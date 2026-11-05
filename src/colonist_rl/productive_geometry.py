"""Small public geometry block; potential access is never a future legal menu."""

from __future__ import annotations

import math

from .contracts import RESOURCES
from .settlement_sites import (
    connectable_road_edges,
    connected_settlement_sites,
    settlement_exclusion,
)

PRODUCTIVE_COLUMNS = tuple(
    sorted(
        (
            "pg_connected_sites",
            "pg_connected_best_pips",
            "pg_one_road_new_sites",
            "pg_one_road_best_pips",
            "pg_expansion_edges_without_site",
            "pg_owned_port_utility",
            "pg_connected_port_gain",
            "pg_one_road_port_gain",
        )
    )
)


def encode_productive_geometry(view, projected=None):
    """Encode nominal pips, stock-aware access and production-supported port gains.

    Uses actual geometry or explicit projection overlays. Settlement access ignores
    hand affordability/phase and the root legal menu. Nominal production includes
    the robber's tile, as in the existing encoder's ``production_*`` features.
    One-road sites are the union of NEW spacing-legal endpoints reachable by one
    unoccupied, publicly connectable edge; no road awards or later menus are inferred.
    """
    own = next(p for p in view.players if p.player_id == view.player_id)
    stock = own if projected is None else projected
    buildings = view.buildings if projected is None else projected.buildings
    roads = view.roads if projected is None else projected.roads
    board = view.board
    blocked = settlement_exclusion(board, buildings)
    sites = set(connected_settlement_sites(board, buildings, roads, view.player_id))
    edges = (
        connectable_road_edges(board, buildings, roads, view.player_id)
        if stock.roads_left > 0
        else ()
    )
    destinations = {
        vertex for edge in edges for vertex in board.edges[edge] if vertex not in blocked
    }
    new_sites = destinations - sites if stock.settlements_left > 0 else set()
    if stock.settlements_left <= 0:
        sites = set()
    without_site = sum(
        stock.settlements_left <= 0 or all(v in blocked for v in board.edges[edge])
        for edge in edges
    )

    # One tile-incidence pass supports both site quality and own production.
    pips = [0.0] * len(board.vertices)
    production = [0.0] * len(RESOURCES)
    owned = {b.vertex: b.level for b in buildings if b.owner == view.player_id}
    for tile, vertices in zip(board.tiles, board.tile_vertices, strict=True):
        if tile.resource not in RESOURCES or not tile.number:
            continue
        amount = float(6 - abs(7 - tile.number))
        resource = RESOURCES.index(tile.resource)
        for vertex in vertices:
            pips[vertex] += amount
            production[resource] += owned.get(vertex, 0) * amount / 36

    # Cache port ratios per vertex in one pass; no per-road geometry reconstruction.
    port_ratios = {}
    for port in board.ports:
        if type(port.ratio) is not int or port.ratio <= 0:
            raise ValueError("positive integer public port ratio required")
        for vertex in port.vertices:
            ratios = port_ratios.setdefault(vertex, [4] * len(RESOURCES))
            for i, resource in enumerate(RESOURCES):
                if port.resource in (None, resource):
                    ratios[i] = min(ratios[i], port.ratio)
    own_ratios = [
        min([4] + [port_ratios[v][i] for v in owned if v in port_ratios])
        for i in range(len(RESOURCES))
    ]

    def port_gain(vertices):
        return max(
            (
                sum(
                    production[i] * (1 / min(ratio, own_ratios[i]) - 1 / own_ratios[i])
                    for i, ratio in enumerate(port_ratios[v])
                )
                for v in vertices
                if v in port_ratios
            ),
            default=0.0,
        )

    result = {
        "pg_connected_sites": float(len(sites)),
        "pg_connected_best_pips": max((pips[v] for v in sites), default=0.0),
        "pg_one_road_new_sites": float(len(new_sites)),
        "pg_one_road_best_pips": max((pips[v] for v in new_sites), default=0.0),
        "pg_expansion_edges_without_site": float(without_site),
        "pg_owned_port_utility": sum(
            production[i] * (1 / ratio - 1 / 4) for i, ratio in enumerate(own_ratios)
        ),
        "pg_connected_port_gain": port_gain(sites),
        "pg_one_road_port_gain": port_gain(new_sites),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite productive geometry features")
    return result
