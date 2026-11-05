"""Compact public ingredients for contextual trading; no teacher action scoring."""

from __future__ import annotations

import torch

from .contracts import RESOURCES, Observation

TRADING_CONTEXT_SCHEMA = "public-trading-context-v1"
TRADING_CONTEXT_DIM = 24


def trading_context_features(view: Observation, *, device="cpu") -> torch.Tensor:
    """Encode piece stock, public goal gates, local geometry and own production.

    Goal gates describe the frozen teacher's recipe availability, including its
    city-stock omission. Geometry, phase and production are additional public
    context, not terms in that teacher's trading utility. No engine or teacher
    method is consulted, and unknown bank stock is intentionally irrelevant.
    """
    own = next(player for player in view.players if player.player_id == view.player_id)
    buildings = tuple(b for b in view.buildings if b.owner == view.player_id)
    settlements = sum(b.level == 1 for b in buildings)
    occupied = {b.vertex for b in view.buildings}
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
    context = view.context
    goals = (
        bool(own.roads_left),
        bool(own.roads_left and own.settlements_left),
        bool(settlements),
        own.development_count < 8,
    )
    production = [0] * len(RESOURCES)
    coverage = [0] * len(RESOURCES)
    for building in buildings:
        adjacent = set()
        for tile, vertices in zip(view.board.tiles, view.board.tile_vertices):
            if building.vertex in vertices and tile.number and tile.resource in RESOURCES:
                resource = RESOURCES.index(tile.resource)
                production[resource] += building.level * (6 - abs(7 - tile.number))
                adjacent.add(resource)
        for resource in adjacent:
            coverage[resource] += 1
    return torch.tensor(
        [
            own.roads_left / 15,
            own.settlements_left / 5,
            own.cities_left / 4,
            settlements / 5,
            own.development_count / 8,
            min(len(connected - blocked), 8) / 8,
            bool(settlements and own.cities_left > 0),
            context.kind in ("TURN", "PLAY_TURN")
            and context.actor == context.turn_owner == view.player_id
            and context.has_rolled,
            context.kind == "TRADE_RESPONSE",
            *goals,
            not any(goals),
            *(pips / 30 for pips in production),
            *(count / len(buildings) if buildings else 0 for count in coverage),
        ],
        dtype=torch.float32,
        device=device,
    )
