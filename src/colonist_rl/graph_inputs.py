"""Bounded cache of immutable board inputs, never learned tensors or dynamic degrees."""

from __future__ import annotations

import time
from contextvars import ContextVar
from functools import lru_cache

import torch
from torch.profiler import record_function

from .contracts import RESOURCES, Observation

NODE_DIM = 40
# Fit-local inclusive encoding timings; no retained observations or trainable values.
encoding_seconds = ContextVar("supervised_encoding_seconds", default=None)


@lru_cache(maxsize=32)
def _static_graph(board, schema_version, device, dtype, player_ids):
    nt, nv, ne = len(board.tiles), len(board.vertices), len(board.edges)
    vertex_offset, edge_offset, player_offset = nt, nt + nv, nt + nv + ne
    # A first evaluation under inference_mode must still leave normal input buffers.
    with torch.inference_mode(False), torch.no_grad():
        features = torch.zeros(
            player_offset + len(player_ids), NODE_DIM, device=device, dtype=dtype
        )
        links = []

        def link(a, b):
            links.extend(((a, b), (b, a)))

        for tile in board.tiles:
            f = features[tile.id]
            f[0] = 1
            f[5 + RESOURCES.index(tile.resource) if tile.resource in RESOURCES else 10] = 1
            f[11] = tile.number / 12
            f[12] = (6 - abs(7 - tile.number)) / 36 if tile.number else 0
            for vertex in board.tile_vertices[tile.id]:
                link(tile.id, vertex_offset + vertex)
        features[vertex_offset:edge_offset, 1] = 1
        features[edge_offset:player_offset, 2] = 1
        for index, (a, b) in enumerate(board.edges):
            link(edge_offset + index, vertex_offset + a)
            link(edge_offset + index, vertex_offset + b)
        for port in board.ports:
            for vertex in port.vertices:
                features[vertex_offset + vertex, 16] = port.ratio / 4
                if port.resource in RESOURCES:
                    features[vertex_offset + vertex, 5 + RESOURCES.index(port.resource)] = 1
        indices = torch.tensor(links, dtype=torch.long, device=device).reshape(-1, 2).T
    return features, indices, (vertex_offset, edge_offset, player_offset)


def graph_features(view: Observation, device, *, dtype=None):
    """Clone cached static buffers, then add this view's complete dynamic state."""
    timing = encoding_seconds.get()
    started = time.perf_counter() if timing is not None else None
    try:
        with record_function("supervised_fit.encoding"):
            return _graph_features(view, torch.device(device), dtype or torch.get_default_dtype())
    finally:
        if timing is not None:
            timing["encoding"] += time.perf_counter() - started


def _graph_features(view, device, dtype):
    board = view.board
    player_ids = tuple(p.player_id for p in view.players)
    base, static_links, (vertex_offset, edge_offset, player_offset) = _static_graph(
        board, view.schema_version, device, dtype, player_ids
    )
    features = base.clone()
    links = []
    player_nodes = {player: player_offset + i for i, player in enumerate(player_ids)}
    player_seats = {player: i for i, player in enumerate(player_ids)}
    own_seat = player_seats[view.player_id]

    def link(a, b):
        links.extend(((a, b), (b, a)))

    for tile in board.tiles:
        features[tile.id, 13] = float(tile.id == view.robber_tile)
    for building in view.buildings:
        node = vertex_offset + building.vertex
        features[node, 14] = building.level / 2
        features[node, 15] = float(building.owner == view.player_id)
        link(node, player_nodes[building.owner])
    for road in view.roads:
        node = edge_offset + road.edge
        features[node, 14] = 1
        features[node, 15] = float(road.owner == view.player_id)
        link(node, player_nodes[road.owner])
    beliefs = {b.player_id: b for b in view.beliefs}
    for player in view.players:
        f = features[player_nodes[player.player_id]]
        f[3] = 1
        f[14] = player.victory_points / view.rules.victory_points
        f[15] = float(player.player_id == view.player_id)
        f[16:25] = torch.tensor(
            [
                player.resource_count / 20,
                player.development_count / 10,
                player.roads_left / 15,
                player.settlements_left / 5,
                player.cities_left / 4,
                player.longest_road / 15,
                player.played_knights / 3,
                float(player.has_longest_road),
                float(player.has_largest_army),
            ],
            device=device,
            dtype=dtype,
        )
        if player.player_id == view.player_id:
            f[25:30] = torch.tensor(view.hand, device=device, dtype=dtype) / 10
        elif player.player_id in beliefs:
            belief = beliefs[player.player_id]
            f[25:30] = torch.tensor(belief.expected, device=device, dtype=dtype) / 10
            f[30:35] = torch.tensor(belief.variance, device=device, dtype=dtype) / 100
            f[35] = float(not belief.exact)
        else:
            f[35] = 1
        # Relative turn position preserves seat order without a fixed player vocabulary.
        f[36] = ((player_seats[player.player_id] - own_seat) % len(view.players)) / len(
            view.players
        )
        f[37] = float(player.player_id == view.context.turn_owner)
    if links:
        dynamic_links = torch.tensor(links, dtype=torch.long, device=device).T
        indices = torch.cat((static_links, dynamic_links), dim=1)
    else:
        indices = static_links.clone()
    return features, indices, (vertex_offset, edge_offset, player_nodes)
