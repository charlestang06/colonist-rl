"""Immutable hex topology and versioned rule profiles, independent of an engine.

IDs are dense tuple indices. ``vertices`` use exact integer pointy-hex lattice
coordinates; their scale is arbitrary. Each tile's six vertices and edges are
ordered clockwise, with edge i joining vertex i to vertex (i + 1) % 6.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .contracts import RESOURCES, ResourceVector, resource_vector


@dataclass(frozen=True)
class Tile:
    id: int
    q: int
    r: int
    resource: str
    number: int = 0


@dataclass(frozen=True)
class Port:
    vertices: tuple[int, int]
    resource: str | None = None
    ratio: int = 3

    def __post_init__(self):
        object.__setattr__(self, "vertices", tuple(self.vertices))


@dataclass(frozen=True)
class BoardSpec:
    tiles: tuple[Tile, ...]
    vertices: tuple[tuple[int, int], ...]
    edges: tuple[tuple[int, int], ...]
    tile_vertices: tuple[tuple[int, ...], ...]
    tile_edges: tuple[tuple[int, ...], ...]
    ports: tuple[Port, ...] = ()
    version: str = "hex-v1"

    def __post_init__(self):
        object.__setattr__(self, "tiles", tuple(self.tiles))
        object.__setattr__(self, "ports", tuple(self.ports))
        for field in ("vertices", "edges", "tile_vertices", "tile_edges"):
            object.__setattr__(self, field, tuple(tuple(v) for v in getattr(self, field)))

    @classmethod
    def from_hexes(
        cls,
        coordinates: Iterable[tuple[int, int]],
        resources: Iterable[str] | None = None,
        numbers: Iterable[int] | None = None,
    ) -> BoardSpec:
        """Build any axial-hex layout while preserving the input tile order.

        Unspecified terrain is DESERT and unspecified number tokens are zero.
        Engine adapters can pass their own tile order and tokens without an
        engine-specific mapping leaking into observations.
        """
        coords = tuple(tuple(c) for c in coordinates)
        if any(len(c) != 2 or any(type(v) is not int for v in c) for c in coords):
            raise ValueError("hex coordinates must be integer axial pairs")
        if len(set(coords)) != len(coords):
            raise ValueError("hex coordinates must be unique")
        terrain = tuple(resources) if resources is not None else ("DESERT",) * len(coords)
        tokens = tuple(numbers) if numbers is not None else (0,) * len(coords)
        if len(terrain) != len(coords) or len(tokens) != len(coords):
            raise ValueError("resources and numbers must match the number of hexes")
        if any(r not in (*RESOURCES, "DESERT") for r in terrain):
            raise ValueError("unsupported resource")
        if any(type(n) is not int or n not in (0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12) for n in tokens):
            raise ValueError("number tokens must be zero or 2..12 excluding seven")
        if any(r == "DESERT" and n != 0 for r, n in zip(terrain, tokens)):
            raise ValueError("desert tiles cannot produce resources")
        vertices: list[tuple[int, int]] = []
        edges: list[tuple[int, int]] = []
        vertex_ids: dict[tuple[int, int], int] = {}
        edge_ids: dict[tuple[int, int], int] = {}
        tile_vertices: list[tuple[int, ...]] = []
        tile_edges: list[tuple[int, ...]] = []
        offsets = ((0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1))
        for q, r in coords:
            local_vertices = []
            for dx, dy in offsets:
                point = (2 * q + r + dx, 3 * r + dy)
                if point not in vertex_ids:
                    vertex_ids[point] = len(vertices)
                    vertices.append(point)
                local_vertices.append(vertex_ids[point])
            local_edges = []
            for i in range(6):
                edge = tuple(sorted((local_vertices[i], local_vertices[(i + 1) % 6])))
                if edge not in edge_ids:
                    edge_ids[edge] = len(edges)
                    edges.append(edge)
                local_edges.append(edge_ids[edge])
            tile_vertices.append(tuple(local_vertices))
            tile_edges.append(tuple(local_edges))
        return cls(
            tuple(Tile(i, q, r, terrain[i], tokens[i]) for i, (q, r) in enumerate(coords)),
            tuple(vertices),
            tuple(edges),
            tuple(tile_vertices),
            tuple(tile_edges),
        )

    @classmethod
    def standard(cls, seed: int = 0) -> BoardSpec:
        coords = tuple(
            (q, r) for r in range(-2, 3) for q in range(max(-2, -r - 2), min(2, -r + 2) + 1)
        )
        return cls._profile(
            coords,
            [4, 3, 4, 4, 3],
            1,
            [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
            9,
            seed,
        )

    @classmethod
    def extended(cls, seed: int = 0) -> BoardSpec:
        # Seven connected rows, lengths 3, 4, 5, 6, 5, 4, 3. This explicit
        # 30-hex layout avoids assumptions about a particular engine's ids.
        coords = tuple((q, r) for r in range(-3, 4) for q in range(max(-3, -r - 3), min(3, -r + 3)))
        return cls._profile(
            coords,
            [6, 5, 6, 6, 5],
            2,
            [2, 2, 12, 12] + [n for n in (3, 4, 5, 6, 8, 9, 10, 11) for _ in range(3)],
            11,
            seed,
        )

    @classmethod
    def _profile(cls, coords, counts, deserts, numbers, port_count, seed):
        rng = random.Random(seed)
        resources = [
            resource for resource, count in zip(RESOURCES, counts) for _ in range(count)
        ] + ["DESERT"] * deserts
        rng.shuffle(resources)
        rng.shuffle(numbers)
        productive = [i for i, resource in enumerate(resources) if resource != "DESERT"]
        rng.shuffle(productive)
        red_numbers = [n for n in numbers if n in (6, 8)]

        def place_red(candidates, chosen):
            if len(chosen) == len(red_numbers):
                return chosen
            for index, tile in enumerate(candidates):
                q, r = coords[tile]
                if all(
                    max(
                        abs(q - coords[other][0]),
                        abs(r - coords[other][1]),
                        abs(q + r - sum(coords[other])),
                    )
                    > 1
                    for other in chosen
                ):
                    result = place_red(candidates[index + 1 :], chosen + [tile])
                    if result is not None:
                        return result
            return None

        red_tiles = place_red(productive, [])
        if red_tiles is None:
            raise ValueError("layout cannot separate its six and eight tokens")
        token_map = dict(zip(red_tiles, red_numbers))
        tokens = iter(n for n in numbers if n not in (6, 8))
        board = cls.from_hexes(
            coords,
            resources,
            [
                0 if resource == "DESERT" else token_map[i] if i in token_map else next(tokens)
                for i, resource in enumerate(resources)
            ],
        )
        boundary_counts = Counter(e for es in board.tile_edges for e in es)
        boundary = [board.edges[e] for e, count in boundary_counts.items() if count == 1]
        neighbors: dict[int, list[int]] = defaultdict(list)
        for a, b in boundary:
            neighbors[a].append(b)
            neighbors[b].append(a)
        start = min(neighbors)
        cycle = [start]
        previous, current = None, start
        while True:
            following = next(v for v in sorted(neighbors[current]) if v != previous)
            if following == start:
                break
            cycle.append(following)
            previous, current = current, following
        port_resources = list(RESOURCES) + [None] * (port_count - 5)
        rng.shuffle(port_resources)
        ports = []
        for i, resource in enumerate(port_resources):
            edge_index = i * len(cycle) // port_count
            endpoints = tuple(sorted((cycle[edge_index], cycle[(edge_index + 1) % len(cycle)])))
            ports.append(Port(endpoints, resource, 3 if resource is None else 2))
        return cls(
            board.tiles,
            board.vertices,
            board.edges,
            board.tile_vertices,
            board.tile_edges,
            tuple(ports),
        )


@dataclass(frozen=True)
class RulesConfig:
    """Finite bank/deck and phase support; player count is data, never a constant."""

    player_count: int = 4
    bank_resources: ResourceVector = (19, 19, 19, 19, 19)
    development_cards: tuple[int, ...] = (14, 2, 2, 2, 5)
    special_build: bool = False
    variant: str = "standard"
    version: str = "catan-base-2015-v1"
    supported_phases: tuple[str, ...] = (
        "initial_settlement",
        "initial_road",
        "main",
        "discard",
        "robber",
        "steal",
        "trade_response",
    )
    victory_points: int = 10
    robber_discard_threshold: int = 7

    def __post_init__(self):
        if type(self.player_count) is not int or self.player_count < 1:
            raise ValueError("player_count must be a positive integer")
        object.__setattr__(self, "bank_resources", resource_vector(self.bank_resources))
        cards = tuple(self.development_cards)
        if len(cards) != 5 or any(type(v) is not int or v < 0 for v in cards):
            raise ValueError("development_cards must contain five nonnegative counts")
        object.__setattr__(self, "development_cards", cards)
        object.__setattr__(self, "supported_phases", tuple(self.supported_phases))

    @classmethod
    def standard(cls, player_count: int = 4) -> RulesConfig:
        return cls(player_count=player_count)

    @classmethod
    def extended(cls, player_count: int = 6) -> RulesConfig:
        return cls(
            player_count,
            (24, 24, 24, 24, 24),
            (20, 3, 3, 3, 5),
            True,
            "extended",
            "catan-extension-special-build-2015-v1",
            cls().supported_phases + ("special_build",),
        )
