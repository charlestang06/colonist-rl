"""Local, opt-in Catanatron board topology corrections."""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from hashlib import sha256
from pathlib import Path

from catanatron.models.board import STATIC_GRAPH, Board
from catanatron.models.enums import SETTLEMENT
from catanatron.models.player import Color
from catanatron.state import State

TOPOLOGY_SHIM_ID = "edge-components-v1"


def topology_shim_sha256() -> str:
    """Digest the exact local backend correction source for study manifests."""
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _find_root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


class CorrectedBoard(Board):
    """Pinned backend board with opponent settlements blocking road expansion."""

    @classmethod
    def from_board(cls, source: Board) -> CorrectedBoard:
        corrected = cls(source.map, initialize=False)
        corrected.__dict__.update(Board.copy(source).__dict__)
        corrected._rebuild_topology()
        return corrected

    def copy(self) -> CorrectedBoard:
        return type(self).from_board(self)

    def buildable_edges(self, color):
        if color in self.buildable_edges_cache:
            return self.buildable_edges_cache[color]
        nodes = set().union(*self.connected_components[color])
        nodes = {node for node in nodes if not self.is_enemy_node(node, color)}
        candidates = self.buildable_subgraph.edges(nodes)
        available = {
            tuple(sorted(edge)) for edge in candidates if self.get_edge_color(edge) is None
        }
        self.buildable_edges_cache[color] = list(available)
        return self.buildable_edges_cache[color]

    def _rebuild_components(self) -> None:
        components = defaultdict(list)
        for color in Color:
            edges = sorted(
                edge for edge, owner in self.roads.items() if edge[0] < edge[1] and owner == color
            )
            parent = list(range(len(edges)))

            incident = defaultdict(list)
            for index, (a, b) in enumerate(edges):
                incident[a].append(index)
                incident[b].append(index)
            for node, indices in incident.items():
                if self.is_enemy_node(node, color):
                    continue
                for index in indices[1:]:
                    parent[_find_root(parent, index)] = _find_root(parent, indices[0])
            grouped = defaultdict(set)
            for index, edge in enumerate(edges):
                grouped[_find_root(parent, index)].update(edge)
            components[color].extend(grouped.values())
            for node, (owner, _) in self.buildings.items():
                if owner == color and not any(node in group for group in components[color]):
                    components[color].append({node})
        self.connected_components = components
        self.buildable_edges_cache = {}

    def _longest_road(self, color: Color) -> int:
        edges = sorted(
            edge for edge, owner in self.roads.items() if edge[0] < edge[1] and owner == color
        )
        if not edges:
            return 0
        incident = defaultdict(list)
        for index, (a, b) in enumerate(edges):
            incident[a].append((index, b))
            incident[b].append((index, a))

        @cache
        def walk(node: int, used: int, arrived: bool) -> int:
            if arrived and self.is_enemy_node(node, color):
                return 0
            return max(
                (
                    1 + walk(other, used | (1 << index), True)
                    for index, other in incident[node]
                    if not used & (1 << index)
                ),
                default=0,
            )

        return max(walk(node, 0, False) for node in incident)

    def _rebuild_topology(self) -> None:
        previous_holder = self.road_color
        self._rebuild_components()
        lengths = {color: self._longest_road(color) for color in Color}
        longest = max(lengths.values(), default=0)
        leaders = {color for color, length in lengths.items() if length == longest}
        if longest < 5:
            holder = None
        elif previous_holder in leaders:
            holder = previous_holder
        elif len(leaders) == 1:
            holder = next(iter(leaders))
        else:
            holder = None
        self.road_lengths = defaultdict(int, lengths)
        self.road_color = holder
        self.road_length = longest

    def build_road(self, color, edge):
        if edge not in self.buildable_edges(color) and edge[::-1] not in self.buildable_edges(
            color
        ):
            raise ValueError("Invalid Road Placement")
        previous_holder = self.road_color
        self.roads[edge] = color
        self.roads[edge[::-1]] = color
        self._rebuild_topology()
        return previous_holder, self.road_color, self.road_lengths

    def build_settlement(self, color, node_id, initial_build_phase=False):
        if node_id not in self.buildable_node_ids(color, initial_build_phase=initial_build_phase):
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )
        if node_id in self.buildings:
            raise ValueError("Invalid Settlement Placement: a building exists there")
        previous_holder = self.road_color
        self.buildings[node_id] = (color, SETTLEMENT)
        self.board_buildable_ids.discard(node_id)
        for neighbor in STATIC_GRAPH.neighbors(node_id):
            self.board_buildable_ids.discard(neighbor)
        self.player_port_resources_cache = {}
        self._rebuild_topology()
        return previous_holder, self.road_color, self.road_lengths


def reconcile_road_award(state: State) -> None:
    """Correct backend VP/flags after an opt-in road or settlement mutation."""
    for color in state.colors:
        key = f"P{state.color_to_index[color]}"
        desired = state.board.road_color == color
        recorded = bool(state.player_state[f"{key}_HAS_ROAD"])
        if desired != recorded:
            delta = 2 if desired else -2
            state.player_state[f"{key}_VICTORY_POINTS"] += delta
            state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += delta
            state.player_state[f"{key}_HAS_ROAD"] = desired
        state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = state.board.road_lengths[color]
