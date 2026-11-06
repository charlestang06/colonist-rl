"""Conservative early paid-road continuation using public settlement access.

This bounded geometry heuristic values nominal future production. It does not
estimate road awards, denial, or guarantee that a replacement plays better.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .contracts import RESOURCES, Action, Observation, Road
from .settlement_sites import connectable_road_edges, settlement_exclusion


@dataclass(frozen=True)
class RoadAccessScore:
    edge: int
    gain: float


@dataclass(frozen=True)
class RoadContinuationDecision:
    action: Action
    changed: bool
    reason: str
    scores: tuple[RoadAccessScore, ...]


def _ineligibility(view: Observation) -> str | None:
    context = view.context
    if (
        context.kind not in ("TURN", "PLAY_TURN")
        or context.actor != view.player_id
        or context.turn_owner != view.player_id
        or not context.has_rolled
        or context.free_roads != 0
        or context.offer is not None
        or context.discard_remaining > 0
    ):
        return "outside_normal_paid_road_turn"
    if sum(road.owner == view.player_id for road in view.roads) > 3:
        return "larger_road_network"
    own = next((player for player in view.players if player.player_id == view.player_id), None)
    if own is None:
        raise ValueError("observation has no public player record for its owner")
    if own.roads_left <= 0:
        return "no_road_stock"
    if own.settlements_left <= 0:
        return "no_settlement_stock"
    if view.own_victory_points >= view.rules.victory_points - 2:
        return "own_near_victory"
    if any(
        player.player_id != view.player_id
        and player.victory_points >= view.rules.victory_points - 2
        for player in view.players
    ):
        return "opponent_near_victory"
    return None


def _distances(view: Observation, roads: tuple[Road, ...], limit: int) -> dict[int, int]:
    """Minimum empty-edge distances; enemy buildings terminate traversal."""
    board = view.board
    own_edges = {road.edge for road in roads if road.owner == view.player_id}
    enemy_edges = {road.edge for road in roads if road.owner != view.player_id}
    enemy_buildings = {building.vertex for building in view.buildings if building.owner != view.player_id}
    sources = {building.vertex for building in view.buildings if building.owner == view.player_id}
    sources.update(vertex for edge in own_edges for vertex in board.edges[edge])
    adjacency: list[list[tuple[int, int]]] = [[] for _ in board.vertices]
    for edge, (a, b) in enumerate(board.edges):
        if edge in enemy_edges:
            continue
        cost = 0 if edge in own_edges else 1
        adjacency[a].append((b, cost))
        adjacency[b].append((a, cost))
    distances = {vertex: 0 for vertex in sources}
    pending = deque(sources)
    while pending:
        vertex = pending.popleft()
        if vertex in enemy_buildings:
            continue
        distance = distances[vertex]
        for neighbor, cost in adjacency[vertex]:
            candidate = distance + cost
            if candidate <= limit and candidate < distances.get(neighbor, limit + 1):
                distances[neighbor] = candidate
                if cost:
                    pending.append(neighbor)
                else:
                    pending.appendleft(neighbor)
    return distances


def _qualities(view: Observation) -> tuple[float, ...]:
    weights = dict(zip(RESOURCES, (1.0, 1.05, 0.85, 1.4, 1.25)))
    qualities = [0.0] * len(view.board.vertices)
    resources: list[set[str]] = [set() for _ in qualities]
    for index, tile in enumerate(view.board.tiles):
        if tile.resource not in weights or tile.number == 0:
            continue
        # Robber location is intentionally ignored for long-term nominal value.
        production = (6 - abs(7 - tile.number)) * weights[tile.resource]
        for vertex in view.board.tile_vertices[index]:
            qualities[vertex] += production
            resources[vertex].add(tile.resource)
    return tuple(quality + 1.4 * len(types) for quality, types in zip(qualities, resources))


def road_access_scores(view: Observation) -> tuple[RoadAccessScore, ...]:
    """Score original connectable legal roads within a three-road horizon."""
    if _ineligibility(view) is not None:
        return ()
    board = view.board
    for road in view.roads:
        if type(road.edge) is not int or not 0 <= road.edge < len(board.edges):
            raise ValueError("road references an invalid board edge")
    own = next(player for player in view.players if player.player_id == view.player_id)
    connectable = set(connectable_road_edges(board, view.buildings, view.roads, view.player_id))
    excluded = settlement_exclusion(board, view.buildings)
    qualities = _qualities(view)
    before = _distances(view, view.roads, min(3, own.roads_left))
    scores = []
    for action in view.legal_actions:
        edge = action.target
        if action.kind != "BUILD_ROAD" or type(edge) is not int or edge not in connectable:
            continue
        after = _distances(
            view, view.roads + (Road(edge, view.player_id),), min(2, own.roads_left - 1)
        )
        gains = []
        for vertex, quality in enumerate(qualities):
            if vertex in excluded or before.get(vertex) == 0:
                continue
            before_term = 2.0 ** -before[vertex] if vertex in before else 0.0
            after_term = 2.0 ** -after[vertex] if vertex in after else 0.0
            gains.append(max(0.0, quality * (after_term - before_term)))
        scores.append(RoadAccessScore(edge, sum(sorted(gains, reverse=True)[:3])))
    return tuple(scores)


def choose_road_continuation(view: Observation, selected: Action) -> RoadContinuationDecision:
    """Replace a zero-access early road only with a positive original menu road."""
    if not any(action is selected for action in view.legal_actions):
        raise ValueError("selected action must be an original legal action object")
    if selected.kind != "BUILD_ROAD":
        return RoadContinuationDecision(selected, False, "selected_action_is_not_a_road", ())
    reason = _ineligibility(view)
    if reason is not None:
        return RoadContinuationDecision(selected, False, reason, ())
    scores = road_access_scores(view)
    gains = {score.edge: score.gain for score in scores}
    if type(selected.target) is not int or selected.target not in gains:
        return RoadContinuationDecision(selected, False, "selected_road_is_not_connectable", scores)
    if gains[selected.target] != 0.0:
        return RoadContinuationDecision(selected, False, "selected_road_has_positive_access", scores)
    alternatives = [
        (action, index)
        for index, action in enumerate(view.legal_actions)
        if action.kind == "BUILD_ROAD"
        and type(action.target) is int
        and gains.get(action.target, 0.0) > 0.0
    ]
    if not alternatives:
        return RoadContinuationDecision(selected, False, "no_positive_access_alternative", scores)
    alternative, _ = min(
        alternatives,
        key=lambda entry: (-gains[entry[0].target], entry[0].target, entry[1]),
    )
    return RoadContinuationDecision(alternative, True, "improved_settlement_access", scores)
