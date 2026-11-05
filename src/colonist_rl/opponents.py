"""Baselines whose complete input is the same immutable player view as the learner."""

from __future__ import annotations

import random

from .contracts import RESOURCES, Action, Observation, TradeOffer

_RECIPES = ((1, 1, 1, 1, 0), (0, 0, 0, 2, 3), (0, 0, 1, 1, 1), (1, 1, 0, 0, 0))


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, view: Observation) -> Action:
        if not view.legal_actions:
            raise ValueError("view contains no legal action")
        return self.rng.choice(view.legal_actions)

    def propose(self, view: Observation) -> TradeOffer | None:
        return None


class HeuristicPolicy(RandomPolicy):
    """Production-aware placement, building progress, and public-information trades."""

    def _production(self, view, vertex):
        value = 0.0
        coverage = set()
        for tile, vertices in zip(view.board.tiles, view.board.tile_vertices):
            if vertex in vertices and tile.number:
                pip = 6 - abs(7 - tile.number)
                weight = {"WOOD": 1.0, "BRICK": 1.05, "SHEEP": 0.85, "WHEAT": 1.4, "ORE": 1.25}[
                    tile.resource
                ]
                value += pip * weight
                coverage.add(tile.resource)
        return value + 1.4 * len(coverage)

    def _goals(self, view):
        players = {player.player_id: player for player in view.players}
        me = players[view.player_id]
        goals = []
        if me.settlements_left and me.roads_left:
            goals.append((_RECIPES[0], 1.2))
        if any(b.owner == view.player_id and b.level == 1 for b in view.buildings):
            goals.append((_RECIPES[1], 1.3))
        if me.development_count < 8:
            goals.append((_RECIPES[2], 0.8))
        if me.roads_left:
            goals.append((_RECIPES[3], 0.5))
        return goals or [(_RECIPES[2], 1)]

    def _utility(self, view, hand):
        scores = [
            weight * (sum(recipe) - sum(max(0, c - h) for c, h in zip(recipe, hand))) / sum(recipe)
            for recipe, weight in self._goals(view)
        ]
        return max(scores) + sum(min(n, 4) for n in hand) * 0.025

    def _road_score(self, view, edge_id):
        # Prefer reaching productive sites that are still outside building distance.
        adjacency = {i: set() for i in range(len(view.board.vertices))}
        for a, b in view.board.edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        blocked = {b.vertex for b in view.buildings}
        for b in view.buildings:
            blocked.update(adjacency[b.vertex])
        a, b = view.board.edges[edge_id]
        frontier, visited = {a, b}, {a, b}
        best = 0.0
        for distance in range(4):
            for vertex in frontier:
                if vertex not in blocked:
                    best = max(best, self._production(view, vertex) / (1 + distance))
            frontier = (
                set().union(*(adjacency[v] for v in frontier)) - visited if frontier else set()
            )
            visited.update(frontier)
        return best

    def act(self, view: Observation) -> Action:
        if not view.legal_actions:
            raise ValueError("view contains no legal action")
        players = {player.player_id: player for player in view.players}

        def score(action):
            kind = action.kind
            jitter = self.rng.random() * 0.02
            if kind == "BUILD_SETTLEMENT":
                return 90 + self._production(view, action.target) + jitter
            if kind == "BUILD_CITY":
                return 95 + self._production(view, action.target) + jitter
            if kind == "ROLL":
                return 80 + jitter
            if kind == "MOVE_ROBBER":
                own = sum(
                    b.level
                    for b in view.buildings
                    if b.owner == view.player_id
                    and b.vertex in view.board.tile_vertices[action.target]
                )
                enemies = sum(
                    b.level * (1 + players[b.owner].victory_points / 5)
                    for b in view.buildings
                    if b.owner != view.player_id
                    and b.vertex in view.board.tile_vertices[action.target]
                )
                tile = view.board.tiles[action.target]
                return (
                    enemies * (6 - abs(7 - tile.number))
                    - own * 12
                    + (3 if action.partner is not None else 0)
                    + jitter
                )
            if kind == "DISCARD_RESOURCE":
                i = RESOURCES.index(action.resource)
                after = tuple(n - int(j == i) for j, n in enumerate(view.hand))
                return self._utility(view, after) + view.hand[i] * 0.01 + jitter
            if kind == "BUILD_ROAD":
                return 30 + self._road_score(view, action.target) + jitter
            if kind == "BUY_DEVELOPMENT_CARD":
                return 45 + jitter
            if kind.startswith("PLAY_"):
                bonus = 0.0
                if kind == "PLAY_MONOPOLY":
                    i = RESOURCES.index(action.resource)
                    bonus = sum(
                        b.expected[i] for b in view.beliefs if b.player_id != view.player_id
                    )
                elif kind == "PLAY_YEAR_OF_PLENTY":
                    extra = [0] * 5
                    for resource in (action.resource, action.second_resource):
                        if resource:
                            extra[RESOURCES.index(resource)] += 1
                    bonus = self._utility(view, tuple(h + e for h, e in zip(view.hand, extra))) * 10
                return 55 + bonus + jitter
            if kind == "MARITIME_TRADE":
                after = tuple(h - g + r for h, g, r in zip(view.hand, action.give, action.receive))
                improvement = self._utility(view, after) - self._utility(view, view.hand)
                return 20 + improvement * 20 + jitter if improvement > 0.005 else -5 + jitter
            if kind == "ACCEPT_TRADE":
                offer = view.context.offer
                give, receive = offer.for_player(view.player_id)
                after = tuple(h - g + r for h, g, r in zip(view.hand, give, receive))
                return 10 if self._utility(view, after) >= self._utility(view, view.hand) else -10
            if kind == "REJECT_TRADE":
                return 0
            return 0 + jitter

        return max(view.legal_actions, key=score)

    def propose(self, view: Observation) -> TradeOffer | None:
        if (
            view.context.kind != "PLAY_TURN"
            or not view.context.has_rolled
            or view.context.actor != view.player_id
            or self.rng.random() > 0.2
        ):
            return None
        surplus = [i for i, h in enumerate(view.hand) if h >= 2]
        missing = [i for i, h in enumerate(view.hand) if h == 0]
        if not surplus or not missing:
            return None
        give_i = max(surplus, key=lambda i: view.hand[i])
        receive_i = self.rng.choice(missing)
        partners = [
            b.player_id
            for b in view.beliefs
            if b.player_id != view.player_id and b.expected[receive_i] >= 0.5
        ]
        if not partners:
            return None
        give = tuple(int(i == give_i) for i in range(5))
        receive = tuple(int(i == receive_i) for i in range(5))
        return TradeOffer(
            f"baseline-{view.player_id}-{view.context.version}",
            view.player_id,
            self.rng.choice(partners),
            give,
            receive,
        )
