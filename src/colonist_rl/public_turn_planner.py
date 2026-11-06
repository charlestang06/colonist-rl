"""Bounded deterministic conservative planning on public current-turn projections.

Children are ephemeral intentions, never Observations or certified future menus.
The policy executes only an original root-menu object and replans on every act.
Scores, budgets and H_view tie seeding are hypotheses, not evidence of strength.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from time import perf_counter_ns

from .board import RulesConfig
from .consequences import PUBLIC_BUILD_COSTS
from .contracts import (
    RESOURCES,
    ZERO,
    Action,
    Building,
    Observation,
    ResourceVector,
    Road,
    TradeOffer,
)
from .opponents import HeuristicPolicy
from .settlement_sites import (
    connectable_road_edges,
    connected_settlement_sites,
    settlement_exclusion,
)

_SUPPORTED = frozenset(
    ("BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "MARITIME_TRADE", "END_TURN")
)
_ROOT_ORDER = {
    kind: index
    for index, kind in enumerate(
        ("BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "MARITIME_TRADE", "END_TURN")
    )
}
_WEIGHTS = (1.0, 1.05, 0.85, 1.4, 1.25)
_GAIN_MICRO = 250_000


@dataclass(frozen=True)
class TurnPlannerConfig:
    max_depth: int = 6
    max_nodes: int = 2048
    beam_width: int = 24
    max_geometry_visits: int = 100_000
    seed: int = 11

    def __post_init__(self):
        for name in ("max_depth", "max_nodes", "max_geometry_visits"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.beam_width) is not int or self.beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


_DEFAULT_CONFIG = TurnPlannerConfig()


@dataclass(frozen=True)
class ProjectedTurn:
    hand: ResourceVector
    buildings: tuple[Building, ...]
    roads: tuple[Road, ...]
    roads_left: int
    settlements_left: int
    cities_left: int
    building_point_gain: int
    bank_lower: ResourceVector
    bank_upper: ResourceVector


PositionEvaluator = Callable[[Observation, ProjectedTurn, int], int]


def _validate_gain_margin(margin: int):
    if type(margin) is not int or margin < 0:
        raise ValueError("gain margin must be a nonnegative integer in evaluator score units")


@dataclass(frozen=True)
class PlannedStep:
    kind: str
    target: int | None = None
    give: ResourceVector = ZERO
    receive: ResourceVector = ZERO


@dataclass(frozen=True)
class TurnPlan:
    first_action: Action
    root_index: int
    steps: tuple[PlannedStep, ...]
    score_before_micro: int
    score_after_micro: int
    nodes: int
    geometry_visits: int
    reason: str
    search_stop: str
    evaluator_calls: int = 0
    evaluator_time_ns: int = 0


def _seed(view: Observation, seed: int, domain: str) -> int:
    canonical = json.dumps(
        [seed, domain, asdict(view)], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return int.from_bytes(sha256(canonical.encode("utf-8")).digest(), "big")


class HeuristicViewPolicy:
    """H_view: production H freshly seeded from this public observation per call."""

    def __init__(self, seed: int = 11):
        self.seed = seed

    def act(self, view: Observation) -> Action:
        return HeuristicPolicy(_seed(view, self.seed, "act")).act(view)

    def propose(self, view: Observation) -> TradeOffer | None:
        return HeuristicPolicy(_seed(view, self.seed, "propose")).propose(view)


class PublicTurnPlannerPolicy:
    def __init__(
        self,
        config: TurnPlannerConfig = _DEFAULT_CONFIG,
        *,
        evaluator: PositionEvaluator | None = None,
        gain_margin_micro: int = _GAIN_MICRO,
    ):
        _validate_gain_margin(gain_margin_micro)
        self.config = config
        self.evaluator = evaluator
        self.gain_margin_micro = gain_margin_micro

    def plan(self, view: Observation) -> TurnPlan:
        return plan_turn(
            view, self.config, evaluator=self.evaluator, gain_margin_micro=self.gain_margin_micro
        )

    def act(self, view: Observation) -> Action:
        return self.plan(view).first_action

    def propose(self, view: Observation) -> TradeOffer | None:
        return HeuristicViewPolicy(self.config.seed).propose(view)


class _GeometryCap(Exception):
    pass


class _PublicContract(Exception):
    pass


@dataclass
class _Work:
    config: TurnPlannerConfig
    nodes: int = 0
    geometry_visits: int = 0

    def charge(self, visits: int):
        # Charge logical edge/vertex work, including cached descriptions.
        available = self.config.max_geometry_visits - self.geometry_visits
        self.geometry_visits += min(visits, available)
        if visits > available:
            raise _GeometryCap


@dataclass(frozen=True)
class _Shape:
    sites: tuple[int, ...]
    edges: tuple[int, ...]
    cities: tuple[int, ...]
    production: float
    access: float


class _Geometry:
    def __init__(self, view: Observation, work: _Work):
        self.view, self.work = view, work
        self.adjacency = [[] for _ in view.board.vertices]
        for edge, (a, b) in enumerate(view.board.edges):
            work.charge(1)
            self.adjacency[a].append((b, edge))
            self.adjacency[b].append((a, edge))
        self.yields = [0.0 for _ in view.board.vertices]
        for index, (tile, vertices) in enumerate(zip(view.board.tiles, view.board.tile_vertices)):
            for vertex in vertices:
                work.charge(1)
                if index != view.robber_tile and tile.number and tile.resource in RESOURCES:
                    self.yields[vertex] += (
                        (6 - abs(7 - tile.number)) / 36 * _WEIGHTS[RESOURCES.index(tile.resource)]
                    )
        self.cache = {}

    def describe(self, state: ProjectedTurn) -> _Shape:
        key = (
            state.buildings,
            state.roads,
            state.roads_left,
            state.settlements_left,
            state.cities_left,
        )
        if key in self.cache:
            shape, visits = self.cache[key]
            self.work.charge(visits)
            return shape
        start = self.work.geometry_visits
        board, player = self.view.board, self.view.player_id
        # Logical work for shared exclusion/connectivity scans, independent of cache state.
        self.work.charge(3 * len(board.edges) + 4 * len(state.roads) + 4 * len(state.buildings))
        sites = connected_settlement_sites(board, state.buildings, state.roads, player)
        edges = connectable_road_edges(board, state.buildings, state.roads, player)
        cities = tuple(
            sorted(b.vertex for b in state.buildings if b.owner == player and b.level == 1)
        )
        production = sum(
            self.yields[b.vertex] * b.level for b in state.buildings if b.owner == player
        )
        access = 0.0
        if state.settlements_left > 0:
            blocked = settlement_exclusion(board, state.buildings)
            enemies = {b.vertex for b in state.buildings if b.owner != player}
            occupied = {r.edge: r.owner for r in state.roads}
            sources = {b.vertex for b in state.buildings if b.owner == player} | {
                v for r in state.roads if r.owner == player for v in board.edges[r.edge]
            }
            distances = {v: 0 for v in sources}
            queue = deque(sorted(sources))
            limit = min(2, state.roads_left)
            while queue:
                vertex = queue.popleft()
                self.work.charge(1)
                if vertex in enemies:
                    continue
                for other, edge in self.adjacency[vertex]:
                    self.work.charge(1)
                    owner = occupied.get(edge)
                    if owner is not None and owner != player:
                        continue
                    cost = int(owner is None)
                    distance = distances[vertex] + cost
                    if distance <= limit and distance < distances.get(other, limit + 1):
                        distances[other] = distance
                        if cost:
                            queue.append(other)
                        else:
                            queue.appendleft(other)
            values = []
            for vertex, distance in distances.items():
                self.work.charge(1)
                if vertex not in blocked:
                    values.append(self.yields[vertex] / 2**distance)
            access = sum(sorted(values, reverse=True)[:3])
        shape = _Shape(sites, edges, cities, production, access)
        self.cache[key] = (shape, self.work.geometry_visits - start)
        return shape


def _initial_projection(view: Observation) -> ProjectedTurn:
    own = next(p for p in view.players if p.player_id == view.player_id)
    supply, hand = view.rules.bank_resources, view.hand
    count = sum(p.resource_count for p in view.players if p.player_id != view.player_id)
    total = sum(supply) - sum(hand) - count
    if own.resource_count != sum(hand) or count < 0 or total < 0:
        raise _PublicContract("resource counts")
    upper = tuple(s - h for s, h in zip(supply, hand))
    lower = tuple(max(0, s - h - count) for s, h in zip(supply, hand))
    if view.bank is not None:
        lower = upper = view.bank
    else:
        # Maximum evidence from this menu, never sum and never re-inject deeper.
        lower = tuple(
            max(l, *(a.receive[i] for a in view.legal_actions if a.kind == "MARITIME_TRADE"))
            if any(a.kind == "MARITIME_TRADE" for a in view.legal_actions)
            else l
            for i, l in enumerate(lower)
        )
    if any(not 0 <= l <= u <= s - h for l, u, s, h in zip(lower, upper, supply, hand)) or not sum(
        lower
    ) <= total <= sum(upper):
        raise _PublicContract("bank evidence")
    return ProjectedTurn(
        hand,
        tuple(sorted(view.buildings, key=lambda b: b.vertex)),
        tuple(sorted(view.roads, key=lambda r: r.edge)),
        own.roads_left,
        own.settlements_left,
        own.cities_left,
        0,
        lower,
        upper,
    )


def _ratio(view: Observation, state: ProjectedTurn, resource: int) -> int:
    owned = {b.vertex for b in state.buildings if b.owner == view.player_id}
    return min(
        [4]
        + [
            port.ratio
            for port in view.board.ports
            if owned.intersection(port.vertices) and port.resource in (None, RESOURCES[resource])
        ]
    )


def _transition(
    view: Observation, state: ProjectedTurn, step: PlannedStep, shape: _Shape
) -> ProjectedTurn | None:
    kind = step.kind
    if kind == "MARITIME_TRADE":
        outgoing = [i for i, n in enumerate(step.give) if n]
        incoming = [i for i, n in enumerate(step.receive) if n]
        if len(outgoing) != 1 or len(incoming) != 1 or outgoing == incoming:
            return None
        i, j = outgoing[0], incoming[0]
        if (
            step.target is not None
            or step.give[i] != _ratio(view, state, i)
            or step.receive[j] != 1
        ):
            return None
        give, receive = step.give, step.receive
    elif kind in ("BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD"):
        if step.give != ZERO or step.receive != ZERO:
            return None
        if kind == "BUILD_CITY" and (state.cities_left <= 0 or step.target not in shape.cities):
            return None
        if kind == "BUILD_SETTLEMENT" and (
            state.settlements_left <= 0 or step.target not in shape.sites
        ):
            return None
        if kind == "BUILD_ROAD" and (state.roads_left <= 0 or step.target not in shape.edges):
            return None
        give, receive = PUBLIC_BUILD_COSTS[kind], ZERO
    else:
        return None
    if any(h < g or l < r for h, g, l, r in zip(state.hand, give, state.bank_lower, receive)):
        return None
    hand = tuple(h - g + r for h, g, r in zip(state.hand, give, receive))
    lower = tuple(l + g - r for l, g, r in zip(state.bank_lower, give, receive))
    upper = tuple(u + g - r for u, g, r in zip(state.bank_upper, give, receive))
    if any(not 0 <= l <= u <= s for l, u, s in zip(lower, upper, view.rules.bank_resources)):
        return None
    updated = replace(state, hand=hand, bank_lower=lower, bank_upper=upper)
    if kind == "BUILD_CITY":
        updated = replace(
            updated,
            buildings=tuple(
                replace(b, level=2) if b.vertex == step.target else b for b in state.buildings
            ),
            cities_left=state.cities_left - 1,
            settlements_left=state.settlements_left + 1,
            building_point_gain=state.building_point_gain + 1,
        )
    elif kind == "BUILD_SETTLEMENT":
        updated = replace(
            updated,
            buildings=tuple(
                sorted(
                    (*state.buildings, Building(step.target, view.player_id)),
                    key=lambda b: b.vertex,
                )
            ),
            settlements_left=state.settlements_left - 1,
            building_point_gain=state.building_point_gain + 1,
        )
    elif kind == "BUILD_ROAD":
        updated = replace(
            updated,
            roads=tuple(
                sorted((*state.roads, Road(step.target, view.player_id)), key=lambda r: r.edge)
            ),
            roads_left=state.roads_left - 1,
        )
    return updated


def _root_transition(
    view: Observation, state: ProjectedTurn, action: Action, shape: _Shape
) -> ProjectedTurn | None:
    """Project an original actual-menu action, only from the exact initial state.

    An engine-certified root road may disagree with conservative public connectivity.
    Its menu membership certifies connectivity alone; future intentions continue to
    use the unchanged strict geometry and transition checks.
    """
    if (
        not any(action is original for original in view.legal_actions)
        or action.kind not in _SUPPORTED
        or not _context_supported(view)
        or view.rules != RulesConfig.standard()
        or any(
            port.ratio not in (2, 3, 4) or port.resource not in (None, *RESOURCES)
            for port in view.board.ports
        )
        or any(
            value is not None
            for value in (action.resource, action.second_resource, action.partner, action.offer_id)
        )
    ):
        return None
    try:
        if state != _initial_projection(view):
            return None
    except (_PublicContract, StopIteration):
        return None
    if any(
        len(vector) != 5 or any(type(n) is not int or n < 0 for n in vector)
        for vector in (state.hand, state.bank_lower, state.bank_upper)
    ) or any(
        type(stock) is not int or stock < 0
        for stock in (state.roads_left, state.settlements_left, state.cities_left)
    ):
        return None
    if action.kind == "END_TURN":
        return state if action.target is None and action.give == action.receive == ZERO else None
    if action.kind.startswith("BUILD_"):
        size = len(view.board.edges if action.kind == "BUILD_ROAD" else view.board.vertices)
        if type(action.target) is not int or not 0 <= action.target < size:
            return None
    if action.kind == "BUILD_ROAD":
        if any(road.edge == action.target for road in state.roads):
            return None
        if action.target not in shape.edges:
            shape = replace(shape, edges=(*shape.edges, action.target))
    return _transition(
        view, state, PlannedStep(action.kind, action.target, action.give, action.receive), shape
    )


def _score(view: Observation, state: ProjectedTurn, shape: _Shape) -> int:
    goals = []
    if state.roads_left and shape.edges:
        goals.append(("BUILD_ROAD", 0.5))
    if state.settlements_left and shape.sites:
        goals.append(("BUILD_SETTLEMENT", 1.2))
    if state.cities_left and shape.cities:
        goals.append(("BUILD_CITY", 1.3))
    balance = max(
        (
            weight
            * (
                1
                - sum(max(0, c - h) for c, h in zip(PUBLIC_BUILD_COSTS[kind], state.hand))
                / sum(PUBLIC_BUILD_COSTS[kind])
            )
            for kind, weight in goals
        ),
        default=0.0,
    )
    balance += 0.025 * sum(min(h, 4) for h in state.hand)
    balance -= 0.03 * max(0, sum(state.hand) - view.rules.robber_discard_threshold)
    public = next(p.victory_points for p in view.players if p.player_id == view.player_id)
    points = public + state.building_point_gain
    opponents = [p.victory_points for p in view.players if p.player_id != view.player_id]
    rank = sum((points > p) - (points < p) for p in opponents) / len(opponents) if opponents else 0
    return round(
        (100 * points + 20 * shape.production + 6 * shape.access + 4 * balance + 12 * rank)
        * 1_000_000
    )


def _step_key(step: PlannedStep):
    return (step.kind, -1 if step.target is None else step.target, step.give, step.receive)


@dataclass(frozen=True)
class _Node:
    state: ProjectedTurn
    shape: _Shape
    root: int
    steps: tuple[PlannedStep, ...]
    score: int
    winning: bool

    def order(self):
        return (
            -self.winning,
            -self.score,
            sum(s.kind == "MARITIME_TRADE" for s in self.steps),
            len(self.steps),
            tuple(_step_key(s) for s in self.steps),
            self.root,
        )


def _candidates(view: Observation, state: ProjectedTurn, shape: _Shape):
    for kind, targets, stock in (
        ("BUILD_CITY", shape.cities, state.cities_left),
        ("BUILD_SETTLEMENT", shape.sites, state.settlements_left),
        ("BUILD_ROAD", shape.edges, state.roads_left),
    ):
        if stock:
            for target in targets:
                yield PlannedStep(kind, target)
    maritime = [
        PlannedStep(
            "MARITIME_TRADE",
            give=tuple(_ratio(view, state, i) if k == i else 0 for k in range(5)),
            receive=tuple(int(k == j) for k in range(5)),
        )
        for i in range(5)
        for j in range(5)
        if i != j
    ]
    yield from sorted(maritime, key=_step_key)


def _context_supported(view: Observation) -> bool:
    c = view.context
    return (
        view.schema_version == 1
        and view.board.version == "hex-v1"
        and c.kind in ("TURN", "PLAY_TURN")
        and c.has_rolled
        and c.actor == c.turn_owner == view.player_id
        and c.free_roads == 0
        and c.offer is None
        and c.discard_remaining == 0
    )


def plan_turn(
    view: Observation,
    config: TurnPlannerConfig = _DEFAULT_CONFIG,
    *,
    evaluator: PositionEvaluator | None = None,
    gain_margin_micro: int = _GAIN_MICRO,
) -> TurnPlan:
    """Pure bounded search; actual execution must reobserve after the first action.

    An optional pure evaluator receives this actual root, a projection and the existing
    handcraft integer score. Its integer score and gain margin must use the same units.
    No projected state is promoted to an Observation or a certified future menu.
    """
    _validate_gain_margin(gain_margin_micro)
    fallback = HeuristicViewPolicy(config.seed).act(view)
    fallback_index = next(i for i, action in enumerate(view.legal_actions) if action is fallback)
    work = _Work(config)
    before = 0
    best = None
    roots_validated = False
    evaluator_calls = evaluator_time_ns = 0

    def evaluate(state, shape):
        nonlocal evaluator_calls, evaluator_time_ns
        handcraft = _score(view, state, shape)
        if evaluator is None:
            return handcraft
        evaluator_calls += 1
        start = perf_counter_ns()
        try:
            value = evaluator(view, state, handcraft)
        finally:
            evaluator_time_ns += perf_counter_ns() - start
        if type(value) is not int:
            raise ValueError("position evaluator must return a finite integer score")
        return value

    def result(reason, stop="not_searched", node=None):
        selected = node.root if node is not None else fallback_index
        return TurnPlan(
            view.legal_actions[selected],
            selected,
            node.steps if node is not None else (),
            before,
            node.score if node is not None else before,
            work.nodes,
            work.geometry_visits,
            reason,
            stop,
            evaluator_calls,
            evaluator_time_ns,
        )

    if not _context_supported(view):
        return result("unsupported_context")
    if view.rules != RulesConfig.standard():
        return result("unsupported_rules")
    if fallback.kind not in _SUPPORTED:
        return result("unsupported_heuristic_action")
    if any(
        port.ratio not in (2, 3, 4) or port.resource not in (None, *RESOURCES)
        for port in view.board.ports
    ):
        return result("public_contract_mismatch")
    try:
        state = _initial_projection(view)
        geometry = _Geometry(view, work)
        shape = geometry.describe(state)
        before = evaluate(state, shape)
        baseline = before
        beam = []
        stopped = "depth_limit"

        def child(parent_state, parent_shape, step, root, steps, action=None):
            projected = (
                _root_transition(view, parent_state, action, parent_shape)
                if action is not None
                else _transition(view, parent_state, step, parent_shape)
            )
            if projected is None:
                return None
            projected_shape = geometry.describe(projected)
            return _Node(
                projected,
                projected_shape,
                root,
                (*steps, step),
                evaluate(projected, projected_shape),
                view.own_victory_points + projected.building_point_gain
                >= view.rules.victory_points,
            )

        # Root menu is authoritative; check supported standard shapes before search.
        roots = sorted(
            enumerate(view.legal_actions),
            key=lambda pair: (
                _ROOT_ORDER.get(pair[1].kind, len(_ROOT_ORDER)),
                *_step_key(
                    PlannedStep(pair[1].kind, pair[1].target, pair[1].give, pair[1].receive)
                ),
                pair[0],
            ),
        )
        for index, action in roots:
            if action.kind not in _SUPPORTED:
                continue
            if work.nodes >= config.max_nodes:
                return result("root_validation_limit", "node_limit")
            # Count the attempt even when its shape is rejected.
            work.nodes += 1
            if action.kind == "END_TURN":
                if _root_transition(view, state, action, shape) is None:
                    return result("public_contract_mismatch")
                continue
            step = PlannedStep(action.kind, action.target, action.give, action.receive)
            node = child(state, shape, step, index, (), action)
            if node is None:
                return result("public_contract_mismatch")
            if index == fallback_index:
                baseline = node.score
            if config.max_depth:
                beam.append(node)
                if best is None or node.order() < best.order():
                    best = node
        # Missing H baseline under a tiny node budget cannot justify overriding H.
        if fallback.kind != "END_TURN" and not any(n.root == fallback_index for n in beam):
            return result("insufficient_gain", stopped)
        roots_validated = True
        beam = sorted(beam, key=lambda n: n.order())[: config.beam_width]
        for _depth in range(1, config.max_depth):
            children = {}
            exhausted = False
            for parent in beam:
                for step in _candidates(view, parent.state, parent.shape):
                    if work.nodes >= config.max_nodes:
                        stopped, exhausted = "node_limit", True
                        break
                    # Invalid and duplicate future intentions also consume a node.
                    work.nodes += 1
                    node = child(parent.state, parent.shape, step, parent.root, parent.steps)
                    if node is None:
                        continue
                    if best is None or node.order() < best.order():
                        best = node
                    key = (node.state, node.root)
                    if key not in children or node.order() < children[key].order():
                        children[key] = node
                if exhausted:
                    break
            if exhausted:
                break
            beam = sorted(children.values(), key=lambda n: n.order())[: config.beam_width]
            if not beam:
                stopped = "empty_beam"
                break
        if (
            best is not None
            and any(s.kind.startswith("BUILD_") for s in best.steps)
            and (best.winning or best.score > max(before, baseline) + gain_margin_micro)
        ):
            return result("planned", stopped, best)
        return result("insufficient_gain", stopped)
    except _GeometryCap:
        # A later cap cannot invalidate a fully constructed known winning prefix.
        # Ordinary prefixes and partially validated root menus still fall back.
        if (
            roots_validated
            and best is not None
            and best.winning
            and any(s.kind.startswith("BUILD_") for s in best.steps)
        ):
            return result("planned", "geometry_limit", best)
        return result("geometry_limit", "geometry_limit")
    except _PublicContract:
        return result("public_contract_mismatch")
