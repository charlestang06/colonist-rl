"""Immutable player views over pinned Catanatron, with atomic domestic trades."""

from __future__ import annotations

import importlib.metadata
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace

from catanatron.game import Game
from catanatron.models.actions import (
    generate_playable_actions,
    monopoly_possibilities,
    year_of_plenty_possibilities,
)
from catanatron.models.enums import Action as EngineAction
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, NodeRef
from catanatron.models.player import Color, Player
from catanatron.state_functions import player_can_play_dev

from .board import BoardSpec, Port, RulesConfig
from .contracts import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    ZERO,
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    Road,
    TradeOffer,
    VisibleEvent,
)

BACKEND_VERSION = "3.2.1"
_RNG_LOCK = threading.RLock()


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    winner: int | None
    info: dict


class FourPlayerGame:
    """One decision per step, including forced phases and trade responses.

    Player ids refer to seating order. ``game`` is diagnostic, never a policy input.
    """

    def __init__(self, rules: RulesConfig | None = None, *, max_steps: int = 20000):
        self.rules = rules or RulesConfig.standard()
        if self.rules.player_count != 4 or self.rules.special_build:
            raise NotImplementedError(
                "Catanatron 3.2.1 supports four players; six-player simulation is unavailable"
            )
        if importlib.metadata.version("catanatron") != BACKEND_VERSION:
            raise RuntimeError(f"catanatron=={BACKEND_VERSION} is required")
        self.max_steps = max_steps
        self.version = 0
        self._rng_state = random.Random(0).getstate()
        self._pending: TradeOffer | None = None
        self._used_offer_ids: set[str] = set()
        self._terminated = self._truncated = False

    @contextmanager
    def _randomness(self):
        with _RNG_LOCK:
            external = random.getstate()
            random.setstate(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = random.getstate()
                random.setstate(external)

    def reset(self, seed: int = 0) -> Observation:
        self._rng_state = random.Random(seed).getstate()
        with self._randomness():
            # Backend treats zero as absent; shift all seeds consistently.
            self.game = Game(
                [Player(c) for c in Color],
                seed=seed + 1,
                discard_limit=self.rules.robber_discard_threshold,
                vps_to_win=self.rules.victory_points,
            )
            self.game.state.resource_freqdeck = list(self.rules.bank_resources)
            self.game.state.development_listdeck = [
                card
                for card, count in zip(DEVELOPMENT_CARDS, self.rules.development_cards)
                for _ in range(count)
            ]
            random.shuffle(self.game.state.development_listdeck)
        self.version = 0
        self._pending = None
        self._used_offer_ids = set()
        self._terminated = self._truncated = False
        self._discard_remaining: dict[int, int] = {}
        self._new_development = [[0] * 5 for _ in range(4)]
        self._events: list[list[VisibleEvent]] = [[] for _ in range(4)]
        self._make_board()
        from .beliefs import BeliefLedger

        self._ledgers = [
            BeliefLedger(
                player_count=4,
                known_hands={q: ZERO for q in range(4)},
                totals=[0] * 4,
                seed=seed + p,
            )
            for p in range(4)
        ]
        self._refresh_actions()
        return self.observe()

    def _make_board(self):
        entries = list(self.game.state.board.map.land_tiles.items())
        board = BoardSpec.from_hexes(
            [(c[0], c[2]) for c, _ in entries],
            [t.resource or "DESERT" for _, t in entries],
            [t.number or 0 for _, t in entries],
        )
        self._coordinate_to_tile = {coord: i for i, (coord, _) in enumerate(entries)}
        self._tile_to_coordinate = {i: coord for coord, i in self._coordinate_to_tile.items()}
        self._node_to_vertex = {}
        for tile_id, (_, tile) in enumerate(entries):
            for ref, vertex in zip(NodeRef, board.tile_vertices[tile_id]):
                previous = self._node_to_vertex.setdefault(tile.nodes[ref], vertex)
                if previous != vertex:
                    raise RuntimeError("backend topology does not match axial vertex geometry")
        self._vertex_to_node = {v: n for n, v in self._node_to_vertex.items()}
        edge_ids = {tuple(sorted(e)): i for i, e in enumerate(board.edges)}
        self._edge_to_id, self._id_to_edge = {}, {}
        for _, tile in entries:
            for edge in tile.edges.values():
                key = tuple(sorted(self._node_to_vertex[n] for n in edge))
                edge_id = edge_ids[key]
                self._edge_to_id[tuple(sorted(edge))] = edge_id
                self._id_to_edge[edge_id] = edge
        ports = []
        for port in self.game.state.board.map.ports_by_id.values():
            nodes = PORT_DIRECTION_TO_NODEREFS[port.direction]
            vertices = tuple(sorted(self._node_to_vertex[port.nodes[n]] for n in nodes))
            ports.append(Port(vertices, port.resource, 3 if port.resource is None else 2))
        self.board = replace(board, ports=tuple(ports))

    @property
    def current_player(self) -> int:
        return self._pending.recipient if self._pending else self.game.state.current_player_index

    @property
    def winner(self) -> int | None:
        turn = self.game.state.current_turn_index
        return turn if self._actual_vp()[turn] >= self.rules.victory_points else None

    def _hand(self, p: int) -> tuple[int, ...]:
        return tuple(self.game.state.player_state[f"P{p}_{r}_IN_HAND"] for r in RESOURCES)

    def _actual_vp(self) -> tuple[int, ...]:
        return tuple(self.game.state.player_state[f"P{p}_ACTUAL_VICTORY_POINTS"] for p in range(4))

    def _refresh_actions(self):
        state = self.game.state
        if state.is_road_building:
            p = state.current_player_index
            if state.free_roads_available and state.player_state[f"P{p}_ROADS_AVAILABLE"]:
                state.playable_actions = [
                    EngineAction(state.current_color(), ActionType.BUILD_ROAD, e)
                    for e in state.board.buildable_edges(state.current_color())
                ]
            else:
                state.playable_actions = []
            if state.playable_actions:
                return
            state.is_road_building = False
            state.free_roads_available = 0
        state.playable_actions = generate_playable_actions(state)
        p = state.current_player_index
        # All development types may be played before rolling; upstream exposes
        # only Knights in that phase. Their normal effects preserve HAS_ROLLED.
        if (
            state.current_prompt == ActionPrompt.PLAY_TURN
            and not state.player_state[f"P{p}_HAS_ROLLED"]
        ):
            if player_can_play_dev(state, state.current_color(), "YEAR_OF_PLENTY"):
                state.playable_actions.extend(
                    year_of_plenty_possibilities(state.current_color(), state.resource_freqdeck)
                )
            if player_can_play_dev(state, state.current_color(), "MONOPOLY"):
                state.playable_actions.extend(monopoly_possibilities(state.current_color()))
        # Upstream incorrectly gates the card itself on affording a paid road.
        if (
            state.current_prompt == ActionPrompt.PLAY_TURN
            and player_can_play_dev(state, state.current_color(), "ROAD_BUILDING")
            and state.player_state[f"P{p}_ROADS_AVAILABLE"] > 0
            and state.board.buildable_edges(state.current_color())
        ):
            free_card = EngineAction(state.current_color(), ActionType.PLAY_ROAD_BUILDING, None)
            if free_card not in state.playable_actions:
                state.playable_actions.append(free_card)
        card_kinds = {
            "PLAY_KNIGHT_CARD": "KNIGHT",
            "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY",
            "PLAY_MONOPOLY": "MONOPOLY",
            "PLAY_ROAD_BUILDING": "ROAD_BUILDING",
        }
        state.playable_actions = [
            a
            for a in state.playable_actions
            if a.action_type.name not in card_kinds
            or state.player_state[f"P{p}_{card_kinds[a.action_type.name]}_IN_HAND"]
            > self._new_development[p][DEVELOPMENT_CARDS.index(card_kinds[a.action_type.name])]
        ]

    def _convert(self, a: EngineAction) -> Action:
        kind, value = a.action_type.name, a.value
        if kind in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            return Action(kind, target=self._node_to_vertex[value])
        if kind == "BUILD_ROAD":
            return Action(kind, target=self._edge_to_id[tuple(sorted(value))])
        if kind == "MOVE_ROBBER":
            coord, color, _ = value
            partner = None if color is None else self.game.state.color_to_index[color]
            return Action(kind, target=self._coordinate_to_tile[coord], partner=partner)
        if kind == "PLAY_YEAR_OF_PLENTY":
            return Action(
                kind, resource=value[0], second_resource=value[1] if len(value) > 1 else None
            )
        if kind == "PLAY_MONOPOLY":
            return Action(kind, resource=value)
        if kind == "MARITIME_TRADE":
            return Action(
                kind,
                give=tuple(value[:-1].count(r) for r in RESOURCES),
                receive=tuple(int(value[-1] == r) for r in RESOURCES),
            )
        return Action(kind)

    def observe(self, player_id: int | None = None) -> Observation:
        p = self.current_player if player_id is None else player_id
        if p not in range(4):
            raise ValueError("invalid player id")
        state = self.game.state
        actor, turn = self.current_player, state.current_turn_index
        own = self._hand(p)
        if self._pending:
            kind, offer = "TRADE_RESPONSE", self._pending
            legal = (Action("REJECT_TRADE", offer_id=offer.offer_id),)
            if all(h >= n for h, n in zip(own, offer.receive)):
                legal += (Action("ACCEPT_TRADE", offer_id=offer.offer_id),)
        elif state.is_discarding:
            kind, offer = "DISCARD", None
            legal = tuple(
                Action("DISCARD_RESOURCE", resource=r) for r, n in zip(RESOURCES, own) if n
            )
        else:
            kind, offer = state.current_prompt.name, None
            legal = tuple(sorted((self._convert(a) for a in state.playable_actions), key=repr))
        if p != actor or self._terminated or self._truncated:
            legal = ()
        players = []
        for q in range(4):
            ps = state.player_state
            players.append(
                PublicPlayer(
                    q,
                    ps[f"P{q}_VICTORY_POINTS"],
                    sum(self._hand(q)),
                    sum(ps[f"P{q}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS),
                    ps[f"P{q}_ROADS_AVAILABLE"],
                    ps[f"P{q}_SETTLEMENTS_AVAILABLE"],
                    ps[f"P{q}_CITIES_AVAILABLE"],
                    ps[f"P{q}_LONGEST_ROAD_LENGTH"],
                    ps[f"P{q}_PLAYED_KNIGHT"],
                    ps[f"P{q}_HAS_ROAD"],
                    ps[f"P{q}_HAS_ARMY"],
                )
            )
        self._ledgers[p].reconcile([q.resource_count for q in players], own_player=p, own_hand=own)
        context = DecisionContext(
            kind,
            actor,
            turn,
            self.version,
            state.num_turns,
            state.player_state[f"P{turn}_HAS_ROLLED"],
            state.player_state[f"P{turn}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
            offer,
            tuple(q for q in range(4) if q != turn),
            self._discard_remaining.get(actor, 0),
            state.free_roads_available if state.is_road_building else 0,
        )
        buildings = tuple(
            sorted(
                (
                    Building(
                        self._node_to_vertex[n], state.color_to_index[c], 2 if t == "CITY" else 1
                    )
                    for n, (c, t) in state.board.buildings.items()
                ),
                key=lambda b: b.vertex,
            )
        )
        roads = tuple(
            sorted(
                {
                    Road(self._edge_to_id[tuple(sorted(edge))], state.color_to_index[c])
                    for edge, c in state.board.roads.items()
                },
                key=lambda r: r.edge,
            )
        )
        kwargs = {
            "player_id": p,
            "board": self.board,
            "rules": self.rules,
            "context": context,
            "players": tuple(players),
            "hand": own,
            "development": tuple(
                state.player_state[f"P{p}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS
            ),
            "buildings": buildings,
            "roads": roads,
            "robber_tile": self._coordinate_to_tile[state.board.robber_coordinate],
            "legal_actions": legal,
            "beliefs": self._ledgers[p].summaries(),
            "events": tuple(self._events[p][-64:]),
        }
        if "bank" in Observation.__dataclass_fields__:
            kwargs["bank"] = None
        return Observation(**kwargs)

    def offer_trade(self, offer: TradeOffer) -> Observation:
        if self._terminated or self._truncated or self._pending:
            raise ValueError("a trade cannot be proposed in this state")
        state = self.game.state
        if (
            offer.proposer != state.current_player_index
            or offer.proposer != state.current_turn_index
            or offer.recipient not in range(4)
            or offer.recipient == offer.proposer
            or state.current_prompt != ActionPrompt.PLAY_TURN
            or state.is_initial_build_phase
            or state.is_road_building
            or not state.player_state[f"P{offer.proposer}_HAS_ROLLED"]
        ):
            raise ValueError("trade requires the active player after rolling")
        if not offer.offer_id or offer.offer_id in self._used_offer_ids:
            raise ValueError("offer id is stale or already used")
        if (
            not any(offer.give)
            or not any(offer.receive)
            or any(g and r for g, r in zip(offer.give, offer.receive))
        ):
            raise ValueError("trade must exchange distinct nonempty resource sets")
        if not all(h >= n for h, n in zip(self._hand(offer.proposer), offer.give)):
            raise ValueError("proposer cannot afford offer")
        self._pending = offer
        self._used_offer_ids.add(offer.offer_id)
        self.version += 1
        return self.observe()

    def step(self, action: Action) -> StepResult:
        if self._terminated or self._truncated:
            raise ValueError("episode has ended; reset before stepping")
        view = self.observe()
        if action not in view.legal_actions:
            raise ValueError("action is not legal in the current decision context")
        before = [self._hand(p) for p in range(4)]
        previous_vp = self._actual_vp()
        actor = self.current_player
        self.version += 1
        if self._pending:
            offer = self._pending
            if action.kind == "ACCEPT_TRADE":
                if not all(h >= n for h, n in zip(before[offer.proposer], offer.give)) or not all(
                    h >= n for h, n in zip(before[offer.recipient], offer.receive)
                ):
                    raise ValueError("offer can no longer be funded")
                for p, give, receive in (
                    (offer.proposer, offer.give, offer.receive),
                    (offer.recipient, offer.receive, offer.give),
                ):
                    for r, g, n in zip(RESOURCES, give, receive):
                        self.game.state.player_state[f"P{p}_{r}_IN_HAND"] += n - g
                self._record_known_changes(before, "DOMESTIC_TRADE", actor)
            else:
                self._emit(VisibleEvent(self.version, "REJECT_TRADE", actor))
            self._pending = None
            self._refresh_actions()
        elif action.kind == "DISCARD_RESOURCE":
            self._discard_one(action.resource, actor)
        else:
            engine_action = next(
                a for a in self.game.state.playable_actions if self._convert(a) == action
            )
            free_before = (
                self.game.state.free_roads_available if self.game.state.is_road_building else 0
            )
            with self._randomness():
                executed = self.game.execute(engine_action)
            if action.kind == "BUILD_ROAD" and free_before > 1:
                # Upstream prematurely stops if the player cannot afford a paid road.
                self.game.state.is_road_building = True
                self.game.state.free_roads_available = free_before - 1
            if action.kind == "ROLL":
                self._emit(
                    VisibleEvent(self.version, "DICE_ROLL", actor, count=sum(executed.value))
                )
            if action.kind == "BUY_DEVELOPMENT_CARD":
                self._new_development[actor][DEVELOPMENT_CARDS.index(executed.value)] += 1
            if action.kind == "END_TURN":
                self._new_development[actor] = [0] * 5
            if action.kind == "ROLL" and sum(executed.value) == 7:
                self._discard_remaining = {
                    p: sum(self._hand(p)) // 2
                    for p in range(4)
                    if sum(self._hand(p)) > self.rules.robber_discard_threshold
                }
            if action.kind == "MOVE_ROBBER" and action.partner is not None:
                resource = executed.value[2]
                for observer, ledger in enumerate(self._ledgers):
                    known = resource if observer in (actor, action.partner) else None
                    ledger.transfer(action.partner, actor, 1, resource=known)
                    self._events[observer].append(
                        VisibleEvent(self.version, "STEAL", actor, action.partner, known, 1)
                    )
            else:
                self._record_known_changes(before, action.kind, actor)
            self._refresh_actions()
        self.assert_resource_conservation()
        self._terminated = self.winner is not None
        self._truncated = self.version >= self.max_steps and not self._terminated
        from .rewards import potential_rewards

        rewards = potential_rewards(
            previous_vp,
            self._actual_vp(),
            self.rules.victory_points,
            winner=self.winner,
            terminated=self._terminated,
        )
        return StepResult(
            self.observe(),
            tuple(rewards),
            self._terminated,
            self._truncated,
            self.winner,
            {"version": self.version, "turn": self.game.state.num_turns},
        )

    def _discard_one(self, resource: str, actor: int):
        state = self.game.state
        if actor not in self._discard_remaining:
            self._discard_remaining[actor] = sum(self._hand(actor)) // 2
        state.player_state[f"P{actor}_{resource}_IN_HAND"] -= 1
        state.resource_freqdeck[RESOURCES.index(resource)] += 1
        self._discard_remaining[actor] -= 1
        for observer, ledger in enumerate(self._ledgers):
            known = tuple(int(r == resource) for r in RESOURCES) if observer == actor else None
            ledger.discard(actor, 1, known=known)
            self._events[observer].append(
                VisibleEvent(
                    self.version,
                    "DISCARD",
                    actor,
                    resource=resource if observer == actor else None,
                    count=1,
                )
            )
        if self._discard_remaining[actor] == 0:
            del self._discard_remaining[actor]
            if self._discard_remaining:
                state.current_player_index = min(self._discard_remaining)
            else:
                state.current_player_index = state.current_turn_index
                state.current_prompt = ActionPrompt.MOVE_ROBBER
                state.is_discarding = False
                state.is_moving_knight = True
        self._refresh_actions()

    def _emit(self, event: VisibleEvent):
        for events in self._events:
            events.append(event)

    def _record_known_changes(self, before, kind: str, actor: int):
        changed = False
        for p in range(4):
            delta = tuple(n - o for n, o in zip(self._hand(p), before[p]))
            if any(delta):
                changed = True
                for ledger in self._ledgers:
                    ledger.apply_delta(p, delta)
                give = tuple(max(0, -n) for n in delta)
                receive = tuple(max(0, n) for n in delta)
                self._emit(VisibleEvent(self.version, kind, actor, p, give=give, receive=receive))
        if not changed:
            self._emit(VisibleEvent(self.version, kind, actor))

    def assert_resource_conservation(self):
        for i, resource in enumerate(RESOURCES):
            bank = self.game.state.resource_freqdeck[i]
            hands = [self._hand(p)[i] for p in range(4)]
            if (
                bank < 0
                or any(n < 0 for n in hands)
                or bank + sum(hands) != self.rules.bank_resources[i]
            ):
                raise AssertionError(
                    f"resource conservation failed for {resource}: {bank}, {hands}"
                )


CatanSimulator = FourPlayerGame
