"""Known public consequences for each action in the current legal menu.

Unknown resource outcomes have an explicit mask. These rows never predict a
future legal menu, a card draw, a robber steal, or an opponent's private hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .consequences import PUBLIC_BUILD_COSTS
from .contracts import RESOURCES, Action, Building, Observation, PublicPlayer, Road
from .settlement_sites import connected_settlement_sites

SUCCESSOR_SCHEMA = "public-candidate-successor-v1"
_SCALARS = (
    "hand_known", "uncertain_resource_result", "unknown_dev_draw",
    "known_point_gain", "wins_now", "road_pieces_after",
    "settlement_pieces_after", "city_pieces_after", "connected_sites_before",
    "connected_sites_after", "city_sites_before", "city_sites_after",
    "point_recipe_gap_before", "point_recipe_gap_after",
    "point_recipe_available_after", "bank_trade_ratio", "own_vp_after",
)
_RESOURCE_PREFIXES = (
    "hand_before_", "hand_after_", "hand_delta_", "road_deficit_after_",
    "settlement_deficit_after_", "city_deficit_after_", "card_deficit_after_",
    "production_nominal_delta_", "production_unblocked_delta_", "port_ratio_after_",
)
SUCCESSOR_COLUMNS = _SCALARS + tuple(
    prefix + resource for prefix in _RESOURCE_PREFIXES for resource in RESOURCES
)
SUCCESSOR_DIM = len(SUCCESSOR_COLUMNS)
_RECIPES = tuple(zip(("road", "settlement", "city", "card"), PUBLIC_BUILD_COSTS.values()))
_UNKNOWN_HAND = frozenset(("ROLL", "MOVE_ROBBER", "PLAY_MONOPOLY", "DISCARD"))
_KNOWN_UNCHANGED = frozenset(
    ("END_TURN", "REJECT_TRADE", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING")
)
_SETTLEMENT = frozenset(("BUILD_SETTLEMENT", "BUILD_INITIAL_SETTLEMENT"))
_ROAD = frozenset(("BUILD_ROAD", "BUILD_INITIAL_ROAD"))
_ZERO = (0, 0, 0, 0, 0)


def _bounded(value: int, capacity: int) -> float:
    return min(capacity, max(0, value)) / capacity


def _point_gap(hand: tuple[int, ...], sites: int, cities: int,
               settlements_left: int, cities_left: int) -> int:
    possible = []
    if sites and settlements_left:
        recipe = PUBLIC_BUILD_COSTS["BUILD_SETTLEMENT"]
        possible.append(sum(max(0, cost - count) for cost, count in zip(recipe, hand)))
    if cities and cities_left:
        recipe = PUBLIC_BUILD_COSTS["BUILD_CITY"]
        possible.append(sum(max(0, cost - count) for cost, count in zip(recipe, hand)))
    return min(possible, default=8)


@dataclass(frozen=True)
class _Prepared:
    view: Observation
    own: PublicPlayer
    sites: int
    cities: int
    point_gap: int
    port_ratios: tuple[int, ...]
    vertex_pips: tuple[tuple[float, ...], ...]
    vertex_unblocked_pips: tuple[tuple[float, ...], ...]


def _prepare(view: Observation) -> _Prepared:
    own = next(player for player in view.players if player.player_id == view.player_id)
    sites = len(connected_settlement_sites(view.board, view.buildings, view.roads, view.player_id))
    cities = sum(b.owner == view.player_id and b.level == 1 for b in view.buildings)
    owned = {b.vertex for b in view.buildings if b.owner == view.player_id}
    ratios = tuple(
        min([4] + [
            port.ratio for port in view.board.ports
            if owned.intersection(port.vertices) and port.resource in (None, resource)
        ])
        for resource in RESOURCES
    )
    pips = [[0.0] * 5 for _ in view.board.vertices]
    unblocked = [[0.0] * 5 for _ in view.board.vertices]
    for tile, vertices in zip(view.board.tiles, view.board.tile_vertices, strict=True):
        if tile.resource not in RESOURCES or not tile.number:
            continue
        resource = RESOURCES.index(tile.resource)
        gain = (6 - abs(7 - tile.number)) / 36
        for vertex in vertices:
            pips[vertex][resource] += gain
            if tile.id != view.robber_tile:
                unblocked[vertex][resource] += gain
    return _Prepared(
        view, own, sites, cities,
        _point_gap(view.hand, sites, cities, own.settlements_left, own.cities_left),
        ratios, tuple(map(tuple, pips)), tuple(map(tuple, unblocked)),
    )


def _second_setup_grant(view: Observation, target: int | None) -> tuple[int, ...] | None:
    """Starting resources are public when standard setup stock guarantees them."""
    if target is None:
        return None
    receive = tuple(
        sum(tile.resource == resource and target in vertices
            for tile, vertices in zip(view.board.tiles, view.board.tile_vertices, strict=True))
        for resource in RESOURCES
    )
    for index, count in enumerate(receive):
        if not count:
            continue
        if view.bank is not None:
            if view.bank[index] < count:
                return None
        elif view.rules.bank_resources[index] < 3 * view.rules.player_count:
            # Earlier second settlements could have taken three of this resource each.
            return None
    return receive


def _hand_after(view: Observation, action: Action) -> tuple[int, ...] | None:
    kind = action.kind
    if kind in _UNKNOWN_HAND:
        return None
    if kind == "MARITIME_TRADE":
        give, receive = action.give, action.receive
    elif kind == "ACCEPT_TRADE":
        if view.context.offer is None:
            raise ValueError("a legal accept requires its public trade offer")
        give, receive = view.context.offer.for_player(view.player_id)
    elif kind in PUBLIC_BUILD_COSTS:
        setup = (
            kind == "BUILD_ROAD" and
            (view.context.free_roads or view.context.kind in ("BUILD_INITIAL_ROAD", "SETUP_ROAD"))
        ) or (
            kind == "BUILD_SETTLEMENT" and
            view.context.kind in ("BUILD_INITIAL_SETTLEMENT", "SETUP_SETTLEMENT")
        )
        give, receive = (_ZERO if setup else PUBLIC_BUILD_COSTS[kind]), _ZERO
        if (setup and kind == "BUILD_SETTLEMENT" and
                any(b.owner == view.player_id for b in view.buildings)):
            receive = _second_setup_grant(view, action.target)
            if receive is None:
                return None
    elif kind in _ROAD | _SETTLEMENT:
        give, receive = _ZERO, _ZERO
        if kind in _SETTLEMENT and any(b.owner == view.player_id for b in view.buildings):
            receive = _second_setup_grant(view, action.target)
            if receive is None:
                return None
    elif kind == "DISCARD_RESOURCE":
        if action.resource not in RESOURCES:
            raise ValueError("a legal discard requires a named resource")
        give = tuple(int(resource == action.resource) for resource in RESOURCES)
        receive = _ZERO
    elif kind == "PLAY_YEAR_OF_PLENTY":
        if action.resource not in RESOURCES or action.second_resource not in RESOURCES:
            return None
        give = _ZERO
        receive = tuple(
            int(resource == action.resource) + int(resource == action.second_resource)
            for resource in RESOURCES
        )
    elif kind in _KNOWN_UNCHANGED:
        return view.hand
    else:
        return None
    hand = tuple(before - spent + gained for before, spent, gained in zip(view.hand, give, receive))
    if any(count < 0 for count in hand):
        raise ValueError("a legal action cannot spend more than the observed hand")
    return hand


def _after_geometry(prepared: _Prepared, action: Action):
    view, own = prepared.view, prepared.own
    buildings, roads = view.buildings, view.roads
    roads_left = own.roads_left
    settlements_left = own.settlements_left
    cities_left = own.cities_left
    point_gain = 0
    nominal_delta = [0.0] * 5
    unblocked_delta = [0.0] * 5
    port_ratios = list(prepared.port_ratios)
    if action.kind in _ROAD and action.target is not None:
        roads = roads + (Road(action.target, view.player_id),)
        roads_left -= 1
    elif action.kind in _SETTLEMENT and action.target is not None:
        buildings = buildings + (Building(action.target, view.player_id),)
        settlements_left -= 1
        point_gain = 1
        nominal_delta = list(prepared.vertex_pips[action.target])
        unblocked_delta = list(prepared.vertex_unblocked_pips[action.target])
        for port in view.board.ports:
            if action.target in port.vertices:
                for i, resource in enumerate(RESOURCES):
                    if port.resource in (None, resource):
                        port_ratios[i] = min(port_ratios[i], port.ratio)
    elif action.kind == "BUILD_CITY" and action.target is not None:
        buildings = tuple(
            Building(b.vertex, b.owner, 2)
            if b.vertex == action.target and b.owner == view.player_id else b
            for b in buildings
        )
        settlements_left += 1
        cities_left -= 1
        point_gain = 1
        nominal_delta = list(prepared.vertex_pips[action.target])
        unblocked_delta = list(prepared.vertex_unblocked_pips[action.target])
    elif action.kind == "MOVE_ROBBER" and action.target is not None:
        if action.target != view.robber_tile:
            for tile in view.board.tiles:
                if (tile.id not in (view.robber_tile, action.target)
                        or tile.resource not in RESOURCES or not tile.number):
                    continue
                occupied = sum(
                    b.level for b in view.buildings
                    if b.owner == view.player_id and b.vertex in view.board.tile_vertices[tile.id]
                )
                sign = 1 if tile.id == view.robber_tile else -1
                unblocked_delta[RESOURCES.index(tile.resource)] += (
                    sign * occupied * (6 - abs(7 - tile.number)) / 36
                )
    sites_after = (
        len(connected_settlement_sites(view.board, buildings, roads, view.player_id))
        if buildings is not view.buildings or roads is not view.roads
        else prepared.sites
    )
    cities_after = sum(b.owner == view.player_id and b.level == 1 for b in buildings)
    return (sites_after, cities_after, roads_left, settlements_left, cities_left,
            point_gain, nominal_delta, unblocked_delta, port_ratios)


def _values(prepared: _Prepared, action: Action) -> dict[str, float]:
    view = prepared.view
    hand = _hand_after(view, action)
    (sites_after, cities_after, roads_left, settlements_left, cities_left,
     point_gain, nominal_delta, unblocked_delta, port_ratios) = _after_geometry(prepared, action)
    own_vp_after = view.own_victory_points + point_gain
    values = dict.fromkeys(SUCCESSOR_COLUMNS, 0.0)
    values.update(
        hand_known=float(hand is not None),
        uncertain_resource_result=float(hand is None),
        unknown_dev_draw=float(action.kind == "BUY_DEVELOPMENT_CARD"),
        known_point_gain=float(point_gain),
        wins_now=float(bool(point_gain and own_vp_after >= view.rules.victory_points)),
        road_pieces_after=_bounded(roads_left, 15),
        settlement_pieces_after=_bounded(settlements_left, 5),
        city_pieces_after=_bounded(cities_left, 4),
        connected_sites_before=min(prepared.sites, 8) / 8,
        connected_sites_after=min(sites_after, 8) / 8,
        city_sites_before=min(prepared.cities, 8) / 8,
        city_sites_after=min(cities_after, 8) / 8,
        point_recipe_gap_before=prepared.point_gap / 8,
        bank_trade_ratio=(sum(action.give) / 4 if action.kind == "MARITIME_TRADE" else 0),
        own_vp_after=own_vp_after / view.rules.victory_points,
    )
    if hand is not None:
        gap = _point_gap(hand, sites_after, cities_after, settlements_left, cities_left)
        values["point_recipe_gap_after"] = gap / 8
        values["point_recipe_available_after"] = float(gap == 0)
    for i, resource in enumerate(RESOURCES):
        values["hand_before_" + resource] = view.hand[i] / 10
        if hand is not None:
            values["hand_after_" + resource] = hand[i] / 10
            values["hand_delta_" + resource] = (hand[i] - view.hand[i]) / 4
            for label, cost in _RECIPES:
                values[f"{label}_deficit_after_{resource}"] = max(0, cost[i] - hand[i]) / 4
        values["production_nominal_delta_" + resource] = nominal_delta[i]
        values["production_unblocked_delta_" + resource] = unblocked_delta[i]
        values["port_ratio_after_" + resource] = port_ratios[i] / 4
    if any(not math.isfinite(number) for number in values.values()):
        raise ValueError("nonfinite successor features")
    return values


def successor_feature_values(view: Observation, action: Action) -> dict[str, float]:
    """Named facts for the exact object in this observation's legal tuple."""
    if not any(action is present for present in view.legal_actions):
        raise ValueError("successor features require the exact current legal action")
    return _values(_prepare(view), action)


def successor_features_for_menu(view: Observation, *, device="cpu") -> torch.Tensor:
    """Candidate rows in original legal order; an empty menu is 0 x D."""
    if not view.legal_actions:
        return torch.empty((0, SUCCESSOR_DIM), device=device)
    prepared = _prepare(view)
    rows = []
    for action in view.legal_actions:
        values = _values(prepared, action)
        rows.append([values[key] for key in SUCCESSOR_COLUMNS])
    return torch.tensor(rows, device=device, dtype=torch.float32)
