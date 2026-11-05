"""Versioned, backend-independent player observations and visible-event exports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from itertools import pairwise
from math import isfinite
from typing import Any

from ..board import BoardSpec, Port, RulesConfig, Tile
from ..contracts import (
    RESOURCES,
    ZERO,
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    ResourceBelief,
    Road,
    TradeOffer,
    VisibleEvent,
    resource_vector,
)
from .errors import PerceptionError


def _envelope(data: Mapping[str, Any], format_name: str) -> None:
    if (
        not isinstance(data, Mapping)
        or data.get("format") != format_name
        or type(data.get("version")) is not int
        or data.get("version") != 1
    ):
        raise PerceptionError(f"expected version 1 {format_name} data")


def _record(cls, data, **changes):
    if not isinstance(data, Mapping) or set(data) - {field.name for field in fields(cls)}:
        raise PerceptionError(f"unknown or malformed {cls.__name__} fields")
    return cls(**(dict(data) | changes))


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _validate_board(board: BoardSpec) -> None:
    if not board.tiles or not board.vertices or len(board.tiles) != len(board.tile_vertices):
        raise PerceptionError("board geometry is missing")
    if len(board.tiles) != len(board.tile_edges):
        raise PerceptionError("incomplete tile geometry")
    for index, tile in enumerate(board.tiles):
        if (
            tile.id != index
            or not _integer(tile.id)
            or type(tile.q) is not int
            or type(tile.r) is not int
            or tile.resource not in (*RESOURCES, "DESERT")
            or type(tile.number) is not int
            or tile.number not in (0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
            or (tile.resource == "DESERT" and tile.number != 0)
        ):
            raise PerceptionError("unknown terrain, token, or tile identifier")
    if any(len(point) != 2 or any(type(v) is not int for v in point) for point in board.vertices):
        raise PerceptionError("vertices must be integer lattice coordinates")
    if len(set(board.vertices)) != len(board.vertices):
        raise PerceptionError("duplicate board vertices")
    for edge in board.edges:
        if (
            len(edge) != 2
            or edge[0] == edge[1]
            or any(not _integer(v) or v >= len(board.vertices) for v in edge)
        ):
            raise PerceptionError("invalid edge geometry")
    for vertices, edges in zip(board.tile_vertices, board.tile_edges):
        if len(vertices) != 6 or len(edges) != 6:
            raise PerceptionError("hexes require six vertices and six edges")
        if any(not _integer(v) or v >= len(board.vertices) for v in vertices):
            raise PerceptionError("tile vertex is absent")
        for index, edge in enumerate(edges):
            if (
                not _integer(edge)
                or edge >= len(board.edges)
                or set(board.edges[edge]) != {vertices[index], vertices[(index + 1) % 6]}
            ):
                raise PerceptionError("inconsistent tile edge geometry")
    for port in board.ports:
        if (
            len(port.vertices) != 2
            or any(not _integer(v) or v >= len(board.vertices) for v in port.vertices)
            or port.resource not in (*RESOURCES, None)
            or not _integer(port.ratio, 2)
        ):
            raise PerceptionError("invalid port")


def observation_to_dict(view: Observation) -> dict[str, Any]:
    """Return a JSON-ready, detached, versioned player-view export."""
    if not isinstance(view, Observation) or view.schema_version != 1:
        raise PerceptionError("unsupported observation schema")
    try:
        payload = json.loads(json.dumps(asdict(view), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise PerceptionError("observation is not finite JSON data") from error
    return {"format": "colonist-rl.observation", "version": 1, "observation": payload}


def _event(data: Mapping[str, Any]) -> VisibleEvent:
    event = _record(
        VisibleEvent,
        data,
        give=resource_vector(data.get("give", ZERO)),
        receive=resource_vector(data.get("receive", ZERO)),
    )
    if (
        not _integer(event.sequence)
        or not isinstance(event.kind, str)
        or not event.kind
        or not _integer(event.count)
        or event.resource not in (*RESOURCES, None)
        or any(
            value is not None and not _integer(value) for value in (event.actor, event.recipient)
        )
    ):
        raise PerceptionError("invalid visible event")
    return event


def _events(data: Sequence[Mapping[str, Any]]) -> tuple[VisibleEvent, ...]:
    events = tuple(_event(item) for item in data)
    if any(a.sequence > b.sequence for a, b in pairwise(events)):
        raise PerceptionError("visible events must be in sequence order")
    return events


def view_from_dict(data: Mapping[str, Any]) -> Observation:
    """Decode full board, rules, and player-view records into immutable tuples."""
    try:
        _envelope(data, "colonist-rl.observation")
        body = data["observation"]
        if body.get("schema_version", 1) != 1:
            raise PerceptionError("unsupported observation schema")
        source = body["board"]
        board = _record(
            BoardSpec,
            source,
            tiles=tuple(_record(Tile, tile) for tile in source["tiles"]),
            ports=tuple(_record(Port, port) for port in source.get("ports", ())),
        )
        _validate_board(board)
        rules = _record(RulesConfig, body["rules"])
        source = body["context"]
        raw_offer = source.get("offer")
        context = _record(
            DecisionContext,
            source,
            offer=_record(TradeOffer, raw_offer) if raw_offer is not None else None,
            eligible_partners=tuple(source.get("eligible_partners", ())),
        )
        players = tuple(_record(PublicPlayer, player) for player in body["players"])
        ids = {player.player_id for player in players}
        if (
            len(ids) != len(players)
            or len(players) != rules.player_count
            or any(not _integer(player.player_id) for player in players)
            or body["player_id"] not in ids
            or context.actor not in ids
            or context.turn_owner not in ids
            or not _integer(context.version)
        ):
            raise PerceptionError("invalid player or decision identity")
        actions = tuple(
            _record(
                Action,
                action,
                give=resource_vector(action.get("give", ZERO)),
                receive=resource_vector(action.get("receive", ZERO)),
            )
            for action in body["legal_actions"]
        )
        beliefs = tuple(
            _record(
                ResourceBelief,
                belief,
                expected=tuple(belief["expected"]),
                variance=tuple(belief["variance"]),
                lower=resource_vector(belief["lower"]),
                upper=resource_vector(belief["upper"]),
            )
            for belief in body.get("beliefs", ())
        )
        for belief in beliefs:
            if (
                belief.player_id not in ids
                or len(belief.expected) != 5
                or len(belief.variance) != 5
                or any(not isfinite(n) or n < 0 for n in (*belief.expected, *belief.variance))
                or any(low > high for low, high in zip(belief.lower, belief.upper))
            ):
                raise PerceptionError("invalid resource belief")
        buildings = tuple(_record(Building, building) for building in body["buildings"])
        roads = tuple(_record(Road, road) for road in body["roads"])
        if any(
            b.owner not in ids
            or not _integer(b.vertex)
            or b.vertex >= len(board.vertices)
            or b.level not in (1, 2)
            for b in buildings
        ):
            raise PerceptionError("invalid building")
        if any(
            r.owner not in ids or not _integer(r.edge) or r.edge >= len(board.edges) for r in roads
        ):
            raise PerceptionError("invalid road")
        if not _integer(body["robber_tile"]) or body["robber_tile"] >= len(board.tiles):
            raise PerceptionError("invalid robber tile")
        view = _record(
            Observation,
            body,
            board=board,
            rules=rules,
            context=context,
            players=players,
            hand=resource_vector(body["hand"]),
            development=resource_vector(body["development"]),
            buildings=buildings,
            roads=roads,
            legal_actions=actions,
            beliefs=beliefs,
            events=_events(body.get("events", ())),
            bank=resource_vector(body["bank"]) if body.get("bank") is not None else None,
        )
        return view
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        if isinstance(error, PerceptionError):
            raise
        raise PerceptionError("malformed observation export") from error


observation_from_dict = view_from_dict


def _visible_to(visibility: Any, observer: int) -> bool:
    if visibility == "public":
        return True
    if isinstance(visibility, list) and all(_integer(player) for player in visibility):
        return observer in visibility
    raise PerceptionError("exports require explicit public or player-list visibility")


def visible_events_from_export(
    data: Mapping[str, Any], observer_id: int
) -> tuple[VisibleEvent, ...]:
    """Project a documented structured export without inferring private identities."""
    _envelope(data, "colonist-rl.visible-events")
    if not _integer(observer_id):
        raise PerceptionError("observer_id must be a nonnegative integer")
    visible = []
    try:
        for raw in data["events"]:
            if not _visible_to(raw.get("visibility"), observer_id):
                continue
            row = dict(raw)
            row.pop("visibility")
            identity_visibility = row.pop("resource_visibility", None)
            if any(key in row for key in ("resource", "give", "receive")) and not _visible_to(
                identity_visibility, observer_id
            ):
                row.update(resource=None, give=ZERO, receive=ZERO)
            visible.append(row)
        return _events(visible)
    except (TypeError, KeyError, ValueError, AttributeError) as error:
        if isinstance(error, PerceptionError):
            raise
        raise PerceptionError("malformed visible event export") from error
