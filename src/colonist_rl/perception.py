"""Versioned offline observations, calibrated geometry, and an opt-in UI boundary.

Annotated captures are useful verified input, not a claim of general screenshot
recognition. This module never opens a browser or discovers private site APIs.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .board import BoardSpec, Port, RulesConfig, Tile
from .contracts import (
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
from .trades import validate_trade_offer


class PerceptionError(ValueError):
    """Input or a UI transition cannot be established unambiguously."""


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


Point = tuple[float, float]
_TARGET_TYPES = {
    "BUILD_SETTLEMENT": "vertex",
    "BUILD_CITY": "vertex",
    "BUILD_ROAD": "edge",
    "MOVE_ROBBER": "tile",
}


@dataclass(frozen=True)
class ScreenCalibration:
    coefficients: tuple[tuple[float, float, float], tuple[float, float, float]]
    confidence: float = 1.0
    max_residual_pixels: float = 0.0

    def __post_init__(self):
        matrix = np.asarray(self.coefficients, dtype=float)
        if (
            matrix.shape != (2, 3)
            or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix[:, :2])) < 1e-10
        ):
            raise PerceptionError("calibration must be a finite nonsingular affine transform")
        if (
            not 0 <= self.confidence <= 1
            or not isfinite(self.max_residual_pixels)
            or self.max_residual_pixels < 0
        ):
            raise PerceptionError("invalid calibration confidence or residual")
        object.__setattr__(
            self, "coefficients", tuple(tuple(float(n) for n in row) for row in matrix)
        )

    @classmethod
    def fit(
        cls,
        anchors: Sequence[tuple[Point, Point]],
        *,
        confidence: float = 1.0,
        max_error_pixels: float = 2.0,
    ) -> ScreenCalibration:
        pairs = np.asarray(anchors, dtype=float)
        if (
            pairs.ndim != 3
            or pairs.shape[1:] != (2, 2)
            or len(pairs) < 3
            or not np.isfinite(pairs).all()
        ):
            raise PerceptionError("at least three finite lattice/pixel anchor pairs are required")
        source = np.column_stack((pairs[:, 0], np.ones(len(pairs))))
        transform, _, rank, _ = np.linalg.lstsq(source, pairs[:, 1], rcond=None)
        if rank < 3:
            raise PerceptionError("calibration anchors are collinear")
        residual = float(np.linalg.norm(source @ transform - pairs[:, 1], axis=1).max())
        if not isfinite(max_error_pixels) or max_error_pixels < 0 or residual > max_error_pixels:
            raise PerceptionError("anchor residual exceeds calibration tolerance")
        return cls(tuple(tuple(row) for row in transform.T), confidence, residual)

    def map_point(self, point: Point, *, min_confidence: float = 0.95) -> Point:
        if self.confidence < min_confidence:
            raise PerceptionError("calibration confidence is too low")
        if len(point) != 2 or not all(isfinite(value) for value in point):
            raise PerceptionError("invalid target point")
        mapped = np.asarray(self.coefficients) @ (point[0], point[1], 1.0)
        return float(mapped[0]), float(mapped[1])

    def action_point(
        self,
        board: BoardSpec,
        action: Action,
        *,
        controls: Mapping[Action, Point] | None = None,
        min_confidence: float = 0.95,
    ) -> Point:
        if self.confidence < min_confidence:
            raise PerceptionError("calibration confidence is too low")
        if controls is not None and action in controls:
            point = controls[action]
            if len(point) != 2 or not all(isfinite(value) for value in point):
                raise PerceptionError("control has invalid screen coordinates")
            return float(point[0]), float(point[1])
        _validate_board(board)
        if action.kind not in _TARGET_TYPES or not _integer(action.target):
            raise PerceptionError("action requires an exact explicitly calibrated control")
        target = action.target
        try:
            if _TARGET_TYPES[action.kind] == "vertex":
                point = board.vertices[target]
            elif _TARGET_TYPES[action.kind] == "edge":
                vertices = [board.vertices[v] for v in board.edges[target]]
                point = tuple(np.mean(vertices, axis=0))
            else:
                vertices = [board.vertices[v] for v in board.tile_vertices[target]]
                point = tuple(np.mean(vertices, axis=0))
        except IndexError as error:
            raise PerceptionError("action target is absent from the board") from error
        return self.map_point(point, min_confidence=min_confidence)


@dataclass(frozen=True)
class CaptureObservation:
    observation: Observation
    confidence: float
    stable: bool
    verified: bool
    target_mode: str | None = None
    screenshot_sha256: str | None = None

    def __post_init__(self):
        if (
            not isinstance(self.observation, Observation)
            or not isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
            or type(self.stable) is not bool
            or type(self.verified) is not bool
        ):
            raise PerceptionError("invalid capture metadata")


class AnnotatedCaptureReader:
    """Read only labeled, user-verified screenshots identified by their SHA-256.

    This is an offline annotation/template registry, not terrain/number OCR.
    Unrecognized screenshots fail closed rather than reuse an earlier state.
    """

    def __init__(self, annotations: Sequence[Mapping[str, Any]]):
        self._captures: dict[str, CaptureObservation] = {}
        for annotation in annotations:
            _envelope(annotation, "colonist-rl.capture")
            if annotation.get("verified") is not True:
                raise PerceptionError("capture must have an explicitly verified annotation")
            digest = annotation.get("screenshot_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise PerceptionError("capture requires a lowercase screenshot SHA-256")
            capture = CaptureObservation(
                view_from_dict(annotation["observation"]),
                annotation["confidence"],
                annotation["stable"],
                True,
                annotation.get("target_mode"),
                digest,
            )
            if digest in self._captures and self._captures[digest] != capture:
                raise PerceptionError("ambiguous annotations for the same screenshot")
            self._captures[digest] = capture

    def read(self, screenshot: bytes) -> CaptureObservation:
        digest = hashlib.sha256(screenshot).hexdigest()
        if digest not in self._captures:
            raise PerceptionError("screenshot has no matching verified annotation")
        try:
            with Image.open(io.BytesIO(screenshot)) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise PerceptionError("capture is not a valid screenshot image") from error
        return self._captures[digest]


def read_capture(annotation_path: str | Path, screenshot_path: str | Path) -> CaptureObservation:
    """Load a user-annotated local screenshot without any browser access."""
    with Path(annotation_path).open() as source:
        annotation = json.load(source)
    return AnnotatedCaptureReader([annotation]).read(Path(screenshot_path).read_bytes())


class BrowserTransport(Protocol):
    def screenshot(self) -> bytes: ...
    def click(self, x: float, y: float) -> None: ...


class CaptureReader(Protocol):
    def read(self, screenshot: bytes) -> CaptureObservation: ...


TransitionVerifier = Callable[[Observation, Action, Observation], bool]
_VERIFIED_ACTIONS = {
    *_TARGET_TYPES,
    "ROLL",
    "END_TURN",
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "DISCARD_RESOURCE",
}


def _expected_transition(before: Observation, action: Action, after: Observation) -> bool:
    actor = before.player_id
    events = tuple(event for event in after.events if event.sequence > before.context.version)
    if action.kind == "BUILD_SETTLEMENT":
        return Building(action.target, actor) in after.buildings and not any(
            b.vertex == action.target for b in before.buildings
        )
    if action.kind == "BUILD_CITY":
        return (
            Building(action.target, actor, 2) in after.buildings
            and Building(action.target, actor) in before.buildings
        )
    if action.kind == "BUILD_ROAD":
        return Road(action.target, actor) in after.roads and not any(
            r.edge == action.target for r in before.roads
        )
    if action.kind == "MOVE_ROBBER":
        return after.robber_tile == action.target and before.robber_tile != action.target
    if action.kind == "ROLL":
        return not before.context.has_rolled and after.context.has_rolled
    if action.kind == "END_TURN":
        return (
            after.context.turn_owner != before.context.turn_owner
            or after.context.turn_number > before.context.turn_number
        )
    if action.kind in ("ACCEPT_TRADE", "REJECT_TRADE"):
        offer = before.context.offer
        if offer is None or after.context.offer == offer:
            return False
        if action.kind == "REJECT_TRADE":
            return any(event.kind == "REJECT_TRADE" and event.actor == actor for event in events)
        give, receive = offer.for_player(actor)
        expected = tuple(h - g + r for h, g, r in zip(before.hand, give, receive))
        return after.hand == expected and any(
            event.kind == "DOMESTIC_TRADE"
            and event.recipient == actor
            and event.give == give
            and event.receive == receive
            for event in events
        )
    if action.kind == "DISCARD_RESOURCE" and action.resource in RESOURCES:
        expected = tuple(h - int(r == action.resource) for h, r in zip(before.hand, RESOURCES))
        return after.hand == expected and any(
            event.kind == "DISCARD"
            and event.actor == actor
            and event.resource == action.resource
            and event.count == 1
            for event in events
        )
    return False


class BrowserAdapter:
    """Injected, explicitly authorized UI execution with before/after checks.

    Controls are keyed by the complete typed action, including offer ID and
    resource choices. Geometry clicks additionally require a verified target
    tool mode. A failed post-click check locks this adapter for manual review.
    No local resource ledger is advanced on a click or predicted outcome.
    """

    def __init__(
        self,
        *,
        calibration: ScreenCalibration,
        transport: BrowserTransport,
        reader: CaptureReader,
        enabled: bool = False,
        authorized: bool = False,
        controls: Mapping[Action, Point] | None = None,
        min_confidence: float = 0.95,
        verify_transition: TransitionVerifier | None = None,
    ):
        if not isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between zero and one")
        self.calibration, self.transport, self.reader = calibration, transport, reader
        self.enabled, self.authorized = enabled, authorized
        self.controls = dict(controls or {})
        self.min_confidence = min_confidence
        self.verify_transition = verify_transition
        self.needs_review = False

    def _read(self) -> CaptureObservation:
        capture = self.reader.read(self.transport.screenshot())
        if (
            not isinstance(capture, CaptureObservation)
            or not capture.verified
            or not capture.stable
            or capture.confidence < self.min_confidence
        ):
            raise PerceptionError("observation must be verified, stable, and confident")
        _validate_board(capture.observation.board)
        return capture

    def execute(self, observation: Observation, action: Action) -> Observation:
        if not (self.enabled and self.authorized):
            raise PermissionError(
                "browser execution requires explicit enablement and authorization"
            )
        if self.needs_review:
            raise PerceptionError("previous ambiguous execution requires manual review")
        capture = self._read()
        current = capture.observation
        if current != observation:
            raise PerceptionError(
                "provided observation is stale or differs from the current capture"
            )
        if action not in current.legal_actions or current.context.actor != current.player_id:
            raise PerceptionError("action is not legal for the observed actor")
        if self.verify_transition is None and action.kind not in _VERIFIED_ACTIONS:
            raise PerceptionError("this action needs an explicit transition verifier")
        if action.kind in ("ACCEPT_TRADE", "REJECT_TRADE"):
            if (
                current.context.kind != "TRADE_RESPONSE"
                or current.context.offer is None
                or action.offer_id != current.context.offer.offer_id
                or current.context.offer.recipient != current.player_id
            ):
                raise PerceptionError("trade response does not match the live offer")
            if action.kind == "ACCEPT_TRADE" and not validate_trade_offer(
                current.context.offer, current, current.context
            ):
                raise PerceptionError("live offer cannot be funded by the acting player")
        if action.kind in _TARGET_TYPES and capture.target_mode != action.kind:
            raise PerceptionError("target requires an explicitly verified UI tool mode")
        if action not in self.controls:
            if action.kind not in _TARGET_TYPES:
                raise PerceptionError("action requires an exact explicitly calibrated control")
            if action.kind == "MOVE_ROBBER" and action.partner is not None:
                raise PerceptionError("robber victim selection requires an explicit UI control")
        point = self.calibration.action_point(
            current.board, action, controls=self.controls, min_confidence=self.min_confidence
        )
        # From this point a partial UI action may have happened; never retry blindly.
        self.needs_review = True
        self.transport.click(*point)
        refreshed = self._read().observation
        baseline_verified = action.kind not in _VERIFIED_ACTIONS or _expected_transition(
            current, action, refreshed
        )
        custom_verified = self.verify_transition is None or self.verify_transition(
            current, action, refreshed
        )
        if (
            refreshed.player_id != current.player_id
            or refreshed.board != current.board
            or refreshed.rules != current.rules
            or refreshed.context.version <= current.context.version
            or not baseline_verified
            or not custom_verified
        ):
            raise PerceptionError("clicked action has no unambiguous reported state change")
        self.needs_review = False
        return refreshed
