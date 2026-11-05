"""Explicitly authorized Colonist UI execution with verified state transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import Protocol

from ...contracts import RESOURCES, Action, Building, Observation, Road
from ...trades import validate_trade_offer
from ..calibration import _TARGET_TYPES, Point, ScreenCalibration
from ..capture import CaptureObservation, CaptureReader
from ..errors import PerceptionError
from ..observation_io import _validate_board


class BrowserTransport(Protocol):
    def screenshot(self) -> bytes: ...
    def click(self, x: float, y: float) -> None: ...


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
