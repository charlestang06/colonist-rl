"""Bounded, explicitly authorized placement and incoming-willingness execution.

The injected reader vouches for the complete screenshot, including UI coordinates.
No gesture is retried and no resource transfer is inferred from UI progress.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import Protocol

from ...contracts import Action, DecisionContext, Observation
from ...trades import validate_trade_offer
from ..calibration import Point, ScreenCalibration
from ..capture import CaptureObservation
from ..errors import PerceptionError
from ..observation_io import _validate_board, observation_to_dict, view_from_dict
from .execution import BrowserTransport, _expected_transition
from .ui import (
    UIControl,
    UIExecutionResult,
    UIState,
    _action,
    _immutable,
    _integer,
    validate_acknowledgement,
    validate_confirmation,
    validate_trade_response,
)


def _point(value: object) -> Point:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, Real) or not isfinite(n) or n < 0
            for n in value
        )
    ):
        raise ValueError("coordinates must be two finite nonnegative real numbers")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class LocatedUIControl:
    control: UIControl
    point: Point

    def __post_init__(self):
        if type(self.control) is not UIControl:
            raise TypeError("located control must contain a UIControl")
        _immutable(self.control)
        object.__setattr__(self, "point", _point(self.point))


@dataclass(frozen=True)
class StagedCapture:
    capture: CaptureObservation
    ui: UIState
    controls: tuple[LocatedUIControl, ...] = ()

    def __post_init__(self):
        if type(self.capture) is not CaptureObservation or type(self.ui) is not UIState:
            raise TypeError("capture and UIState records are required")
        _immutable(self.capture)
        _immutable(self.ui)
        if not isinstance(self.controls, Sequence) or isinstance(self.controls, (str, bytes)):
            raise TypeError("controls must be a sequence of located controls")
        controls = tuple(self.controls)
        if any(type(item) is not LocatedUIControl for item in controls):
            raise TypeError("controls must contain LocatedUIControl records")
        if len({item.control.control_id for item in controls}) != len(controls):
            raise ValueError("duplicate located control identities")
        if any(item.control.revision != self.ui.revision for item in controls):
            raise ValueError("located controls must match the UI revision")
        version = self.capture.observation.context.version
        _integer(version, "observation version")
        if self.ui.observation_version != version:
            raise ValueError("UI version must match the game observation")
        object.__setattr__(self, "controls", controls)


class StagedCaptureReader(Protocol):
    def read(self, screenshot: bytes) -> StagedCapture: ...


class StagedBrowserAdapter:
    """Explicit placement and one incoming response, with a global capture budget."""

    def __init__(
        self,
        *,
        calibration: ScreenCalibration,
        transport: BrowserTransport,
        reader: StagedCaptureReader,
        enabled: bool = False,
        authorized: bool = False,
        min_confidence: float = 0.95,
        max_captures: int = 8,
        poll_interval: float = 0.05,
        wait: Callable[[float], None] = time.sleep,
    ):
        if type(calibration) is not ScreenCalibration:
            raise TypeError("calibration must be a ScreenCalibration")
        if type(enabled) is not bool or type(authorized) is not bool:
            raise TypeError("enablement and authorization must be booleans")
        if type(max_captures) is not int or max_captures < 1:
            raise ValueError("max_captures must be a positive integer")
        for name, number in (("min_confidence", min_confidence), ("poll_interval", poll_interval)):
            if (
                isinstance(number, bool)
                or not isinstance(number, Real)
                or not isfinite(number)
                or not 0 <= number <= 1
            ):
                raise ValueError(f"{name} must be finite and between zero and one")
        if not callable(wait):
            raise TypeError("wait must be callable")
        if not callable(getattr(reader, "read", None)) or not all(
            callable(getattr(transport, name, None)) for name in ("click", "screenshot")
        ):
            raise TypeError("reader and transport must implement the staged boundary")
        self.calibration, self.transport, self.reader = calibration, transport, reader
        self.enabled, self.authorized = enabled, authorized
        self.min_confidence, self.max_captures = min_confidence, max_captures
        self.poll_interval, self.wait = poll_interval, wait
        self.needs_review = False

    def _read(self) -> StagedCapture:
        screenshot = self.transport.screenshot()
        if type(screenshot) is not bytes:
            raise PerceptionError("transport must supply screenshot bytes")
        frame = self.reader.read(screenshot)
        if type(frame) is not StagedCapture:
            raise PerceptionError("reader must return a complete StagedCapture")
        # Revalidate the immutable public boundary, including detached coordinates.
        StagedCapture(frame.capture, frame.ui, frame.controls)
        capture = frame.capture
        if (
            type(capture.verified) is not bool
            or not capture.verified
            or type(capture.stable) is not bool
            or isinstance(capture.confidence, bool)
            or not isinstance(capture.confidence, Real)
            or not isfinite(capture.confidence)
            or not self.min_confidence <= capture.confidence <= 1
        ):
            raise PerceptionError("capture must be verified and confident")
        if capture.screenshot_sha256 is not None and (
            capture.screenshot_sha256 != hashlib.sha256(screenshot).hexdigest()
        ):
            raise PerceptionError("screenshot digest does not match the complete evidence")
        observation = capture.observation
        _validate_board(observation.board)
        view_from_dict(observation_to_dict(observation))
        if frame.ui.phase != observation.context.kind:
            raise PerceptionError("UI phase does not match the game observation")
        return frame

    @staticmethod
    def _control_point(frame: StagedCapture, control: UIControl) -> Point:
        matches = tuple(item.point for item in frame.controls if item.control == control)
        if len(matches) != 1:
            raise PerceptionError("exact current control has no unambiguous located binding")
        return matches[0]

    @staticmethod
    def _placement_stage(
        frame: StagedCapture, action: Action, *, completed: bool = False
    ) -> UIControl | None:
        modes = (frame.capture.target_mode, frame.ui.active_tool)
        cleared = (
            completed
            and modes == (None, None)
            and frame.ui.selected_action is None
            and frame.ui.confirmation is None
        )
        if modes != (action.kind, action.kind) and not cleared:
            raise PerceptionError("placement requires verified active target mode and tool")
        if frame.ui.selected_action not in (None, action) or frame.ui.trade_rows:
            raise PerceptionError("incompatible placement selection or trade dialog")
        if frame.ui.confirmation is not None:
            if frame.ui.dialog_stage not in (None, "confirmation"):
                raise PerceptionError("incompatible placement dialog")
            return validate_confirmation(frame.ui, action, expected_revision=frame.ui.revision)
        if frame.ui.dialog_stage is not None:
            raise PerceptionError("unexpected placement dialog")
        return None

    def execute(
        self,
        observation: Observation,
        action: Action,
        *,
        expected_revision: int,
        row_id: str | None = None,
    ) -> UIExecutionResult:
        # Invalid arguments must not be wrapped into a result containing an invalid Action.
        if type(observation) is not Observation:
            raise TypeError("observation must be an Observation")
        if type(observation.context) is not DecisionContext:
            raise TypeError("observation context must be a DecisionContext")
        _immutable(observation)
        _integer(observation.context.version, "observation version")
        _action(action)
        _integer(expected_revision, "expected_revision")
        if row_id is not None and (not isinstance(row_id, str) or not row_id.strip()):
            raise ValueError("row_id must be a nonempty string or None")
        if not (self.enabled and self.authorized):
            raise PermissionError("browser execution requires enablement and authorization")

        latest_revision = expected_revision

        def failure(reason: str) -> UIExecutionResult:
            return UIExecutionResult(
                "ambiguous" if self.needs_review else "blocked",
                action,
                observation.context.version,
                latest_revision,
                reason=reason,
            )

        if self.needs_review:
            return UIExecutionResult(
                "blocked",
                action,
                observation.context.version,
                expected_revision,
                reason="previous ambiguous execution requires manual review",
            )
        placement = action.kind in ("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD")
        incoming = action.kind == "ACCEPT_TRADE"
        if not (
            (
                placement
                and action.target is not None
                and action == Action(action.kind, target=action.target)
                and row_id is None
            )
            or (
                incoming
                and action.offer_id is not None
                and action == Action("ACCEPT_TRADE", offer_id=action.offer_id)
            )
        ):
            return failure("unsupported action kind or noncanonical shape")

        captures = 0

        def read() -> StagedCapture:
            nonlocal captures, latest_revision
            if captures >= self.max_captures:
                raise PerceptionError("capture budget exhausted")
            captures += 1
            frame = self._read()
            latest_revision = frame.ui.revision
            return frame

        def click(point: Point) -> None:
            if captures >= self.max_captures:
                raise PerceptionError("no capture remaining to verify another gesture")
            self.needs_review = True
            self.transport.click(*point)

        try:
            before = read()
            if not before.capture.stable:
                raise PerceptionError("initial capture must be stable")
            if before.capture.observation != observation or before.ui.revision != expected_revision:
                raise PerceptionError("provided game observation or UI revision is stale")
            if (
                type(observation.context.actor) is not int
                or type(observation.player_id) is not int
                or observation.context.actor != observation.player_id
                or action not in observation.legal_actions
            ):
                raise PerceptionError("action is not legal for the current acting player")
            if self.calibration.confidence < self.min_confidence:
                raise PerceptionError("calibration confidence is too low")

            if placement:
                confirmation = self._placement_stage(before, action)
                point = (
                    self._control_point(before, confirmation)
                    if confirmation is not None
                    else _point(
                        self.calibration.action_point(
                            observation.board,
                            action,
                            min_confidence=self.min_confidence,
                        )
                    )
                )
                confirmed = confirmation is not None
            else:
                offer = observation.context.offer
                if (
                    observation.context.kind != "TRADE_RESPONSE"
                    or offer is None
                    or row_id is None
                    or not validate_trade_offer(offer, observation, observation.context)
                    or before.ui.dialog_stage is not None
                    or before.ui.confirmation is not None
                    or before.ui.active_tool is not None
                    or before.capture.target_mode is not None
                ):
                    raise PerceptionError(
                        "incoming willingness requires the current funded offer and row"
                    )
                control = validate_trade_response(
                    before.ui,
                    action,
                    offer,
                    row_id=row_id,
                    expected_revision=expected_revision,
                    player_id=observation.player_id,
                )
                row = next(row for row in before.ui.trade_rows if row.row_id == row_id)
                if any(marker.player_id == observation.player_id for marker in row.responses):
                    raise PerceptionError("acting recipient already has a response marker")
                point = self._control_point(before, control)

            click(point)
            previous = before
            while captures < self.max_captures:
                current = read()
                game = current.capture.observation
                if (
                    game.player_id != observation.player_id
                    or game.board != observation.board
                    or game.rules != observation.rules
                    or game.context.version < previous.capture.observation.context.version
                    or current.ui.revision < previous.ui.revision
                ):
                    raise PerceptionError(
                        "post-input identity, board, rules, or revision changed unexpectedly"
                    )
                payload = (game, current.ui, current.capture.target_mode, current.controls)
                prior_payload = (
                    previous.capture.observation,
                    previous.ui,
                    previous.capture.target_mode,
                    previous.controls,
                )
                if current.ui.revision == previous.ui.revision and payload != prior_payload:
                    raise PerceptionError("semantic payload changed under the same UI revision")
                newer = current.ui.revision > before.ui.revision
                if placement:
                    control = self._placement_stage(current, action, completed=game != observation)

                if game != observation:
                    if (
                        placement
                        and current.capture.stable
                        and newer
                        and game.context.version > observation.context.version
                        and _expected_transition(observation, action, game)
                    ):
                        result = UIExecutionResult(
                            "committed",
                            action,
                            observation.context.version,
                            current.ui.revision,
                            observation=game,
                            evidence=(
                                f"visible board: {action.kind} target={action.target} actor={observation.player_id} version={observation.context.version}->{game.context.version}",
                            ),
                        )
                        self.needs_review = False
                        return result
                    # A verified unstable render may settle, but a different placement may not.
                    if not (
                        placement
                        and not current.capture.stable
                        and newer
                        and game.context.version > observation.context.version
                        and _expected_transition(observation, action, game)
                    ):
                        raise PerceptionError(
                            "unexpected game change without exact verified placement"
                        )
                elif placement:
                    if confirmed:
                        if control is None and payload != prior_payload:
                            raise PerceptionError(
                                "confirmation disappeared without placement commitment"
                            )
                    elif control is not None:
                        if not newer:
                            raise PerceptionError(
                                "target confirmation requires a newer UI revision"
                            )
                        point = self._control_point(current, control)
                        if current.capture.stable:
                            click(point)
                            confirmed = True
                            previous = current
                            continue
                    elif payload != prior_payload:
                        raise PerceptionError("target click produced unexpected UI evidence")
                else:
                    if (
                        current.ui.phase != before.ui.phase
                        or current.ui.selected_action != action
                        or current.ui.active_tool is not None
                        or current.ui.confirmation is not None
                        or current.ui.dialog_stage not in (None, "willing")
                        or current.capture.target_mode is not None
                    ):
                        raise PerceptionError("incoming response changed selection or dialog")
                    rows = tuple(row for row in current.ui.trade_rows if row.row_id == row_id)
                    if len(rows) != 1:
                        raise PerceptionError("incoming offer row disappeared")
                    after_row = rows[0]
                    if after_row.revision != current.ui.revision or (
                        after_row.offer_id,
                        after_row.proposer,
                        after_row.give,
                        after_row.receive,
                    ) != (row.offer_id, row.proposer, row.give, row.receive):
                        raise PerceptionError("incoming offer row identity or terms changed")
                    own = tuple(
                        marker
                        for marker in after_row.responses
                        if marker.player_id == observation.player_id
                    )
                    if own and own[0].status == "rejected":
                        raise PerceptionError(
                            "acting recipient visibly rejected the incoming offer"
                        )
                    if own and own[0].status == "willing" and newer and current.capture.stable:
                        result = validate_acknowledgement(
                            before.ui,
                            current.ui,
                            observation,
                            game,
                            action,
                        )
                        result = replace(
                            result,
                            evidence=(
                                f"visible willing marker: row={row_id} offer={offer.offer_id} actor={observation.player_id} revision={current.ui.revision}",
                            ),
                        )
                        self.needs_review = False
                        return result
                    if payload != prior_payload and current.capture.stable:
                        raise PerceptionError(
                            "incoming response lacks the acting recipient's willing marker"
                        )
                previous = current
                if captures < self.max_captures:
                    self.wait(self.poll_interval)
            return failure("capture budget exhausted without verified outcome")
        except Exception as error:  # noqa: BLE001 - all ordinary boundary failures fail closed
            # BaseException deliberately propagates with the pre-input review lock intact.
            return failure(f"{type(error).__name__}: {error}")
