"""Pure, immutable records for caller-supplied, player-visible Colonist UI evidence.

UI revisions are independent of Observation.context.version. These records do
not recognize screenshots, execute gestures, or verify resource transfers.
Sequence inputs are detached to tuples; mutable nested payloads are rejected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

from ...contracts import (
    RESOURCES,
    ZERO,
    Action,
    Observation,
    ResourceVector,
    TradeOffer,
    resource_vector,
)
from ..errors import PerceptionError


def _integer(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _identity(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _optional_identity(value: object, name: str) -> None:
    if value is not None:
        _identity(value, name)


def _immutable(value: object) -> None:
    """Reject mutable payloads, including those inside shared frozen records."""
    if value is None or type(value) in (str, int, bool, float):
        return
    if type(value) is tuple:
        for item in value:
            _immutable(item)
        return
    if is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        for field in fields(value):
            _immutable(getattr(value, field.name))
        return
    raise TypeError("nested UI evidence must be immutable")


def _action(action: Action) -> None:
    if type(action) is not Action:
        raise TypeError("action must be an Action")
    _identity(action.kind, "action kind")
    for name in ("target", "partner"):
        if getattr(action, name) is not None:
            _integer(getattr(action, name), name)
    for name in ("resource", "second_resource"):
        value = getattr(action, name)
        if value is not None and (not isinstance(value, str) or value not in RESOURCES):
            raise ValueError(f"{name} must be a resource")
    _optional_identity(action.offer_id, "offer_id")
    _immutable(action)


def _sequence(value: Sequence, item_type: type, name: str) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if any(type(item) is not item_type for item in result):
        raise TypeError(f"{name} contains an invalid record")
    return result


@dataclass(frozen=True)
class UIControl:
    """Visible control binding.

    Fields: control_id: str, revision: int, action: Action,
    gesture: Literal['click', 'drag'], row_id: str | None = None,
    partner: int | None = None. Identities are supplied by the caller.
    """

    control_id: str
    revision: int
    action: Action
    gesture: Literal["click", "drag"]
    row_id: str | None = None
    partner: int | None = None

    def __post_init__(self):
        _identity(self.control_id, "control_id")
        _integer(self.revision, "revision")
        _action(self.action)
        if self.gesture not in ("click", "drag"):
            raise ValueError("gesture must be click or drag")
        _optional_identity(self.row_id, "row_id")
        if self.partner is not None:
            _integer(self.partner, "partner")


@dataclass(frozen=True)
class UITradeResponse:
    """Row-local marker: player_id: int, status: Literal['willing', 'rejected'].

    A willing marker describes interest only and never certifies a transfer.
    """

    player_id: int
    status: Literal["willing", "rejected"]

    def __post_init__(self):
        _integer(self.player_id, "player_id")
        if self.status not in ("willing", "rejected"):
            raise ValueError("unknown trade response status")


@dataclass(frozen=True)
class UITradeRow:
    """Visible offer with proposer-perspective terms.

    Fields: row_id: str, offer_id: str, proposer: int, give: ResourceVector,
    receive: ResourceVector, revision: int,
    responses: tuple[UITradeResponse, ...] = (),
    controls: tuple[UIControl, ...] = (). row_id is independent of offer_id;
    never derive it from a position, terms, or hash.
    """

    row_id: str
    offer_id: str
    proposer: int
    give: ResourceVector
    receive: ResourceVector
    revision: int
    responses: tuple[UITradeResponse, ...] = ()
    controls: tuple[UIControl, ...] = ()

    def __post_init__(self):
        _identity(self.row_id, "row_id")
        _identity(self.offer_id, "offer_id")
        _integer(self.proposer, "proposer")
        _integer(self.revision, "revision")
        object.__setattr__(self, "give", resource_vector(self.give))
        object.__setattr__(self, "receive", resource_vector(self.receive))
        responses = _sequence(self.responses, UITradeResponse, "responses")
        controls = _sequence(self.controls, UIControl, "controls")
        if len({response.player_id for response in responses}) != len(responses):
            raise ValueError("duplicate partner response markers")
        if len({control.control_id for control in controls}) != len(controls) or len(
            {(control.action, control.row_id, control.partner) for control in controls}
        ) != len(controls):
            raise ValueError("duplicate control binding")
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "controls", controls)


@dataclass(frozen=True)
class UIState:
    """Independent UI snapshot.

    Fields: revision: int, observation_version: int, phase: str,
    focused_control: str | None = None, active_tool: str | None = None,
    selected_action: Action | None = None, dialog_stage: str | None = None,
    confirmation: UIControl | None = None, trade_rows: tuple[UITradeRow, ...] = ().
    """

    revision: int
    observation_version: int
    phase: str
    focused_control: str | None = None
    active_tool: str | None = None
    selected_action: Action | None = None
    dialog_stage: str | None = None
    confirmation: UIControl | None = None
    trade_rows: tuple[UITradeRow, ...] = ()

    def __post_init__(self):
        _integer(self.revision, "revision")
        _integer(self.observation_version, "observation_version")
        _identity(self.phase, "phase")
        for name in ("focused_control", "active_tool", "dialog_stage"):
            _optional_identity(getattr(self, name), name)
        if self.selected_action is not None:
            _action(self.selected_action)
        if self.confirmation is not None and type(self.confirmation) is not UIControl:
            raise TypeError("confirmation must be a UIControl")
        rows = _sequence(self.trade_rows, UITradeRow, "trade_rows")
        if len({row.row_id for row in rows}) != len(rows):
            raise ValueError("duplicate row identities")
        controls = tuple(control for row in rows for control in row.controls)
        if self.confirmation is not None:
            controls += (self.confirmation,)
        if len({control.control_id for control in controls}) != len(controls):
            raise ValueError("duplicate visible control identities")
        object.__setattr__(self, "trade_rows", rows)


@dataclass(frozen=True)
class UIExecutionResult:
    """Caller-supplied evidence record, not an independent commitment verifier.

    Fields: status: Literal['committed', 'pending', 'blocked', 'ambiguous'],
    action: Action, starting_version: int, ui_revision: int,
    observation: Observation | None = None, evidence: tuple[str, ...] = (),
    reason: str | None = None. Committed requires later observation and evidence;
    blocked/ambiguous require a reason. Other statuses cannot carry observation.
    """

    status: Literal["committed", "pending", "blocked", "ambiguous"]
    action: Action
    starting_version: int
    ui_revision: int
    observation: Observation | None = None
    evidence: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        if self.status not in ("committed", "pending", "blocked", "ambiguous"):
            raise ValueError("unknown execution status")
        _action(self.action)
        _integer(self.starting_version, "starting_version")
        _integer(self.ui_revision, "ui_revision")
        evidence = _sequence(self.evidence, str, "evidence")
        for item in evidence:
            _identity(item, "evidence")
        object.__setattr__(self, "evidence", evidence)
        _optional_identity(self.reason, "reason")
        if self.status == "committed":
            if type(self.observation) is not Observation or not evidence:
                raise ValueError("committed requires observation and commitment evidence")
            _immutable(self.observation)
            _integer(self.observation.context.version, "observation version")
            if self.observation.context.version <= self.starting_version:
                raise ValueError("committed requires a later game observation")
        elif self.observation is not None:
            raise ValueError("only committed may carry an observation")
        if self.status in ("blocked", "ambiguous") and self.reason is None:
            raise ValueError("blocked or ambiguous requires a reason")


def _current(state: UIState, action: Action, expected_revision: int) -> None:
    if type(state) is not UIState or type(action) is not Action:
        raise PerceptionError("UI state and complete Action are required")
    try:
        _action(action)
    except (TypeError, ValueError) as error:
        raise PerceptionError("malformed action binding") from error
    if type(expected_revision) is not int or state.revision != expected_revision:
        raise PerceptionError("stale UI revision")
    if state.selected_action != action:
        raise PerceptionError("selection does not match the complete action")


def _confirmation_action(action: Action) -> bool:
    targets = ("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MOVE_ROBBER")
    if action.kind in targets:
        return action.target is not None
    if action.kind in ("PLAY_MONOPOLY", "DISCARD_RESOURCE"):
        return action.resource is not None
    if action.kind == "PLAY_YEAR_OF_PLENTY":
        return action.resource is not None
    return action.kind in (
        "ROLL",
        "END_TURN",
        "BUY_DEVELOPMENT_CARD",
        "PLAY_KNIGHT_CARD",
        "PLAY_ROAD_BUILDING",
    )


def validate_confirmation(state: UIState, action: Action, *, expected_revision: int) -> UIControl:
    """Return the exact current action-bound click confirmation, or fail closed."""
    _current(state, action, expected_revision)
    control = state.confirmation
    if control is None or not _confirmation_action(action):
        raise PerceptionError("unsupported or missing confirmation")
    if control.revision != state.revision:
        raise PerceptionError("stale UI revision")
    if control.action != action:
        raise PerceptionError("confirmation does not match the complete action")
    if (
        control.gesture != "click"
        or control.row_id is not None
        or control.partner != action.partner
    ):
        raise PerceptionError("confirmation has inconsistent control binding")
    return control


def _row(state: UIState, row_id: str) -> UITradeRow:
    rows = tuple(row for row in state.trade_rows if row.row_id == row_id)
    if len(rows) != 1 or rows[0].revision != state.revision:
        raise PerceptionError("missing, ambiguous, or stale trade row")
    return rows[0]


def _trade_control(row: UITradeRow, action: Action, partner: int) -> UIControl:
    controls = tuple(control for control in row.controls if control.action == action)
    if len(controls) != 1:
        raise PerceptionError("missing or ambiguous exact trade control")
    control = controls[0]
    if (
        control.row_id != row.row_id
        or control.partner != partner
        or control.revision != row.revision
        or control.gesture != "click"
    ):
        raise PerceptionError("trade control does not match row, partner, or revision")
    return control


def validate_trade_response(
    state: UIState,
    action: Action,
    offer: TradeOffer,
    *,
    row_id: str,
    expected_revision: int,
    player_id: int,
) -> UIControl:
    """Bind canonical incoming ACCEPT_TRADE/REJECT_TRADE to a shared offer.

    Control.partner identifies the acting recipient; shared response Actions
    contain only kind and offer_id. Terms always come from the supplied offer.
    """
    _current(state, action, expected_revision)
    if type(offer) is not TradeOffer or type(player_id) is not int or player_id < 0:
        raise PerceptionError("shared offer and acting recipient are required")
    if (
        type(offer.proposer) is not int
        or type(offer.recipient) is not int
        or offer.proposer < 0
        or offer.recipient < 0
        or action.kind not in ("ACCEPT_TRADE", "REJECT_TRADE")
        or action != Action(action.kind, offer_id=offer.offer_id)
        or offer.recipient != player_id
        or offer.proposer == player_id
    ):
        raise PerceptionError("response does not match the acting recipient and offer")
    row = _row(state, row_id)
    if (
        row.offer_id != offer.offer_id
        or row.proposer != offer.proposer
        or row.give != offer.give
        or row.receive != offer.receive
    ):
        raise PerceptionError("trade row does not match the exact shared offer")
    return _trade_control(row, action, player_id)


def validate_trade_finalization(
    state: UIState,
    action: Action,
    *,
    row_id: str,
    expected_revision: int,
    player_id: int,
) -> UIControl:
    """Return the chosen willing partner's exact proposer-side FINALIZE_TRADE control."""
    _current(state, action, expected_revision)
    row = _row(state, row_id)
    if (
        type(player_id) is not int
        or player_id < 0
        or row.proposer != player_id
        or action.kind != "FINALIZE_TRADE"
        or action.partner in (None, player_id)
        or action
        != Action(
            "FINALIZE_TRADE",
            partner=action.partner,
            give=row.give,
            receive=row.receive,
            offer_id=row.offer_id,
        )
        or row.give == ZERO
        or row.receive == ZERO
    ):
        raise PerceptionError("finalization does not match proposer, partner, and exact terms")
    responses = tuple(
        response for response in row.responses if response.player_id == action.partner
    )
    if len(responses) != 1 or responses[0].status != "willing":
        raise PerceptionError("chosen partner has no unambiguous willing marker")
    return _trade_control(row, action, action.partner)


def validate_acknowledgement(
    before_ui: UIState,
    after_ui: UIState,
    before_observation: Observation,
    after_observation: Observation,
    action: Action,
) -> UIExecutionResult:
    """Record pending UI progress only when the entire game Observation is equal."""
    if (
        type(before_ui) is not UIState
        or type(after_ui) is not UIState
        or type(before_observation) is not Observation
        or type(after_observation) is not Observation
        or type(action) is not Action
    ):
        raise PerceptionError("UI snapshots, game observations, and Action are required")
    _current(after_ui, action, after_ui.revision)
    if (
        after_ui.revision <= before_ui.revision
        or before_observation != after_observation
        or before_ui.observation_version != before_observation.context.version
        or after_ui.observation_version != after_observation.context.version
        or after_ui.selected_action != action
    ):
        raise PerceptionError("acknowledgement requires newer UI and exactly unchanged game state")
    return UIExecutionResult(
        "pending", action, before_observation.context.version, after_ui.revision
    )
