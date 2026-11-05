"""Offline UI evidence contracts; every fixture is synthetic, not a capture reader."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from colonist_rl.adapters.colonist.ui import (
    UIControl,
    UIExecutionResult,
    UIState,
    UITradeResponse,
    UITradeRow,
    validate_acknowledgement,
    validate_confirmation,
    validate_trade_finalization,
    validate_trade_response,
)
from colonist_rl.adapters.errors import PerceptionError
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer, TradeOffer

GIVE = (0, 0, 0, 0, 1)
RECEIVE = (0, 0, 1, 0, 0)


def observation():
    return Observation(
        player_id=0,
        board=BoardSpec.from_hexes(((0, 0),)),
        rules=RulesConfig(),
        context=DecisionContext("TURN", 0, 0, 12),
        players=(PublicPlayer(0, 2, 3, 0), PublicPlayer(1, 2, 3, 0)),
        hand=(1, 0, 0, 1, 1),
        development=(0, 0, 0, 0, 0),
        buildings=(),
        roads=(),
        robber_tile=0,
        legal_actions=(Action("ROLL"),),
    )


def state(action=None, **changes):
    values = {"revision": 7, "observation_version": 12, "phase": "TURN", "selected_action": action}
    values.update(changes)
    return UIState(**values)


@pytest.fixture
def confirmed_state():
    action = Action("BUILD_SETTLEMENT", target=4)
    return state(action, confirmation=UIControl("house-confirm", 7, action, "click"))


@pytest.fixture
def incoming():
    action = Action("ACCEPT_TRADE", offer_id="offer-a")
    control = UIControl("incoming-check", 7, action, "click", row_id="row-a", partner=0)
    row = UITradeRow("row-a", "offer-a", 1, GIVE, RECEIVE, 7, controls=(control,))
    offer = TradeOffer("offer-a", 1, 0, GIVE, RECEIVE)
    return state(action, trade_rows=(row,)), action, offer


@pytest.fixture
def outgoing():
    action = Action("FINALIZE_TRADE", partner=1, give=GIVE, receive=RECEIVE, offer_id="offer-a")
    control = UIControl("partner-check", 7, action, "click", row_id="row-a", partner=1)
    row = UITradeRow(
        "row-a",
        "offer-a",
        0,
        GIVE,
        RECEIVE,
        7,
        responses=(UITradeResponse(1, "willing"),),
        controls=(control,),
    )
    return state(action, trade_rows=(row,)), action


def test_confirmation_returns_exact_bound_control(confirmed_state):
    assert (
        validate_confirmation(
            confirmed_state, Action("BUILD_SETTLEMENT", target=4), expected_revision=7
        )
        is confirmed_state.confirmation
    )


@pytest.mark.parametrize(
    "action",
    [
        Action("BUILD_SETTLEMENT", target=99),
        Action("BUILD_SETTLEMENT"),
        Action("BUILD_SETTLEMENT", target=4, resource="ORE"),
    ],
)
def test_confirmation_rejects_same_kind_other_complete_action(confirmed_state, action):
    with pytest.raises(PerceptionError):
        validate_confirmation(confirmed_state, action, expected_revision=7)


@pytest.mark.parametrize("revision", [6, 8, True, -1])
def test_confirmation_rejects_wrong_expected_revision(confirmed_state, revision):
    with pytest.raises(PerceptionError):
        validate_confirmation(
            confirmed_state, confirmed_state.selected_action, expected_revision=revision
        )


def test_confirmation_rejects_stale_control(confirmed_state):
    stale = replace(confirmed_state, confirmation=replace(confirmed_state.confirmation, revision=6))
    with pytest.raises(PerceptionError):
        validate_confirmation(stale, stale.selected_action, expected_revision=7)


@pytest.mark.parametrize("field", ["selected_action", "confirmation"])
def test_confirmation_rejects_incomplete_binding(confirmed_state, field):
    with pytest.raises(PerceptionError):
        validate_confirmation(
            replace(confirmed_state, **{field: None}),
            confirmed_state.selected_action,
            expected_revision=7,
        )


def test_confirmation_rejects_unknown_action():
    action = Action("UNSUPPORTED")
    ui = state(action, confirmation=UIControl("check", 7, action, "click"))
    with pytest.raises(PerceptionError):
        validate_confirmation(ui, action, expected_revision=7)


def test_confirmation_rejects_changed_resource():
    action = Action("PLAY_MONOPOLY", resource="ORE")
    ui = state(action, confirmation=UIControl("resource-check", 7, action, "click"))
    with pytest.raises(PerceptionError):
        validate_confirmation(ui, Action("PLAY_MONOPOLY", resource="SHEEP"), expected_revision=7)


def test_incoming_response_binds_shared_offer_and_recipient(incoming):
    ui, action, offer = incoming
    assert (
        validate_trade_response(ui, action, offer, row_id="row-a", expected_revision=7, player_id=0)
        is ui.trade_rows[0].controls[0]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"offer_id": "other"},
        {"give": (1, 0, 0, 0, 0)},
        {"receive": (0, 1, 0, 0, 0)},
        {"proposer": 2},
        {"revision": 6},
    ],
)
def test_incoming_response_rejects_replaced_row(incoming, changes):
    ui, action, offer = incoming
    changed = replace(ui, trade_rows=(replace(ui.trade_rows[0], **changes),))
    with pytest.raises(PerceptionError):
        validate_trade_response(
            changed, action, offer, row_id="row-a", expected_revision=7, player_id=0
        )


@pytest.mark.parametrize("changes", [{"recipient": 2}, {"proposer": 0}, {"offer_id": "new"}])
def test_incoming_response_rejects_other_shared_offer(incoming, changes):
    ui, action, offer = incoming
    with pytest.raises(PerceptionError):
        validate_trade_response(
            ui, action, replace(offer, **changes), row_id="row-a", expected_revision=7, player_id=0
        )


@pytest.mark.parametrize("field,value", [("row_id", "other"), ("partner", 2), ("revision", 6)])
def test_response_rejects_control_with_wrong_own_binding(incoming, field, value):
    ui, action, offer = incoming
    row = ui.trade_rows[0]
    row = replace(row, controls=(replace(row.controls[0], **{field: value}),))
    with pytest.raises(PerceptionError):
        validate_trade_response(
            replace(ui, trade_rows=(row,)),
            action,
            offer,
            row_id="row-a",
            expected_revision=7,
            player_id=0,
        )


def test_response_rejects_kind_only_control_match(incoming):
    ui, action, offer = incoming
    row = ui.trade_rows[0]
    wrong = replace(row.controls[0], action=replace(action, offer_id="other"))
    with pytest.raises(PerceptionError):
        validate_trade_response(
            replace(ui, trade_rows=(replace(row, controls=(wrong,)),)),
            action,
            offer,
            row_id="row-a",
            expected_revision=7,
            player_id=0,
        )


def test_rejection_is_an_exact_incoming_response(incoming):
    ui, _, offer = incoming
    action = Action("REJECT_TRADE", offer_id="offer-a")
    row = ui.trade_rows[0]
    control = replace(row.controls[0], action=action)
    ui = replace(ui, selected_action=action, trade_rows=(replace(row, controls=(control,)),))
    assert (
        validate_trade_response(ui, action, offer, row_id="row-a", expected_revision=7, player_id=0)
        is control
    )


def test_finalization_requires_proposer_willing_partner_and_exact_terms(outgoing):
    ui, action = outgoing
    assert (
        validate_trade_finalization(ui, action, row_id="row-a", expected_revision=7, player_id=0)
        is ui.trade_rows[0].controls[0]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"partner": 2},
        {"give": (1, 0, 0, 0, 0)},
        {"offer_id": "other"},
        {"kind": "ACCEPT_TRADE"},
        {"partner": None},
    ],
)
def test_finalization_rejects_wrong_complete_action(outgoing, changes):
    ui, action = outgoing
    with pytest.raises(PerceptionError):
        validate_trade_finalization(
            ui, replace(action, **changes), row_id="row-a", expected_revision=7, player_id=0
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"give": (1, 0, 0, 0, 0)},
        {"proposer": 1},
        {"revision": 6},
        {"responses": ()},
        {"responses": (UITradeResponse(1, "rejected"),)},
    ],
)
def test_finalization_rejects_replaced_or_unwilling_row(outgoing, changes):
    ui, action = outgoing
    with pytest.raises(PerceptionError):
        validate_trade_finalization(
            replace(ui, trade_rows=(replace(ui.trade_rows[0], **changes),)),
            action,
            row_id="row-a",
            expected_revision=7,
            player_id=0,
        )


def test_finalization_rejects_non_proposer(outgoing):
    ui, action = outgoing
    with pytest.raises(PerceptionError):
        validate_trade_finalization(ui, action, row_id="row-a", expected_revision=7, player_id=1)


@pytest.mark.parametrize("duplicate", ["rows", "controls", "responses"])
def test_ambiguous_duplicate_bindings_fail_closed(outgoing, duplicate):
    ui, action = outgoing
    row = ui.trade_rows[0]
    with pytest.raises((ValueError, TypeError)):
        if duplicate == "rows":
            ui = replace(ui, trade_rows=(row, replace(row, offer_id="other")))
        elif duplicate == "controls":
            ui = replace(
                ui,
                trade_rows=(
                    replace(
                        row,
                        controls=(
                            row.controls[0],
                            replace(row.controls[0], control_id="another-check"),
                        ),
                    ),
                ),
            )
        else:
            ui = replace(
                ui,
                trade_rows=(
                    replace(
                        row,
                        responses=(UITradeResponse(1, "willing"), UITradeResponse(1, "rejected")),
                    ),
                ),
            )
        validate_trade_finalization(ui, action, row_id="row-a", expected_revision=7, player_id=0)


@pytest.mark.parametrize("after_revision", [6, 7])
def test_acknowledgement_rejects_identical_or_lower_revision(after_revision):
    action = Action("PLAY_KNIGHT_CARD")
    with pytest.raises(PerceptionError):
        validate_acknowledgement(
            state(action),
            state(action, revision=after_revision),
            observation(),
            observation(),
            action,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"hand": (2, 0, 0, 1, 1)},
        {"robber_tile": 1},
        {"context": DecisionContext("TURN", 0, 0, 13)},
        {"legal_actions": (Action("END_TURN"),)},
    ],
)
def test_acknowledgement_rejects_changed_full_observation(changes):
    action = Action("PLAY_KNIGHT_CARD")
    with pytest.raises(PerceptionError):
        validate_acknowledgement(
            state(action),
            state(action, revision=8),
            observation(),
            replace(observation(), **changes),
            action,
        )


@pytest.mark.parametrize("which", ["before", "after", "selection"])
def test_acknowledgement_rejects_mismatched_stored_version_or_selection(which):
    action = Action("PLAY_KNIGHT_CARD")
    before, after = state(action), state(action, revision=8)
    if which == "before":
        before = replace(before, observation_version=11)
    elif which == "after":
        after = replace(after, observation_version=13)
    else:
        after = replace(after, selected_action=Action("PLAY_MONOPOLY", resource="ORE"))
    with pytest.raises(PerceptionError):
        validate_acknowledgement(before, after, observation(), observation(), action)


def test_willing_marker_with_unchanged_hand_is_pending(incoming):
    before, action, _ = incoming
    after_row = replace(
        before.trade_rows[0], revision=8, responses=(UITradeResponse(0, "willing"),), controls=()
    )
    after = state(Action("ACCEPT_TRADE", offer_id="offer-a"), revision=8, trade_rows=(after_row,))
    result = validate_acknowledgement(before, after, observation(), observation(), action)
    assert result.status == "pending"
    assert result.observation is None
    assert result.starting_version == 12
    assert result.ui_revision == 8
    assert result.action == Action("ACCEPT_TRADE", offer_id="offer-a")


@pytest.mark.parametrize("status", ["pending", "blocked", "ambiguous", "committed"])
def test_execution_result_records_all_statuses(status):
    later = replace(observation(), context=DecisionContext("TURN", 0, 0, 13))
    result = UIExecutionResult(
        status,
        Action("ROLL"),
        12,
        8,
        observation=later if status == "committed" else None,
        evidence=["visible-log:roll"] if status == "committed" else (),
        reason="requires review" if status in ("blocked", "ambiguous") else None,
    )
    assert result.status == status
    if status == "committed":
        assert result.evidence == ("visible-log:roll",)
        assert result.observation is later
    if status in ("blocked", "ambiguous"):
        assert result.reason == "requires review"


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "unknown"},
        {"status": "committed"},
        {"status": "committed", "observation": observation(), "evidence": ("log",)},
        {"observation": observation()},
        {"status": "blocked"},
        {"status": "ambiguous", "reason": ""},
        {"evidence": (["mutable"],)},
    ],
)
def test_execution_result_rejects_incomplete_or_mutable_evidence(changes):
    values = {
        "status": "pending",
        "action": Action("ROLL"),
        "starting_version": 12,
        "ui_revision": 8,
    }
    values.update(changes)
    with pytest.raises((ValueError, TypeError)):
        UIExecutionResult(**values)


def test_result_rejects_mutable_nested_observation():
    later = replace(observation(), context=DecisionContext("TURN", 0, 0, 13), players=[])
    with pytest.raises((ValueError, TypeError)):
        UIExecutionResult(
            "committed", Action("ROLL"), 12, 8, observation=later, evidence=("caller evidence",)
        )


def test_records_detach_sequences_and_are_frozen(outgoing):
    ui, action = outgoing
    row = ui.trade_rows[0]
    responses, controls = list(row.responses), list(row.controls)
    give, receive = list(GIVE), list(RECEIVE)
    detached = UITradeRow("detached", "offer-a", 0, give, receive, 7, responses, controls)
    rows = [detached]
    snapshot = state(action, trade_rows=rows)
    responses.clear()
    controls.clear()
    give[4] = 9
    receive[2] = 9
    rows.clear()
    assert snapshot.trade_rows == (detached,)
    assert detached.give == (0, 0, 0, 0, 1)
    assert detached.receive == (0, 0, 1, 0, 0)
    assert detached.responses == (UITradeResponse(1, "willing"),)
    assert detached.controls == (row.controls[0],)
    for record in (snapshot, detached, detached.responses[0], detached.controls[0]):
        with pytest.raises(FrozenInstanceError):
            record.arbitrary = "mutation"


@pytest.mark.parametrize("revision", [-1, True, 1.5, "7"])
def test_records_reject_noninteger_or_negative_revisions(revision):
    with pytest.raises((ValueError, TypeError)):
        state(revision=revision)
    with pytest.raises((ValueError, TypeError)):
        UIControl("check", revision, Action("ROLL"), "click")
    with pytest.raises((ValueError, TypeError)):
        UITradeRow("row", "offer", 0, GIVE, RECEIVE, revision)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UIControl("", 7, Action("ROLL"), "click"),
        lambda: UIControl("check", 7, Action("ROLL"), "keypress"),
        lambda: UITradeRow("row", "", 0, GIVE, RECEIVE, 7),
        lambda: UITradeRow("row", "offer", 0, [1], RECEIVE, 7),
        lambda: UITradeRow("row", "offer", 0, [True, 0, 0, 0, 0], RECEIVE, 7),
        lambda: UITradeResponse(1, "unknown"),
        lambda: UIControl("check", 7, Action(["mutable"]), "click"),
        lambda: state(focused_control=["mutable"]),
    ],
)
def test_records_reject_missing_identity_malformed_vectors_and_mutable_payloads(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_pure_module_imports_no_browser_execution_or_training_dependencies():
    code = """
import importlib, json, sys
importlib.import_module('colonist_rl.adapters.colonist.ui')
forbidden = ('playwright', 'torch', 'catanatron', 'colonist_rl.simulator',
             'colonist_rl.adapters.colonist.browser', 'colonist_rl.adapters.colonist.execution')
print(json.dumps([name for name in sys.modules
                 if any(name == x or name.startswith(x + '.') for x in forbidden)]))
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def fixture_control(data):
    values = dict(data)
    values["action"] = Action(**values["action"])
    return UIControl(**values)


def fixture_state(data):
    action = Action(**data["selected_action"]) if data.get("selected_action") else None
    rows = []
    for row_data in data.get("trade_rows", ()):
        row = dict(row_data)
        row["responses"] = tuple(UITradeResponse(**item) for item in row.get("responses", ()))
        row["controls"] = tuple(fixture_control(item) for item in row.get("controls", ()))
        rows.append(UITradeRow(**row))
    return state(
        action,
        revision=data["revision"],
        phase=data["phase"],
        focused_control=data.get("focused_control"),
        active_tool=data.get("active_tool"),
        dialog_stage=data.get("dialog_stage"),
        confirmation=fixture_control(data["confirmation"]) if data.get("confirmation") else None,
        trade_rows=rows,
    )


CATALOG_PATH = Path(__file__).parent / "fixtures" / "colonist-ui-stages.json"
CATALOG = json.loads(CATALOG_PATH.read_text())


@pytest.mark.parametrize("scenario", CATALOG["scenarios"], ids=lambda x: x["name"])
def test_synthetic_catalog_exact_stage_bindings(scenario):
    assert CATALOG["provenance"] == "synthetic"
    assert scenario["basis"] in ("controls_notes", "observed_image")
    assert (Path(__file__).parents[1] / scenario["controls"]).is_file()
    for screenshot in scenario["screenshots"]:
        assert (Path(__file__).parents[1] / screenshot).is_file()
    before, after = fixture_state(scenario["before"]), fixture_state(scenario["after"])
    action = Action(**scenario["action"])
    if scenario["binding"] == "confirmation":
        if scenario["valid"]:
            assert (
                validate_confirmation(after, action, expected_revision=after.revision)
                is after.confirmation
            )
        else:
            with pytest.raises(PerceptionError):
                validate_confirmation(after, action, expected_revision=after.revision)
    elif scenario["binding"] == "acknowledgement":
        if scenario["valid"]:
            assert (
                validate_acknowledgement(before, after, observation(), observation(), action).status
                == "pending"
            )
        else:
            with pytest.raises(PerceptionError):
                validate_acknowledgement(before, after, observation(), observation(), action)
    else:
        options = {
            "row_id": scenario["row_id"],
            "expected_revision": after.revision,
            "player_id": scenario["player_id"],
        }
        if scenario["binding"] == "response":
            validator = validate_trade_response
            args = (after, action, TradeOffer(**scenario["offer"]))
        else:
            validator = validate_trade_finalization
            args = (after, action)
        if scenario["valid"]:
            control = validator(*args, **options)
            row = next(row for row in after.trade_rows if row.row_id == scenario["row_id"])
            assert control is row.controls[0]
        else:
            with pytest.raises(PerceptionError):
                validator(*args, **options)


def test_confirmation_rejects_numerically_equal_malformed_action(confirmed_state):
    with pytest.raises(PerceptionError):
        validate_confirmation(
            confirmed_state, Action("BUILD_SETTLEMENT", target=4.0), expected_revision=7
        )


@pytest.mark.parametrize("changes", [{"proposer": True}, {"recipient": False}])
def test_response_rejects_numerically_equal_malformed_shared_player_ids(incoming, changes):
    ui, action, offer = incoming
    with pytest.raises(PerceptionError):
        validate_trade_response(
            ui, action, replace(offer, **changes), row_id="row-a", expected_revision=7, player_id=0
        )


def test_state_rejects_control_identity_shared_by_confirmation_and_trade_row(outgoing):
    ui, _ = outgoing
    action = Action("BUILD_SETTLEMENT", target=4)
    with pytest.raises(ValueError):
        replace(
            ui, selected_action=action, confirmation=UIControl("partner-check", 7, action, "click")
        )
