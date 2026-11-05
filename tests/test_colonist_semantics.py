"""Conservative semantic candidates from supplied visible ordered log tokens."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from colonist_rl.adapters.colonist.logs import normalize_snapshot

FIXTURES = Path(__file__).parent / "fixtures"


def text(value):
    return {"kind": "text", "text": value}


def icon(value):
    return {"kind": "icon", "alt": value}


def source(*tokens):
    return normalize_snapshot([{"tokens": tokens}]).rows[0]


def test_all_resource_icons_and_repetitions_count_only_visible_icons():
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_row

    row = source(
        text("Eilis got "),
        icon("Lumber"),
        icon("Brick"),
        icon("Wool"),
        icon("Grain"),
        icon("Ore"),
        icon("Ore"),
    )
    result = project_row(row, [VisiblePlayerName("Eilis", 2)])
    assert result.source is row
    assert result.issues == ()
    assert result.candidate.kind == "production"
    assert result.candidate.actor_name == "Eilis"
    assert result.candidate.actor == 2
    assert result.candidate.give.known == (0, 0, 0, 0, 0)
    assert result.candidate.receive.known == (1, 1, 1, 1, 2)
    assert result.candidate.receive.unknown_count == 0
    assert result.candidate.dice == ()


def test_split_adjacent_text_and_whitespace_match_without_changing_source():
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_row

    row = source(text(" \tEi"), text("lis\n g"), text("ot  "), icon("Ore"), text("\n"))
    result = project_row(row, (VisiblePlayerName("Eilis", 0),))
    assert result.candidate.receive.known == (0, 0, 0, 0, 1)
    assert result.candidate.actor == 0
    assert result.source.tokens[0].value == " \tEi"
    assert result.source is row


@pytest.mark.parametrize("duplicate_id", [0, 1])
def test_duplicate_name_bindings_never_select_even_identical_ids(duplicate_id):
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_row

    result = project_row(
        source(text("Eilis got "), icon("Ore")),
        [VisiblePlayerName("Eilis", 0), VisiblePlayerName("Eilis", duplicate_id)],
    )
    assert result.candidate.actor is None
    assert result.issues == ("ambiguous_actor",)


def test_missing_mapping_never_guesses_an_opponent_id():
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_row

    result = project_row(
        source(text("Unknown Person got "), icon("Brick")), [VisiblePlayerName("Eilis", 0)]
    )
    assert result.candidate.actor_name == "Unknown Person"
    assert result.candidate.actor is None
    assert result.issues == ("unresolved_actor",)


def test_card_backs_preserve_unknown_counts_in_both_resource_slots():
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_row

    result = project_row(
        source(
            text("Eilis gave bank "),
            icon("Ore"),
            icon("CardBack"),
            icon("card back"),
            text(" and took "),
            icon("cardback"),
        ),
        [VisiblePlayerName("Eilis", 0)],
    )
    assert result.candidate.give.known == (0, 0, 0, 0, 1)
    assert result.candidate.give.unknown_count == 2
    assert result.candidate.receive.known == (0, 0, 0, 0, 0)
    assert result.candidate.receive.unknown_count == 1
    assert result.issues == ("unknown_resources",)


@pytest.mark.parametrize("alt", ["MysteryOre", "DevelopmentCard", "road", "dice_2", ""])
def test_unsupported_resource_icon_never_completes_an_amount(alt):
    from colonist_rl.adapters.colonist.semantics import project_row

    row = source(text("Eilis got "), icon("Ore"), icon(alt))
    result = project_row(row, ())
    assert result.candidate is None
    assert "unsupported_resource_icon" in result.issues
    assert result.source is row


@pytest.mark.parametrize(
    "tokens",
    [
        [text("Eilis got Ore")],
        [text("Eilis got "), icon("Ore"), text(" extra")],
        [text("Eilis got "), icon("Ore"), text(" and "), icon("Brick")],
        [text("Eilis got ")],
        [text(" got "), icon("Ore")],
        [text("Eilis gave bank "), icon("Ore"), text(" and took ")],
        [text("Eilis gave bank "), icon("Ore"), text(" and took "), icon("Wool"), text("!")],
    ],
)
def test_resource_grammars_require_the_entire_row(tokens):
    from colonist_rl.adapters.colonist.semantics import project_row

    result = project_row(source(*tokens), ())
    assert result.candidate is None
    assert result.issues


@pytest.mark.parametrize(
    "tail",
    [
        [],
        [icon("dice_1")],
        [icon("dice_1"), icon("dice_2"), icon("dice_3")],
        [icon("dice_1"), icon("dice_7")],
        [icon("dice_1"), icon("Ore")],
        [icon("dice_1"), text("2")],
        [icon("dice_1"), icon("dice_2"), text(" extra")],
    ],
)
def test_roll_requires_exactly_two_recognized_faces_and_no_extra_material(tail):
    from colonist_rl.adapters.colonist.semantics import project_row

    result = project_row(source(text("Kati rolled "), *tail), ())
    assert result.candidate is None
    assert result.issues == ("invalid_dice",)


def test_roll_detaches_dice_and_never_produces_resource_counts():
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_row

    result = project_row(
        source(text("Kati rolled "), icon("dice_5"), text(" \n "), icon("dice_6")),
        [VisiblePlayerName("Kati", 3)],
    )
    assert result.candidate.kind == "roll"
    assert result.candidate.dice == (5, 6)
    assert result.candidate.give.known == result.candidate.receive.known == (0, 0, 0, 0, 0)
    assert result.issues == ()


@pytest.mark.parametrize("fixture", ["colonist-log-001.json", "colonist-log-002.json"])
def test_actual_dom_fixture_production_and_complete_roll_pairs(fixture):
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_snapshot

    raw = json.loads((FIXTURES / fixture).read_text())
    snapshot = normalize_snapshot(**raw)
    bindings = [
        VisiblePlayerName(name, index)
        for index, name in enumerate(["Eilis", "Kati", "Almire", "Donni#0773"])
    ]
    results = project_snapshot(snapshot, bindings)
    rolls = [item.candidate for item in results if item.candidate and item.candidate.kind == "roll"]
    production = [
        item.candidate for item in results if item.candidate and item.candidate.kind == "production"
    ]
    assert [(item.actor_name, item.actor, item.dice) for item in rolls] == [
        ("Kati", 1, (1, 2)),
        ("Donni#0773", 3, (3, 2)),
    ]
    assert [(item.actor_name, item.receive.known) for item in production] == [
        ("Eilis", (1, 0, 0, 0, 1)),
        ("Almire", (0, 0, 0, 0, 1)),
        ("Donni#0773", (0, 2, 0, 0, 0)),
    ]
    assert all(result.source is row for result, row in zip(results, snapshot.rows, strict=True))


def test_real_offer_is_not_a_transfer():
    from colonist_rl.adapters.colonist.semantics import project_snapshot

    raw = json.loads((FIXTURES / "colonist-log-001.json").read_text())
    rows = project_snapshot(normalize_snapshot(**raw), ())
    assert rows[-1].candidate is None
    assert rows[-1].source.tokens == normalize_snapshot(**raw).rows[-1].tokens


@pytest.mark.parametrize(
    "tokens",
    [
        [text("Eilis wants to give "), icon("Ore"), text(" for "), icon("Wool")],
        [text("Eilis accepted Donni#0773's trade")],
        [text("Eilis gave "), icon("Ore"), text(" and got "), icon("Wool"), text(" from Kati")],
        [text("Eilis built a Settlement "), icon("settlement"), text(" (+1 VP)")],
        [text("Eilis stole "), icon("CardBack"), text(" from Kati")],
        [text("Eilis discarded "), icon("Ore")],
        [text("Game Resumed")],
        [],
    ],
)
def test_unsupported_rows_retain_evidence_without_transfers(tokens):
    from colonist_rl.adapters.colonist.semantics import project_row

    row = source(*tokens)
    result = project_row(row, ())
    assert result.candidate is None
    assert result.source is row
    assert result.issues


def test_synthetic_fixture_is_explicitly_synthetic_and_projects_bank_and_unknowns():
    from colonist_rl.adapters.colonist.semantics import VisiblePlayerName, project_snapshot

    raw = json.loads((FIXTURES / "colonist-semantic-synthetic.json").read_text())
    assert raw["evidence_kind"] == "synthetic_ordered_tokens"
    results = project_snapshot(
        normalize_snapshot(raw["rows"]), [VisiblePlayerName("Donni#0773", 0)]
    )
    bank = results[0].candidate
    assert bank.kind == "bank_trade"
    assert bank.give.known == (4, 0, 0, 0, 0)
    assert bank.receive.known == (0, 0, 0, 1, 0)
    assert bank.actor == 0
    assert results[1].candidate.receive.known == (1, 1, 1, 1, 2)
    assert results[2].candidate.receive.unknown_count == 2
    assert results[3].candidate is None
    assert "unsupported_resource_icon" in results[3].issues


def test_snapshot_preserves_duplicate_rows_independently_without_event_identity():
    from colonist_rl.adapters.colonist.semantics import project_snapshot

    tokens = [text("Eilis got "), icon("Ore")]
    snapshot = normalize_snapshot([{"tokens": tokens}, {"tokens": tokens}])
    results = project_snapshot(snapshot, [])
    assert isinstance(results, tuple)
    assert len(results) == 2
    assert results[0].candidate == results[1].candidate
    assert results[0] is not results[1]
    assert results[0].source is snapshot.rows[0]
    assert results[1].source is snapshot.rows[1]
    assert results[0].source.row_id is results[1].source.row_id is None


def test_frozen_records_detach_sequences_and_validate_operation_invariants():
    from colonist_rl.adapters.colonist.semantics import (
        ResourceEvidence,
        RowProjection,
        SemanticCandidate,
        VisiblePlayerName,
    )

    counts = [1, 0, 0, 0, 0]
    evidence = ResourceEvidence(counts, 0)
    counts[0] = 99
    assert evidence.known == (1, 0, 0, 0, 0)
    empty = ResourceEvidence([0, 0, 0, 0, 0], 0)
    dice = [1, 2]
    candidate = SemanticCandidate("roll", "Kati", None, empty, empty, dice)
    dice[0] = 6
    assert candidate.dice == (1, 2)
    issues = ["unresolved_actor"]
    projection = RowProjection(
        source(text("Kati rolled "), icon("dice_1"), icon("dice_2")), candidate, issues
    )
    issues.clear()
    assert projection.issues == ("unresolved_actor",)
    for record, field, value in [
        (evidence, "unknown_count", 1),
        (candidate, "actor", 2),
        (projection, "candidate", None),
        (VisiblePlayerName("Kati", 1), "player_id", 2),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, value)
    for invalid in [[1, 2], [-1, 0, 0, 0, 0], [True, 0, 0, 0, 0]]:
        with pytest.raises(ValueError):
            ResourceEvidence(invalid, 0)
    for invalid in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            ResourceEvidence([0, 0, 0, 0, 0], invalid)
    for name, player_id in [("", 0), (" \t", 0), ("Kati", -1), ("Kati", True)]:
        with pytest.raises(ValueError):
            VisiblePlayerName(name, player_id)
    for kind, give, receive, faces in [
        ("offer", empty, empty, ()),
        ("roll", empty, empty, (1,)),
        ("roll", empty, empty, (1, 7)),
        ("roll", evidence, empty, (1, 2)),
        ("production", evidence, evidence, ()),
        ("production", empty, empty, ()),
        ("bank_trade", empty, evidence, ()),
        ("production", empty, evidence, (1, 2)),
    ]:
        with pytest.raises(ValueError):
            SemanticCandidate(kind, "Kati", 0, give, receive, faces)
    with pytest.raises(ValueError):
        SemanticCandidate("production", "", 0, empty, evidence, ())
    with pytest.raises(ValueError):
        SemanticCandidate("production", "Kati", True, empty, evidence, ())


def test_import_in_fresh_process_excludes_browser_simulator_policy_and_ledger():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import colonist_rl.adapters.colonist.semantics
forbidden = ('playwright', 'colonist_rl.browser', 'colonist_rl.simulator',
    'colonist_rl.policy', 'colonist_rl.beliefs',
    'colonist_rl.adapters.colonist.browser', 'colonist_rl.adapters.colonist.execution')
assert not any(name == prefix or name.startswith(prefix + '.')
    for name in sys.modules for prefix in forbidden)
print('offline imports only')
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": os.environ.get("PYTHONPATH", "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.stdout.strip() == "offline imports only"
