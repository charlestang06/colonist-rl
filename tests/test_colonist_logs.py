"""Offline visible-token preservation and conservative virtualized-log history."""

import json
from pathlib import Path

import pytest


def row(text, *, row_id=None):
    return {"row_id": row_id, "tokens": [{"kind": "text", "text": text}]}


def test_normalization_preserves_order_raw_values_and_repeated_resources():
    from colonist_rl.adapters.colonist.logs import normalize_snapshot

    raw = [
        {
            "tokens": [
                {"kind": "text", "text": "Eilis"},
                {"kind": "text", "text": " got "},
                {"kind": "icon", "alt": "Ore"},
                {"kind": "text", "text": " "},
                {"kind": "icon", "alt": "Ore"},
                {"kind": "icon", "alt": "Lumber"},
                {"kind": "icon", "alt": "Brick"},
                {"kind": "icon", "alt": "Wool"},
                {"kind": "icon", "alt": "Grain"},
            ]
        }
    ]
    snapshot = normalize_snapshot(raw, capture_id="capture-1", captured_at="capture time")
    tokens = snapshot.rows[0].tokens
    assert [(token.kind, token.value) for token in tokens[:5]] == [
        ("text", "Eilis"),
        ("text", " got "),
        ("icon", "Ore"),
        ("text", " "),
        ("icon", "Ore"),
    ]
    assert [token.resource for token in tokens if token.kind == "icon"] == [
        "ORE",
        "ORE",
        "WOOD",
        "BRICK",
        "SHEEP",
        "WHEAT",
    ]
    assert snapshot.capture_id == "capture-1"
    assert snapshot.captured_at == "capture time"
    raw[0]["tokens"][0]["text"] = "modified"
    assert tokens[0].value == "Eilis"


def test_unknown_and_nonresource_icons_never_become_resource_facts():
    from colonist_rl.adapters.colonist.logs import normalize_snapshot

    labels = [
        "dice_2",
        "dice_6",
        "Development Card",
        "road",
        "longest road",
        "CardBack",
        "settlement",
        "MysteryOre",
        "",
        "dice_7",
    ]
    snapshot = normalize_snapshot([{"tokens": [{"kind": "icon", "alt": alt} for alt in labels]}])
    tokens = snapshot.rows[0].tokens
    assert [token.value for token in tokens] == labels
    assert [token.category for token in tokens] == [
        "dice",
        "dice",
        "development_card",
        "road",
        "longest_road",
        "card_back",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
    ]
    assert [token.dice for token in tokens[:2]] == [2, 6]
    assert all(token.resource is None for token in tokens)
    assert snapshot.rows[0].uncertain


def test_normalization_keeps_empty_rows_and_repeated_distinct_rows():
    from colonist_rl.adapters.colonist.logs import normalize_snapshot

    snapshot = normalize_snapshot([row("built a Road"), row("built a Road"), {"tokens": []}])
    assert len(snapshot.rows) == 3
    assert snapshot.rows[0] == snapshot.rows[1]
    assert snapshot.rows[0].row_id is None
    assert snapshot.rows[2].tokens == ()
    assert snapshot.rows[2].uncertain


@pytest.mark.parametrize(
    "raw",
    [
        [{"tokens": [{"kind": "html", "text": "<img>"}]}],
        [{"tokens": [{"kind": "icon"}]}],
        [{"tokens": [{"kind": "text", "text": 3}]}],
        [{"row_id": "", "tokens": []}],
        [row("one", row_id="same"), row("two", row_id="same")],
    ],
)
def test_malformed_tokens_and_duplicate_explicit_ids_are_rejected(raw):
    from colonist_rl.adapters.colonist.logs import normalize_snapshot

    with pytest.raises(ValueError):
        normalize_snapshot(raw)


def test_actual_dom_fixture_preserves_placeholder_rows_dice_and_unknown_icons():
    from colonist_rl.adapters.colonist.logs import normalize_snapshot

    path = Path(__file__).parent / "fixtures" / "colonist-log-001.json"
    captured = json.loads(path.read_text())
    snapshot = normalize_snapshot(**captured)
    assert len(snapshot.rows) == 10
    assert snapshot.rows[2].tokens == ()
    assert snapshot.rows[6].tokens == ()
    assert snapshot.rows[1].tokens[2].value == "settlement"
    assert snapshot.rows[1].tokens[2].category == "unknown"
    assert [token.dice for token in snapshot.rows[3].tokens if token.kind == "icon"] == [1, 2]
    assert all(row.row_id is None for row in snapshot.rows)


def test_initial_history_is_incomplete_and_keeps_repeated_distinct_rows():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    snapshot = normalize_snapshot([row("built a Road"), row("built a Road")], capture_id="one")
    update = history.ingest(snapshot)
    assert update.status == "initial"
    assert len(update.new_rows) == 2
    assert "initial_incomplete" in update.issues
    assert not update.history_complete
    assert history.snapshots == (snapshot,)


def test_unique_overlap_appends_only_new_ordered_rows():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A"), row("B"), row("C")]))
    update = history.ingest(normalize_snapshot([row("B"), row("C"), row("D"), row("D")]))
    assert update.status == "overlap"
    assert update.overlap_lengths == (2,)
    assert [item.tokens[0].value for item in update.new_rows] == ["D", "D"]
    assert update.unresolved_rows == ()
    assert not update.history_complete


def test_exact_capture_replay_is_idempotent_without_rebasing_latest_window():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    first = normalize_snapshot([row("A"), row("B")], capture_id="one")
    second = normalize_snapshot([row("B"), row("C")], capture_id="two")
    history.ingest(first)
    history.ingest(second)
    replay = history.ingest(first)
    assert replay.status == "unchanged"
    assert replay.new_rows == ()
    assert history.snapshots == (first, second)
    update = history.ingest(normalize_snapshot([row("C"), row("D")], capture_id="three"))
    assert update.status == "overlap"
    assert [item.tokens[0].value for item in update.new_rows] == ["D"]


def test_exact_anonymous_snapshot_replay_does_not_duplicate_rows():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A"), row("A")]))
    update = history.ingest(normalize_snapshot([row("A"), row("A")]))
    assert update.status == "unchanged"
    assert update.new_rows == ()
    assert len(history.snapshots) == 1


def test_disjoint_windows_report_gap_and_keep_unresolved_evidence():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A"), row("B")]))
    snapshot = normalize_snapshot([row("C"), row("D")])
    update = history.ingest(snapshot)
    assert update.status == "gap"
    assert update.overlap_lengths == ()
    assert update.new_rows == ()
    assert update.unresolved_rows == snapshot.rows
    assert "history_gap" in update.issues
    assert "initial_incomplete" in history.issues
    assert "history_gap" in history.issues
    assert history.snapshots[-1] == snapshot


def test_repeated_patterns_report_every_possible_overlap_without_guessing():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A"), row("A")], capture_id="one"))
    snapshot = normalize_snapshot([row("A"), row("A"), row("B")], capture_id="two")
    update = history.ingest(snapshot)
    assert update.status == "ambiguous"
    assert update.overlap_lengths == (1, 2)
    assert update.new_rows == ()
    assert update.unresolved_rows == snapshot.rows
    assert "ambiguous_overlap" in update.issues
    assert len(history.snapshots) == 2


def test_equal_content_in_different_captures_is_not_a_fabricated_event_identity():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A"), row("A")], capture_id="one"))
    update = history.ingest(normalize_snapshot([row("A"), row("A")], capture_id="two"))
    assert update.status == "ambiguous"
    assert update.overlap_lengths == (1, 2)
    assert update.new_rows == ()


def test_explicit_stable_ids_distinguish_equal_text_events():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("Road", row_id="1"), row("Road", row_id="2")]))
    update = history.ingest(normalize_snapshot([row("Road", row_id="2"), row("Road", row_id="3")]))
    assert update.status == "overlap"
    assert update.overlap_lengths == (1,)
    assert [item.row_id for item in update.new_rows] == ["3"]


def test_conflicting_stable_row_id_keeps_both_captures_and_reports_uncertainty():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("Road", row_id="1")]))
    snapshot = normalize_snapshot([row("City", row_id="1")])
    update = history.ingest(snapshot)
    assert update.status == "ambiguous"
    assert "row_id_conflict" in update.issues
    assert update.new_rows == ()
    assert update.unresolved_rows == snapshot.rows
    assert len(history.snapshots) == 2


def test_blank_placeholder_alone_is_not_an_overlap_anchor():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A"), {"tokens": []}]))
    snapshot = normalize_snapshot([{"tokens": []}, row("B")])
    update = history.ingest(snapshot)
    assert update.status == "ambiguous"
    assert update.overlap_lengths == (1,)
    assert "empty_overlap" in update.issues
    assert update.new_rows == ()


def test_reused_capture_id_with_changed_contents_is_rejected_without_mutation():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    snapshot = normalize_snapshot([row("A")], capture_id="one")
    history.ingest(snapshot)
    with pytest.raises(ValueError, match="capture_id"):
        history.ingest(normalize_snapshot([row("B")], capture_id="one"))
    assert history.snapshots == (snapshot,)


def test_actual_consecutive_dom_fixtures_have_one_overlap_and_one_new_offer():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    directory = Path(__file__).parent / "fixtures"
    history = LogHistory()
    history.ingest(
        normalize_snapshot(**json.loads((directory / "colonist-log-001.json").read_text()))
    )
    update = history.ingest(
        normalize_snapshot(**json.loads((directory / "colonist-log-002.json").read_text()))
    )
    assert update.status == "overlap"
    assert update.overlap_lengths == (8,)
    assert len(update.new_rows) == 1
    assert [token.value for token in update.new_rows[0].tokens] == [
        "Donni#0773",
        " wants to give ",
        "Brick",
        " ",
        "Brick",
        " for ",
        "Ore",
    ]
    assert not update.history_complete


def test_previous_explicit_row_id_cannot_reappear_as_a_new_event():
    from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

    history = LogHistory()
    history.ingest(normalize_snapshot([row("A", row_id="1"), row("B", row_id="2")]))
    snapshot = normalize_snapshot([row("B", row_id="2"), row("A", row_id="1")])
    update = history.ingest(snapshot)
    assert update.status == "ambiguous"
    assert "row_id_reappeared" in update.issues
    assert update.new_rows == ()
    assert update.unresolved_rows == snapshot.rows


def test_whitespace_only_rows_remain_uncertain_visible_evidence():
    from colonist_rl.adapters.colonist.logs import normalize_snapshot

    snapshot = normalize_snapshot([row(" \n\t")])
    assert snapshot.rows[0].tokens[0].value == " \n\t"
    assert snapshot.rows[0].uncertain
