"""Offline visible DOM tokens and conservative virtualized-log overlap.

Inputs are explicitly supplied ordered rows, never HTML or browser state.
These records describe visible evidence, not parsed game events or ledgers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

TokenCategory = Literal[
    "text", "resource", "dice", "development_card", "road", "longest_road", "card_back", "unknown"
]

_RESOURCES = {
    "Lumber": "WOOD",
    "Brick": "BRICK",
    "Wool": "SHEEP",
    "Grain": "WHEAT",
    "Ore": "ORE",
}
_ICONS: dict[str, TokenCategory] = {
    "DevelopmentCard": "development_card",
    "Development Card": "development_card",
    "road": "road",
    "longestroad": "longest_road",
    "longest road": "longest_road",
    "CardBack": "card_back",
    "card back": "card_back",
    "cardback": "card_back",
}


@dataclass(frozen=True)
class LogToken:
    kind: Literal["text", "icon"]
    value: str
    category: TokenCategory
    resource: str | None = None
    dice: int | None = None


@dataclass(frozen=True)
class LogRow:
    tokens: tuple[LogToken, ...]
    row_id: str | None = None

    @property
    def uncertain(self) -> bool:
        """An empty row or unknown icon cannot supply complete visible evidence."""
        return not any(token.value.strip() for token in self.tokens) or any(
            token.category == "unknown" for token in self.tokens
        )


@dataclass(frozen=True)
class LogSnapshot:
    rows: tuple[LogRow, ...]
    capture_id: str | None = None
    captured_at: str | None = None


def _optional_text(value: Any, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be nonempty text or None")
    return value


def _sequence(value: Any, name: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an ordered sequence")
    return value


def _token(raw: Any) -> LogToken:
    if not isinstance(raw, Mapping):
        raise TypeError("tokens must be text or icon records")
    kind = raw.get("kind")
    field = "text" if kind == "text" else "alt"
    if (
        kind not in ("text", "icon")
        or set(raw) != {"kind", field}
        or not isinstance(raw[field], str)
    ):
        raise ValueError("tokens require kind plus exact text or image-alt text")
    value = raw[field]
    if kind == "text":
        return LogToken("text", value, "text")
    if value in _RESOURCES:
        return LogToken("icon", value, "resource", resource=_RESOURCES[value])
    if value in tuple(f"dice_{n}" for n in range(1, 7)):
        return LogToken("icon", value, "dice", dice=int(value[-1]))
    return LogToken("icon", value, _ICONS.get(value, "unknown"))


def normalize_snapshot(
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_id: str | None = None,
    captured_at: str | None = None,
) -> LogSnapshot:
    """Detach ordered ``{row_id?, tokens: [{kind, text|alt}]}`` records.

    Resource aliases and die faces describe icons only. Names, punctuation,
    empty rows, unknown labels, token boundaries, and repetitions are preserved.
    Optional IDs and capture times are supplied by the caller, never generated.
    """
    capture_id = _optional_text(capture_id, "capture_id")
    captured_at = _optional_text(captured_at, "captured_at")
    result = []
    ids = set()
    for raw in _sequence(rows, "rows"):
        if not isinstance(raw, Mapping) or set(raw) - {"row_id", "tokens"} or "tokens" not in raw:
            raise ValueError("rows require tokens and an optional explicit row_id")
        row_id = _optional_text(raw.get("row_id"), "row_id")
        if row_id is not None:
            if row_id in ids:
                raise ValueError("explicit row IDs must be unique within a snapshot")
            ids.add(row_id)
        result.append(
            LogRow(tuple(_token(token) for token in _sequence(raw["tokens"], "tokens")), row_id)
        )
    return LogSnapshot(tuple(result), capture_id, captured_at)


@dataclass(frozen=True)
class HistoryUpdate:
    status: Literal["initial", "unchanged", "overlap", "gap", "ambiguous"]
    snapshot: LogSnapshot
    new_rows: tuple[LogRow, ...] = ()
    unresolved_rows: tuple[LogRow, ...] = ()
    overlap_lengths: tuple[int, ...] = ()
    issues: tuple[str, ...] = ()
    history_complete: Literal[False] = False


class LogHistory:
    """Retain captures and report every possible suffix/prefix overlap.

    Equal row content is only an overlap candidate, never an event identity.
    Multiple possible overlaps and disjoint windows retain unresolved evidence
    without guessing which rows are new. Even a unique overlap cannot recover
    events missing before or between captures, so history is never complete.
    """

    def __init__(self):
        self._snapshots: list[LogSnapshot] = []
        self._updates: list[HistoryUpdate] = []
        self._captures: dict[str, LogSnapshot] = {}
        self._row_ids: dict[str, LogRow] = {}
        self._issues: list[str] = []

    @property
    def snapshots(self) -> tuple[LogSnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def updates(self) -> tuple[HistoryUpdate, ...]:
        return tuple(self._updates)

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(self._issues)

    def ingest(self, snapshot: LogSnapshot) -> HistoryUpdate:
        """Compare with the latest window without assigning event IDs or times.

        An exact replay of an explicit capture ID is idempotent even after newer
        captures. Without capture IDs, only an exact replay of the latest entire
        snapshot is treated as unchanged; no historical content set is used.
        """
        if snapshot.capture_id in self._captures:
            if self._captures[snapshot.capture_id] != snapshot:
                raise ValueError("capture_id was reused with different capture data")
            return HistoryUpdate("unchanged", snapshot)
        if self._snapshots and snapshot == self._snapshots[-1]:
            return HistoryUpdate("unchanged", snapshot)

        conflicts = any(
            row.row_id in self._row_ids and self._row_ids[row.row_id] != row
            for row in snapshot.rows
            if row.row_id is not None
        )
        if not self._snapshots:
            update = HistoryUpdate(
                "initial", snapshot, new_rows=snapshot.rows, issues=("initial_incomplete",)
            )
        elif conflicts:
            update = HistoryUpdate(
                "ambiguous", snapshot, unresolved_rows=snapshot.rows, issues=("row_id_conflict",)
            )
        else:
            before = self._snapshots[-1].rows
            overlaps = tuple(
                length
                for length in range(1, min(len(before), len(snapshot.rows)) + 1)
                if before[-length:] == snapshot.rows[:length]
            )
            if not overlaps:
                update = HistoryUpdate(
                    "gap", snapshot, unresolved_rows=snapshot.rows, issues=("history_gap",)
                )
            elif len(overlaps) > 1:
                update = HistoryUpdate(
                    "ambiguous",
                    snapshot,
                    unresolved_rows=snapshot.rows,
                    overlap_lengths=overlaps,
                    issues=("ambiguous_overlap",),
                )
            elif not any(
                row.row_id is not None or any(token.value.strip() for token in row.tokens)
                for row in snapshot.rows[: overlaps[0]]
            ):
                update = HistoryUpdate(
                    "ambiguous",
                    snapshot,
                    unresolved_rows=snapshot.rows,
                    overlap_lengths=overlaps,
                    issues=("empty_overlap",),
                )
            elif any(
                row.row_id in self._row_ids
                for row in snapshot.rows[overlaps[0] :]
                if row.row_id is not None
            ):
                update = HistoryUpdate(
                    "ambiguous",
                    snapshot,
                    unresolved_rows=snapshot.rows,
                    overlap_lengths=overlaps,
                    issues=("row_id_reappeared",),
                )
            else:
                update = HistoryUpdate(
                    "overlap",
                    snapshot,
                    new_rows=snapshot.rows[overlaps[0] :],
                    overlap_lengths=overlaps,
                )

        self._snapshots.append(snapshot)
        self._updates.append(update)
        if snapshot.capture_id is not None:
            self._captures[snapshot.capture_id] = snapshot
        for row in snapshot.rows:
            if row.row_id is not None:
                self._row_ids.setdefault(row.row_id, row)
        for issue in update.issues:
            if issue not in self._issues:
                self._issues.append(issue)
        return update
