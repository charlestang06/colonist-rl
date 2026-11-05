"""Project supplied visible log tokens into conservative semantic candidates.

Candidates carry no event identity, event sequence, or ledger authority. This
module is offline and does not reconcile captures or infer hidden resources.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ...contracts import RESOURCES, ZERO, ResourceVector, resource_vector
from .logs import LogRow, LogSnapshot, LogToken


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("visible names must be nonblank text")
    return " ".join(value.split())


def _player_id(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("player IDs must be nonnegative integers")


def _ordered(values: Sequence, name: str) -> tuple:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an ordered sequence")
    return tuple(values)


@dataclass(frozen=True)
class VisiblePlayerName:
    name: str
    player_id: int

    def __post_init__(self):
        _name(self.name)
        _player_id(self.player_id)


@dataclass(frozen=True)
class ResourceEvidence:
    known: ResourceVector
    unknown_count: int

    def __post_init__(self):
        object.__setattr__(self, "known", resource_vector(self.known))
        if type(self.unknown_count) is not int or self.unknown_count < 0:
            raise ValueError("unknown_count must be a nonnegative integer")


_EMPTY = ResourceEvidence(ZERO, 0)


def _amount(evidence: ResourceEvidence) -> int:
    return sum(evidence.known) + evidence.unknown_count


@dataclass(frozen=True)
class SemanticCandidate:
    kind: Literal["production", "bank_trade", "roll"]
    actor_name: str
    actor: int | None
    give: ResourceEvidence
    receive: ResourceEvidence
    dice: tuple[int, int] | tuple[()]

    def __post_init__(self):
        _name(self.actor_name)
        if self.actor is not None:
            _player_id(self.actor)
        if not isinstance(self.give, ResourceEvidence) or not isinstance(
            self.receive, ResourceEvidence
        ):
            raise TypeError("give and receive must be ResourceEvidence")
        dice = _ordered(self.dice, "dice")
        object.__setattr__(self, "dice", dice)
        if self.kind == "roll":
            if (
                len(dice) != 2
                or any(type(face) is not int or not 1 <= face <= 6 for face in dice)
                or _amount(self.give)
                or _amount(self.receive)
            ):
                raise ValueError("roll requires exactly two die faces and no resource amounts")
        elif self.kind == "production":
            if dice or _amount(self.give) or not _amount(self.receive):
                raise ValueError("production requires only a nonempty receive amount")
        elif self.kind == "bank_trade":
            if dice or not _amount(self.give) or not _amount(self.receive):
                raise ValueError("bank_trade requires nonempty give and receive amounts")
        else:
            raise ValueError("unsupported candidate kind")


@dataclass(frozen=True)
class RowProjection:
    source: LogRow
    candidate: SemanticCandidate | None
    issues: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.source, LogRow):
            raise TypeError("source must be a LogRow")
        if self.candidate is not None and not isinstance(self.candidate, SemanticCandidate):
            raise TypeError("candidate must be a SemanticCandidate or None")
        issues = _ordered(self.issues, "issues")
        if any(not isinstance(issue, str) or not issue.strip() for issue in issues):
            raise ValueError("issues must contain nonblank issue codes")
        object.__setattr__(self, "issues", issues)


def _parts(row: LogRow) -> list[str | LogToken]:
    """Join adjacent text before whitespace folding; keep every icon separate."""
    parts: list[str | LogToken] = []
    text: list[str] = []
    for token in row.tokens:
        if token.kind == "text":
            text.append(token.value)
        else:
            folded = " ".join("".join(text).split())
            if folded:
                parts.append(folded)
            text.clear()
            parts.append(token)
    folded = " ".join("".join(text).split())
    if folded:
        parts.append(folded)
    return parts


def _resources(parts: list[str | LogToken]) -> tuple[ResourceEvidence | None, tuple[str, ...]]:
    if not parts or any(isinstance(part, str) for part in parts):
        return None, ("unparsed_row",)
    known = [0, 0, 0, 0, 0]
    unknown = 0
    for token in parts:
        if token.category == "resource" and token.resource in RESOURCES:
            known[RESOURCES.index(token.resource)] += 1
        elif token.category == "card_back":
            unknown += 1
        else:
            return None, ("unsupported_resource_icon",)
    return ResourceEvidence(resource_vector(known), unknown), ()


def _bindings(players: Sequence[VisiblePlayerName]) -> tuple[VisiblePlayerName, ...]:
    result = _ordered(players, "players")
    if any(not isinstance(player, VisiblePlayerName) for player in result):
        raise TypeError("players must contain VisiblePlayerName bindings")
    return result


def _actor(name: str, players: tuple[VisiblePlayerName, ...]) -> tuple[int | None, tuple[str, ...]]:
    matches = [player.player_id for player in players if _name(player.name) == name]
    if not matches:
        return None, ("unresolved_actor",)
    if len(matches) != 1:
        return None, ("ambiguous_actor",)
    return matches[0], ()


def project_row(row: LogRow, players: Sequence[VisiblePlayerName]) -> RowProjection:
    """Match a complete row, preserving its exact source and unresolved evidence.

    A missing or repeated visible-name binding leaves ``actor=None``. Card-back
    icons count as unknown resources; unsupported resource icons reject a row.
    """
    if not isinstance(row, LogRow):
        raise TypeError("row must be a LogRow")
    bindings = _bindings(players)
    parts = _parts(row)
    if not parts:
        return RowProjection(row, None, ("empty_row",))
    prefix = parts[0] if isinstance(parts[0], str) else ""
    match = re.fullmatch(r"(.+) (got|gave bank|rolled)", prefix)
    if match is None:
        issues = ("unparsed_row",)
        if any(token.kind == "icon" and token.category == "unknown" for token in row.tokens):
            issues += ("unsupported_icon",)
        return RowProjection(row, None, issues)
    name, verb = match.groups()
    tail = parts[1:]
    give, receive = _EMPTY, _EMPTY
    dice: tuple[int, int] | tuple[()] = ()
    if verb == "rolled":
        if len(tail) != 2 or any(
            not isinstance(token, LogToken)
            or token.category != "dice"
            or type(token.dice) is not int
            or not 1 <= token.dice <= 6
            for token in tail
        ):
            return RowProjection(row, None, ("invalid_dice",))
        dice = (tail[0].dice, tail[1].dice)
        kind = "roll"
    elif verb == "got":
        receive, issues = _resources(tail)
        if receive is None:
            return RowProjection(row, None, issues)
        kind = "production"
    else:
        delimiters = [index for index, part in enumerate(tail) if part == "and took"]
        if len(delimiters) != 1:
            return RowProjection(row, None, ("unparsed_row",))
        index = delimiters[0]
        give, give_issues = _resources(tail[:index])
        receive, receive_issues = _resources(tail[index + 1 :])
        if give is None or receive is None:
            return RowProjection(row, None, tuple(dict.fromkeys(give_issues + receive_issues)))
        kind = "bank_trade"
    actor, issues = _actor(name, bindings)
    if give.unknown_count or receive.unknown_count:
        issues += ("unknown_resources",)
    return RowProjection(row, SemanticCandidate(kind, name, actor, give, receive, dice), issues)


def project_snapshot(
    snapshot: LogSnapshot, players: Sequence[VisiblePlayerName]
) -> tuple[RowProjection, ...]:
    """Project every supplied row in order, including identical repeated rows."""
    if not isinstance(snapshot, LogSnapshot):
        raise TypeError("snapshot must be a LogSnapshot")
    bindings = _bindings(players)
    return tuple(project_row(row, bindings) for row in snapshot.rows)
