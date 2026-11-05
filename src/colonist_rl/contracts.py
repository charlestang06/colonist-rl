"""Immutable, backend-independent player-view contracts.

Resource order is always WOOD, BRICK, SHEEP, WHEAT, ORE. Engine objects
and hidden simulator state must never be stored in these records.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .board import BoardSpec, RulesConfig

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVELOPMENT_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
ResourceVector = tuple[int, int, int, int, int]
ZERO: ResourceVector = (0, 0, 0, 0, 0)


def resource_vector(values: Iterable[int]) -> ResourceVector:
    result = tuple(values)
    if len(result) != 5 or any(type(v) is not int or v < 0 for v in result):
        raise ValueError("resources must be five nonnegative integer counts")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class TradeOffer:
    """Give/receive are always from the proposer's perspective."""

    offer_id: str
    proposer: int
    recipient: int
    give: ResourceVector
    receive: ResourceVector

    def __post_init__(self):
        object.__setattr__(self, "give", resource_vector(self.give))
        object.__setattr__(self, "receive", resource_vector(self.receive))

    def for_player(self, player: int) -> tuple[ResourceVector, ResourceVector]:
        if player == self.proposer:
            return self.give, self.receive
        if player == self.recipient:
            return self.receive, self.give
        raise ValueError("player is not a party to this offer")


@dataclass(frozen=True)
class DecisionContext:
    kind: str
    actor: int
    turn_owner: int
    version: int
    turn_number: int = 0
    has_rolled: bool = False
    has_played_development_card: bool = False
    offer: TradeOffer | None = None
    eligible_partners: tuple[int, ...] = ()
    discard_remaining: int = 0
    free_roads: int = 0


@dataclass(frozen=True)
class Action:
    kind: str
    target: int | None = None
    resource: str | None = None
    second_resource: str | None = None
    partner: int | None = None
    give: ResourceVector = ZERO
    receive: ResourceVector = ZERO
    offer_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "give", resource_vector(self.give))
        object.__setattr__(self, "receive", resource_vector(self.receive))


@dataclass(frozen=True)
class Building:
    vertex: int
    owner: int
    level: int = 1


@dataclass(frozen=True)
class Road:
    edge: int
    owner: int


@dataclass(frozen=True)
class PublicPlayer:
    player_id: int
    victory_points: int
    resource_count: int
    development_count: int
    roads_left: int = 15
    settlements_left: int = 5
    cities_left: int = 4
    longest_road: int = 0
    played_knights: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False


@dataclass(frozen=True)
class ResourceBelief:
    player_id: int
    expected: tuple[float, ...]
    lower: ResourceVector
    upper: ResourceVector
    variance: tuple[float, ...]
    exact: bool = False


@dataclass(frozen=True)
class VisibleEvent:
    sequence: int
    kind: str
    actor: int | None = None
    recipient: int | None = None
    resource: str | None = None
    count: int = 0
    give: ResourceVector = ZERO
    receive: ResourceVector = ZERO


@dataclass(frozen=True)
class Observation:
    player_id: int
    board: BoardSpec
    rules: RulesConfig
    context: DecisionContext
    players: tuple[PublicPlayer, ...]
    hand: ResourceVector
    development: tuple[int, ...]
    buildings: tuple[Building, ...]
    roads: tuple[Road, ...]
    robber_tile: int
    legal_actions: tuple[Action, ...]
    beliefs: tuple[ResourceBelief, ...] = ()
    events: tuple[VisibleEvent, ...] = ()
    schema_version: int = 1
    bank: ResourceVector | None = None

    @property
    def own_victory_points(self) -> int:
        public = next(p.victory_points for p in self.players if p.player_id == self.player_id)
        return public + self.development[DEVELOPMENT_CARDS.index("VICTORY_POINT")]
