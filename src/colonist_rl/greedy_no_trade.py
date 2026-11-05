"""Frozen plain greedy policy: immediate purchases, raw pips, and no trades."""

from __future__ import annotations

from torch import nn

from .contracts import RESOURCES, Action, Observation


def _pips(view: Observation, vertex: int | None) -> int:
    return sum(
        6 - abs(7 - tile.number)
        for tile, vertices in zip(view.board.tiles, view.board.tile_vertices)
        if vertex in vertices and tile.resource in RESOURCES and tile.number
    )


def _canonical(action: Action) -> tuple:
    """Resource order, then numeric target/victim; missing values sort last.

    Python's stable min preserves original menu position on an exact tie.
    """
    return (
        RESOURCES.index(action.resource) if action.resource in RESOURCES else len(RESOURCES),
        (
            RESOURCES.index(action.second_resource)
            if action.second_resource in RESOURCES
            else len(RESOURCES)
        ),
        (action.target is None, action.target),
        (action.partner is None, action.partner),
    )


class GreedyNoTrade(nn.Module):
    """Stateless deterministic ``plain-greedy-no-trade-v1`` baseline.

    The module interface permits use by the existing evaluator's actor adapter;
    there is no model forward, learned value, recurrent memory, or RNG use.
    """

    @property
    def identity(self) -> str:
        return "plain-greedy-no-trade-v1"

    def select_action(self, view: Observation) -> Action:
        legal = view.legal_actions
        for kind in ("REJECT_TRADE", "DISCARD_RESOURCE", "DISCARD", "MOVE_ROBBER", "ROLL"):
            actions = [action for action in legal if action.kind == kind]
            if not actions:
                continue
            if kind == "DISCARD_RESOURCE":
                return min(
                    actions,
                    key=lambda action: (
                        -view.hand[RESOURCES.index(action.resource)],
                        _canonical(action),
                    ),
                )
            return min(actions, key=_canonical)

        for kind in ("BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "BUY_DEVELOPMENT_CARD"):
            actions = [action for action in legal if action.kind == kind]
            if not actions:
                continue
            if kind in ("BUILD_CITY", "BUILD_SETTLEMENT"):
                return min(
                    actions, key=lambda action: (-_pips(view, action.target), _canonical(action))
                )
            return min(actions, key=_canonical)

        for kind in (
            "PLAY_ROAD_BUILDING",
            "PLAY_KNIGHT_CARD",
            "PLAY_YEAR_OF_PLENTY",
            "PLAY_MONOPOLY",
            "END_TURN",
        ):
            actions = [action for action in legal if action.kind == kind]
            if actions:
                return min(actions, key=_canonical)
        raise ValueError("no supported safe legal action for plain-greedy-no-trade-v1")

    def act(
        self, view: Observation, memory=None, *, deterministic: bool = False
    ) -> tuple[Action, None]:
        """Ignore supplied memory and sampling mode; never retain either."""
        return self.select_action(view), None

    def propose(self, view: Observation) -> None:
        return None
