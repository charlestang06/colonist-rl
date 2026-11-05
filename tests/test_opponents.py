"""Baselines consume player ids as identities rather than tuple indices."""

from dataclasses import replace

import pytest

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import (
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    TradeOffer,
)
from colonist_rl.opponents import HeuristicPolicy


def baseline_view():
    board = BoardSpec.standard(seed=5)
    return Observation(
        player_id=2,
        board=board,
        rules=RulesConfig.standard(),
        context=DecisionContext("PLAY_TURN", 2, 2, 0, has_rolled=True),
        players=tuple(PublicPlayer(p, 2 + p, 4, 0) for p in range(4)),
        hand=(4, 0, 1, 1, 0),
        development=(0, 0, 0, 0, 0),
        buildings=(
            Building(board.tile_vertices[0][0], 1),
            Building(board.tile_vertices[4][0], 3),
            Building(board.tile_vertices[10][0], 2),
        ),
        roads=(),
        robber_tile=18,
        legal_actions=(),
    )


def relabel(view):
    ids = (10, 20, 30, 40)
    context = view.context
    offer = context.offer
    if offer is not None:
        offer = replace(offer, proposer=ids[offer.proposer], recipient=ids[offer.recipient])
    return replace(
        view,
        player_id=ids[view.player_id],
        players=tuple(replace(p, player_id=ids[p.player_id]) for p in view.players),
        buildings=tuple(replace(b, owner=ids[b.owner]) for b in view.buildings),
        context=replace(
            context, actor=ids[context.actor], turn_owner=ids[context.turn_owner], offer=offer
        ),
        legal_actions=tuple(
            replace(a, partner=ids[a.partner]) if a.partner is not None else a
            for a in view.legal_actions
        ),
    )


@pytest.mark.parametrize("scenario", ["discard", "maritime", "incoming_trade", "robber"])
def test_heuristic_actions_are_invariant_to_arbitrary_player_ids(scenario):
    view = baseline_view()
    if scenario == "discard":
        view = replace(
            view,
            context=replace(view.context, kind="DISCARD"),
            legal_actions=(
                Action("DISCARD_RESOURCE", resource="WOOD"),
                Action("DISCARD_RESOURCE", resource="SHEEP"),
            ),
        )
    elif scenario == "maritime":
        view = replace(
            view,
            legal_actions=(
                Action("END_TURN"),
                Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1)),
            ),
        )
    elif scenario == "incoming_trade":
        offer = TradeOffer("incoming", 1, 2, (1, 0, 0, 0, 0), (0, 0, 0, 1, 0))
        view = replace(
            view,
            context=replace(view.context, kind="TRADE_RESPONSE", turn_owner=1, offer=offer),
            legal_actions=(
                Action("ACCEPT_TRADE", offer_id="incoming"),
                Action("REJECT_TRADE", offer_id="incoming"),
            ),
        )
    else:
        view = replace(
            view,
            context=replace(view.context, kind="MOVE_ROBBER"),
            legal_actions=(
                Action("MOVE_ROBBER", target=0, partner=1),
                Action("MOVE_ROBBER", target=4, partner=3),
            ),
        )
    original = HeuristicPolicy(seed=8).act(view)
    actual = HeuristicPolicy(seed=8).act(relabel(view))
    expected = (
        replace(original, partner=(10, 20, 30, 40)[original.partner])
        if original.partner is not None
        else original
    )
    assert actual == expected
