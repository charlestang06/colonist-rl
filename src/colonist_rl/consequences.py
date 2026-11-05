"""Deterministic public hand consequences for present legal action candidates.

Readiness describes normal-build assumptions, never an additional legal action.
The legal tuple establishes exchange feasibility, including an unknown bank.
"""

from __future__ import annotations

import torch

from .contracts import Action, Observation
from .settlement_sites import connected_settlement_site_count

CONSEQUENCE_SCHEMA = "public-consequence-v1"
CONSEQUENCE_DIM = 34
_RECIPES = (
    (1, 1, 0, 0, 0),  # road
    (1, 1, 1, 1, 0),  # settlement
    (0, 0, 0, 2, 3),  # city
    (0, 0, 1, 1, 1),  # development (resource sufficiency only)
)


def consequence_features(view: Observation, action: Action, *, device="cpu") -> torch.Tensor:
    """Encode only known transitions; unsupported legal kinds are all zero."""
    if action not in view.legal_actions:
        raise ValueError("consequence features require a present legal action")
    values = torch.zeros(CONSEQUENCE_DIM, device=device)
    if action.kind not in ("MARITIME_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "END_TURN"):
        return values
    hand = view.hand
    if action.kind in ("MARITIME_TRADE", "ACCEPT_TRADE"):
        if action.kind == "ACCEPT_TRADE":
            if view.context.offer is None:
                raise ValueError("a legal accept requires its public trade offer")
            give, receive = view.context.offer.for_player(view.player_id)
        else:
            give, receive = action.give, action.receive
        hand = tuple(h - g + r for h, g, r in zip(hand, give, receive))
        if any(h < 0 for h in hand):
            raise ValueError("a legal exchange cannot spend more than the observed hand")
    values[0] = 1
    values[1:6] = torch.tensor(hand, device=device) / 10
    sufficient = []
    for index, recipe in enumerate(_RECIPES):
        deficits = [max(0, cost - have) for cost, have in zip(recipe, hand)]
        values[6 + 5 * index : 11 + 5 * index] = torch.tensor(deficits, device=device) / 4
        sufficient.append(not any(deficits))
    values[26:30] = torch.tensor(sufficient, device=device)

    sites = connected_settlement_site_count(view)
    own = next(player for player in view.players if player.player_id == view.player_id)
    context = view.context
    normal_build = (
        context.kind in ("TURN", "PLAY_TURN")
        and context.has_rolled
        and context.actor == context.turn_owner == view.player_id
    )
    city_site = own.cities_left > 0 and any(
        building.owner == view.player_id and building.level == 1 for building in view.buildings
    )
    values[30] = min(sites, 8) / 8
    values[31] = normal_build and sufficient[1] and sites > 0 and own.settlements_left > 0
    values[32] = city_site
    values[33] = normal_build and city_site and sufficient[2]
    return values
