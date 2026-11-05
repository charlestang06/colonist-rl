import pytest

from colonist_rl.contracts import Action, TradeOffer, resource_vector


def test_resource_vectors_reject_negative_and_fractional_counts():
    for values in [(-1, 0, 0, 0, 0), (1.2, 0, 0, 0, 0), (0, 0)]:
        with pytest.raises(ValueError):
            resource_vector(values)


def test_offer_direction_is_explicit_and_actions_are_hashable():
    offer = TradeOffer("o1", 0, 1, (1, 0, 0, 0, 0), (0, 0, 0, 1, 0))
    assert offer.for_player(1) == (offer.receive, offer.give)
    assert offer.for_player(0) == (offer.give, offer.receive)
    assert len({Action("ROLL"), Action("ROLL")}) == 1


def test_action_copies_mutable_resource_vectors_and_remains_hashable():
    give = [1, 0, 0, 0, 0]
    receive = [0, 0, 0, 1, 0]
    action = Action("MARITIME_TRADE", give=give, receive=receive)
    give[0] = 99
    receive[3] = 99
    expected = Action("MARITIME_TRADE", give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0))
    assert action == expected
    assert len({action, expected}) == 1


@pytest.mark.parametrize("field", ["give", "receive"])
def test_action_rejects_invalid_resource_vectors(field):
    with pytest.raises(ValueError):
        Action("MARITIME_TRADE", **{field: [-1, 0, 0, 0, 0]})
