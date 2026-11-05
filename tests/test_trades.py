from dataclasses import FrozenInstanceError, fields, replace
from threading import Event
from time import monotonic

import pytest

from colonist_rl.contracts import DecisionContext, Observation, PublicPlayer, TradeOffer
from colonist_rl.trades import (
    TradeAssessment,
    TradeDecision,
    TradeOrchestrator,
    validate_trade_decision,
)

WOOD = (1, 0, 0, 0, 0)
WHEAT = (0, 0, 0, 1, 0)
ZERO = (0, 0, 0, 0, 0)


def offer(identifier="o1", proposer=0, recipient=1, give=WOOD, receive=WHEAT):
    return TradeOffer(identifier, proposer, recipient, give, receive)


def view(*, incoming=False, hand=(3, 2, 1, 2, 0), version=7, trade=None):
    actor = 1 if incoming else 0
    context = DecisionContext(
        "TRADE_RESPONSE" if incoming else "TURN",
        actor,
        0,
        version,
        has_rolled=True,
        offer=trade if trade is not None else (offer() if incoming else None),
        eligible_partners=(1, 2),
    )
    return Observation(
        player_id=actor,
        board=None,
        rules=None,
        context=context,
        players=tuple(PublicPlayer(i, 2, 5, 0) for i in range(4)),
        hand=hand,
        development=(0, 0, 0, 0, 0),
        buildings=(),
        roads=(),
        robber_tile=0,
        legal_actions=(),
    )


class FixedGenerator:
    def __init__(self, offers):
        self.offers = offers

    def propose(self, player_view, context, max_candidates):
        return self.offers


class FixedCritic:
    def __init__(self, benefit=2.0):
        self.benefit = benefit

    def score(self, player_view, context, offers):
        return [TradeAssessment(item, self.benefit, 0.2) for item in offers]


def enabled(generator=None, critic=None, **kwargs):
    return TradeOrchestrator(generator=generator, critic=critic, enabled=True, **kwargs)


def test_plugins_are_disabled_by_default_and_incoming_uses_binary_policy():
    class Explodes:
        def propose(self, *args):
            raise AssertionError("disabled generator ran")

        def score(self, *args):
            raise AssertionError("disabled critic ran")

    outgoing = view()
    controller = TradeOrchestrator(generator=Explodes(), critic=Explodes())
    assert controller.choose(outgoing, outgoing.context).kind == "NO_OFFER"
    incoming = view(incoming=True)
    controller = TradeOrchestrator(
        generator=Explodes(), critic=Explodes(), binary_callback=lambda v, c: True
    )
    decision = controller.choose(incoming, incoming.context)
    assert (decision.kind, decision.offer) == ("ACCEPT", incoming.context.offer)
    assert validate_trade_decision(decision, incoming, incoming.context)


def test_assessment_is_immutable_and_invalid_numbers_are_rejected():
    assessment = TradeAssessment(offer(), 1.0, 0.1, 0.75, 2.0)
    with pytest.raises(FrozenInstanceError):
        assessment.uncertainty = 1
    for kwargs in [
        {"benefit_relative_to_no_trade": float("nan")},
        {"uncertainty": -1},
        {"acceptance_probability": 1.1},
        {"benefit_if_accepted": float("inf")},
    ]:
        values = {"offer": offer(), "benefit_relative_to_no_trade": 1.0, "uncertainty": 0.1}
        values.update(kwargs)
        with pytest.raises(ValueError):
            TradeAssessment(**values)


def test_outgoing_scores_are_optional_and_no_trade_wins_nonpositive_scores():
    outgoing = view()
    controller = enabled(FixedGenerator([offer()]), FixedCritic(-0.1), include_assessments=True)
    decision = controller.choose(outgoing, outgoing.context)
    assert decision.kind == "NO_OFFER"
    assert any(
        item.offer is None and item.benefit_relative_to_no_trade == 0
        for item in decision.assessments
    )
    decision = enabled(FixedGenerator([offer()]), FixedCritic()).choose(outgoing, outgoing.context)
    assert (decision.kind, decision.offer) == ("PROPOSE", offer())
    assert decision.assessments == ()
    assert outgoing.hand == (3, 2, 1, 2, 0)


def test_generators_and_critics_are_independently_interchangeable():
    class YieldingGenerator:
        def propose(self, player_view, context, max_candidates):
            yield offer("second", recipient=2)
            yield offer("first")

    class QuantityCritic:
        def score(self, player_view, context, offers):
            return [TradeAssessment(item, float(item.recipient), 0) for item in offers]

    outgoing = view()
    choices = []
    for generator in [FixedGenerator([offer(), offer("second", recipient=2)]), YieldingGenerator()]:
        for critic in [FixedCritic(), QuantityCritic()]:
            decision = enabled(generator, critic).choose(outgoing, outgoing.context)
            assert decision.kind == "PROPOSE"
            choices.append(decision.offer)
    assert choices[-1].recipient == 2
    assessments = QuantityCritic().score(outgoing, outgoing.context, [offer()])
    assert assessments[0].benefit_relative_to_no_trade == 1


def test_cap_deduplicates_terms_and_bounds_unbounded_iteration():
    class InfiniteGenerator:
        iterations = 0

        def propose(self, player_view, context, max_candidates):
            while True:
                self.iterations += 1
                yield offer(str(self.iterations))

    generator = InfiniteGenerator()
    outgoing = view()
    decision = enabled(generator, FixedCritic(), max_candidates=3, include_assessments=True).choose(
        outgoing, outgoing.context
    )
    assert decision.kind == "PROPOSE"
    assert generator.iterations <= 12
    assert len([item for item in decision.assessments if item.offer]) == 1


def test_default_candidate_cap_limits_unique_candidates_before_scoring():
    class ManyGenerator:
        iterations = 0

        def propose(self, player_view, context, max_candidates):
            for quantity in range(1, 200):
                self.iterations += 1
                yield offer(str(quantity), give=(quantity, 0, 0, 0, 0))

    generator = ManyGenerator()
    outgoing = view(hand=(500, 0, 0, 0, 0))
    decision = enabled(generator, FixedCritic(), include_assessments=True).choose(
        outgoing, outgoing.context
    )
    assert generator.iterations == 32
    assert len(decision.assessments) == 33


@pytest.mark.parametrize(
    "bad_offer",
    [
        offer(give=ZERO),
        offer(receive=ZERO),
        offer(receive=WOOD),
        offer(give=(4, 0, 0, 0, 0)),
        offer(recipient=0),
        offer(recipient=3),
        offer(recipient=99),
        offer(proposer=2),
        offer(identifier=""),
    ],
)
def test_invalid_outgoing_offers_never_reach_selection(bad_offer):
    outgoing = view()
    decision = enabled(FixedGenerator([bad_offer]), FixedCritic()).choose(
        outgoing, outgoing.context
    )
    assert decision.kind == "NO_OFFER"


@pytest.mark.parametrize(
    "replacement",
    [
        {"has_rolled": False},
        {"kind": "DISCARD"},
        {"turn_owner": 2},
        {"actor": 2},
    ],
)
def test_proposals_are_only_available_to_the_rolled_turn_owner(replacement):
    outgoing = view()
    context = replace(outgoing.context, **replacement)
    outgoing = replace(outgoing, context=context)
    decision = enabled(FixedGenerator([offer()]), FixedCritic()).choose(outgoing, context)
    assert decision.kind == "NO_OFFER"


def test_incoming_uses_recipient_inventory_and_original_offer_direction():
    incoming = view(incoming=True, hand=WHEAT)
    before = incoming.hand
    decision = enabled(critic=FixedCritic()).choose(incoming, incoming.context)
    assert (decision.kind, decision.offer) == ("ACCEPT", incoming.context.offer)
    assert incoming.hand == before
    insufficient = view(incoming=True, hand=WOOD)
    decision = enabled(critic=FixedCritic(), binary_callback=lambda v, c: True).choose(
        insufficient, insufficient.context
    )
    assert decision.kind == "REJECT"


def test_models_receive_only_public_observation_and_own_hand():
    class PublicOnlyCritic:
        def score(self, player_view, context, offers):
            names = {field.name for field in fields(player_view)}
            assert "hands" not in names and "engine" not in names and "state" not in names
            assert player_view.hand == WHEAT
            assert all(not hasattr(player, "hand") for player in player_view.players)
            with pytest.raises(FrozenInstanceError):
                player_view.hand = WOOD
            return [TradeAssessment(item, 1, 0) for item in offers]

    incoming = view(incoming=True, hand=WHEAT)
    incoming = replace(
        incoming, players=tuple(replace(p, resource_count=0) for p in incoming.players)
    )
    assert enabled(critic=PublicOnlyCritic()).choose(incoming, incoming.context).kind == "ACCEPT"


def test_stale_context_and_changed_offer_identity_are_rejected_before_models():
    incoming = view(incoming=True)
    controller = enabled(critic=FixedCritic(), binary_callback=lambda v, c: True)
    stale = replace(incoming.context, version=6)
    assert controller.choose(incoming, stale).kind == "REJECT"
    wrong_identity = replace(incoming.context, offer=offer("another"))
    assert controller.choose(incoming, wrong_identity).kind == "REJECT"
    decision = controller.choose(incoming, incoming.context)
    next_context = replace(incoming.context, version=8)
    next_view = replace(incoming, context=next_context)
    assert not validate_trade_decision(decision, next_view, next_context)
    changed_context = replace(incoming.context, offer=offer("changed"))
    assert not validate_trade_decision(
        decision, replace(incoming, context=changed_context), changed_context
    )


@pytest.mark.parametrize("failure", ["raise", "malformed", "wrong_offer"])
def test_incoming_critic_failure_falls_back_to_injected_binary_response(failure):
    class BrokenCritic:
        def score(self, player_view, context, offers):
            if failure == "raise":
                raise RuntimeError("offline")
            if failure == "malformed":
                return [object()]
            return [TradeAssessment(offer("other"), 5, 0)]

    incoming = view(incoming=True)
    decision = enabled(critic=BrokenCritic(), binary_callback=lambda v, c: True).choose(
        incoming, incoming.context
    )
    assert decision.kind == "ACCEPT"


def test_outgoing_generator_or_critic_failure_returns_no_offer():
    class BrokenGenerator:
        def propose(self, *args):
            raise RuntimeError("offline")

    class BrokenCritic:
        def score(self, *args):
            raise RuntimeError("offline")

    outgoing = view()
    controllers = [
        enabled(BrokenGenerator(), FixedCritic()),
        enabled(FixedGenerator([offer()]), BrokenCritic()),
    ]
    for controller in controllers:
        assert controller.choose(outgoing, outgoing.context).kind == "NO_OFFER"


def test_blocking_plugins_return_within_budget_and_do_not_spawn_again_while_busy():
    release = Event()

    class BlockedCritic:
        calls = 0

        def score(self, player_view, context, offers):
            self.calls += 1
            release.wait()
            return [TradeAssessment(item, 10, 0) for item in offers]

    critic = BlockedCritic()
    incoming = view(incoming=True)
    controller = enabled(critic=critic, timeout_seconds=0.02, binary_callback=lambda v, c: False)
    try:
        start = monotonic()
        assert controller.choose(incoming, incoming.context).kind == "REJECT"
        assert monotonic() - start < 0.3
        assert controller.choose(incoming, incoming.context).kind == "REJECT"
        assert critic.calls == 1
    finally:
        release.set()


def test_binary_callback_is_also_bounded_and_invalid_output_rejects():
    release = Event()
    incoming = view(incoming=True)
    controller = TradeOrchestrator(
        binary_callback=lambda v, c: release.wait(), timeout_seconds=0.01
    )
    try:
        start = monotonic()
        assert controller.choose(incoming, incoming.context).kind == "REJECT"
        assert monotonic() - start < 0.3
    finally:
        release.set()
    controller = TradeOrchestrator(binary_callback=lambda v, c: "ACCEPT")
    assert controller.choose(incoming, incoming.context).kind == "REJECT"


def test_decision_revalidation_rejects_unsupported_kinds_and_modified_terms():
    outgoing = view()
    decision = enabled(FixedGenerator([offer()]), FixedCritic()).choose(outgoing, outgoing.context)
    assert validate_trade_decision(decision, outgoing, outgoing.context)
    bad = replace(decision, offer=offer(give=(8, 0, 0, 0, 0)))
    assert not validate_trade_decision(bad, outgoing, outgoing.context)
    with pytest.raises(ValueError):
        TradeDecision("EXECUTE", offer())


def test_generator_and_critic_can_be_enabled_independently():
    incoming = view(incoming=True)
    controller = TradeOrchestrator(critic=FixedCritic(), critic_enabled=True)
    assert controller.choose(incoming, incoming.context).kind == "ACCEPT"
    outgoing = view()
    controller = TradeOrchestrator(generator=FixedGenerator([offer()]), generator_enabled=True)
    assert controller.choose(outgoing, outgoing.context).kind == "NO_OFFER"


def test_no_trade_wins_exact_ties_and_expected_benefit_controls_selection():
    class OptimisticCritic:
        def score(self, player_view, context, offers):
            return [TradeAssessment(item, 0, 0.5, 0.001, 1000) for item in offers]

    outgoing = view()
    decision = enabled(FixedGenerator([offer()]), OptimisticCritic()).choose(
        outgoing, outgoing.context
    )
    assert decision.kind == "NO_OFFER"


def test_no_trade_diagnostics_exist_when_models_are_disabled():
    outgoing = view()
    decision = TradeOrchestrator(include_assessments=True).choose(outgoing, outgoing.context)
    assert len(decision.assessments) == 1
    assert decision.assessments[0].offer is None
    assert decision.assessments[0].benefit_relative_to_no_trade == 0


def test_changed_terms_under_same_incoming_id_invalidate_an_old_decision():
    incoming = view(incoming=True)
    decision = enabled(critic=FixedCritic()).choose(incoming, incoming.context)
    changed = replace(incoming.context, offer=offer(receive=(0, 0, 0, 2, 0)))
    assert not validate_trade_decision(decision, replace(incoming, context=changed), changed)


def test_a_blocking_generator_is_bounded_without_calling_critic():
    release = Event()

    class BlockingGenerator:
        def propose(self, player_view, context, max_candidates):
            release.wait()
            return [offer()]

    outgoing = view()
    controller = enabled(BlockingGenerator(), FixedCritic(), timeout_seconds=0.01)
    try:
        start = monotonic()
        assert controller.choose(outgoing, outgoing.context).kind == "NO_OFFER"
        assert monotonic() - start < 0.3
    finally:
        release.set()


@pytest.mark.parametrize(
    "vector", [(1, 0), (1.5, 0, 0, 0, 0), (-1, 0, 0, 0, 0), (True, 0, 0, 0, 0)]
)
def test_revalidation_defends_against_malformed_plugin_resource_vectors(vector):
    malformed = offer()
    object.__setattr__(malformed, "give", vector)
    outgoing = view()
    decision = enabled(FixedGenerator([malformed]), FixedCritic()).choose(
        outgoing, outgoing.context
    )
    assert decision.kind == "NO_OFFER"


def test_duplicate_offer_ids_with_different_terms_are_not_scored_twice():
    outgoing = view()
    candidates = [offer(), offer(receive=(0, 1, 0, 0, 0))]
    controller = enabled(FixedGenerator(candidates), FixedCritic(), include_assessments=True)
    decision = controller.choose(outgoing, outgoing.context)
    assert len(decision.assessments) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_candidates": 0},
        {"max_candidates": True},
        {"max_input_items": 0},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
    ],
)
def test_invalid_budgets_are_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        TradeOrchestrator(**kwargs)


def test_enabled_plugins_propose_from_real_post_roll_simulator_observation():
    from colonist_rl.opponents import HeuristicPolicy
    from colonist_rl.simulator import FourPlayerGame

    game = FourPlayerGame()
    current = game.reset(seed=17)
    policy = HeuristicPolicy(seed=17)
    for _ in range(200):
        if current.context.kind == "PLAY_TURN" and current.context.has_rolled and any(current.hand):
            break
        current = game.step(policy.act(current)).observation
    else:
        pytest.fail("never reached a funded post-roll turn")
    given_index = next(index for index, count in enumerate(current.hand) if count)
    received_index = (given_index + 1) % 5
    candidate = TradeOffer(
        "plugin-real-turn",
        current.player_id,
        current.context.eligible_partners[0],
        tuple(int(index == given_index) for index in range(5)),
        tuple(int(index == received_index) for index in range(5)),
    )

    class InvokedGenerator:
        calls = 0

        def propose(self, player_view, context, max_candidates):
            self.calls += 1
            return [candidate]

    generator = InvokedGenerator()
    decision = enabled(generator, FixedCritic()).choose(current, current.context)
    assert decision.kind == "PROPOSE"
    assert generator.calls == 1
    assert validate_trade_decision(decision, current, current.context)
    pending = game.offer_trade(decision.offer)
    assert pending.context.offer == candidate
    game.assert_resource_conservation()
