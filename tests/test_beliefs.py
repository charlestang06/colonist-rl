"""Belief updates must respect public evidence without reading simulator hands."""

import pytest

from colonist_rl.beliefs import BeliefInconsistency, BeliefLedger


def by_player(ledger):
    return {b.player_id: b for b in ledger.summaries()}


def test_exact_visible_changes_and_known_transfer_conserve_cards():
    ledger = BeliefLedger({0: (2, 1, 0, 0, 0), 3: (0, 0, 0, 1, 0)})
    ledger.apply_delta(0, (-1, 0, 0, 0, 2))
    ledger.transfer(0, 3, 1, resource="ORE")
    ledger.discard(3, 1, known=(0, 0, 0, 1, 0))
    beliefs = by_player(ledger)
    assert beliefs[0].expected == (1, 1, 0, 0, 1)
    assert beliefs[3].expected == (0, 0, 0, 0, 1)
    assert all(b.exact and b.lower == b.upper for b in beliefs.values())


def test_hidden_steal_is_weighted_by_victim_hand_and_jointly_correlated():
    ledger = BeliefLedger({0: (3, 1, 0, 0, 0), 1: (0, 0, 0, 0, 0)})
    ledger.transfer(0, 1, 1)
    beliefs = by_player(ledger)
    assert beliefs[1].expected == pytest.approx((0.75, 0.25, 0, 0, 0))
    assert beliefs[1].variance == pytest.approx((0.1875, 0.1875, 0, 0, 0))
    assert not beliefs[1].exact
    for particle in ledger.particles:
        assert tuple(a + b for a, b in zip(particle[0], particle[1])) == (3, 1, 0, 0, 0)
    ledger.reconcile({0: 3, 1: 1}, own_player=1, own_hand=(0, 1, 0, 0, 0))
    assert by_player(ledger)[0].lower == (3, 0, 0, 0, 0)
    assert all(b.exact for b in ledger.summaries())


def test_missing_history_and_sample_agreement_never_certify_exact_hands():
    ledger = BeliefLedger(totals={0: 15, 1: 0}, seed=4, max_particles=1)
    beliefs = by_player(ledger)
    assert not beliefs[0].exact
    assert beliefs[0].lower == (0, 0, 0, 0, 0)
    assert beliefs[0].upper == (15, 15, 15, 15, 15)
    assert beliefs[1].exact
    assert len(ledger.particles) <= 1
    twin = BeliefLedger(totals={0: 15, 1: 0}, seed=4, max_particles=1)
    assert ledger.particles == twin.particles
    assert ledger.summaries() == twin.summaries()


def test_hidden_discard_and_repeated_steals_are_nonnegative_and_bounded():
    ledger = BeliefLedger({0: (10, 10, 10, 10, 10), 1: (0, 0, 0, 0, 0)}, max_particles=8, seed=2)
    ledger.transfer(0, 1, 7)
    ledger.discard(0, 5)
    assert len(ledger.particles) <= 8
    for particle in ledger.particles:
        assert sum(particle[0]) == 38
        assert sum(particle[1]) == 7
        assert all(n >= 0 for hand in particle.values() for n in hand)
    assert not by_player(ledger)[1].exact
    assert sum(by_player(ledger)[0].expected) == pytest.approx(38)


def test_contradictory_visible_evidence_raises_without_mutating_state():
    ledger = BeliefLedger({0: (1, 0, 0, 0, 0), 1: (0, 0, 0, 0, 0)})
    before = ledger.summaries()
    with pytest.raises(BeliefInconsistency):
        ledger.apply_delta(0, (0, -1, 0, 0, 0))
    assert ledger.summaries() == before
    with pytest.raises(BeliefInconsistency):
        ledger.reconcile({0: 2, 1: 0})
    assert ledger.summaries() == before
    with pytest.raises(BeliefInconsistency):
        ledger.transfer(0, 1, 2)
    assert ledger.summaries() == before


def test_public_spending_tightens_unknown_hands_without_false_exactness():
    ledger = BeliefLedger(totals={0: 4}, seed=9, max_particles=2)
    ledger.apply_delta(0, (-1, 0, 0, 0, 0))
    belief = by_player(ledger)[0]
    assert sum(belief.expected) == pytest.approx(3)
    assert not belief.exact
    assert belief.upper == (3, 3, 3, 3, 3)
    assert all(hand[0][0] >= 0 for hand in ledger.particles)


def test_known_discard_must_match_public_count():
    ledger = BeliefLedger({0: (2, 0, 0, 0, 0)})
    with pytest.raises(ValueError):
        ledger.discard(0, 1, known=(2, 0, 0, 0, 0))


def test_conditioning_after_sampling_retains_known_global_inventory():
    ledger = BeliefLedger({0: (3, 1, 0, 0, 0), 1: (0, 0, 0, 0, 0)}, max_particles=1, seed=0)
    ledger.transfer(0, 1, 1)
    # Supply the outcome absent from this particle, forcing rejuvenation.
    absent = (0, 1, 0, 0, 0) if ledger.particles[0][1][0] else (1, 0, 0, 0, 0)
    ledger.reconcile({0: 3, 1: 1}, own_player=1, own_hand=absent)
    source = tuple(a - b for a, b in zip((3, 1, 0, 0, 0), absent))
    assert by_player(ledger)[0].lower == source
    assert by_player(ledger)[0].upper == source
    assert by_player(ledger)[0].exact
    assert ledger.particles[0][0] == source
    assert ledger.rejuvenations == 1


def test_missing_history_own_hand_can_be_observed_after_particle_depletion():
    ledger = BeliefLedger(totals={0: 8, 2: 10}, max_particles=1, seed=1)
    own_hand = (0, 0, 0, 0, 8)
    ledger.reconcile({0: 8, 2: 10}, own_player=0, own_hand=own_hand)
    assert by_player(ledger)[0].lower == own_hand
    assert by_player(ledger)[0].exact
    assert not by_player(ledger)[2].exact
    assert by_player(ledger)[2].upper == (10, 10, 10, 10, 10)


def test_public_spending_conditions_correlated_hidden_transfer():
    ledger = BeliefLedger({0: (1, 1, 0, 0, 0), 1: (0, 0, 0, 0, 0)})
    ledger.transfer(0, 1, 1)
    ledger.apply_delta(1, (-1, 0, 0, 0, 0))
    assert by_player(ledger)[0].lower == (0, 1, 0, 0, 0)
    assert by_player(ledger)[0].exact
    assert by_player(ledger)[1].exact


def test_all_resource_totals_can_only_be_reconciled_with_matching_own_hand():
    ledger = BeliefLedger(totals=[0, 4, 0], max_particles=1)
    before = ledger.summaries()
    with pytest.raises(BeliefInconsistency):
        ledger.reconcile([0, 4, 0], own_player=1, own_hand=(1, 0, 0, 0, 0))
    assert ledger.summaries() == before
