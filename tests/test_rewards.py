import numpy as np

from colonist_rl.rewards import potential_rewards


def test_shaping_telescopes_and_award_cycles_cancel():
    initial = (2, 2, 2, 2)
    gained = (4, 2, 2, 2)
    lost = (2, 2, 2, 2)
    a = potential_rewards(initial, gained, 10)
    b = potential_rewards(gained, lost, 10)
    np.testing.assert_allclose(np.array(a) + b, np.zeros(4), atol=1e-10)
    end = potential_rewards(lost, (9, 10, 8, 5), 10, winner=1, terminated=True)
    np.testing.assert_allclose(np.array(a) + b + end, [-0.04, 0.96, -0.04, -0.04])


def test_unchanged_vp_and_no_winner_cutoff_are_not_losses():
    assert potential_rewards((2, 3), (2, 3), 10) == (0.0, 0.0)
