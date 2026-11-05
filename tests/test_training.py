import numpy as np
import torch

from colonist_rl.policy import GraphPolicy
from colonist_rl.training import advantages, load_checkpoint, save_checkpoint


def test_gae_bootstraps_cutoff_and_stops_at_episode_boundary():
    adv, returns = advantages([0.0, 1.0], [0.2, 0.3], [0.3, 0.0], [False, True], lam=1.0)
    np.testing.assert_allclose(returns, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(adv, [0.8, 0.7], atol=1e-6)
    adv, returns = advantages([0.0], [0.2], [0.7], [True], lam=1.0)
    np.testing.assert_allclose(returns, [0.7])
    np.testing.assert_allclose(adv, [0.5])


def test_checkpoint_roundtrip_and_resume_optimizer(tmp_path):
    original = GraphPolicy(hidden_size=32)
    optimizer = torch.optim.Adam(original.parameters(), lr=3e-4)
    path = tmp_path / "model.pt"
    save_checkpoint(path, original, optimizer, steps=12)
    loaded, state = load_checkpoint(path)
    assert state["steps"] == 12
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])
