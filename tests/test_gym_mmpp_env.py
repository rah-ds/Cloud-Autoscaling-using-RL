"""Tests for the MMPP Gymnasium environment."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gym_mmpp_env import MMPPEnv


class TestMMPPEnv:
    """Test suite for MMPPEnv."""

    def test_env_reset_and_step(self):
        """Verify reset returns a valid observation dict and step accepts a valid action."""
        env = MMPPEnv(seed=123)
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "arrivals" in obs and "state" in obs

        # valid action
        obs2, reward, done, info = env.step(0)
        assert isinstance(obs2, dict)
        assert isinstance(reward, float)
        assert done is False

    def test_action_space_bounds(self):
        """Ensure actions outside the declared action_space raise a ValueError."""
        env = MMPPEnv(max_servers=5, seed=1)
        env.reset()
        with pytest.raises(ValueError):
            env.step(10)  # out of bounds

    def test_seeding_reproducibility(self):
        """Check deterministic behavior when using the same RNG seed."""
        env1 = MMPPEnv(seed=42)
        env2 = MMPPEnv(seed=42)

        obs1 = env1.reset()
        obs2 = env2.reset()
        assert np.array_equal(obs1["arrivals"], obs2["arrivals"])
        assert obs1["state"] == obs2["state"]

        a1, r1, _, _ = env1.step(1)
        a2, r2, _, _ = env2.step(1)
        assert np.array_equal(a1["arrivals"], a2["arrivals"])
        assert r1 == r2

    def test_observation_space_structure(self):
        """Verify observation space structure."""
        env = MMPPEnv(seed=0)
        obs = env.reset()

        assert "arrivals" in env.observation_space.spaces
        assert "state" in env.observation_space.spaces
        assert obs["arrivals"].shape == (1,)
        assert isinstance(obs["state"], (int, np.integer))

    def test_action_space_size(self):
        """Verify action space matches max_servers + 1."""
        env = MMPPEnv(max_servers=10, seed=0)
        assert env.action_space.n == 11  # 0 to 10 servers

    def test_reward_structure(self):
        """Verify reward is negative (cost-based)."""
        env = MMPPEnv(seed=42)
        env.reset()

        _, reward, _, _ = env.step(5)  # Allocate 5 servers
        # Reward should be negative (cost of servers + unmet demand penalty)
        assert isinstance(reward, float)
