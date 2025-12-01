import pytest
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from gym_mmpp_env import MMPPEnv


def test_env_reset_and_step():
    """Verify reset returns a valid observation dict and step accepts a valid action.

    This test checks that reset() returns a dictionary containing 'arrivals' and 'state',
    and that calling step with a valid action returns the expected (obs, reward, done, info)
    tuple with appropriate types.
    """
    env = MMPPEnv(seed=123)
    obs = env.reset()
    assert isinstance(obs, dict)
    assert "arrivals" in obs and "state" in obs

    # valid action
    obs2, reward, done, info = env.step(0)
    assert isinstance(obs2, dict)
    assert isinstance(reward, float)
    assert done is False


def test_action_space_bounds():
    """Ensure actions outside the declared action_space raise a ValueError."""
    env = MMPPEnv(max_servers=5, seed=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(10)  # out of bounds


def test_seeding_reproducibility():
    """Check deterministic behavior when using the same RNG seed.

    Two environments created with the same seed should produce identical
    observations and rewards for the same sequence of actions.
    """
    env1 = MMPPEnv(seed=42)
    env2 = MMPPEnv(seed=42)

    obs1 = env1.reset()
    obs2 = env2.reset()
    assert np.array_equal(obs1["arrivals"], obs2["arrivals"]) and obs1["state"] == obs2["state"]

    a1, r1, _, _ = env1.step(1)
    a2, r2, _, _ = env2.step(1)
    assert np.array_equal(a1["arrivals"], a2["arrivals"]) and r1 == r2
