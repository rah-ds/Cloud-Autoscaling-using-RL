"""
Pytest configuration and shared fixtures for Cloud Autoscaling RL tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session", autouse=True)
def setup_path():
    """Ensure src is in path for all tests."""
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def random_seed():
    """Provide consistent random seed."""
    return 42


@pytest.fixture
def sample_workload():
    """Generate sample workload data."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)
    workload = 50 + 30 * np.sin(t) + np.random.normal(0, 5, 200)
    return np.clip(workload, 10, 100)


@pytest.fixture
def constant_workload():
    """Generate constant workload data."""
    return np.array([50.0] * 100)


@pytest.fixture
def state_space_shape():
    """Default state space shape."""
    return (3, 5, 3)  # (utilization_levels, capacity_levels, trend_levels)


@pytest.fixture
def n_actions():
    """Default number of actions."""
    return 3  # scale_down, hold, scale_up
