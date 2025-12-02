"""Tests for the Cloud Autoscaling Environment."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.cloud_autoscaling_env import CloudAutoscalingEnv


class TestCloudAutoscalingEnv:
    """Test suite for CloudAutoscalingEnv."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        workload = np.array([50.0] * 100)  # Constant workload
        return CloudAutoscalingEnv(workload_data=workload, seed=42)

    @pytest.fixture
    def env_variable_workload(self):
        """Create environment with variable workload."""
        np.random.seed(42)
        workload = 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, 200))
        return CloudAutoscalingEnv(workload_data=workload, seed=42)

    def test_reset_returns_valid_state(self, env):
        """Test that reset returns valid state and info."""
        state, info = env.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == 3  # (utilization_level, capacity_level, demand_trend)
        assert "utilization" in info
        assert "capacity" in info

    def test_action_space(self, env):
        """Test action space is correct."""
        assert env.action_space.n == 3  # scale down, hold, scale up

    def test_observation_space(self, env):
        """Test observation space shape."""
        assert env.observation_space.shape == (3,)

    def test_step_returns_valid_tuple(self, env):
        """Test step returns (state, reward, terminated, truncated, info)."""
        env.reset()
        result = env.step(1)  # Hold
        
        assert len(result) == 5
        state, reward, terminated, truncated, info = result
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_scale_up_increases_capacity(self, env):
        """Test that scale up action increases capacity."""
        env.reset()
        initial_capacity = env.current_capacity
        
        env.step(2)  # Scale up
        
        assert env.current_capacity == initial_capacity + 1

    def test_scale_down_decreases_capacity(self, env):
        """Test that scale down action decreases capacity."""
        env.reset()
        env.current_capacity = 3  # Ensure we have room to scale down
        initial_capacity = env.current_capacity
        
        env.step(0)  # Scale down
        
        assert env.current_capacity == initial_capacity - 1

    def test_hold_maintains_capacity(self, env):
        """Test that hold action maintains capacity."""
        env.reset()
        initial_capacity = env.current_capacity
        
        env.step(1)  # Hold
        
        assert env.current_capacity == initial_capacity

    def test_capacity_bounds(self, env):
        """Test capacity stays within bounds."""
        env.reset()
        
        # Scale down repeatedly
        for _ in range(20):
            env.step(0)
        assert env.current_capacity >= env.min_capacity
        
        # Scale up repeatedly
        for _ in range(20):
            env.step(2)
        assert env.current_capacity <= env.max_capacity

    def test_episode_terminates(self, env):
        """Test episode terminates when workload is exhausted."""
        env.reset()
        
        terminated = False
        for _ in range(200):
            _, _, terminated, _, _ = env.step(1)
            if terminated:
                break
        
        assert terminated

    def test_seeding_reproducibility(self):
        """Test same seed produces same results."""
        workload = np.array([50.0] * 50)
        
        env1 = CloudAutoscalingEnv(workload_data=workload.copy(), seed=42)
        env2 = CloudAutoscalingEnv(workload_data=workload.copy(), seed=42)
        
        state1, _ = env1.reset()
        state2, _ = env2.reset()
        
        assert np.array_equal(state1, state2)

    def test_reward_sla_violation(self, env_variable_workload):
        """Test that high utilization incurs SLA penalty."""
        env = env_variable_workload
        env.reset()
        env.current_capacity = 1  # Force high utilization
        env.current_demand = 100  # High demand
        
        _, reward, _, _, info = env.step(1)  # Hold with high utilization
        
        # Reward should be negative due to SLA violation
        assert reward < 0

    def test_info_contains_metrics(self, env):
        """Test info dict contains expected metrics."""
        env.reset()
        _, _, _, _, info = env.step(1)
        
        assert "utilization" in info
        assert "demand" in info
        assert "capacity" in info
        assert "total_cost" in info
        assert "sla_violations" in info
