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

    def test_info_contains_per_step_sla_violation_key(self, env):
        """Test info dict contains 'sla_violation' (singular) for per-step tracking.

        This test was added to catch the bug where training code expected
        'sla_violation' but environment only provided 'sla_violations'.
        """
        env.reset()
        _, _, _, _, info = env.step(1)

        # Must have BOTH keys
        assert "sla_violation" in info, "Missing 'sla_violation' (per-step flag)"
        assert "sla_violations" in info, "Missing 'sla_violations' (cumulative count)"

    def test_sla_violation_is_per_step_flag(self, env):
        """Test that sla_violation is 0 or 1 (per-step flag, not cumulative)."""
        env.reset()

        for _ in range(10):
            _, _, terminated, _, info = env.step(1)
            if terminated:
                break
            # Per-step flag must be 0 or 1
            assert info["sla_violation"] in [0, 1], (
                f"sla_violation should be 0 or 1, got {info['sla_violation']}"
            )

    def test_sla_violations_is_cumulative(self, env):
        """Test that sla_violations is cumulative count (non-decreasing)."""
        env.reset()

        prev_violations = 0
        for _ in range(20):
            _, _, terminated, _, info = env.step(1)
            if terminated:
                break
            # Cumulative count must be non-decreasing
            assert info["sla_violations"] >= prev_violations, (
                f"sla_violations should be non-decreasing: {prev_violations} -> {info['sla_violations']}"
            )
            prev_violations = info["sla_violations"]

    def test_sla_violation_triggers_on_high_utilization(self):
        """Test that SLA violation is detected when utilization exceeds threshold."""
        # Create workload that will cause high utilization
        high_demand_workload = np.array([100.0] * 50)  # High demand
        env = CloudAutoscalingEnv(workload_data=high_demand_workload, seed=42)
        env.reset()

        # Force low capacity to trigger SLA violation
        env.current_capacity = 1  # Minimum capacity

        # Take a step - utilization will be high
        _, _, _, _, info = env.step(1)  # Hold

        utilization = info["utilization"]
        sla_threshold = env.sla_violation_threshold

        if utilization >= sla_threshold:
            assert info["sla_violation"] == 1, (
                f"Expected sla_violation=1 when utilization={utilization:.2%} >= threshold={sla_threshold:.2%}"
            )
        else:
            assert info["sla_violation"] == 0, (
                f"Expected sla_violation=0 when utilization={utilization:.2%} < threshold={sla_threshold:.2%}"
            )

    def test_sla_violation_count_matches_step_flags(self):
        """Test that cumulative sla_violations equals sum of per-step sla_violation flags."""
        # Use variable workload to trigger some violations
        workload = np.concatenate(
            [
                np.full(20, 30),  # Low demand
                np.full(20, 95),  # High demand - should cause violations
                np.full(20, 40),  # Medium demand
            ]
        )
        env = CloudAutoscalingEnv(workload_data=workload, seed=42)
        env.reset()

        step_violations_sum = 0
        cumulative_violations = 0

        for _ in range(55):
            _, _, terminated, _, info = env.step(1)  # Hold to let violations happen
            if terminated:
                break
            step_violations_sum += info["sla_violation"]
            cumulative_violations = info["sla_violations"]

        assert step_violations_sum == cumulative_violations, (
            f"Sum of per-step flags ({step_violations_sum}) != cumulative count ({cumulative_violations})"
        )

    def test_info_keys_consistency_across_steps(self, env):
        """Test that info dict has consistent keys across all steps."""
        env.reset()

        expected_keys = {
            "utilization",
            "demand",
            "capacity",
            "total_cost",
            "sla_violation",
            "sla_violations",
            "capacity_changes",
        }

        for _ in range(20):
            _, _, terminated, _, info = env.step(1)
            if terminated:
                break

            missing_keys = expected_keys - set(info.keys())
            assert not missing_keys, f"Missing keys in step info: {missing_keys}"

    def test_capacity_changes_tracking(self, env):
        """Test that capacity_changes is tracked correctly."""
        env.reset()

        # Do some scaling actions
        env.step(2)  # Scale up
        env.step(2)  # Scale up
        env.step(1)  # Hold (no change)
        _, _, _, _, info = env.step(0)  # Scale down

        # Should have 3 capacity changes (2 up + 1 down, hold doesn't count)
        assert info["capacity_changes"] == 3, (
            f"Expected 3 capacity changes, got {info['capacity_changes']}"
        )
