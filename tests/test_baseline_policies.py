"""Tests for Baseline Policies."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.baseline_policies import (
    RandomPolicy,
    ThresholdPolicy,
    ReactivePolicy,
    ProactivePolicy,
)


class TestRandomPolicy:
    """Test suite for Random Policy."""

    def test_returns_valid_action(self):
        """Test random policy returns valid action."""
        policy = RandomPolicy(seed=42)
        state = np.array([1, 2, 1])

        action = policy.select_action(state)

        assert action in [0, 1, 2]

    def test_produces_all_actions(self):
        """Test random policy produces all possible actions."""
        policy = RandomPolicy(seed=42)
        state = np.array([1, 2, 1])

        actions = [policy.select_action(state) for _ in range(100)]
        unique_actions = set(actions)

        assert unique_actions == {0, 1, 2}


class TestThresholdPolicy:
    """Test suite for Threshold Policy."""

    @pytest.fixture
    def policy(self):
        """Create threshold policy."""
        return ThresholdPolicy()

    def test_high_utilization_scales_up(self, policy):
        """Test high utilization triggers scale up."""
        state = np.array([2, 2, 1])  # High utilization

        action = policy.select_action(state)

        assert action == 2  # Scale up

    def test_low_utilization_scales_down(self, policy):
        """Test low utilization triggers scale down."""
        state = np.array([0, 2, 1])  # Low utilization

        action = policy.select_action(state)

        assert action == 0  # Scale down

    def test_medium_utilization_holds(self, policy):
        """Test medium utilization holds steady."""
        state = np.array([1, 2, 1])  # Medium utilization

        action = policy.select_action(state)

        assert action == 1  # Hold


class TestReactivePolicy:
    """Test suite for Reactive Policy."""

    @pytest.fixture
    def policy(self):
        """Create reactive policy."""
        return ReactivePolicy()

    def test_rising_demand_high_util_scales_up(self, policy):
        """Test rising demand with high utilization scales up."""
        state = np.array([2, 2, 2])  # High util, rising trend

        action = policy.select_action(state)

        assert action == 2  # Scale up

    def test_falling_demand_low_util_scales_down(self, policy):
        """Test falling demand with low utilization scales down."""
        state = np.array([0, 2, 0])  # Low util, falling trend

        action = policy.select_action(state)

        assert action == 0  # Scale down

    def test_flat_demand_medium_util_holds(self, policy):
        """Test flat demand with medium utilization holds."""
        state = np.array([1, 2, 1])  # Medium util, flat trend

        action = policy.select_action(state)

        assert action == 1  # Hold


class TestProactivePolicy:
    """Test suite for Proactive Policy."""

    @pytest.fixture
    def policy(self):
        """Create proactive policy."""
        return ProactivePolicy()

    def test_rising_demand_scales_up_preemptively(self, policy):
        """Test proactive policy scales up when demand is rising."""
        state = np.array([1, 2, 2])  # Medium util, rising trend

        action = policy.select_action(state)

        # Proactive should scale up even with medium utilization
        assert action == 2  # Scale up


class TestPolicyComparison:
    """Tests comparing different policies."""

    def test_all_policies_return_valid_actions(self):
        """Test all policies return valid actions."""
        policies = [
            RandomPolicy(seed=42),
            ThresholdPolicy(),
            ReactivePolicy(),
            ProactivePolicy(),
        ]

        state = np.array([1, 2, 1])

        for policy in policies:
            action = policy.select_action(state)
            assert action in [0, 1, 2], f"{policy} returned invalid action {action}"

    def test_policy_str_representation(self):
        """Test policies have string representation."""
        policies = [
            RandomPolicy(),
            ThresholdPolicy(),
            ReactivePolicy(),
            ProactivePolicy(),
        ]

        for policy in policies:
            assert len(str(policy)) > 0
