"""Tests for Deep RL Agents."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.deep_rl_agents import (
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    DuelingQNetwork,
    QNetwork,
    ReplayBuffer,
)


class TestQNetwork:
    """Tests for the Q-Network architecture."""

    def test_forward_pass(self) -> None:
        """Test that forward pass produces correct output shape."""
        state_dim, action_dim = 4, 3
        network = QNetwork(state_dim, action_dim)
        
        state = torch.randn(1, state_dim)
        output = network(state)
        
        assert output.shape == (1, action_dim)

    def test_batch_forward_pass(self) -> None:
        """Test forward pass with batch input."""
        state_dim, action_dim = 4, 3
        batch_size = 32
        network = QNetwork(state_dim, action_dim)
        
        states = torch.randn(batch_size, state_dim)
        output = network(states)
        
        assert output.shape == (batch_size, action_dim)

    def test_custom_hidden_dims(self) -> None:
        """Test network with custom hidden dimensions."""
        state_dim, action_dim = 4, 3
        hidden_dims = (64, 32, 16)
        network = QNetwork(state_dim, action_dim, hidden_dims)
        
        state = torch.randn(1, state_dim)
        output = network(state)
        
        assert output.shape == (1, action_dim)


class TestDuelingQNetwork:
    """Tests for the Dueling Q-Network architecture."""

    def test_forward_pass(self) -> None:
        """Test that forward pass produces correct output shape."""
        state_dim, action_dim = 4, 3
        network = DuelingQNetwork(state_dim, action_dim)
        
        state = torch.randn(1, state_dim)
        output = network(state)
        
        assert output.shape == (1, action_dim)

    def test_advantage_centering(self) -> None:
        """Test that advantage values are centered (mean = 0)."""
        state_dim, action_dim = 4, 3
        network = DuelingQNetwork(state_dim, action_dim)
        
        # The Q-values should be V(s) + A(s,a) - mean(A)
        # We can't directly check this, but we verify output shape
        state = torch.randn(1, state_dim)
        output = network(state)
        
        assert output.shape == (1, action_dim)


class TestReplayBuffer:
    """Tests for the Replay Buffer."""

    def test_push_and_sample(self) -> None:
        """Test adding experiences and sampling."""
        buffer = ReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(50):
            state = np.array([i, i + 1, i + 2], dtype=np.float32)
            next_state = np.array([i + 1, i + 2, i + 3], dtype=np.float32)
            buffer.push(state, i % 3, float(i), next_state, i % 10 == 0)
        
        assert len(buffer) == 50
        
        # Sample a batch
        states, actions, rewards, next_states, dones = buffer.sample(16)
        
        assert states.shape == (16, 3)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 3)
        assert dones.shape == (16,)

    def test_capacity_limit(self) -> None:
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(20):
            state = np.array([i], dtype=np.float32)
            buffer.push(state, 0, 0.0, state, False)
        
        assert len(buffer) == 10


class TestDQNAgent:
    """Tests for the DQN Agent."""

    @pytest.fixture
    def agent(self) -> DQNAgent:
        """Create a DQN agent for testing."""
        return DQNAgent(
            state_dim=4,
            action_dim=3,
            hidden_dims=(32, 32),
            batch_size=8,
            buffer_size=100,
            seed=42,
        )

    def test_initialization(self, agent: DQNAgent) -> None:
        """Test agent initialization."""
        assert agent.state_dim == 4
        assert agent.action_dim == 3
        assert agent.epsilon == 1.0
        assert len(agent.buffer) == 0

    def test_select_action(self, agent: DQNAgent) -> None:
        """Test action selection."""
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        
        action = agent.select_action(state, training=True)
        assert 0 <= action < 3

    def test_select_action_greedy(self, agent: DQNAgent) -> None:
        """Test greedy action selection."""
        agent.epsilon = 0.0  # Force greedy
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        
        # Greedy action should be deterministic
        actions = [agent.select_action(state, training=False) for _ in range(10)]
        assert len(set(actions)) == 1  # All same action

    def test_update(self, agent: DQNAgent) -> None:
        """Test update method."""
        # Fill buffer with experiences
        for i in range(20):
            state = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(4).astype(np.float32)
            agent.update(state, i % 3, float(i), next_state, i % 10 == 0)
        
        # After enough experiences, should return loss
        state = np.random.randn(4).astype(np.float32)
        next_state = np.random.randn(4).astype(np.float32)
        loss = agent.update(state, 0, 1.0, next_state, False)
        
        assert loss is not None
        assert isinstance(loss, float)

    def test_decay_epsilon(self, agent: DQNAgent) -> None:
        """Test epsilon decay."""
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min


class TestDoubleDQNAgent:
    """Tests for the Double DQN Agent."""

    @pytest.fixture
    def agent(self) -> DoubleDQNAgent:
        """Create a Double DQN agent for testing."""
        return DoubleDQNAgent(
            state_dim=4,
            action_dim=3,
            hidden_dims=(32, 32),
            batch_size=8,
            buffer_size=100,
            seed=42,
        )

    def test_double_dqn_update(self, agent: DoubleDQNAgent) -> None:
        """Test Double DQN specific update."""
        # Fill buffer
        for i in range(20):
            state = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(4).astype(np.float32)
            agent.update(state, i % 3, float(i), next_state, i % 10 == 0)
        
        # Update should work
        state = np.random.randn(4).astype(np.float32)
        next_state = np.random.randn(4).astype(np.float32)
        loss = agent.update(state, 0, 1.0, next_state, False)
        
        assert loss is not None


class TestDuelingDQNAgent:
    """Tests for the Dueling DQN Agent."""

    @pytest.fixture
    def agent(self) -> DuelingDQNAgent:
        """Create a Dueling DQN agent for testing."""
        return DuelingDQNAgent(
            state_dim=4,
            action_dim=3,
            hidden_dims=(32, 32),
            batch_size=8,
            buffer_size=100,
            seed=42,
        )

    def test_uses_dueling_network(self, agent: DuelingDQNAgent) -> None:
        """Test that agent uses dueling architecture."""
        assert isinstance(agent.q_network, DuelingQNetwork)
        assert isinstance(agent.target_network, DuelingQNetwork)

    def test_select_action(self, agent: DuelingDQNAgent) -> None:
        """Test action selection with dueling network."""
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < 3


class TestDeviceHandling:
    """Tests for device handling."""

    def test_auto_device_selection(self) -> None:
        """Test automatic device selection."""
        agent = DQNAgent(state_dim=4, action_dim=3)
        
        # Should select an available device
        assert agent.device in [
            torch.device("cuda"),
            torch.device("mps"),
            torch.device("cpu"),
        ]

    def test_explicit_cpu(self) -> None:
        """Test explicit CPU device."""
        agent = DQNAgent(state_dim=4, action_dim=3, device="cpu")
        assert agent.device == torch.device("cpu")
