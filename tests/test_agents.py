"""Tests for RL Agents (Q-Learning and SARSA)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent


class TestQLearningAgent:
    """Test suite for Q-Learning Agent."""

    @pytest.fixture
    def agent(self):
        """Create a test Q-Learning agent."""
        return QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            seed=42
        )

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
        assert agent.epsilon == 1.0
        assert agent.n_actions == 3
        assert agent.q_table.shape == (3, 5, 3, 3)  # state_shape + n_actions

    def test_q_table_initialized_to_zeros(self, agent):
        """Test Q-table starts with zeros."""
        assert np.all(agent.q_table == 0)

    def test_select_action_returns_valid_action(self, agent):
        """Test action selection returns valid action."""
        state = np.array([1, 2, 1])  # Medium util, capacity 3, flat trend
        
        action = agent.select_action(state)
        
        assert action in [0, 1, 2]

    def test_select_action_exploration(self, agent):
        """Test that epsilon=1.0 leads to random actions."""
        state = np.array([1, 2, 1])
        agent.epsilon = 1.0
        
        # With epsilon=1.0, should explore (random actions)
        actions = [agent.select_action(state) for _ in range(100)]
        unique_actions = set(actions)
        
        # Should see multiple different actions
        assert len(unique_actions) > 1

    def test_select_action_exploitation(self, agent):
        """Test that epsilon=0 leads to greedy actions."""
        state = np.array([1, 2, 1])
        agent.epsilon = 0.0
        
        # Set Q-values so action 2 is best
        agent.q_table[1, 2, 1, 2] = 10.0
        
        action = agent.select_action(state)
        
        assert action == 2

    def test_update_modifies_q_table(self, agent):
        """Test that update modifies Q-table."""
        state = np.array([1, 2, 1])
        action = 1
        reward = 5.0
        next_state = np.array([1, 2, 1])
        
        old_q = agent.q_table[1, 2, 1, action]
        agent.update(state, action, reward, next_state, done=False)
        new_q = agent.q_table[1, 2, 1, action]
        
        assert new_q != old_q
        assert new_q > old_q  # Q should increase with positive reward

    def test_decay_epsilon(self, agent):
        """Test epsilon decay."""
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon == initial_epsilon * agent.epsilon_decay

    def test_epsilon_min_bound(self, agent):
        """Test epsilon doesn't go below minimum."""
        agent.epsilon = agent.epsilon_min
        agent.decay_epsilon()
        
        assert agent.epsilon >= agent.epsilon_min

    def test_greedy_action_returns_best(self, agent):
        """Test greedy action returns highest Q-value action."""
        state = np.array([0, 0, 0])
        
        # Set specific Q-values
        agent.q_table[0, 0, 0, 0] = 1.0
        agent.q_table[0, 0, 0, 1] = 5.0  # Best
        agent.q_table[0, 0, 0, 2] = 3.0
        
        action = agent.select_action(state, training=False)
        
        assert action == 1


class TestSARSAAgent:
    """Test suite for SARSA Agent."""

    @pytest.fixture
    def agent(self):
        """Create a test SARSA agent."""
        return SARSAAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            seed=42
        )

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
        assert agent.q_table.shape == (3, 5, 3, 3)

    def test_select_action_returns_valid_action(self, agent):
        """Test action selection returns valid action."""
        state = np.array([1, 2, 1])
        
        action = agent.select_action(state)
        
        assert action in [0, 1, 2]

    def test_update_uses_next_action(self, agent):
        """Test SARSA update uses actual next action (on-policy)."""
        state = np.array([1, 2, 1])
        action = 1
        reward = 5.0
        next_state = np.array([1, 2, 1])
        next_action = 0  # Specific next action
        
        # Set Q-value for next_action
        agent.q_table[1, 2, 1, next_action] = 10.0
        
        old_q = agent.q_table[1, 2, 1, action]
        agent.update(state, action, reward, next_state, next_action, done=False)
        new_q = agent.q_table[1, 2, 1, action]
        
        # Q should increase toward reward + gamma * Q(s', a')
        assert new_q != old_q

    def test_sarsa_vs_qlearning_update(self):
        """Test that SARSA and Q-Learning updates differ."""
        # SARSA uses actual next action, Q-Learning uses max
        q_agent = QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=0.5,
            seed=42
        )
        sarsa_agent = SARSAAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=0.5,
            seed=42
        )
        
        state = np.array([1, 2, 1])
        next_state = np.array([1, 2, 1])
        
        # Set different Q-values for next state actions
        q_agent.q_table[1, 2, 1, 0] = 1.0
        q_agent.q_table[1, 2, 1, 1] = 5.0  # Max for Q-learning
        q_agent.q_table[1, 2, 1, 2] = 2.0
        
        sarsa_agent.q_table[1, 2, 1, 0] = 1.0
        sarsa_agent.q_table[1, 2, 1, 1] = 5.0
        sarsa_agent.q_table[1, 2, 1, 2] = 2.0
        
        # Q-Learning update (uses max Q)
        q_agent.update(state, 0, 1.0, next_state, done=False)
        
        # SARSA update (uses actual next_action=2, not max)
        sarsa_agent.update(state, 0, 1.0, next_state, next_action=2, done=False)
        
        # Updates should be different because Q-learning uses max(5.0)
        # while SARSA uses Q(next_state, 2) = 2.0
        q_value_q = q_agent.q_table[1, 2, 1, 0]
        q_value_sarsa = sarsa_agent.q_table[1, 2, 1, 0]
        
        assert q_value_q != q_value_sarsa


class TestAgentSeeding:
    """Test reproducibility with seeding."""

    def test_qlearning_seed_reproducibility(self):
        """Test Q-Learning agent produces same results with same seed."""
        agent1 = QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            epsilon=0.5,
            seed=42
        )
        agent2 = QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            epsilon=0.5,
            seed=42
        )
        
        state = np.array([1, 2, 1])
        
        # Generate multiple actions
        actions1 = [agent1.select_action(state) for _ in range(10)]
        
        # Reset seed and generate again
        np.random.seed(42)
        agent2 = QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            epsilon=0.5,
            seed=42
        )
        actions2 = [agent2.select_action(state) for _ in range(10)]
        
        assert actions1 == actions2
