"""
SARSA Agent for Cloud Autoscaling
"""

import numpy as np
from typing import Tuple, Optional
import pickle


class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) agent with epsilon-greedy exploration.
    
    SARSA update rule:
    Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    Key difference from Q-Learning: Uses actual next action a' instead of max_a' Q(s',a')
    """
    
    def __init__(
        self,
        state_space_shape: Tuple[int, ...],
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize SARSA agent.
        
        Args:
            state_space_shape: Shape of state space (e.g., (3, 5, 3))
            n_actions: Number of possible actions
            learning_rate: Learning rate α (alpha)
            discount_factor: Discount factor γ (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            seed: Random seed
        """
        self.state_space_shape = state_space_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        q_shape = tuple(state_space_shape) + (n_actions,)
        self.q_table = np.zeros(q_shape)
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
        if seed is not None:
            np.random.seed(seed)
    
    def _state_to_indices(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert state array to tuple of indices for Q-table lookup."""
        return tuple(state.astype(int))
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action
        """
        state_idx = self._state_to_indices(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state_idx])
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool
    ):
        """
        Update Q-value using SARSA update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        
        Note: Uses the actual next_action, not max over next actions.
        """
        state_idx = self._state_to_indices(state)
        next_state_idx = self._state_to_indices(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx + (action,)]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # SARSA: use Q-value of actual next action
            next_q = self.q_table[next_state_idx + (next_action,)]
            target_q = reward + self.discount_factor * next_q
        
        # SARSA update
        td_error = target_q - current_q
        self.q_table[state_idx + (action,)] += self.learning_rate * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save Q-table and parameters to file."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table and parameters from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        print(f"Agent loaded from {filepath}")
    
    def get_policy(self) -> np.ndarray:
        """
        Extract deterministic policy from Q-table.
        
        Returns:
            Array of shape state_space_shape with best action for each state
        """
        return np.argmax(self.q_table, axis=-1)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in given state."""
        state_idx = self._state_to_indices(state)
        return self.q_table[state_idx]


def train_sarsa(
    env,
    agent: SARSAAgent,
    n_episodes: int = 1000,
    verbose: bool = True,
    verbose_freq: int = 100
) -> Tuple[SARSAAgent, dict]:
    """
    Train SARSA agent.
    
    Args:
        env: Gym environment
        agent: SARSA agent
        n_episodes: Number of training episodes
        verbose: Whether to print progress
        verbose_freq: Print frequency
    
    Returns:
        Trained agent and training metrics
    """
    episode_rewards = []
    episode_lengths = []
    sla_violations_history = []
    costs_history = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # SARSA: Select initial action
        action = agent.select_action(state, training=True)
        
        while not done:
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # SARSA: Select next action before update
            if not done:
                next_action = agent.select_action(next_state, training=True)
            else:
                next_action = 0  # Dummy value (won't be used)
            
            # Update Q-values using SARSA
            agent.update(state, action, reward, next_state, next_action, done)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            
            # Move to next state-action pair
            state = next_state
            action = next_action
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sla_violations_history.append(info.get('sla_violations', 0))
        costs_history.append(info.get('total_cost', 0))
        
        # Print progress
        if verbose and (episode + 1) % verbose_freq == 0:
            avg_reward = np.mean(episode_rewards[-verbose_freq:])
            avg_sla = np.mean(sla_violations_history[-verbose_freq:])
            avg_cost = np.mean(costs_history[-verbose_freq:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg SLA Violations: {avg_sla:.2f} | "
                  f"Avg Cost: {avg_cost:.2f}")
    
    agent.episode_rewards = episode_rewards
    agent.episode_lengths = episode_lengths
    
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'sla_violations': sla_violations_history,
        'costs': costs_history
    }
    
    return agent, metrics


def evaluate_agent(
    env,
    agent: SARSAAgent,
    n_episodes: int = 100,
    verbose: bool = True
) -> dict:
    """
    Evaluate trained agent.
    
    Args:
        env: Gym environment
        agent: Trained agent
        n_episodes: Number of evaluation episodes
        verbose: Whether to print results
    
    Returns:
        Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    sla_violations = []
    costs = []
    utilizations = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_utils = []
        done = False
        
        while not done:
            # Greedy action selection (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            episode_utils.append(info.get('utilization', 0))
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sla_violations.append(info.get('sla_violations', 0))
        costs.append(info.get('total_cost', 0))
        utilizations.append(np.mean(episode_utils))
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_sla_violations': np.mean(sla_violations),
        'mean_cost': np.mean(costs),
        'mean_utilization': np.mean(utilizations)
    }
    
    if verbose:
        print("\n" + "="*80)
        print("EVALUATION RESULTS (SARSA)")
        print("="*80)
        print(f"Episodes: {n_episodes}")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_length']:.2f}")
        print(f"Mean SLA Violations: {metrics['mean_sla_violations']:.2f}")
        print(f"Mean Cost: {metrics['mean_cost']:.2f}")
        print(f"Mean Utilization: {metrics['mean_utilization']:.2%}")
        print("="*80)
    
    return metrics
