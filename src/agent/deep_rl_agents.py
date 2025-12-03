"""
Deep Reinforcement Learning Agents for Cloud Autoscaling

This module provides neural network-based RL agents:
- DQN (Deep Q-Network)
- Double DQN
- Dueling DQN
- PPO (Proximal Policy Optimization) wrapper
- A2C (Advantage Actor-Critic) wrapper

All agents use PyTorch and are compatible with Gymnasium environments.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm


# ============================================================================
# Neural Network Architectures
# ============================================================================

class QNetwork(nn.Module):
    """
    Standard Q-Network for DQN.
    
    Architecture: state -> FC -> ReLU -> FC -> ReLU -> FC -> Q-values
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        
        layers: List[nn.Module] = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.
    
    Separates value and advantage streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    
    Reference: Wang et al., "Dueling Network Architectures for Deep RL" (2016)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else 64),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else 64, 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else 64),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else 64, action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    """
    
    def __init__(self, capacity: int = 100000) -> None:
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent.
    
    Features:
    - Experience replay
    - Target network with periodic updates
    - Epsilon-greedy exploration
    
    Reference: Mnih et al., "Human-level control through deep RL" (2015)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor γ
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            seed: Random seed
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.train_step = 0
        self.episode_rewards: List[float] = []
        self.losses: List[float] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        Store experience and train if buffer is full enough.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_rewards': self.episode_rewards,
        }, filepath)
        print(f"DQN saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"DQN loaded from {filepath}")


# ============================================================================
# Double DQN Agent
# ============================================================================

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent.
    
    Uses online network for action selection and target network for evaluation,
    reducing overestimation bias.
    
    Reference: van Hasselt et al., "Deep RL with Double Q-learning" (2016)
    """
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """Train using Double DQN update rule."""
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Action selection with online network
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Action evaluation with target network
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val


# ============================================================================
# Dueling DQN Agent
# ============================================================================

class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent.
    
    Uses dueling architecture that separates value and advantage estimation.
    
    Reference: Wang et al., "Dueling Network Architectures" (2016)
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize parent but we'll replace the networks
        super().__init__(*args, **kwargs)
        
        # Replace with dueling networks
        state_dim = self.state_dim
        action_dim = self.action_dim
        hidden_dims = kwargs.get('hidden_dims', (128, 128))
        
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Recreate optimizer
        lr = kwargs.get('learning_rate', 1e-3)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)


# ============================================================================
# Training Functions
# ============================================================================

def train_dqn(
    env: Any,
    agent: Union[DQNAgent, DoubleDQNAgent, DuelingDQNAgent],
    n_episodes: int = 1000,
    verbose: bool = True,
    verbose_freq: int = 100,
) -> Tuple[Union[DQNAgent, DoubleDQNAgent, DuelingDQNAgent], Dict[str, List[float]]]:
    """
    Train a DQN-based agent.
    
    Args:
        env: Gymnasium environment
        agent: DQN agent (DQN, Double DQN, or Dueling DQN)
        n_episodes: Number of training episodes
        verbose: Whether to show progress
        verbose_freq: Frequency of verbose output
        
    Returns:
        Trained agent and metrics dictionary
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    sla_violations: List[float] = []
    losses: List[float] = []
    
    agent_name = agent.__class__.__name__
    
    for episode in tqdm(range(n_episodes), desc=f"{agent_name} Training", disable=not verbose):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        
        episode_reward: float = 0
        episode_length: int = 0
        episode_losses: List[float] = []
        done: bool = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            
            # Update agent
            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sla_violations.append(info.get('sla_violations', 0))
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        if verbose and (episode + 1) % verbose_freq == 0:
            avg_reward = np.mean(episode_rewards[-verbose_freq:])
            avg_loss = np.mean(losses[-verbose_freq:]) if losses else 0
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Loss: {avg_loss:.4f}")
    
    agent.episode_rewards = episode_rewards
    
    metrics: Dict[str, List[float]] = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'sla_violations': sla_violations,
        'losses': losses,
    }
    
    return agent, metrics


def evaluate_dqn(
    env: Any,
    agent: Union[DQNAgent, DoubleDQNAgent, DuelingDQNAgent],
    n_episodes: int = 100,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained DQN-based agent.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent
        n_episodes: Number of evaluation episodes
        verbose: Whether to print results
        
    Returns:
        Evaluation metrics
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    sla_violations: List[float] = []
    costs: List[float] = []
    
    for _ in range(n_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        
        episode_reward: float = 0
        episode_length: int = 0
        done: bool = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sla_violations.append(info.get('sla_violations', 0))
        costs.append(info.get('total_cost', 0))
    
    metrics: Dict[str, float] = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_sla_violations': float(np.mean(sla_violations)),
        'mean_cost': float(np.mean(costs)),
    }
    
    if verbose:
        agent_name = agent.__class__.__name__
        print("\n" + "=" * 80)
        print(f"EVALUATION RESULTS ({agent_name})")
        print("=" * 80)
        print(f"Episodes: {n_episodes}")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_length']:.2f}")
        print(f"Mean SLA Violations: {metrics['mean_sla_violations']:.2f}")
        print(f"Mean Cost: {metrics['mean_cost']:.2f}")
        print("=" * 80)
    
    return metrics


# ============================================================================
# Stable-Baselines3 Wrappers
# ============================================================================

def create_sb3_dqn(
    env: Any,
    learning_rate: float = 1e-4,
    buffer_size: int = 100000,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.1,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1000,
    verbose: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
) -> Any:
    """
    Create a Stable-Baselines3 DQN agent.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        batch_size: Training batch size
        gamma: Discount factor
        exploration_fraction: Fraction of training for exploration decay
        exploration_final_eps: Final exploration rate
        target_update_interval: Steps between target updates
        verbose: Verbosity level
        device: Device to use
        seed: Random seed
        
    Returns:
        SB3 DQN model
    """
    try:
        from stable_baselines3 import DQN
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            target_update_interval=target_update_interval,
            verbose=verbose,
            device=device,
            seed=seed,
        )
        return model
    except ImportError:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")


def create_sb3_ppo(
    env: Any,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    verbose: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
) -> Any:
    """
    Create a Stable-Baselines3 PPO agent.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        verbose: Verbosity level
        device: Device to use
        seed: Random seed
        
    Returns:
        SB3 PPO model
    """
    try:
        from stable_baselines3 import PPO
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=verbose,
            device=device,
            seed=seed,
        )
        return model
    except ImportError:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")


def create_sb3_a2c(
    env: Any,
    learning_rate: float = 7e-4,
    n_steps: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 1.0,
    verbose: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
) -> Any:
    """
    Create a Stable-Baselines3 A2C agent.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        n_steps: Steps per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        verbose: Verbosity level
        device: Device to use
        seed: Random seed
        
    Returns:
        SB3 A2C model
    """
    try:
        from stable_baselines3 import A2C
        
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            verbose=verbose,
            device=device,
            seed=seed,
        )
        return model
    except ImportError:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")


if __name__ == "__main__":
    # Quick test
    print("Testing Deep RL Agents...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Test network architectures
    state_dim, action_dim = 3, 3
    
    q_net = QNetwork(state_dim, action_dim)
    print(f"\nQNetwork: {sum(p.numel() for p in q_net.parameters())} parameters")
    
    dueling_net = DuelingQNetwork(state_dim, action_dim)
    print(f"DuelingQNetwork: {sum(p.numel() for p in dueling_net.parameters())} parameters")
    
    # Test with dummy input
    dummy_state = torch.randn(1, state_dim)
    print(f"\nQNetwork output: {q_net(dummy_state)}")
    print(f"DuelingQNetwork output: {dueling_net(dummy_state)}")
    
    print("\n✓ All deep RL agent tests passed!")
