"""
REINFORCE (Monte Carlo Policy Gradient) Agent for Cloud Autoscaling.

This implements a policy gradient method that learns a stochastic policy
directly, as opposed to value-based methods like DQN that learn Q-values.

Key differences from DQN:
- Learns policy directly (π(a|s)) instead of Q(s,a)
- Uses Monte Carlo returns (full episode)
- On-policy: must use trajectories from current policy
- Higher variance but can learn stochastic policies

Based on Williams (1992) "Simple Statistical Gradient-Following Algorithms
for Connectionist Reinforcement Learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


class PolicyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities.
    
    Unlike DQN which outputs Q-values, this outputs a probability
    distribution over actions using softmax.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities for given state."""
        return self.network(state)


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    
    This agent learns by:
    1. Collecting a full episode trajectory
    2. Computing discounted returns for each step
    3. Updating policy to increase probability of actions
       that led to higher returns
    
    Attributes:
        gamma: Discount factor for future rewards
        policy: Neural network that outputs action probabilities
        optimizer: Optimizer for policy network
        log_probs: Log probabilities of actions taken in current episode
        rewards: Rewards received in current episode
        baseline: Optional baseline for variance reduction
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        use_baseline: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for policy network
            gamma: Discount factor
            hidden_dim: Hidden layer size
            use_baseline: Whether to use baseline for variance reduction
            device: Device to run on ('cpu' or 'cuda')
        """
        self.gamma = gamma
        self.action_dim = action_dim
        self.use_baseline = use_baseline
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode storage
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        
        # Baseline (running average of returns)
        self.baseline = 0.0
        self.baseline_alpha = 0.1  # Baseline update rate
        
        # Training stats
        self.episode_count = 0
        self.total_loss = 0.0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state observation
            training: If True, sample from policy. If False, take argmax.
            
        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_tensor)
        
        if training:
            # Sample from the probability distribution
            dist = Categorical(probs)
            action = dist.sample()
            # Store log probability for policy gradient
            self.log_probs.append(dist.log_prob(action))
        else:
            # Greedy action selection for evaluation
            action = probs.argmax(dim=-1)
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update(self) -> float:
        """
        Update policy at end of episode using REINFORCE.
        
        Returns:
            Policy loss value
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns (G_t)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Update baseline
        if self.use_baseline:
            mean_return = returns.mean().item()
            self.baseline = (1 - self.baseline_alpha) * self.baseline + \
                           self.baseline_alpha * mean_return
            # Subtract baseline for variance reduction
            advantages = returns - self.baseline
        else:
            advantages = returns
        
        # Normalize advantages for training stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate policy gradient loss
        # Loss = -sum(log_prob * advantage)
        # We minimize negative because we want to maximize expected return
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store loss for logging
        loss_value = loss.item()
        self.total_loss += loss_value
        self.episode_count += 1
        
        # Clear episode storage
        self.log_probs = []
        self.rewards = []
        
        return loss_value
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for visualization/analysis.
        
        Args:
            state: Current state
            
        Returns:
            Array of action probabilities
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy(state_tensor)
            return probs.cpu().numpy().flatten()
    
    def get_entropy(self, state: np.ndarray) -> float:
        """
        Get entropy of action distribution (measure of exploration).
        
        Higher entropy = more exploration/uncertainty
        Lower entropy = more exploitation/confidence
        
        Args:
            state: Current state
            
        Returns:
            Entropy value
        """
        probs = self.get_action_probs(state)
        # H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline': self.baseline,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.baseline = checkpoint.get('baseline', 0.0)
        self.episode_count = checkpoint.get('episode_count', 0)
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'avg_loss': self.total_loss / max(1, self.episode_count),
            'baseline': self.baseline,
        }


class REINFORCEWithBaseline(REINFORCEAgent):
    """
    REINFORCE with learned value function baseline.
    
    This variant learns a separate value function V(s) to use as
    a baseline, which can significantly reduce variance compared
    to using a simple running average.
    
    Also known as "Actor-Critic" when the value function is used
    for bootstrapping, but here we still use full Monte Carlo returns.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        value_learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        device: Optional[str] = None
    ):
        # Initialize parent without baseline (we'll use learned one)
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            hidden_dim=hidden_dim,
            use_baseline=False,
            device=device
        )
        
        # Value network (critic)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=value_learning_rate
        )
        
        # Store states for value function update
        self.states: List[np.ndarray] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action and store state."""
        if training:
            self.states.append(state.copy())
        return super().select_action(state, training)
    
    def update(self) -> float:
        """Update both policy and value function."""
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        # Get value predictions
        values = self.value_network(states).squeeze()
        
        # Compute advantages (returns - baseline)
        advantages = returns - values.detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()
        
        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        
        # Clear episode storage
        loss_value = policy_loss.item()
        self.total_loss += loss_value
        self.episode_count += 1
        
        self.log_probs = []
        self.rewards = []
        self.states = []
        
        return loss_value
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'gamma': self.gamma,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
