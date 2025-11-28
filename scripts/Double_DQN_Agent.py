import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from DQN_Utils import QNetwork, DuelingQNetwork, ReplayBuffer 


class DoubleDQNAgent:
    """Interacts with and learns from the environment using Double DQN."""

    def __init__(
        self,
        observation_space_shape,
        action_space_size,
        seed,
        device=None,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        lr=5e-4,
        tau=1e-3,
        update_every=4,
    ):
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size

        # Allow caller to pass in a device, but default to cuda if available
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # Q-Networks
        self.qnetwork_local = QNetwork(observation_space_shape, action_space_size).to(self.device)
        self.qnetwork_target = QNetwork(observation_space_shape, action_space_size).to(self.device)

        # Initialize target with the same weights as local
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory and learn every update_every steps."""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn.
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    # === Interface wrapper to match DQN agent ===
    def act(self, state, eps=0.0):
        """Return actions for given state as per current policy (epsilon-greedy)."""
        return self.select_action(state, eps)

    def select_action(self, state, eps=0.0):
        """Epsilon-greedy action selection using the local Q-network."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            # Exploit
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore
            return random.choice(np.arange(self.action_space_size))

    def learn(self, experiences, gamma):
        """Update value parameters using a batch of experience tuples.

        Double DQN:
          1) Use local network to select the best action for next_states.
          2) Use target network to evaluate that action.
        """
        states, actions, rewards, next_states, dones = experiences

        # 1) Use local network to choose best action at next_state
        with torch.no_grad():
            next_q_local = self.qnetwork_local(next_states)
            next_action_indices = next_q_local.argmax(1).unsqueeze(1)

            # 2) Use target network to value that action
            next_q_target = self.qnetwork_target(next_states)
            Q_targets_next = next_q_target.gather(1, next_action_indices)

            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # Optional: if your DQN agent has these, add them too for compatibility
    def save(self, filepath):
        torch.save(self.qnetwork_local.state_dict(), filepath)

    def load(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        self.qnetwork_local.load_state_dict(state_dict)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

