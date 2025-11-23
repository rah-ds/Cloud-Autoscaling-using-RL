import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
from DQN_Utils import QNetwork, ReplayBuffer 


class DoubleDQNAgent:
    """Interacts with and learns from the environment using Double DQN."""

    def __init__(self, observation_space_shape, action_space_size, device, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, lr=5e-4, tau=1e-3, update_every=4):
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size
        self.device = device # Store device as an attribute
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        self.qnetwork_local = QNetwork(observation_space_shape, action_space_size).to(self.device)
        self.qnetwork_target = QNetwork(observation_space_shape, action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def select_action(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # Ensure action is on CPU for numpy conversion
        else:
            return random.choice(np.arange(self.action_space_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from local model
        Q_targets_next_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get Q values from target model for selected actions from local model
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_targets_next_local)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
