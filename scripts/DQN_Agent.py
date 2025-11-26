
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
from DQN_Utils import QNetwork, ReplayBuffer


# --- DQNAgent Definition
class DQNAgent:
    def __init__(self, observation_space_shape, action_space_size, seed, device, buffer_size=int(1e5), batch_size=64, gamma=0.99, lr=5e-4, tau=1e-3, update_every=4):
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.device = device
        self.qnetwork_local = QNetwork(observation_space_shape, action_space_size).to(self.device)
        self.qnetwork_target = QNetwork(observation_space_shape, action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, devic=self.device,seed=seed)
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
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expecteds = self.qnetwork_local(states).gather(1, actions)
        loss = nn.MSELoss()(Q_expecteds, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



