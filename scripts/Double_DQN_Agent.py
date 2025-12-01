import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

from DQN_Utils import ReplayBuffer, QNetwork, DuelingQNetwork


class DoubleDQNAgent:
    """Double DQN agent that can use either standard QNetwork or DuelingQNetwork."""

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
        network_class=QNetwork
    ):
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # Use whichever network class the user selects
        self.qnetwork_local = network_class(observation_space_shape, action_space_size).to(self.device)
        self.qnetwork_target = network_class(observation_space_shape, action_space_size).to(self.device)

        # Make target identical at start
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay buffer must receive device
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        self.t_step = 0

    # ----------------------
    # Public Interface
    # ----------------------
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(np.argmax(q_values.cpu().data.numpy()))
        else:
            return random.randrange(self.action_space_size)

    # ----------------------
    # Learning
    # ----------------------
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            # 1. Best next actions from local network
            next_q_local = self.qnetwork_local(next_states)
            best_next_actions = next_q_local.argmax(dim=1, keepdim=True)

            # 2. Evaluate these actions using target network
            next_q_target = self.qnetwork_target(next_states)
            Q_targets_next = next_q_target.gather(1, best_next_actions)

            # 3. Bellman target
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Current Q-values
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local, target, tau):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    # ----------------------
    # Persistence
    # ----------------------
    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(state)
        self.qnetwork_target.load_state_dict(state)

