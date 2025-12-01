"""
DQN Utilities for Cloud Autoscaling Reinforcement Learning

This module provides implementations and utilities for training and evaluating
Deep Q-Network (DQN) agents for the cloud autoscaling problem.

Key Components:
    - QNetwork: Standard MLP-based Q-network for direct Q-value approximation
    - DuelingQNetwork: Advanced dueling architecture with separate value and
      advantage streams for improved learning
    - ReplayBuffer: Experience replay buffer for storing and sampling training data
    - train_agent(): Training loop with epsilon-greedy exploration
    - evaluate_agent(): Evaluation loop that produces detailed step-by-step metrics
    - sample_start_index_for_split(): Utility for sampling episode windows from
      different train/val/test splits
    - reset_random_window_for_split(): Environment reset with random windowing

Data Split Configuration:
    SPLIT_RANGES defines three fixed ranges over a 10,000-step dataset:
    - train: steps 0-6999
    - val: steps 7000-8499
    - test: steps 8500-9999

The module supports both standard and random-window-based episode sampling to enable
flexible training and evaluation strategies on time-series data.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import pandas as pd


# Global split ranges for the 10,000-step df_usage
SPLIT_RANGES = {
    "train": (0, 6999),     # inclusive
    "val":   (7000, 8499),  # inclusive
    "test":  (8500, 9999),  # inclusive
}


# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- Plain QNetwork -----
class QNetwork(nn.Module):
    """Standard MLP Q-network: directly outputs Q(s,a)."""

    def __init__(self, observation_space_shape, action_space_size, seed=0):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        if isinstance(observation_space_shape, int):
            state_size = observation_space_shape
        else:
            state_size = observation_space_shape[0]

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_size)

    def forward(self, x):
        # x: [batch, state_dim] or [state_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # [batch, action_space_size]
        return q_values



# ----- Dueling QNetwork -----
class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network:
      - Shared feature layer
      - Separate value and advantage streams
      - Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(self, observation_space_shape, action_space_size, seed=0):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        if isinstance(observation_space_shape, int):
            state_size = observation_space_shape
        else:
            state_size = observation_space_shape[0]

        self.action_size = action_space_size

        # Shared feature layer
        self.fc1 = nn.Linear(state_size, 128)

        # Value stream
        self.fc_value = nn.Linear(128, 128)
        self.value_out = nn.Linear(128, 1)

        # Advantage stream
        self.fc_advantage = nn.Linear(128, 128)
        self.advantage_out = nn.Linear(128, action_space_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))

        # Value stream
        value = F.relu(self.fc_value(x))
        value = self.value_out(value)          # [batch, 1]

        # Advantage stream
        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage_out(advantage)  # [batch, A]

        advantage_mean = advantage.mean(dim=1, keepdim=True)  # [batch, 1]

        q_values = value + (advantage - advantage_mean)       # [batch, A]
        return q_values




from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed=0):
        """Fixed-size buffer to store experience tuples."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e[0] for e in experiences if e is not None])
        ).float().to(self.device)

        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])
        ).long().to(self.device)

        rewards = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None])
        ).float().to(self.device)

        next_states = torch.from_numpy(
            np.vstack([e[3] for e in experiences if e is not None])
        ).float().to(self.device)

        dones = torch.from_numpy(
            np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



def train_agent(agent, env, num_episodes, max_steps_per_episode,
                epsilon_start, epsilon_end, epsilon_decay, split: str = "train",
                use_random_windows: bool = False):

    scores = []
    epsilon = epsilon_start

    for i_episode in range(num_episodes):

        if use_random_windows:
            state, info = reset_random_window_for_split(env, episode_length=max_steps_per_episode,
            split = split
            )
        else:
            state, info = env.reset()

        score = 0.0

        for t in range(max_steps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, terminated or truncated)

            score += reward
            state = next_state

            if terminated or truncated:
                break

        scores.append(score)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores[-10:]):.2f}')

    print("Training finished.")
    return scores

def evaluate_agent(agent, env, num_evaluation_episodes, max_steps_per_evaluation_episode, split: str = "test", use_random_windows: bool = False):
    eval_scores = []
    evaluation_results = []

    for i_episode in range(1, num_evaluation_episodes + 1):
        if use_random_windows:
            state, info = reset_random_window_for_split(env, episode_length=max_steps_per_evaluation_episode, split = split )
        else:
            state, info = env.reset()
        score = 0
        episode_results = []

        for t in range(max_steps_per_evaluation_episode):
            action = agent.act(state, eps=0.)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_results.append({
                'step': t,
                'action': action,
                'reward': reward,
                'current_capacity': info['current_capacity'],
                'utilization': info['utilization'],
                'demand_cpu': info['demand_cpu'],
                'cost_penalty': info['reward_components'].get('cost_penalty', np.nan),
                'sla_penalty': info['reward_components'].get('sla_penalty', np.nan),
                'util_deviation_penalty': info['reward_components'].get('util_deviation_penalty', np.nan)
            })

            state = next_state
            score += reward

            if terminated or truncated:
                break

        eval_scores.append(score)
        evaluation_results.extend(episode_results)

        print(f'Evaluation Episode {i_episode}\tScore: {score:.2f}')

    print(f'\nAverage Evaluation Score over {num_evaluation_episodes} episodes: {np.mean(eval_scores):.2f}')

    evaluation_results_df = pd.DataFrame(evaluation_results)
    evaluation_results_df.set_index('step', inplace=True)

    return eval_scores, evaluation_results_df
import numpy as np


def sample_start_index_for_split(split: str, episode_length: int, max_steps: int) -> int:
    """
    Sample a random start index for a contiguous episode window of length `episode_length`
    restricted to the specified split.

    Args:
        split: one of {"train", "val", "test"}.
        episode_length: number of steps in the episode (e.g., 1000).
        max_steps: total number of steps in the underlying df_usage (env.max_steps).

    Returns:
        start_idx (int): starting index for the episode window.
    """
    if split not in SPLIT_RANGES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(SPLIT_RANGES.keys())}.")

    split_start, split_end = SPLIT_RANGES[split]

    # Safety: ensure split does not exceed dataset length
    split_start = max(split_start, 0)
    split_end = min(split_end, max_steps - 1)

    split_length = split_end - split_start + 1
    if episode_length > split_length:
        raise ValueError(
            f"Episode length {episode_length} is too long for split '{split}' "
            f"with length {split_length} (range [{split_start}, {split_end}])."
        )

    # max_start is the largest index such that start_idx + episode_length - 1 <= split_end
    max_start_idx = split_end - episode_length + 1
    start_idx = np.random.randint(split_start, max_start_idx + 1)

    return start_idx

def reset_random_window_for_split(env, episode_length: int, split: str):
    """
    Reset the environment and jump to a random starting index within the given split.

    Args:
        env: the AutoScalingEnv instance.
        episode_length: number of steps in the episode (e.g., 1000).
        split: one of {"train", "val", "test"}.

    Returns:
        obs: initial observation at the chosen start index.
        info: info dict, augmented with window_start/window_end.
    """
    # Reset env internal state (capacity, cooldown, etc.)
    obs, info = env.reset()

    # Sample a valid start index for the split
    start_idx = sample_start_index_for_split(split, episode_length, env.max_steps)

    # Jump to that index and recompute the observation
    env.current_step = start_idx
    obs = env._get_obs()  # use env's own observation builder

    # Optionally record window bounds in info
    info = dict(info)  # make a shallow copy
    info["window_start"] = start_idx
    info["window_end"] = start_idx + episode_length - 1

    return obs, info
