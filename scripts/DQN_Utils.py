
"""## DQN Utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_size, hidden_size=64):
        super(QNetwork, self).__init__()
        input_features = observation_space_shape[0]
        self.fc1 = nn.Linear(input_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- ReplayBuffer Definition
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = (None, None, None, None, None)
        self.seed = random.seed(0)

    def add(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)



def train_agent(agent, env, num_episodes, max_steps_per_episode, epsilon_start, epsilon_end, epsilon_decay):
    scores = []
    epsilon = epsilon_start

    for i_episode in range(1, num_episodes + 1):
        state, info = env.reset()
        score = 0
        done = False
        truncated = False

        for t in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, terminated or truncated)
            state = next_state
            score += reward

            if terminated or truncated:
                break

        scores.append(score)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores[-10:]):.2f}')

    print("Training finished.")
    return scores

def evaluate_agent(agent, env, num_evaluation_episodes, max_steps_per_evaluation_episode):
    eval_scores = []
    evaluation_results = []

    for i_episode in range(1, num_evaluation_episodes + 1):
        state, info = env.reset()
        score = 0
        episode_results = []

        for t in range(max_steps_per_evaluation_episode):
            action = agent.select_action(state, eps=0.)
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

