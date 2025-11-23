import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Define a data class to hold environment configuration parameters
@dataclass
class AutoScalingEnvConfig:
    initial_capacity: int = 10 # The starting number of machines in the cluster.
    target_utilization: float = 0.6 # The desired average CPU utilization for the cluster.
    sla_threshold: float = 0.8 # The utilization threshold above which SLA violations occur.
    cost_per_machine_per_minute: float = 0.002 # The cost of running one machine for one minute.
    sla_penalty_weight: float = 10.0 # The penalty applied for exceeding the SLA threshold.
    min_capacity: int = 1 # The minimum number of machines the cluster can scale down to.
    max_capacity: int = 50 # The maximum number of machines the cluster can scale up to.
    cooldown_period: int = 5 # The number of steps the agent must wait after a scaling action before taking another.


# Define the RL Environment
class AutoScalingEnv(gym.Env):
    def __init__(self, usage_dataframe: pd.DataFrame, config: AutoScalingEnvConfig = AutoScalingEnvConfig()):
        """
        Initializes the AutoScalingEnv reinforcement learning environment.

        Args:
            usage_dataframe (pd.DataFrame): DataFrame containing time-series usage data.
            config (AutoScalingEnvConfig): Configuration object for the environment.
        """
        super(AutoScalingEnv, self).__init__()

        self.df_usage = usage_dataframe
        self.current_step = 0
        self.max_steps = len(self.df_usage) - 1

        # Store configuration
        self.config = config

        # Simulation parameters, initialized from config
        self.initial_capacity = self.config.initial_capacity
        self.current_capacity = self.initial_capacity
        self.target_utilization = self.config.target_utilization
        self.sla_threshold = self.config.sla_threshold
        self.cost_per_machine_per_minute = self.config.cost_per_machine_per_minute
        self.sla_penalty_weight = self.config.sla_penalty_weight
        self.min_capacity = self.config.min_capacity
        self.max_capacity = self.config.max_capacity
        self.cooldown_period = self.config.cooldown_period
        self.cooldown = 0 # To implement cooldown between scaling actions


        # Define action and observation space
        # Action: 0 (scale down), 1 (hold), 2 (scale up)
        self.action_space = spaces.Discrete(3)

        # Observation: [avg_cpu, avg_mem, current_capacity]
        # We'll add lagged values and trends later if needed.
        # The upper bound is set to infinity as these values can vary widely.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

        # Define initial state
        self.state = self._get_obs()

    def _get_obs(self) -> np.ndarray | None:
        """
        Gets the current observation of the environment.

        Returns:
            np.ndarray | None: The observation array or None if the episode is done.
        """
        if self.current_step > self.max_steps:
            return None

        # Get the current row of usage data
        row = self.df_usage.iloc[self.current_step]
        # Observation includes current workload (avg_cpu, avg_mem) and the agent's managed capacity
        obs = np.array([row['avg_cpu'], row['avg_mem'], self.current_capacity], dtype=np.float32)
        return obs


    def step(self, action: int) -> tuple[np.ndarray | None, float, bool, dict]:
        """
        Takes a step in the environment based on the agent's action.

        Args:
            action (int): The action taken by the agent (0: scale down, 1: hold, 2: scale up).

        Returns:
            tuple: A tuple containing the next observation, reward, done flag, and info dictionary.
        """
        # Apply action based on cooldown
        if self.cooldown == 0:
            if action == 2: # Scale up
                self.current_capacity = min(self.current_capacity + 1, self.max_capacity)
                self.cooldown = self.cooldown_period # Reset cooldown
            elif action == 0: # Scale down
                 self.current_capacity = max(self.current_capacity - 1, self.min_capacity)
                 self.cooldown = self.cooldown_period # Reset cooldown
            # Action 1 is hold, capacity doesn't change
        else:
            # Decrement cooldown if not zero
            self.cooldown -= 1

        # Advance the simulation step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step > self.max_steps

        # Calculate reward if not done
        reward = 0.0
        utilization = 0.0 # Initialize utilization
        estimated_total_cpu_load = 0.0

        if not done:
            # Get data for the step *before* moving to the next
            row = self.df_usage.iloc[self.current_step - 1]
            # Assuming avg_cpu is a fraction, estimate total CPU load based on actual active machines
            estimated_total_cpu_load = row['avg_cpu'] * row['active_machines']

            # Calculate utilization based on simulated capacity
            utilization = estimated_total_cpu_load / self.current_capacity if self.current_capacity > 0 else 0.0

            # Reward components
            # Penalize cost based on the number of active machines managed by the agent
            cost_penalty = self.current_capacity * self.cost_per_machine_per_minute

            # Penalize high utilization (SLA violation)
            sla_penalty = 0.0
            if utilization > self.sla_threshold:
                sla_penalty = (utilization - self.sla_threshold) * self.sla_penalty_weight

            # Penalize deviation from target utilization
            util_deviation_penalty = -abs(utilization - self.target_utilization)

            # Total reward (example combination)
            # We want to minimize cost and SLA violations, and ideally stay near target utilization
            # Using negative rewards for penalties and positive for goals (deviation is a penalty here).
            reward = -cost_penalty - sla_penalty + util_deviation_penalty

        # Get next observation
        next_obs = self._get_obs() if not done else None

        # Additional info (optional)
        info = {
            'current_capacity': self.current_capacity,
            'utilization': utilization, # Report utilization for the step
            'estimated_total_cpu_load': estimated_total_cpu_load,
            'reward_components': { # Optional: include components for debugging
                'cost_penalty': -cost_penalty,
                'sla_penalty': -sla_penalty,
                'util_deviation_penalty': util_deviation_penalty
            }
        }

        # In Gymnasium 0.28+, the step method returns (observation, reward, terminated, truncated, info)
        # We need to return 'terminated' and 'truncated' flags. In this simple env, 'done' covers both.
        terminated = done
        truncated = False # Or set to True based on other conditions if applicable (e.g., time limit)

        return next_obs, reward, terminated, truncated, info


    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int | None): An optional seed for the random number generator.
            options (dict | None): Additional options for resetting.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed) # Call the superclass reset with seed

        self.current_step = 0
        self.current_capacity = self.initial_capacity
        self.cooldown = 0
        self.state = self._get_obs()

        # Return initial observation and info dictionary (optional)
        info = {
            'initial_capacity': self.initial_capacity
        }
        return self.state, info

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close (self):
        # Clean up resources if needed
        pass
