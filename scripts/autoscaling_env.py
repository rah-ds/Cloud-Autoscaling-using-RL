import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Define a data class to hold environment configuration parameters
@dataclass
class AutoScalingEnvConfig:
    initial_capacity: int = 10          # The starting number of machines in the cluster.
    target_utilization: float = 0.6     # The desired average CPU utilization for the cluster.
    sla_threshold: float = 0.8          # The utilization threshold above which SLA violations occur.
    cost_per_machine_per_minute: float = 0.002  # The cost of running one machine for one minute.
    min_capacity: int = 1               # The minimum number of machines the cluster can scale down to.
    max_capacity: int = 20             # The maximum number of machines the cluster can scale up to.
    cooldown_period: int = 5            # Steps the agent must wait after a scaling action.
    demand_scale: float = 1000.0       #  scales avg_cpu up
    cost_weight: float = 1.0          # weights for different reward terms
    util_weight: float = 1.0
    sla_weight: float = 10.0


class AutoScalingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, usage_dataframe: pd.DataFrame,
                 config: AutoScalingEnvConfig = AutoScalingEnvConfig()):
        """
        Initializes the AutoScalingEnv reinforcement learning environment.

        Args:
            usage_dataframe (pd.DataFrame): DataFrame containing time-series usage data
                                            with columns ['time_window', 'avg_cpu'].
            config (AutoScalingEnvConfig): Configuration object for the environment.
        """
        super().__init__()

        self.df_usage = usage_dataframe.reset_index(drop=True)
        self.current_step = 0
        # Use the full length as the number of steps; indices go 0..len-1
        self.max_steps = len(self.df_usage)

        # Store configuration
        self.config = config
        self.initial_capacity = self.config.initial_capacity
        self.current_capacity = self.initial_capacity
        self.target_utilization = self.config.target_utilization
        self.sla_threshold = self.config.sla_threshold
        self.cost_per_machine_per_minute = self.config.cost_per_machine_per_minute
        self.sla_penalty_weight = self.config.sla_penalty_weight
        self.min_capacity = self.config.min_capacity
        self.max_capacity = self.config.max_capacity
        self.cooldown_period = self.config.cooldown_period
        self.cooldown = 0  # cooldown counter

        # Action: 0 (scale down), 1 (hold), 2 (scale up)
        self.action_space = spaces.Discrete(3)

        # Observation: [avg_cpu]
        # Bound CPU/mem from 0 to +inf, capacity between min and max capacity.
        self.observation_space = spaces.Box(
        low=np.array([0.0, float(self.min_capacity)], dtype=np.float32),
        high=np.array([np.inf, float(self.max_capacity)], dtype=np.float32),
        dtype=np.float32
)


        self.state = self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Gets the current observation of the environment.

        Returns:
            np.ndarray: The observation array.
        """
        # Clamp step to valid range to avoid IndexError
        idx = min(self.current_step, self.max_steps - 1)
        row = self.df_usage.iloc[idx]
        obs = np.array([row["avg_cpu"], self.current_capacity],
                       dtype=np.float32)
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the environment based on the agent's action.

        Args:
            action (int): 0 (scale down), 1 (hold), 2 (scale up).

        Returns:
            observation (np.ndarray): Next observation.
            reward (float): Scalar reward.
            terminated (bool): Episode ended because task is solved/failed.
            truncated (bool): Episode ended because of time limit or external constraint.
            info (dict): Additional diagnostic info.
        """
        # Ensure valid action
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Use usage data for the *current* time step to compute reward
        idx = min(self.current_step, self.max_steps - 1)
        row = self.df_usage.iloc[idx]

        # Apply action with cooldown
        if self.cooldown == 0:
            if action == 2:  # Scale up
                self.current_capacity = min(self.current_capacity + 1, self.max_capacity)
                self.cooldown = self.cooldown_period
            elif action == 0:  # Scale down
                self.current_capacity = max(self.current_capacity - 1, self.min_capacity)
                self.cooldown = self.cooldown_period
            # action == 1 is "hold", do nothing
        else:
            # Decrement cooldown if not zero
            self.cooldown = max(self.cooldown - 1, 0)


        raw_cpu = row["avg_cpu"]
        demand_cpu = raw_cpu * self.config.demand_scale
        utilization = (
            demand_cpu / self.current_capacity
            if self.current_capacity > 0 else 0.0
        )
        # ------------------------------------------------------------

        # Cost term
        raw_cost = self.current_capacity * self.cost_per_machine_per_minute
        cost_term = -self.config.cost_weight * raw_cost

        # SLA term
        sla_term = 0.0
        if utilization > self.sla_threshold:
            sla_excess = utilization - self.sla_threshold
            sla_term = -self.config.sla_weight * sla_excess

        # Utilization deviation term
        util_deviation = abs(utilization - self.target_utilization)
        util_term = -self.config.util_weight * util_deviation

        # Total reward
        reward = cost_term + sla_term + util_term

        # Advance time
        self.current_step += 1

        # Termination logic: we've consumed the dataset
        terminated = self.current_step >= self.max_steps
        truncated = False  # you can set this using a separate time limit if desired

        # Next observation (Gymnasium requires an obs even when terminated/truncated)
        if not terminated:
            self.state = self._get_obs()
        else:
            # Reuse the last real observation but with the *current* capacity
            self.state = np.array(
                [row["avg_cpu"], self.current_capacity],
                dtype=np.float32
            )



        info = {
            "current_capacity": self.current_capacity,
            "utilization": utilization,
            "demand_cpu": demand_cpu,
            "reward_components": {
                "cost_term": cost_term,
                "sla_term": sla_term,
                "util_term": util_term,
                }
}


        return self.state, reward, terminated, truncated, info

    def reset(self, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int | None): Optional RNG seed.
            options (dict | None): Additional options.

        Returns:
            observation (np.ndarray): Initial observation.
            info (dict): Additional info.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_capacity = self.initial_capacity
        self.cooldown = 0
        self.state = self._get_obs()

        info = {"initial_capacity": self.initial_capacity}
        return self.state, info

    def render(self, mode: str = "human"):
        # You can add plotting or logging here if you want.
        print(
            f"Step: {self.current_step}, "
            f"Capacity: {self.current_capacity}"
        )

    def close(self):
        # Clean up resources if needed
        pass
