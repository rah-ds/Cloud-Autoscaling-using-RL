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
    max_capacity: int = 12              # The maximum number of machines the cluster can scale up to.
    cooldown_period: int = 2            # Steps the agent must wait after a scaling action.
    demand_scale: float = 1500.0        # Scales avg_cpu up to "demand" CPU.
    cost_weight: float = 1.0            # Weight for cost term in reward.
    util_weight: float = 3.0            # Weight for utilization deviation term.
    sla_weight: float = 15.0            # Weight for SLA violation term.


# -------------------------------------------------------------------
# Preset configurations for different environment profiles (A, B, C)
# -------------------------------------------------------------------
ENV_CONFIGS: dict[str, AutoScalingEnvConfig] = {
    "A": AutoScalingEnvConfig(
        # Baseline / balanced
        initial_capacity=10,
        target_utilization=0.60,
        sla_threshold=0.80,
        cost_per_machine_per_minute=0.002,
        min_capacity=1,
        max_capacity=12,
        cooldown_period=2,
        demand_scale=1500.0,
        cost_weight=1.0,
        util_weight=3.0,
        sla_weight=15.0,
    ),
    "B": AutoScalingEnvConfig(
        # High-demand / high-penalty
        initial_capacity=12,
        target_utilization=0.55,
        sla_threshold=0.75,
        cost_per_machine_per_minute=0.002,
        min_capacity=1,
        max_capacity=20,
        cooldown_period=4,
        demand_scale=2500.0,
        cost_weight=0.8,
        util_weight=4.0,
        sla_weight=20.0,
    ),
    "C": AutoScalingEnvConfig(
        # Low-load / cost-focused
        initial_capacity=8,
        target_utilization=0.70,
        sla_threshold=0.85,
        cost_per_machine_per_minute=0.002,
        min_capacity=1,
        max_capacity=10,
        cooldown_period=3,
        demand_scale=900.0,
        cost_weight=2.0,
        util_weight=2.0,
        sla_weight=10.0,
    ),
}


def get_env_config(profile: str) -> AutoScalingEnvConfig:
    """
    Look up a preset configuration by profile name ('A', 'B', 'C').
    """
    key = profile.upper()
    if key not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment profile '{profile}'. Use 'A', 'B', or 'C'.")
    return ENV_CONFIGS[key]


class AutoScalingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        usage_dataframe: pd.DataFrame,
        profile: str = "A",
        config: AutoScalingEnvConfig | None = None,
    ):
        """
        Initializes the AutoScalingEnv reinforcement learning environment.

        Args:
            usage_dataframe (pd.DataFrame): DataFrame containing time-series usage data
                                            with columns ['time_window', 'avg_cpu'].
            profile (str): Environment profile key: 'A', 'B', or 'C'. Ignored if
                           a config is explicitly provided.
            config (AutoScalingEnvConfig | None): Configuration object for the
                                                  environment. If None, the
                                                  profile's preset is used.
        """
        super().__init__()

        # If user didn't pass a config, look up by profile
        if config is None:
            config = get_env_config(profile)

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
        self.sla_weight = self.config.sla_weight
        self.min_capacity = self.config.min_capacity
        self.max_capacity = self.config.max_capacity
        self.cooldown_period = self.config.cooldown_period
        self.cooldown = 0  # cooldown counter

        # Action: 0 (scale down), 1 (hold), 2 (scale up)
        self.action_space = spaces.Discrete(3)

        # Observation: [avg_cpu, current_capacity]
        self.observation_space = spaces.Box(
            low=np.array([0.0, float(self.min_capacity)], dtype=np.float32),
            high=np.array([np.inf, float(self.max_capacity)], dtype=np.float32),
            dtype=np.float32,
        )

        self.state = self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Gets the current observation of the environment.

        Returns:
            np.ndarray: The observation array [avg_cpu, current_capacity].
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
