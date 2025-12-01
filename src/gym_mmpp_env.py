from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MMPPEnv(gym.Env):
    """Markov-Modulated Poisson Process environment.

    Observation: Dict with 'arrivals' (float array shape (1,)) and 'state' (discrete state index).
    Action: Discrete number of servers to allocate (0..max_servers).

    Reward: negative cost = - (server_cost * servers + penalty_cost * unmet_demand)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rates: Optional[list] = None,
        transitions: Optional[list] = None,
        max_servers: int = 10,
        server_capacity: int = 10,
        server_cost: float = 0.1,
        penalty_cost: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.rates = rates if rates is not None else [5, 50, 200]
        self.transitions = (
            transitions
            if transitions is not None
            else [[0.90, 0.09, 0.01], [0.15, 0.80, 0.05], [0.05, 0.15, 0.80]]
        )
        self.max_servers = int(max_servers)
        self.server_capacity = int(server_capacity)
        self.server_cost = float(server_cost)
        self.penalty_cost = float(penalty_cost)

        # RNG
        self._seed = seed
        self.rng = np.random.RandomState(seed)

        # internal state
        self.state = 0
        self.t = 0

        # spaces
        self.observation_space = spaces.Dict(
            {
                "arrivals": spaces.Box(low=0.0, high=1e9, shape=(1,), dtype=np.float32),
                "state": spaces.Discrete(len(self.rates)),
            }
        )
        self.action_space = spaces.Discrete(self.max_servers + 1)

    def seed(self, seed: Optional[int] = None):
        """Set the RNG seed for reproducibility."""
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        return [seed]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state. Accepts gym/gymnasium-style seed and options.

        Returns the initial observation (keeps legacy single-return format for compatibility).
        """
        if seed is not None:
            # reseed RNG for reproducibility when requested by caller
            self.seed(seed)

        self.state = 0
        self.t = 0
        arrivals = int(self.rng.poisson(self.rates[self.state]))
        obs = {
            "arrivals": np.array([arrivals], dtype=np.float32),
            "state": int(self.state),
        }
        return obs

    def step(self, action: int):
        """Take one environment step.

        Args:
            action: number of servers to allocate (int between 0 and max_servers).

        Returns:
            obs, reward, done, info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # sample arrivals from Poisson according to current state's rate
        arrivals = int(self.rng.poisson(self.rates[self.state]))

        # compute served and unmet demand
        servers = int(action)
        capacity = servers * self.server_capacity
        served = min(capacity, arrivals)
        unmet = max(0, arrivals - served)

        # cost-based reward (negative cost)
        reward = -(servers * self.server_cost + self.penalty_cost * unmet)

        # transition to next state
        self.state = int(
            self.rng.choice(len(self.rates), p=self.transitions[self.state])
        )
        self.t += 1

        obs = {
            "arrivals": np.array([arrivals], dtype=np.float32),
            "state": int(self.state),
        }
        done = False
        info: Dict[str, Any] = {}
        return obs, float(reward), done, info

    def render(self, mode: str = "human"):
        print(f"t={self.t}, state={self.state}")

    def close(self):
        return
