"""
Cloud Autoscaling Environment
A Gymnasium-compatible environment for cloud resource autoscaling simulation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional, Any


class CloudAutoscalingEnv(gym.Env):
    """
    Cloud Autoscaling Environment using Gymnasium interface.
    
    State Space:
        - utilization_level: 0 (low), 1 (medium), 2 (high)
        - capacity_level: 0-4 (representing 1-5 capacity units)
        - demand_trend: 0 (falling), 1 (flat), 2 (rising)
    
    Action Space:
        - 0: Scale down (remove capacity)
        - 1: Hold steady (no change)
        - 2: Scale up (add capacity)
    
    Reward Function:
        - Positive reward for optimal utilization (40-80%)
        - Large penalty for SLA violations (utilization > 90%)
        - Small penalty for wasted capacity (utilization < 20%)
        - Penalty for frequent capacity changes
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        workload_data: Optional[np.ndarray] = None,
        max_capacity: int = 5,
        min_capacity: int = 1,
        optimal_util_min: float = 0.40,
        optimal_util_max: float = 0.80,
        sla_violation_threshold: float = 0.90,
        waste_threshold: float = 0.20,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.optimal_util_min = optimal_util_min
        self.optimal_util_max = optimal_util_max
        self.sla_violation_threshold = sla_violation_threshold
        self.waste_threshold = waste_threshold
        
        # State space: (utilization_level, capacity_level, demand_trend)
        # utilization_level: 0 (low <40%), 1 (medium 40-80%), 2 (high >80%)
        # capacity_level: 0-4 (representing 1-5 units)
        # demand_trend: 0 (falling), 1 (flat), 2 (rising)
        self.observation_space = spaces.MultiDiscrete([3, 5, 3])
        
        # Action space: 0 (scale down), 1 (hold), 2 (scale up)
        self.action_space = spaces.Discrete(3)
        
        # Workload data
        if workload_data is not None:
            self.workload_data = workload_data
        else:
            # Generate synthetic workload if none provided
            self.workload_data = self._generate_synthetic_workload()
        
        # Environment state
        self.current_step = 0
        self.current_capacity = 3  # Start with medium capacity
        self.previous_demand = 0
        self.current_demand = 0
        self.previous_action = 1  # Start with "hold"
        
        # Tracking metrics
        self.total_cost = 0
        self.sla_violations = 0
        self.capacity_changes = 0
        
        if seed is not None:
            self.seed(seed)
    
    def _generate_synthetic_workload(self, length: int = 1000) -> np.ndarray:
        """Generate synthetic workload with trends and noise."""
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, length)
        
        # Combine multiple patterns
        daily_pattern = 50 + 30 * np.sin(t)  # Daily cycle
        weekly_pattern = 10 * np.sin(t / 7)  # Weekly variation
        noise = np.random.normal(0, 5, length)  # Random noise
        spikes = np.random.choice([0, 20], size=length, p=[0.95, 0.05])  # Occasional spikes
        
        workload = daily_pattern + weekly_pattern + noise + spikes
        workload = np.clip(workload, 10, 100)  # Keep in reasonable range
        
        return workload
    
    def _get_utilization_level(self, utilization: float) -> int:
        """Convert continuous utilization to discrete level."""
        if utilization < 0.40:
            return 0  # Low
        elif utilization < 0.80:
            return 1  # Medium
        else:
            return 2  # High
    
    def _get_demand_trend(self) -> int:
        """Determine if demand is rising, flat, or falling."""
        diff = self.current_demand - self.previous_demand
        threshold = 2.0  # Threshold for considering change significant
        
        if diff < -threshold:
            return 0  # Falling
        elif diff > threshold:
            return 2  # Rising
        else:
            return 1  # Flat
    
    def _calculate_utilization(self) -> float:
        """Calculate current utilization as demand / capacity."""
        return self.current_demand / self.current_capacity
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on current state and action.
        
        Reward components:
        1. Optimal utilization: +10 if in [40%, 80%]
        2. SLA violation: -50 if utilization > 90%
        3. Wasted capacity: -5 if utilization < 20%
        4. Capacity change penalty: -2 if action changed capacity
        5. Efficiency bonus: +5 if utilization in [60%, 70%] (sweet spot)
        """
        reward = 0.0
        utilization = self._calculate_utilization()
        
        # 1. Optimal utilization reward
        if self.optimal_util_min <= utilization <= self.optimal_util_max:
            reward += 10.0
            
            # Efficiency bonus for being in the sweet spot
            if 0.60 <= utilization <= 0.70:
                reward += 5.0
        
        # 2. SLA violation penalty (critical)
        if utilization >= self.sla_violation_threshold:
            penalty = -50.0 * (1 + (utilization - self.sla_violation_threshold))
            reward += penalty
            self.sla_violations += 1
        
        # 3. Wasted capacity penalty
        if utilization < self.waste_threshold:
            waste_penalty = -5.0 * (self.waste_threshold - utilization)
            reward += waste_penalty
        
        # 4. Capacity change penalty (encourage stability)
        if action != 1:  # If not holding steady
            reward -= 2.0
            self.capacity_changes += 1
        
        # 5. Cost penalty (proportional to capacity)
        cost_penalty = -0.5 * self.current_capacity
        reward += cost_penalty
        self.total_cost += self.current_capacity
        
        return reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_capacity = 3  # Start with medium capacity
        self.previous_demand = self.workload_data[0]
        self.current_demand = self.workload_data[0]
        self.previous_action = 1
        
        # Reset metrics
        self.total_cost = 0
        self.sla_violations = 0
        self.capacity_changes = 0
        
        # Get initial state
        utilization = self._calculate_utilization()
        utilization_level = self._get_utilization_level(utilization)
        capacity_level = self.current_capacity - 1  # 0-indexed
        demand_trend = 1  # Start with flat trend
        
        state = np.array([utilization_level, capacity_level, demand_trend], dtype=np.int32)
        info = {
            'utilization': utilization,
            'demand': self.current_demand,
            'capacity': self.current_capacity
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (scale down), 1 (hold), 2 (scale up)
        
        Returns:
            observation: Current state
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Update capacity based on action
        if action == 0:  # Scale down
            self.current_capacity = max(self.min_capacity, self.current_capacity - 1)
        elif action == 2:  # Scale up
            self.current_capacity = min(self.max_capacity, self.current_capacity + 1)
        # action == 1 (hold) doesn't change capacity
        
        # Move to next time step
        self.current_step += 1
        
        # Update demand
        self.previous_demand = self.current_demand
        if self.current_step < len(self.workload_data):
            self.current_demand = self.workload_data[self.current_step]
        else:
            # Episode done
            terminated = True
            truncated = False
            
            # Calculate final state and reward
            utilization = self._calculate_utilization()
            reward = self._calculate_reward(action)
            
            utilization_level = self._get_utilization_level(utilization)
            capacity_level = self.current_capacity - 1
            demand_trend = self._get_demand_trend()
            
            state = np.array([utilization_level, capacity_level, demand_trend], dtype=np.int32)
            info = {
                'utilization': utilization,
                'demand': self.current_demand,
                'capacity': self.current_capacity,
                'total_cost': self.total_cost,
                'sla_violations': self.sla_violations,
                'capacity_changes': self.capacity_changes
            }
            
            return state, reward, terminated, truncated, info
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Get current state
        utilization = self._calculate_utilization()
        utilization_level = self._get_utilization_level(utilization)
        capacity_level = self.current_capacity - 1
        demand_trend = self._get_demand_trend()
        
        state = np.array([utilization_level, capacity_level, demand_trend], dtype=np.int32)
        
        terminated = False
        truncated = False
        
        info = {
            'utilization': utilization,
            'demand': self.current_demand,
            'capacity': self.current_capacity,
            'total_cost': self.total_cost,
            'sla_violations': self.sla_violations,
            'capacity_changes': self.capacity_changes
        }
        
        self.previous_action = action
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state (optional)."""
        if self.current_step % 100 == 0:
            utilization = self._calculate_utilization()
            print(f"Step: {self.current_step}, "
                  f"Demand: {self.current_demand:.2f}, "
                  f"Capacity: {self.current_capacity}, "
                  f"Utilization: {utilization:.2%}, "
                  f"Cost: {self.total_cost}, "
                  f"SLA Violations: {self.sla_violations}")
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
