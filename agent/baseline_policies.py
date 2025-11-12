"""
Baseline Policies for Cloud Autoscaling
"""

import numpy as np
from typing import Tuple


class BaselinePolicy:
    """Base class for baseline policies."""
    
    def select_action(self, state: np.ndarray, info: dict = None) -> int:
        """Select action based on state."""
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__


class RandomPolicy(BaselinePolicy):
    """Randomly select actions."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
    
    def select_action(self, state: np.ndarray, info: dict = None) -> int:
        return np.random.randint(3)  # 0, 1, or 2


class ThresholdPolicy(BaselinePolicy):
    """
    Simple threshold-based policy.
    
    Rules:
    - If utilization is high (>80%): scale up
    - If utilization is low (<40%): scale down
    - Otherwise: hold steady
    """
    
    def select_action(self, state: np.ndarray, info: dict = None) -> int:
        utilization_level = state[0]
        
        if utilization_level == 2:  # High utilization
            return 2  # Scale up
        elif utilization_level == 0:  # Low utilization
            return 0  # Scale down
        else:  # Medium utilization
            return 1  # Hold steady


class ReactivePolicy(BaselinePolicy):
    """
    Reactive policy that responds to demand changes.
    
    Rules:
    - If demand is rising and utilization is medium/high: scale up
    - If demand is falling and utilization is low/medium: scale down
    - Otherwise: hold steady
    """
    
    def select_action(self, state: np.ndarray, info: dict = None) -> int:
        utilization_level = state[0]
        demand_trend = state[2]
        
        # Rising demand
        if demand_trend == 2:
            if utilization_level >= 1:  # Medium or high
                return 2  # Scale up
            else:
                return 1  # Hold
        
        # Falling demand
        elif demand_trend == 0:
            if utilization_level <= 1:  # Low or medium
                return 0  # Scale down
            else:
                return 1  # Hold
        
        # Flat demand
        else:
            if utilization_level == 2:  # High
                return 2  # Scale up
            elif utilization_level == 0:  # Low
                return 0  # Scale down
            else:
                return 1  # Hold


class ProactivePolicy(BaselinePolicy):
    """
    Proactive policy that anticipates demand changes.
    
    Rules:
    - If demand is rising: scale up preemptively
    - If demand is falling and utilization is low: scale down
    - If utilization is high: always scale up
    - Otherwise: adjust based on utilization
    """
    
    def select_action(self, state: np.ndarray, info: dict = None) -> int:
        utilization_level = state[0]
        capacity_level = state[1]
        demand_trend = state[2]
        
        # Critical: High utilization always scales up
        if utilization_level == 2:
            return 2  # Scale up
        
        # Proactive: Rising demand - prepare early
        if demand_trend == 2:
            if utilization_level >= 1 or capacity_level < 3:
                return 2  # Scale up
            else:
                return 1  # Hold
        
        # Falling demand - be cautious about scaling down
        elif demand_trend == 0:
            if utilization_level == 0 and capacity_level > 1:
                return 0  # Scale down
            else:
                return 1  # Hold
        
        # Flat demand - maintain based on utilization
        else:
            if utilization_level == 0:
                return 0  # Scale down
            else:
                return 1  # Hold


class ConservativePolicy(BaselinePolicy):
    """
    Conservative policy that minimizes capacity changes.
    
    Rules:
    - Only change capacity for extreme situations
    - Prefers to hold steady
    """
    
    def select_action(self, state: np.ndarray, info: dict = None) -> int:
        utilization_level = state[0]
        capacity_level = state[1]
        
        # Only scale up if utilization is critically high
        if utilization_level == 2 and capacity_level < 4:
            return 2  # Scale up
        
        # Only scale down if utilization is very low and we have excess capacity
        elif utilization_level == 0 and capacity_level > 2:
            return 0  # Scale down
        
        # Default: hold steady
        else:
            return 1  # Hold


def evaluate_baseline(
    env,
    policy: BaselinePolicy,
    n_episodes: int = 100,
    verbose: bool = True
) -> dict:
    """
    Evaluate a baseline policy.
    
    Args:
        env: Gym environment
        policy: Baseline policy to evaluate
        n_episodes: Number of episodes
        verbose: Whether to print results
    
    Returns:
        Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    sla_violations = []
    costs = []
    utilizations = []
    capacity_changes_list = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_utils = []
        done = False
        
        while not done:
            # Select action using baseline policy
            action = policy.select_action(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            episode_utils.append(info.get('utilization', 0))
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sla_violations.append(info.get('sla_violations', 0))
        costs.append(info.get('total_cost', 0))
        utilizations.append(np.mean(episode_utils))
        capacity_changes_list.append(info.get('capacity_changes', 0))
    
    metrics = {
        'policy_name': str(policy),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_sla_violations': np.mean(sla_violations),
        'mean_cost': np.mean(costs),
        'mean_utilization': np.mean(utilizations),
        'mean_capacity_changes': np.mean(capacity_changes_list)
    }
    
    if verbose:
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS - {metrics['policy_name']}")
        print("="*80)
        print(f"Episodes: {n_episodes}")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_length']:.2f}")
        print(f"Mean SLA Violations: {metrics['mean_sla_violations']:.2f}")
        print(f"Mean Cost: {metrics['mean_cost']:.2f}")
        print(f"Mean Utilization: {metrics['mean_utilization']:.2%}")
        print(f"Mean Capacity Changes: {metrics['mean_capacity_changes']:.2f}")
        print("="*80)
    
    return metrics


def compare_baselines(env, n_episodes: int = 100) -> dict:
    """
    Compare all baseline policies.
    
    Args:
        env: Gym environment
        n_episodes: Number of episodes per policy
    
    Returns:
        Dictionary of metrics for each policy
    """
    policies = [
        RandomPolicy(seed=42),
        ThresholdPolicy(),
        ReactivePolicy(),
        ProactivePolicy(),
        ConservativePolicy()
    ]
    
    results = {}
    
    print("\n" + "="*80)
    print("COMPARING BASELINE POLICIES")
    print("="*80)
    
    for policy in policies:
        metrics = evaluate_baseline(env, policy, n_episodes, verbose=True)
        results[str(policy)] = metrics
    
    # Print comparison table
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Policy':<20} {'Reward':>10} {'SLA Viol':>10} {'Cost':>10} {'Util':>10}")
    print("-"*80)
    
    for policy_name, metrics in results.items():
        print(f"{policy_name:<20} "
              f"{metrics['mean_reward']:>10.2f} "
              f"{metrics['mean_sla_violations']:>10.2f} "
              f"{metrics['mean_cost']:>10.2f} "
              f"{metrics['mean_utilization']:>9.1%}")
    
    print("="*80)
    
    return results
