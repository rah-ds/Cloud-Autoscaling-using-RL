#!/usr/bin/env python3
"""
Run all baseline experiments for Cloud Autoscaling RL project.

This script orchestrates all experiments:
1. Baseline policies (random, threshold)
2. Q-Learning with various hyperparameters
3. SARSA with various hyperparameters
4. Algorithm comparison

Usage:
    python scripts/run_baselines.py                    # Run all experiments
    python scripts/run_baselines.py --quick            # Quick test run
    python scripts/run_baselines.py --algo q-learning  # Run specific algorithm
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.cloud_autoscaling_env import CloudAutoscalingEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent
from agent.baseline_policies import (
    RandomPolicy,
    ThresholdPolicy,
)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("run_experiments")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()  # Clear any existing handlers
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (concise)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# Global logger (will be initialized in main)
logger = logging.getLogger("run_experiments")

# Plot settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def generate_workload(length: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate synthetic cloud workload with realistic patterns."""
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, length)
    
    # Combine multiple patterns
    daily_pattern = 50 + 30 * np.sin(t)
    weekly_pattern = 10 * np.sin(t / 7)
    noise = np.random.normal(0, 5, length)
    spikes = np.random.choice([0, 20], size=length, p=[0.95, 0.05])
    
    workload = daily_pattern + weekly_pattern + noise + spikes
    workload = np.clip(workload, 10, 100)
    
    return workload


def run_baseline_policies(
    env: CloudAutoscalingEnv,
    n_episodes: int = 100
) -> Dict[str, Any]:
    """Run baseline policies and collect metrics."""
    logger.info("=" * 50)
    logger.info("Running baseline policies...")
    logger.info(f"Number of episodes: {n_episodes}")
    
    results = {}
    
    # Random policy
    logger.debug("Initializing Random policy")
    random_policy = RandomPolicy(seed=42)
    random_rewards = []
    random_sla = []
    
    for ep in tqdm(range(n_episodes), desc="Random Policy", leave=False):
        state, info = env.reset()
        total_reward = 0
        sla_violations = 0
        done = False
        
        while not done:
            action = random_policy.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated
        
        random_rewards.append(total_reward)
        random_sla.append(sla_violations)
    
    results["random"] = {
        "mean_reward": np.mean(random_rewards),
        "std_reward": np.std(random_rewards),
        "mean_sla_violations": np.mean(random_sla),
    }
    logger.info(f"Random Policy completed: reward={results['random']['mean_reward']:.2f} ± {results['random']['std_reward']:.2f}")
    
    # Threshold policy
    logger.debug("Initializing Threshold policy")
    threshold_policy = ThresholdPolicy()
    threshold_rewards = []
    threshold_sla = []
    
    for ep in tqdm(range(n_episodes), desc="Threshold Policy", leave=False):
        state, info = env.reset()
        total_reward = 0
        sla_violations = 0
        done = False
        
        while not done:
            action = threshold_policy.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated
        
        threshold_rewards.append(total_reward)
        threshold_sla.append(sla_violations)
    
    results["threshold"] = {
        "mean_reward": np.mean(threshold_rewards),
        "std_reward": np.std(threshold_rewards),
        "mean_sla_violations": np.mean(threshold_sla),
    }
    
    logger.info(f"Threshold Policy completed: reward={results['threshold']['mean_reward']:.2f} ± {results['threshold']['std_reward']:.2f}")
    logger.info("Baseline policies evaluation complete")
    
    return results


def train_q_learning_agent(
    env: CloudAutoscalingEnv,
    n_episodes: int = 1000,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    seed: int = 42
) -> Dict[str, Any]:
    """Train Q-Learning agent and return metrics."""
    logger.info("=" * 50)
    logger.info("Training Q-Learning Agent")
    logger.info(f"  Episodes: {n_episodes}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Discount factor: {discount_factor}")
    logger.info(f"  Initial epsilon: {epsilon}")
    logger.info(f"  Epsilon decay: {epsilon_decay}")
    logger.info(f"  Seed: {seed}")
    
    agent = QLearningAgent(
        state_space_shape=(3, 5, 3),
        n_actions=3,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        seed=seed
    )
    
    episode_rewards = []
    episode_sla = []
    
    for ep in tqdm(range(n_episodes), desc="Q-Learning", leave=False):
        state, info = env.reset()
        total_reward = 0
        sla_violations = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning update
            agent.update(state, action, reward, next_state, terminated)
            
            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_sla.append(sla_violations)
    
    final_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_sla = np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    logger.info(f"Q-Learning training complete: final_reward={final_reward:.2f}, final_sla={final_sla:.2f}")
    
    return {
        "agent": agent,
        "episode_rewards": episode_rewards,
        "episode_sla": episode_sla,
        "final_epsilon": agent.epsilon,
        "mean_reward": final_reward,
        "mean_sla": final_sla,
    }


def train_sarsa_agent(
    env: CloudAutoscalingEnv,
    n_episodes: int = 1000,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    seed: int = 42
) -> Dict[str, Any]:
    """Train SARSA agent and return metrics."""
    logger.info("=" * 50)
    logger.info("Training SARSA Agent")
    logger.info(f"  Episodes: {n_episodes}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Discount factor: {discount_factor}")
    logger.info(f"  Initial epsilon: {epsilon}")
    logger.info(f"  Epsilon decay: {epsilon_decay}")
    logger.info(f"  Seed: {seed}")
    
    agent = SARSAAgent(
        state_space_shape=(3, 5, 3),
        n_actions=3,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        seed=seed
    )
    
    episode_rewards = []
    episode_sla = []
    
    with tqdm(range(n_episodes), desc="SARSA Training", leave=False) as pbar:
        for ep in pbar:
            state, info = env.reset()
            action = agent.select_action(state)
            total_reward = 0
            sla_violations = 0
            done = False
            
            while not done:
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = agent.select_action(next_state)
                
                # SARSA update (on-policy)
                agent.update(state, action, reward, next_state, next_action, terminated)
                
                total_reward += reward
                sla_violations += info.get("sla_violation", 0)
                state = next_state
                action = next_action
                done = terminated or truncated
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_sla.append(sla_violations)
            
            if (ep + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_sla = np.mean(episode_sla[-100:])
                pbar.set_postfix(avg_reward=f"{avg_reward:.2f}", avg_sla=f"{avg_sla:.2f}", eps=f"{agent.epsilon:.3f}")
                logger.info(f"  Episode {ep+1}/{n_episodes}: avg_reward={avg_reward:.2f}, avg_sla={avg_sla:.2f}, eps={agent.epsilon:.3f}")
    
    final_reward = np.mean(episode_rewards[-100:])
    final_sla = np.mean(episode_sla[-100:])
    logger.info(f"SARSA training complete: final_reward={final_reward:.2f}, final_sla={final_sla:.2f}")
    
    return {
        "agent": agent,
        "episode_rewards": episode_rewards,
        "episode_sla": episode_sla,
        "final_epsilon": agent.epsilon,
        "mean_reward": np.mean(episode_rewards[-100:]),
        "mean_sla": np.mean(episode_sla[-100:]),
    }


def run_hyperparameter_sweep(
    env: CloudAutoscalingEnv,
    algorithm: str = "q-learning",
    n_episodes: int = 500
) -> Dict[str, Any]:
    """Run hyperparameter sweep for given algorithm."""
    logger.info("=" * 50)
    logger.info(f"Running hyperparameter sweep for {algorithm}")
    logger.info(f"Episodes per configuration: {n_episodes}")
    
    learning_rates = [0.01, 0.1, 0.3]
    discount_factors = [0.9, 0.95, 0.99]
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Discount factors: {discount_factors}")
    
    results = {}
    total_configs = len(learning_rates) * len(discount_factors)
    config_num = 0
    
    with tqdm(total=total_configs, desc="Hyperparameter Sweep", leave=False) as pbar:
        for lr in learning_rates:
            for gamma in discount_factors:
                config_num += 1
                config_name = f"lr={lr}_gamma={gamma}"
                pbar.set_description(f"Sweep: {config_name}")
                logger.info(f"  [{config_num}/{total_configs}] Config: {config_name}")
                
                if algorithm == "q-learning":
                    metrics = train_q_learning_agent(
                        env, n_episodes=n_episodes,
                        learning_rate=lr, discount_factor=gamma
                    )
                else:
                    metrics = train_sarsa_agent(
                        env, n_episodes=n_episodes,
                        learning_rate=lr, discount_factor=gamma
                    )
                
                results[config_name] = {
                    "learning_rate": lr,
                    "discount_factor": gamma,
                    "mean_reward": metrics["mean_reward"],
                    "mean_sla": metrics["mean_sla"],
                }
                logger.debug(f"    Result: reward={metrics['mean_reward']:.2f}, sla={metrics['mean_sla']:.2f}")
                pbar.update(1)
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]["mean_reward"])
    logger.info(f"Best configuration: {best_config[0]} (reward={best_config[1]['mean_reward']:.2f})")
    
    return results


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save experiment results to JSON and plots."""
    logger.info("=" * 50)
    logger.info("Saving experiment results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results (excluding non-serializable objects)
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {
                k: v for k, v in value.items()
                if not hasattr(v, "__dict__")  # Skip agent objects
            }
        else:
            json_results[key] = value
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"
    
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"JSON results saved to {json_path}")
    
    # Create comparison plot
    if "q_learning" in results and "sarsa" in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rewards plot
        ax = axes[0]
        q_rewards = results["q_learning"].get("episode_rewards", [])
        sarsa_rewards = results["sarsa"].get("episode_rewards", [])
        
        if q_rewards:
            window = min(50, len(q_rewards) // 10)
            q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
            ax.plot(q_smooth, label="Q-Learning", alpha=0.8)
        
        if sarsa_rewards:
            window = min(50, len(sarsa_rewards) // 10)
            sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
            ax.plot(sarsa_smooth, label="SARSA", alpha=0.8)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (Moving Avg)")
        ax.set_title("Training Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Comparison bar chart
        ax = axes[1]
        algorithms = []
        rewards = []
        
        if "baselines" in results:
            for name, data in results["baselines"].items():
                algorithms.append(name.capitalize())
                rewards.append(data["mean_reward"])
        
        if "q_learning" in results:
            algorithms.append("Q-Learning")
            rewards.append(results["q_learning"]["mean_reward"])
        
        if "sarsa" in results:
            algorithms.append("SARSA")
            rewards.append(results["sarsa"]["mean_reward"])
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algorithms)))
        ax.bar(algorithms, rewards, color=colors)
        ax.set_ylabel("Mean Reward")
        ax.set_title("Algorithm Comparison")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = output_dir / f"comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Plot saved to {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run RL experiments for cloud autoscaling")
    parser.add_argument("--quick", action="store_true", help="Quick test run with fewer episodes")
    parser.add_argument("--algo", choices=["all", "q-learning", "sarsa", "baselines"],
                        default="all", help="Algorithm to run")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="artifacts/results",
                        help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    global logger
    output_dir = PROJECT_ROOT / args.output_dir
    logger = setup_logging(output_dir, args.log_level)
    
    # Adjust for quick mode
    n_episodes = 100 if args.quick else args.episodes
    
    logger.info("=" * 60)
    logger.info("Cloud Autoscaling RL Experiments")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Episodes: {n_episodes}")
    logger.info(f"  Algorithm: {args.algo}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Quick mode: {args.quick}")
    logger.info(f"  Log level: {args.log_level}")
    
    # Generate workload
    logger.info("Generating synthetic workload data...")
    workload = generate_workload(length=1000, seed=args.seed)
    logger.debug(f"Workload stats: min={workload.min():.2f}, max={workload.max():.2f}, mean={workload.mean():.2f}")
    
    env = CloudAutoscalingEnv(workload_data=workload, seed=args.seed)
    logger.info(f"Environment created: state_space={env.observation_space}, action_space={env.action_space}")
    
    results = {}
    start_time = datetime.now()
    
    # Run baselines
    if args.algo in ["all", "baselines"]:
        results["baselines"] = run_baseline_policies(env, n_episodes=min(100, n_episodes))
    
    # Run Q-Learning
    if args.algo in ["all", "q-learning"]:
        results["q_learning"] = train_q_learning_agent(
            env, n_episodes=n_episodes, seed=args.seed
        )
    
    # Run SARSA
    if args.algo in ["all", "sarsa"]:
        results["sarsa"] = train_sarsa_agent(
            env, n_episodes=n_episodes, seed=args.seed
        )
    
    # Save results
    save_results(results, output_dir)
    
    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    if "baselines" in results:
        logger.info("Baseline Policies:")
        for name, data in results["baselines"].items():
            logger.info(f"  {name.capitalize()}: reward={data['mean_reward']:.2f}, sla_violations={data['mean_sla_violations']:.2f}")
    
    if "q_learning" in results:
        logger.info(f"Q-Learning: reward={results['q_learning']['mean_reward']:.2f}, sla={results['q_learning']['mean_sla']:.2f}")
    
    if "sarsa" in results:
        logger.info(f"SARSA: reward={results['sarsa']['mean_reward']:.2f}, sla={results['sarsa']['mean_sla']:.2f}")
    
    logger.info("=" * 60)
    logger.info(f"Total runtime: {elapsed_time}")
    logger.info("Experiments complete!")
    
    env.close()


if __name__ == "__main__":
    main()