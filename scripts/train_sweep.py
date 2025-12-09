#!/usr/bin/env python3
"""
Training script called by wandb sweep agent.

This script is designed to be called directly by wandb sweep agents,
receiving hyperparameters as command-line arguments.

Usage (called automatically by wandb agent):
    python scripts/train_sweep.py --algorithm=q-learning --learning_rate=0.1 ...
"""

import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from agent.cloud_autoscaling_env import CloudAutoscalingEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def generate_workload(length: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate synthetic cloud workload."""
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, length)
    
    # Combine patterns
    daily_pattern = 50 + 30 * np.sin(t)
    weekly_pattern = 10 * np.sin(t / 7)
    noise = np.random.normal(0, 5, length)
    spikes = np.random.choice([0, 20], size=length, p=[0.95, 0.05])
    
    workload = daily_pattern + weekly_pattern + noise + spikes
    return np.clip(workload, 10, 100)


def train():
    """Main training function called by wandb sweep agent."""
    logger = setup_logging()
    
    # Initialize wandb - sweep agent will have already set the config
    run = wandb.init()
    config = wandb.config
    
    logger.info(f"Starting training with config: {dict(config)}")
    
    # Extract hyperparameters from wandb config
    algorithm = config.get("algorithm", "q-learning")
    learning_rate = config.get("learning_rate", 0.1)
    discount_factor = config.get("discount_factor", 0.95)
    epsilon_decay = config.get("epsilon_decay", 0.995)
    epsilon_init = config.get("epsilon_init", 1.0)
    n_episodes = config.get("n_episodes", 500)
    seed = config.get("seed", 42)
    
    # Create environment
    workload = generate_workload(seed=seed)
    env = CloudAutoscalingEnv(workload_data=workload)
    
    # Create agent based on algorithm
    agent_kwargs = {
        "state_space_shape": (3, 5, 3),
        "n_actions": 3,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon_init,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": 0.01,
        "seed": seed,
    }
    
    if algorithm == "sarsa":
        agent = SARSAAgent(**agent_kwargs)
    else:
        agent = QLearningAgent(**agent_kwargs)
    
    logger.info(f"Training {algorithm} agent for {n_episodes} episodes")
    
    # Training loop
    episode_rewards = []
    episode_sla = []
    
    for ep in tqdm(range(n_episodes), desc=f"{algorithm} training", leave=False):
        state, info = env.reset()
        
        if algorithm == "sarsa":
            action = agent.select_action(state)
        
        total_reward = 0
        sla_violations = 0
        done = False
        
        while not done:
            if algorithm == "sarsa":
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, terminated)
                action = next_action
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
            
            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_sla.append(sla_violations)
        
        # Log to wandb
        wandb.log({
            "episode": ep,
            "reward": total_reward,
            "sla_violations": sla_violations,
            "epsilon": agent.epsilon,
            "cumulative_reward": sum(episode_rewards),
        })
    
    # Calculate final metrics
    final_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_sla = np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    best_reward = max(episode_rewards)
    
    # Log summary metrics
    wandb.summary["final_reward"] = final_reward
    wandb.summary["final_sla"] = final_sla
    wandb.summary["best_reward"] = best_reward
    wandb.summary["total_episodes"] = n_episodes
    wandb.summary["final_epsilon"] = agent.epsilon
    
    logger.info(f"Training complete! Final reward: {final_reward:.2f}")
    
    wandb.finish()


def main():
    """Main entry point."""
    if not WANDB_AVAILABLE:
        print("Error: wandb not installed. Please run: pip install wandb")
        return 1
    
    train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
