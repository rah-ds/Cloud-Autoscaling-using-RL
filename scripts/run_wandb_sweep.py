#!/usr/bin/env python3
"""
Wandb sweep runner for Cloud Autoscaling RL hyperparameter optimization.

This script creates and runs wandb sweeps for hyperparameter tuning.

Usage:
    python scripts/run_wandb_sweep.py --sweep q-learning --count 20
    python scripts/run_wandb_sweep.py --sweep dqn --count 50
    python scripts/run_wandb_sweep.py --sweep full --count 100
    python scripts/run_wandb_sweep.py --create-only --sweep q-learning  # Just create sweep
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wandb_utils import (
    init_wandb,
    log_episode,
    log_summary,
    log_learning_curve,
    finish_run,
    create_sweep,
    run_sweep_agent,
    get_sweep_config,
    load_wandb_key,
    WANDB_AVAILABLE,
)

if WANDB_AVAILABLE:
    import wandb

from agent.cloud_autoscaling_env import CloudAutoscalingEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent
from agent.deep_rl_agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent


LOGS_DIR = PROJECT_ROOT / "artifacts" / "logs"


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"wandb_sweep_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
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


def train_tabular_agent(config: Dict[str, Any] = None) -> None:
    """Training function for tabular RL agents (Q-Learning, SARSA)."""
    # Initialize wandb run with sweep config
    run = init_wandb(
        config=config,
        job_type="sweep",
        group="hyperparameter-sweep",
    )
    
    if run is None:
        print("Failed to initialize wandb")
        return
    
    # Get config from wandb
    config = wandb.config
    
    # Extract hyperparameters
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
    
    # Training loop
    episode_rewards = []
    episode_sla = []
    
    for ep in tqdm(range(n_episodes), desc=f"{algorithm} sweep", leave=False):
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
        log_episode(
            episode=ep,
            reward=total_reward,
            sla_violations=sla_violations,
            epsilon=agent.epsilon,
        )
    
    # Calculate final metrics
    final_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_sla = np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    best_reward = max(episode_rewards)
    
    # Log summary
    log_summary(
        final_reward=final_reward,
        final_sla=final_sla,
        best_reward=best_reward,
        additional_summary={
            "total_episodes": n_episodes,
            "final_epsilon": agent.epsilon,
        }
    )
    
    finish_run()


def train_deep_agent(config: Dict[str, Any] = None) -> None:
    """Training function for deep RL agents (DQN, Double DQN, Dueling DQN)."""
    # Initialize wandb run with sweep config
    run = init_wandb(
        config=config,
        job_type="sweep",
        group="deep-rl-sweep",
    )
    
    if run is None:
        print("Failed to initialize wandb")
        return
    
    # Get config from wandb
    config = wandb.config
    
    # Extract hyperparameters
    algorithm = config.get("algorithm", "dqn")
    learning_rate = config.get("learning_rate", 1e-3)
    discount_factor = config.get("discount_factor", 0.99)
    batch_size = config.get("batch_size", 64)
    hidden_dim = config.get("hidden_dim", 128)
    target_update_freq = config.get("target_update_freq", 50)
    n_episodes = config.get("n_episodes", 500)
    seed = config.get("seed", 42)
    
    # Create environment
    workload = generate_workload(seed=seed)
    env = CloudAutoscalingEnv(workload_data=workload)
    
    # Select agent class
    agent_classes = {
        "dqn": DQNAgent,
        "double-dqn": DoubleDQNAgent,
        "dueling-dqn": DuelingDQNAgent,
    }
    agent_class = agent_classes.get(algorithm, DQNAgent)
    
    # Create agent
    state_dim = len(env.observation_space.nvec)
    action_dim = env.action_space.n
    
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        batch_size=batch_size,
        hidden_dims=(hidden_dim, hidden_dim),
        target_update_freq=target_update_freq,
        seed=seed,
    )
    
    # Training loop
    episode_rewards = []
    episode_sla = []
    
    for ep in tqdm(range(n_episodes), desc=f"{algorithm} sweep", leave=False):
        state, info = env.reset()
        total_reward = 0
        sla_violations = 0
        done = False
        losses = []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # DQN update method stores transition internally
            loss = agent.update(state, action, reward, next_state, terminated)
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_sla.append(sla_violations)
        
        # Log to wandb
        log_episode(
            episode=ep,
            reward=total_reward,
            sla_violations=sla_violations,
            epsilon=agent.epsilon,
            loss=np.mean(losses) if losses else None,
        )
    
    # Calculate final metrics
    final_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_sla = np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    best_reward = max(episode_rewards)
    
    # Log summary
    log_summary(
        final_reward=final_reward,
        final_sla=final_sla,
        best_reward=best_reward,
        additional_summary={
            "total_episodes": n_episodes,
            "final_epsilon": agent.epsilon,
        }
    )
    
    finish_run()


def train_with_sweep_config() -> None:
    """
    Universal training function that routes to appropriate agent based on config.
    This is the function called by wandb.agent().
    """
    if not WANDB_AVAILABLE:
        print("wandb not available")
        return
    
    # Initialize run first to get config
    run = init_wandb(
        job_type="sweep",
        group="hyperparameter-sweep",
    )
    
    if run is None:
        return
    
    config = wandb.config
    algorithm = config.get("algorithm", "q-learning")
    
    # Close this run - the specific train function will create its own
    finish_run(quiet=True)
    
    # Route to appropriate training function
    if algorithm in ["q-learning", "sarsa"]:
        train_tabular_agent()
    else:
        train_deep_agent()


def main():
    parser = argparse.ArgumentParser(
        description="Run wandb hyperparameter sweeps for Cloud Autoscaling RL"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["q-learning", "sarsa", "dqn", "full"],
        default="q-learning",
        help="Sweep configuration to use"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep runs to execute"
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create the sweep, don't run agents"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Join an existing sweep instead of creating a new one"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="cloud-autoscaling-rl",
        help="Wandb project name"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Starting wandb sweep runner")
    logger.info(f"  Sweep config: {args.sweep}")
    logger.info(f"  Count: {args.count}")
    
    if not WANDB_AVAILABLE:
        logger.error("wandb not installed. Please run: pip install wandb")
        return 1
    
    # Login to wandb
    api_key = load_wandb_key()
    if not api_key:
        logger.error("No wandb API key found. Please add it to keys/wandb_key.txt")
        return 1
    
    wandb.login(key=api_key)
    
    # Get or create sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        logger.info(f"Joining existing sweep: {sweep_id}")
    else:
        sweep_config = get_sweep_config(args.sweep)
        sweep_id = create_sweep(sweep_config, project=args.project)
        logger.info(f"Created sweep: {sweep_id}")
        print(f"\n✅ Sweep ID: {sweep_id}")
        print(f"   View at: https://wandb.ai/{args.project}/sweeps/{sweep_id}")
    
    if args.create_only:
        logger.info("Sweep created. Run agents with --sweep-id option.")
        print(f"\n📋 To run agents for this sweep:")
        print(f"   python scripts/run_wandb_sweep.py --sweep-id {sweep_id} --count {args.count}")
        return 0
    
    # Select training function based on sweep type
    if args.sweep in ["q-learning", "sarsa"]:
        train_func = train_tabular_agent
    elif args.sweep == "dqn":
        train_func = train_deep_agent
    else:
        train_func = train_with_sweep_config
    
    # Run sweep agent
    logger.info(f"Starting {args.count} sweep runs...")
    run_sweep_agent(
        sweep_id=sweep_id,
        function=train_func,
        count=args.count,
        project=args.project,
    )
    
    logger.info("Sweep complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
