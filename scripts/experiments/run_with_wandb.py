#!/usr/bin/env python3
"""
Run experiments with wandb logging integration.

This script wraps the existing experiment runners with wandb tracking.
Supports both individual runs and experiment groups.

Usage:
    python scripts/run_with_wandb.py --algo all --episodes 500
    python scripts/run_with_wandb.py --algo q-learning --episodes 1000
    python scripts/run_with_wandb.py --algo deep --episodes 500 --offline
"""

import argparse
import logging
import sys
import time
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
    log_model,
    finish_run,
    load_wandb_key,
    WANDB_AVAILABLE,
)

if WANDB_AVAILABLE:
    import wandb

from agent.cloud_autoscaling_env import CloudAutoscalingEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent
from agent.baseline_policies import RandomPolicy, ThresholdPolicy
from agent.deep_rl_agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent


LOGS_DIR = PROJECT_ROOT / "artifacts" / "logs"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"wandb_run_{timestamp}.log"

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

    daily_pattern = 50 + 30 * np.sin(t)
    weekly_pattern = 10 * np.sin(t / 7)
    noise = np.random.normal(0, 5, length)
    spikes = np.random.choice([0, 20], size=length, p=[0.95, 0.05])

    workload = daily_pattern + weekly_pattern + noise + spikes
    return np.clip(workload, 10, 100)


def run_baseline_with_wandb(
    env: CloudAutoscalingEnv,
    policy_name: str,
    n_episodes: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run a baseline policy with wandb logging."""

    # Initialize wandb run
    config = {
        "algorithm": policy_name,
        "n_episodes": n_episodes,
        "policy_type": "baseline",
    }

    init_wandb(
        name=f"{policy_name}-baseline",
        config=config,
        tags=["baseline", policy_name],
        group="baselines",
        job_type="eval",
    )

    # Create policy
    if policy_name == "random":
        policy = RandomPolicy(seed=42)
    else:
        policy = ThresholdPolicy()

    rewards = []
    sla_violations_list = []

    for ep in tqdm(range(n_episodes), desc=policy_name, leave=False):
        state, info = env.reset()
        total_reward = 0
        sla_violations = 0
        done = False

        while not done:
            action = policy.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated

        rewards.append(total_reward)
        sla_violations_list.append(sla_violations)

        log_episode(
            episode=ep,
            reward=total_reward,
            sla_violations=sla_violations,
        )

    # Log summary
    log_summary(
        final_reward=np.mean(rewards),
        final_sla=np.mean(sla_violations_list),
        additional_summary={
            "std_reward": np.std(rewards),
            "std_sla": np.std(sla_violations_list),
        },
    )

    finish_run()

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_sla": np.mean(sla_violations_list),
    }


def run_q_learning_with_wandb(
    env: CloudAutoscalingEnv,
    n_episodes: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Run Q-Learning with wandb logging."""

    config = {
        "algorithm": "q-learning",
        "n_episodes": n_episodes,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon_decay": epsilon_decay,
        "seed": seed,
    }

    init_wandb(
        name=f"q-learning-lr{learning_rate}-g{discount_factor}",
        config=config,
        tags=["tabular", "q-learning"],
        group="q-learning",
        job_type="train",
    )

    agent = QLearningAgent(
        state_space_shape=(3, 5, 3),
        n_actions=3,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        seed=seed,
    )

    start_time = time.time()
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
            agent.update(state, action, reward, next_state, terminated)

            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            done = terminated or truncated

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_sla.append(sla_violations)

        log_episode(
            episode=ep,
            reward=total_reward,
            sla_violations=sla_violations,
            epsilon=agent.epsilon,
        )

    training_time = time.time() - start_time
    final_reward = (
        np.mean(episode_rewards[-100:])
        if len(episode_rewards) >= 100
        else np.mean(episode_rewards)
    )
    final_sla = (
        np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    )

    log_summary(
        final_reward=final_reward,
        final_sla=final_sla,
        training_time=training_time,
        best_reward=max(episode_rewards),
    )

    finish_run()

    return {
        "episode_rewards": episode_rewards,
        "episode_sla": episode_sla,
        "mean_reward": final_reward,
        "mean_sla": final_sla,
        "training_time": training_time,
    }


def run_sarsa_with_wandb(
    env: CloudAutoscalingEnv,
    n_episodes: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Run SARSA with wandb logging."""

    config = {
        "algorithm": "sarsa",
        "n_episodes": n_episodes,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon_decay": epsilon_decay,
        "seed": seed,
    }

    init_wandb(
        name=f"sarsa-lr{learning_rate}-g{discount_factor}",
        config=config,
        tags=["tabular", "sarsa"],
        group="sarsa",
        job_type="train",
    )

    agent = SARSAAgent(
        state_space_shape=(3, 5, 3),
        n_actions=3,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        seed=seed,
    )

    start_time = time.time()
    episode_rewards = []
    episode_sla = []

    for ep in tqdm(range(n_episodes), desc="SARSA", leave=False):
        state, info = env.reset()
        action = agent.select_action(state)
        total_reward = 0
        sla_violations = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)

            total_reward += reward
            sla_violations += info.get("sla_violation", 0)
            state = next_state
            action = next_action
            done = terminated or truncated

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_sla.append(sla_violations)

        log_episode(
            episode=ep,
            reward=total_reward,
            sla_violations=sla_violations,
            epsilon=agent.epsilon,
        )

    training_time = time.time() - start_time
    final_reward = (
        np.mean(episode_rewards[-100:])
        if len(episode_rewards) >= 100
        else np.mean(episode_rewards)
    )
    final_sla = (
        np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    )

    log_summary(
        final_reward=final_reward,
        final_sla=final_sla,
        training_time=training_time,
        best_reward=max(episode_rewards),
    )

    finish_run()

    return {
        "episode_rewards": episode_rewards,
        "episode_sla": episode_sla,
        "mean_reward": final_reward,
        "mean_sla": final_sla,
        "training_time": training_time,
    }


def run_dqn_with_wandb(
    env: CloudAutoscalingEnv,
    n_episodes: int,
    agent_type: str = "dqn",
    learning_rate: float = 1e-3,
    discount_factor: float = 0.99,
    batch_size: int = 64,
    hidden_dim: int = 128,
    seed: int = 42,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Run DQN variants with wandb logging."""

    config = {
        "algorithm": agent_type,
        "n_episodes": n_episodes,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "seed": seed,
    }

    init_wandb(
        name=f"{agent_type}-lr{learning_rate:.0e}",
        config=config,
        tags=["deep-rl", agent_type],
        group="deep-rl",
        job_type="train",
    )

    # Select agent class
    agent_classes = {
        "dqn": DQNAgent,
        "double-dqn": DoubleDQNAgent,
        "dueling-dqn": DuelingDQNAgent,
    }
    agent_class = agent_classes.get(agent_type, DQNAgent)

    state_dim = len(env.observation_space.nvec)
    action_dim = env.action_space.n

    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        batch_size=batch_size,
        hidden_dims=(hidden_dim, hidden_dim),
        seed=seed,
    )

    start_time = time.time()
    episode_rewards = []
    episode_sla = []

    for ep in tqdm(range(n_episodes), desc=agent_type.upper(), leave=False):
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

        log_episode(
            episode=ep,
            reward=total_reward,
            sla_violations=sla_violations,
            epsilon=agent.epsilon,
            loss=np.mean(losses) if losses else None,
        )

    training_time = time.time() - start_time
    final_reward = (
        np.mean(episode_rewards[-100:])
        if len(episode_rewards) >= 100
        else np.mean(episode_rewards)
    )
    final_sla = (
        np.mean(episode_sla[-100:]) if len(episode_sla) >= 100 else np.mean(episode_sla)
    )

    # Save and log model
    model_path = MODELS_DIR / f"{agent_type}_wandb.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))
    log_model(model_path, name=f"{agent_type}-model", artifact_type="model")

    log_summary(
        final_reward=final_reward,
        final_sla=final_sla,
        training_time=training_time,
        best_reward=max(episode_rewards),
    )

    finish_run()

    return {
        "episode_rewards": episode_rewards,
        "episode_sla": episode_sla,
        "mean_reward": final_reward,
        "mean_sla": final_sla,
        "training_time": training_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Cloud Autoscaling RL experiments with wandb logging"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=[
            "all",
            "baselines",
            "tabular",
            "deep",
            "q-learning",
            "sarsa",
            "dqn",
            "double-dqn",
            "dueling-dqn",
        ],
        default="all",
        help="Algorithm(s) to run",
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test run (100 episodes)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no wandb sync)"
    )
    parser.add_argument(
        "--project", type=str, default="cloud-autoscaling-rl", help="Wandb project name"
    )

    args = parser.parse_args()

    logger = setup_logging()

    if args.quick:
        args.episodes = 100

    logger.info("=" * 60)
    logger.info("Cloud Autoscaling RL - Wandb Experiment Runner")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {args.algo}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Mode: {'offline' if args.offline else 'online'}")

    if not WANDB_AVAILABLE:
        logger.error("wandb not installed. Please run: pip install wandb")
        return 1

    # Set wandb mode
    if args.offline:
        import os

        os.environ["WANDB_MODE"] = "offline"

    # Login to wandb
    api_key = load_wandb_key()
    if api_key and not args.offline:
        wandb.login(key=api_key)

    # Create environment
    workload = generate_workload(seed=args.seed)
    env = CloudAutoscalingEnv(workload_data=workload)

    results = {}

    # Run selected algorithms
    if args.algo in ["all", "baselines"]:
        logger.info("\n📊 Running Baseline Policies...")
        results["random"] = run_baseline_with_wandb(
            env, "random", args.episodes, logger
        )
        results["threshold"] = run_baseline_with_wandb(
            env, "threshold", args.episodes, logger
        )

    if args.algo in ["all", "tabular", "q-learning"]:
        logger.info("\n🎯 Running Q-Learning...")
        results["q-learning"] = run_q_learning_with_wandb(
            env, args.episodes, seed=args.seed, logger=logger
        )

    if args.algo in ["all", "tabular", "sarsa"]:
        logger.info("\n🎯 Running SARSA...")
        results["sarsa"] = run_sarsa_with_wandb(
            env, args.episodes, seed=args.seed, logger=logger
        )

    if args.algo in ["all", "deep", "dqn"]:
        logger.info("\n🧠 Running DQN...")
        results["dqn"] = run_dqn_with_wandb(
            env, args.episodes, agent_type="dqn", seed=args.seed, logger=logger
        )

    if args.algo in ["all", "deep", "double-dqn"]:
        logger.info("\n🧠 Running Double DQN...")
        results["double-dqn"] = run_dqn_with_wandb(
            env, args.episodes, agent_type="double-dqn", seed=args.seed, logger=logger
        )

    if args.algo in ["all", "deep", "dueling-dqn"]:
        logger.info("\n🧠 Running Dueling DQN...")
        results["dueling-dqn"] = run_dqn_with_wandb(
            env, args.episodes, agent_type="dueling-dqn", seed=args.seed, logger=logger
        )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    for algo, result in results.items():
        reward = result.get("mean_reward", 0)
        sla = result.get("mean_sla", 0)
        logger.info(f"  {algo:20s}: reward={reward:8.2f}, sla={sla:.2f}")

    logger.info("\n✅ All experiments complete!")
    logger.info(f"📊 View results at: https://wandb.ai/{args.project}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
