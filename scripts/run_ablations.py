#!/usr/bin/env python3
"""
Run ablation studies for Cloud Autoscaling RL project.

This script runs systematic ablation studies to understand the impact
of different hyperparameters and components on agent performance.

Usage:
    python scripts/run_ablations.py                     # Run default ablations
    python scripts/run_ablations.py --quick             # Quick test run
    python scripts/run_ablations.py --study lr          # Run learning rate ablation
    python scripts/run_ablations.py --study exploration # Run exploration ablation
    python scripts/run_ablations.py --study all         # Run all ablation studies
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.cloud_autoscaling_env import CloudAutoscalingEnv
from agent.q_learning_agent import QLearningAgent, train_q_learning, evaluate_agent
from agent.sarsa_agent import SARSAAgent, train_sarsa, evaluate_agent as evaluate_sarsa_agent
from ablations import (
    AblationStudy,
    run_hyperparameter_ablation,
    run_component_ablation,
    run_grid_ablation,
    plot_ablation_results,
    plot_ablation_heatmap,
    plot_learning_curve_comparison,
    create_ablation_table,
    get_learning_rate_ablation_values,
    get_discount_factor_ablation_values,
    get_epsilon_decay_ablation_values,
    get_exploration_ablation_configs,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    log_dir = PROJECT_ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ablation_{timestamp}.log"
    
    logger = logging.getLogger("ablations")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# Factory functions for ablation studies
def create_env(seed: int = 42) -> CloudAutoscalingEnv:
    """Create environment with given seed."""
    workload = generate_workload(length=1000, seed=seed)
    env = CloudAutoscalingEnv(workload_data=workload)
    return env


def generate_workload(length: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate synthetic workload."""
    np.random.seed(seed)
    
    # Base load with daily pattern
    t = np.linspace(0, 4 * np.pi, length)
    base_load = 50 + 30 * np.sin(t)
    
    # Add noise and spikes
    noise = np.random.normal(0, 5, length)
    spikes = np.random.choice([0, 1], size=length, p=[0.95, 0.05]) * np.random.uniform(20, 50, length)
    
    workload = base_load + noise + spikes
    return np.clip(workload, 0, 100)


def create_q_learning_agent(
    state_space_shape=(3, 5, 3),
    n_actions=3,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    seed=42,
    **kwargs  # Ignore extra params
) -> QLearningAgent:
    """Create Q-Learning agent with given parameters."""
    return QLearningAgent(
        state_space_shape=state_space_shape,
        n_actions=n_actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def create_sarsa_agent(
    state_space_shape=(3, 5, 3),
    n_actions=3,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    seed=42,
    **kwargs
) -> SARSAAgent:
    """Create SARSA agent with given parameters."""
    return SARSAAgent(
        state_space_shape=state_space_shape,
        n_actions=n_actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


# Wrapper for train functions to match expected signature
def train_q_wrapper(env, agent, n_episodes=500, verbose=False):
    """Wrapper for Q-learning training."""
    return train_q_learning(env, agent, n_episodes=n_episodes, verbose=verbose, verbose_freq=100)


def train_sarsa_wrapper(env, agent, n_episodes=500, verbose=False):
    """Wrapper for SARSA training."""
    return train_sarsa(env, agent, n_episodes=n_episodes, verbose=verbose, verbose_freq=100)


def eval_q_wrapper(env, agent, n_episodes=50, verbose=False):
    """Wrapper for Q-learning evaluation."""
    return evaluate_agent(env, agent, n_episodes=n_episodes, verbose=verbose)


def eval_sarsa_wrapper(env, agent, n_episodes=50, verbose=False):
    """Wrapper for SARSA evaluation."""
    return evaluate_sarsa_agent(env, agent, n_episodes=n_episodes, verbose=verbose)


def run_learning_rate_ablation(
    results_dir: Path,
    plots_dir: Path,
    n_episodes: int = 500,
    n_seeds: int = 3,
    algorithm: str = "q-learning",
    logger: logging.Logger = None,
) -> AblationStudy:
    """Run learning rate ablation study."""
    logger = logger or logging.getLogger("ablations")
    
    logger.info("=" * 60)
    logger.info(f"Running Learning Rate Ablation ({algorithm})")
    logger.info(f"Episodes: {n_episodes}, Seeds: {n_seeds}")
    
    base_params = {
        "state_space_shape": (3, 5, 3),
        "n_actions": 3,
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
    }
    
    if algorithm == "q-learning":
        agent_factory = create_q_learning_agent
        train_fn = train_q_wrapper
        eval_fn = eval_q_wrapper
    else:
        agent_factory = create_sarsa_agent
        train_fn = train_sarsa_wrapper
        eval_fn = eval_sarsa_wrapper
    
    study = run_hyperparameter_ablation(
        env_factory=create_env,
        agent_factory=agent_factory,
        train_fn=train_fn,
        eval_fn=eval_fn,
        param_name="learning_rate",
        param_values=get_learning_rate_ablation_values(),
        base_params=base_params,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
        study_name=f"learning_rate_{algorithm}",
    )
    
    # Save and plot
    study.save(results_dir)
    plot_ablation_results(study, output_path=plots_dir / f"ablation_lr_{algorithm}.png")
    plot_learning_curve_comparison(study, output_path=plots_dir / f"ablation_lr_curves_{algorithm}.png")
    
    # Print table
    logger.info("\nLearning Rate Ablation Results:")
    logger.info("\n" + create_ablation_table(study, sort_by="mean_reward_mean"))
    
    best = study.get_best_config("mean_reward_mean")
    logger.info(f"\nBest learning rate: {best.config.params['learning_rate']}")
    
    return study


def run_discount_factor_ablation(
    results_dir: Path,
    plots_dir: Path,
    n_episodes: int = 500,
    n_seeds: int = 3,
    algorithm: str = "q-learning",
    logger: logging.Logger = None,
) -> AblationStudy:
    """Run discount factor ablation study."""
    logger = logger or logging.getLogger("ablations")
    
    logger.info("=" * 60)
    logger.info(f"Running Discount Factor Ablation ({algorithm})")
    
    base_params = {
        "state_space_shape": (3, 5, 3),
        "n_actions": 3,
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
    }
    
    if algorithm == "q-learning":
        agent_factory = create_q_learning_agent
        train_fn = train_q_wrapper
        eval_fn = eval_q_wrapper
    else:
        agent_factory = create_sarsa_agent
        train_fn = train_sarsa_wrapper
        eval_fn = eval_sarsa_wrapper
    
    study = run_hyperparameter_ablation(
        env_factory=create_env,
        agent_factory=agent_factory,
        train_fn=train_fn,
        eval_fn=eval_fn,
        param_name="discount_factor",
        param_values=get_discount_factor_ablation_values(),
        base_params=base_params,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
        study_name=f"discount_factor_{algorithm}",
    )
    
    study.save(results_dir)
    plot_ablation_results(study, output_path=plots_dir / f"ablation_gamma_{algorithm}.png")
    
    logger.info("\nDiscount Factor Ablation Results:")
    logger.info("\n" + create_ablation_table(study, sort_by="mean_reward_mean"))
    
    return study


def run_exploration_ablation(
    results_dir: Path,
    plots_dir: Path,
    n_episodes: int = 500,
    n_seeds: int = 3,
    algorithm: str = "q-learning",
    logger: logging.Logger = None,
) -> AblationStudy:
    """Run exploration strategy ablation study."""
    logger = logger or logging.getLogger("ablations")
    
    logger.info("=" * 60)
    logger.info(f"Running Exploration Ablation ({algorithm})")
    
    base_params = {
        "state_space_shape": (3, 5, 3),
        "n_actions": 3,
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
    }
    
    if algorithm == "q-learning":
        agent_factory = create_q_learning_agent
        train_fn = train_q_wrapper
        eval_fn = eval_q_wrapper
    else:
        agent_factory = create_sarsa_agent
        train_fn = train_sarsa_wrapper
        eval_fn = eval_sarsa_wrapper
    
    study = run_component_ablation(
        env_factory=create_env,
        agent_factory=agent_factory,
        train_fn=train_fn,
        eval_fn=eval_fn,
        components=get_exploration_ablation_configs(),
        base_params=base_params,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
    )
    study.name = f"exploration_{algorithm}"
    
    study.save(results_dir)
    plot_ablation_results(study, output_path=plots_dir / f"ablation_exploration_{algorithm}.png")
    
    logger.info("\nExploration Ablation Results:")
    logger.info("\n" + create_ablation_table(study, sort_by="mean_reward_mean"))
    
    return study


def run_grid_ablation_lr_gamma(
    results_dir: Path,
    plots_dir: Path,
    n_episodes: int = 500,
    n_seeds: int = 1,
    algorithm: str = "q-learning",
    logger: logging.Logger = None,
) -> AblationStudy:
    """Run grid ablation over learning rate and discount factor."""
    logger = logger or logging.getLogger("ablations")
    
    logger.info("=" * 60)
    logger.info(f"Running Grid Ablation: LR x Gamma ({algorithm})")
    
    base_params = {
        "state_space_shape": (3, 5, 3),
        "n_actions": 3,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
    }
    
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.3],
        "discount_factor": [0.9, 0.95, 0.99],
    }
    
    if algorithm == "q-learning":
        agent_factory = create_q_learning_agent
        train_fn = train_q_wrapper
        eval_fn = eval_q_wrapper
    else:
        agent_factory = create_sarsa_agent
        train_fn = train_sarsa_wrapper
        eval_fn = eval_sarsa_wrapper
    
    study = run_grid_ablation(
        env_factory=create_env,
        agent_factory=agent_factory,
        train_fn=train_fn,
        eval_fn=eval_fn,
        param_grid=param_grid,
        base_params=base_params,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
    )
    study.name = f"grid_lr_gamma_{algorithm}"
    
    study.save(results_dir)
    plot_ablation_heatmap(
        study,
        param1="learning_rate",
        param2="discount_factor",
        output_path=plots_dir / f"ablation_grid_{algorithm}.png"
    )
    
    logger.info("\nGrid Ablation Results:")
    logger.info("\n" + create_ablation_table(study, sort_by="mean_reward_mean"))
    
    return study


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with fewer episodes/seeds")
    parser.add_argument("--study", type=str, default="all",
                        choices=["lr", "gamma", "exploration", "grid", "all"],
                        help="Which ablation study to run")
    parser.add_argument("--algorithm", type=str, default="q-learning",
                        choices=["q-learning", "sarsa"],
                        help="Algorithm to ablate")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes per configuration")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Setup directories
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.log_level)
    
    # Adjust for quick mode
    n_episodes = 100 if args.quick else args.episodes
    n_seeds = 1 if args.quick else args.seeds
    
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("Cloud Autoscaling RL - Ablation Studies")
    logger.info("=" * 60)
    logger.info(f"Study: {args.study}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Plots dir: {plots_dir}")
    
    studies = []
    
    # Run selected studies
    if args.study in ["lr", "all"]:
        study = run_learning_rate_ablation(
            results_dir, plots_dir, n_episodes, n_seeds, args.algorithm, logger
        )
        studies.append(study)
    
    if args.study in ["gamma", "all"]:
        study = run_discount_factor_ablation(
            results_dir, plots_dir, n_episodes, n_seeds, args.algorithm, logger
        )
        studies.append(study)
    
    if args.study in ["exploration", "all"]:
        study = run_exploration_ablation(
            results_dir, plots_dir, n_episodes, n_seeds, args.algorithm, logger
        )
        studies.append(study)
    
    if args.study in ["grid", "all"]:
        study = run_grid_ablation_lr_gamma(
            results_dir, plots_dir, n_episodes, n_seeds, args.algorithm, logger
        )
        studies.append(study)
    
    # Summary
    elapsed = datetime.now() - start_time
    
    logger.info("=" * 60)
    logger.info("ABLATION STUDIES COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {elapsed}")
    logger.info(f"Studies completed: {len(studies)}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Plots directory: {plots_dir}")
    
    # Print best configs from each study
    logger.info("\nBest Configurations:")
    for study in studies:
        best = study.get_best_config("mean_reward_mean")
        logger.info(f"  {study.name}: {best.config.name} (reward={best.metrics['mean_reward_mean']:.2f})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
