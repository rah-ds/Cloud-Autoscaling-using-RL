"""
Optuna-based Bayesian optimization for hyperparameter tuning.

This module provides:
- Hyperparameter optimization for DQN agents
- Integration with wandb for logging
- Pruning of unpromising trials
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

# Add project root to path for imports when running as script
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.cloud_autoscaling_env import CloudAutoscalingEnv
from src.agent.deep_rl_agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from src.wandb_utils import (
    WANDB_AVAILABLE,
    finish_run,
    init_wandb,
    load_wandb_key,
    log_episode,
)

# Agent mapping
AGENTS = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
}


def create_objective(
    agent_type: str = "dqn",
    n_episodes: int = 500,
    seed: int = 42,
    use_wandb: bool = True,
):
    """
    Create an Optuna objective function for hyperparameter optimization.

    Args:
        agent_type: Type of agent ("dqn", "double_dqn", "dueling_dqn")
        n_episodes: Number of training episodes
        seed: Random seed
        use_wandb: Whether to log to wandb

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        discount_factor = trial.suggest_categorical("discount_factor", [0.95, 0.99])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        target_update_freq = trial.suggest_categorical("target_update_freq", [10, 50, 100])
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)

        # Create environment
        env = CloudAutoscalingEnv(seed=seed)
        env.reset(seed=seed)
        np.random.seed(seed)

        # Create agent
        agent_class = AGENTS[agent_type]
        agent = agent_class(
            state_dim=len(env.observation_space.nvec),
            action_dim=env.action_space.n,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            hidden_dims=(hidden_dim, hidden_dim),
            target_update_freq=target_update_freq,
            epsilon=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.01,
            seed=seed,
        )

        # Initialize wandb for this trial
        if use_wandb and WANDB_AVAILABLE:
            init_wandb(
                name=f"optuna_trial_{trial.number}",
                config={
                    "agent_type": agent_type,
                    "trial_number": trial.number,
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "target_update_freq": target_update_freq,
                    "epsilon_decay": epsilon_decay,
                    "n_episodes": n_episodes,
                    "seed": seed,
                },
                tags=["optuna", agent_type],
                group="optuna_sweep",
            )

        # Training loop
        episode_rewards = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                log_episode(
                    episode=episode,
                    reward=total_reward,
                    epsilon=agent.epsilon,
                )

            # Report intermediate value for pruning
            if episode % 50 == 0 and episode > 0:
                intermediate_value = np.mean(episode_rewards[-50:])
                trial.report(intermediate_value, episode)

                # Prune if unpromising
                if trial.should_prune():
                    if use_wandb and WANDB_AVAILABLE:
                        finish_run()
                    raise optuna.TrialPruned()

        # Calculate final performance (last 10% of episodes)
        final_rewards = episode_rewards[int(len(episode_rewards) * 0.9) :]
        final_mean = np.mean(final_rewards)

        if use_wandb and WANDB_AVAILABLE:
            finish_run()

        return final_mean

    return objective


def run_optuna_optimization(
    agent_type: str = "dqn",
    n_trials: int = 50,
    n_episodes: int = 500,
    seed: int = 42,
    use_wandb: bool = True,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        agent_type: Type of agent to optimize
        n_trials: Number of optimization trials
        n_episodes: Episodes per trial
        seed: Random seed
        use_wandb: Whether to log to wandb
        study_name: Name for the Optuna study
        storage: Database URL for persistent storage (e.g., "sqlite:///optuna.db")
        pruner: Optuna pruner for early stopping

    Returns:
        Completed Optuna study
    """
    # Setup wandb key
    if use_wandb and WANDB_AVAILABLE:
        api_key = load_wandb_key()
        if api_key:
            import os

            os.environ["WANDB_API_KEY"] = api_key

    # Create pruner if not provided
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=50,
        )

    # Create or load study
    study_name = study_name or f"{agent_type}_optimization"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    # Create objective
    objective = create_objective(
        agent_type=agent_type,
        n_episodes=n_episodes,
        seed=seed,
        use_wandb=use_wandb,
    )

    # Setup wandb callback if available
    callbacks = []
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_callback = WeightsAndBiasesCallback(
                metric_name="final_reward",
                wandb_kwargs={
                    "project": "cloud-autoscaling-rl",
                    "group": f"optuna_{agent_type}",
                },
            )
            callbacks.append(wandb_callback)
        except Exception:
            pass  # Continue without wandb callback if it fails

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks if callbacks else None,
        show_progress_bar=True,
    )

    return study


def print_optimization_results(study: optuna.Study) -> Dict[str, Any]:
    """Print and return optimization results."""
    print("\n" + "=" * 60)
    print("OPTUNA OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (final reward): {study.best_value:.2f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nTotal trials: {len(study.trials)}")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Completed: {len(complete_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")

    return {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_complete": len(complete_trials),
        "n_pruned": len(pruned_trials),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=["dqn", "double_dqn", "dueling_dqn"],
        help="Agent type to optimize",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 trials, 100 episodes)")
    args = parser.parse_args()

    if args.quick:
        n_trials = 10
        n_episodes = 100
    else:
        n_trials = args.trials
        n_episodes = args.episodes

    print(f"Running Optuna optimization for {args.agent}")
    print(f"  Trials: {n_trials}")
    print(f"  Episodes per trial: {n_episodes}")
    print(f"  Wandb: {'disabled' if args.no_wandb else 'enabled'}")

    study = run_optuna_optimization(
        agent_type=args.agent,
        n_trials=n_trials,
        n_episodes=n_episodes,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )

    results = print_optimization_results(study)
