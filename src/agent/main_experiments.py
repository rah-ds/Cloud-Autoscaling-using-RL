"""
Main Training Script for Cloud Autoscaling RL Project

This script implements experiments comparing SARSA and Q-Learning agents
with different hyperparameters for cloud resource autoscaling.
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from cloud_autoscaling_env import CloudAutoscalingEnv
from q_learning_agent import QLearningAgent, train_q_learning, evaluate_agent as eval_q
from sarsa_agent import SARSAAgent, train_sarsa, evaluate_agent as eval_sarsa
from baseline_policies import compare_baselines


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"main_experiments_{timestamp}.log"

    logger = logging.getLogger("main_experiments")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # File handler
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


# Global logger
logger = logging.getLogger("main_experiments")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def experiment_1_basic_training(workload_data, n_episodes=1000):
    """
    Experiment 1: Basic training of Q-Learning and SARSA agents.
    Compare performance with baseline policies.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 1: BASIC TRAINING")
    logger.info("=" * 80)
    logger.info(f"Episodes: {n_episodes}")

    # Create environment
    env = CloudAutoscalingEnv(workload_data=workload_data, seed=42)

    # Train Q-Learning agent
    logger.info("Training Q-Learning Agent...")
    q_agent = QLearningAgent(
        state_space_shape=(3, 5, 3),
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )

    q_agent, q_metrics = train_q_learning(
        env, q_agent, n_episodes=n_episodes, verbose=True, verbose_freq=100
    )
    logger.info(f"Q-Learning training complete. Final epsilon: {q_agent.epsilon:.4f}")

    # Train SARSA agent
    logger.info("Training SARSA Agent...")
    sarsa_agent = SARSAAgent(
        state_space_shape=(3, 5, 3),
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )

    sarsa_agent, sarsa_metrics = train_sarsa(
        env, sarsa_agent, n_episodes=n_episodes, verbose=True, verbose_freq=100
    )
    logger.info(f"SARSA training complete. Final epsilon: {sarsa_agent.epsilon:.4f}")

    # Evaluate agents
    logger.info("Evaluating agents...")
    q_eval = eval_q(env, q_agent, n_episodes=100, verbose=True)
    sarsa_eval = eval_sarsa(env, sarsa_agent, n_episodes=100, verbose=True)

    logger.info(f"Q-Learning eval: reward={q_eval['mean_reward']:.2f}")
    logger.info(f"SARSA eval: reward={sarsa_eval['mean_reward']:.2f}")

    # Compare with baselines
    logger.info("Comparing with baselines...")
    baseline_results = compare_baselines(env, n_episodes=100)

    # Save agents
    q_agent.save("models/q_learning_basic.pkl")
    sarsa_agent.save("models/sarsa_basic.pkl")
    logger.info("Models saved to models/ directory")

    # Save metrics
    results = {
        "q_learning": {"training": q_metrics, "evaluation": q_eval},
        "sarsa": {"training": sarsa_metrics, "evaluation": sarsa_eval},
        "baselines": baseline_results,
    }

    with open("results/experiment1_results.pkl", "wb") as f:
        pickle.dump(results, f)
    logger.info("Results saved to results/experiment1_results.pkl")

    return results


def experiment_2_exploration_rates(workload_data, n_episodes=1000):
    """
    Experiment 2: Compare different exploration rates (epsilon values).
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 2: EXPLORATION RATE COMPARISON")
    logger.info("=" * 80)

    epsilon_values = [1.0, 0.5, 0.3]
    epsilon_decays = [0.999, 0.995, 0.99]

    logger.info(f"Epsilon values: {epsilon_values}")
    logger.info(f"Epsilon decays: {epsilon_decays}")

    results = {}
    total_configs = len(epsilon_values) * len(epsilon_decays)
    config_num = 0

    with tqdm(total=total_configs, desc="Exploration Rate Sweep", leave=False) as pbar:
        for eps_init in epsilon_values:
            for eps_decay in epsilon_decays:
                config_num += 1
                config_name = f"eps_{eps_init}_decay_{eps_decay}"
                pbar.set_description(f"Sweep: eps={eps_init}, decay={eps_decay}")
                logger.info(
                    f"[{config_num}/{total_configs}] Configuration: {config_name}"
                )

            env = CloudAutoscalingEnv(workload_data=workload_data, seed=42)

            # Q-Learning
            q_agent = QLearningAgent(
                state_space_shape=(3, 5, 3),
                n_actions=3,
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=eps_init,
                epsilon_decay=eps_decay,
                epsilon_min=0.01,
                seed=42,
            )

            q_agent, q_metrics = train_q_learning(
                env, q_agent, n_episodes=n_episodes, verbose=False
            )

            q_eval = eval_q(env, q_agent, n_episodes=100, verbose=False)

            results[config_name] = {
                "config": {"epsilon": eps_init, "decay": eps_decay},
                "training": q_metrics,
                "evaluation": q_eval,
            }

            logger.info(
                f"  Result: reward={q_eval['mean_reward']:.2f}, sla={q_eval['mean_sla_violations']:.2f}"
            )
            pbar.update(1)

    with open("results/experiment2_results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info("Experiment 2 complete. Results saved.")

    return results


def experiment_3_episode_lengths(workload_data, episode_lengths=[500, 1000, 2000]):
    """
    Experiment 3: Compare different episode lengths.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 3: EPISODE LENGTH COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Episode lengths: {episode_lengths}")

    results = {}

    for i, n_episodes in tqdm(
        enumerate(episode_lengths),
        total=len(episode_lengths),
        desc="Episode Length Sweep",
        leave=False,
    ):
        logger.info(
            f"[{i + 1}/{len(episode_lengths)}] Training with {n_episodes} episodes"
        )

        env = CloudAutoscalingEnv(workload_data=workload_data, seed=42)

        # Q-Learning
        logger.debug("Training Q-Learning...")
        q_agent = QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=42,
        )

        q_agent, q_metrics = train_q_learning(
            env,
            q_agent,
            n_episodes=n_episodes,
            verbose=True,
            verbose_freq=max(100, n_episodes // 10),
        )

        q_eval = eval_q(env, q_agent, n_episodes=100, verbose=True)

        # SARSA
        sarsa_agent = SARSAAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=42,
        )

        sarsa_agent, sarsa_metrics = train_sarsa(
            env,
            sarsa_agent,
            n_episodes=n_episodes,
            verbose=True,
            verbose_freq=max(100, n_episodes // 10),
        )

        sarsa_eval = eval_sarsa(env, sarsa_agent, n_episodes=100, verbose=True)

        results[f"n_{n_episodes}"] = {
            "q_learning": {"training": q_metrics, "evaluation": q_eval},
            "sarsa": {"training": sarsa_metrics, "evaluation": sarsa_eval},
        }

        logger.info(f"  Q-Learning: reward={q_eval['mean_reward']:.2f}")
        logger.info(f"  SARSA: reward={sarsa_eval['mean_reward']:.2f}")

    with open("results/experiment3_results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info("Experiment 3 complete. Results saved.")

    return results


def experiment_4_learning_rates(workload_data, n_episodes=1000):
    """
    Experiment 4: Compare different learning rates.
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT 4: LEARNING RATE COMPARISON")
    logger.info("=" * 80)

    learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
    logger.info(f"Learning rates: {learning_rates}")

    results = {}

    for i, lr in tqdm(
        enumerate(learning_rates),
        total=len(learning_rates),
        desc="Learning Rate Sweep",
        leave=False,
    ):
        logger.info(f"[{i + 1}/{len(learning_rates)}] Learning Rate: {lr}")

        env = CloudAutoscalingEnv(workload_data=workload_data, seed=42)

        # Q-Learning
        q_agent = QLearningAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=lr,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=42,
        )

        q_agent, q_metrics = train_q_learning(
            env, q_agent, n_episodes=n_episodes, verbose=False
        )

        q_eval = eval_q(env, q_agent, n_episodes=100, verbose=False)

        # SARSA
        sarsa_agent = SARSAAgent(
            state_space_shape=(3, 5, 3),
            n_actions=3,
            learning_rate=lr,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=42,
        )

        sarsa_agent, sarsa_metrics = train_sarsa(
            env, sarsa_agent, n_episodes=n_episodes, verbose=False
        )

        sarsa_eval = eval_sarsa(env, sarsa_agent, n_episodes=100, verbose=False)

        results[f"lr_{lr}"] = {
            "q_learning": {"training": q_metrics, "evaluation": q_eval},
            "sarsa": {"training": sarsa_metrics, "evaluation": sarsa_eval},
        }

        logger.info(
            f"  Q-Learning: reward={q_eval['mean_reward']:.2f}, sla={q_eval['mean_sla_violations']:.2f}"
        )
        logger.info(
            f"  SARSA: reward={sarsa_eval['mean_reward']:.2f}, sla={sarsa_eval['mean_sla_violations']:.2f}"
        )

    with open("results/experiment4_results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info("Experiment 4 complete. Results saved.")

    return results


def plot_training_curves(results, save_path="figures/training_curves.png"):
    """Plot training curves for Q-Learning and SARSA."""
    logger.info(f"Generating training curves plot: {save_path}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Moving average function
    def moving_average(data, window=50):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    # Plot 1: Rewards
    ax = axes[0, 0]
    q_rewards = results["q_learning"]["training"]["episode_rewards"]
    sarsa_rewards = results["sarsa"]["training"]["episode_rewards"]

    ax.plot(moving_average(q_rewards), label="Q-Learning", alpha=0.8)
    ax.plot(moving_average(sarsa_rewards), label="SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (Moving Avg)")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: SLA Violations
    ax = axes[0, 1]
    q_sla = results["q_learning"]["training"]["sla_violations"]
    sarsa_sla = results["sarsa"]["training"]["sla_violations"]

    ax.plot(moving_average(q_sla), label="Q-Learning", alpha=0.8)
    ax.plot(moving_average(sarsa_sla), label="SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("SLA Violations (Moving Avg)")
    ax.set_title("SLA Violations During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Costs
    ax = axes[1, 0]
    q_costs = results["q_learning"]["training"]["costs"]
    sarsa_costs = results["sarsa"]["training"]["costs"]

    ax.plot(moving_average(q_costs), label="Q-Learning", alpha=0.8)
    ax.plot(moving_average(sarsa_costs), label="SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost (Moving Avg)")
    ax.set_title("Total Cost During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Episode Lengths
    ax = axes[1, 1]
    q_lengths = results["q_learning"]["training"]["episode_lengths"]
    sarsa_lengths = results["sarsa"]["training"]["episode_lengths"]

    ax.plot(moving_average(q_lengths), label="Q-Learning", alpha=0.8)
    ax.plot(moving_average(sarsa_lengths), label="SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (Moving Avg)")
    ax.set_title("Episode Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Training curves saved to {save_path}")
    plt.show()


def plot_comparison_bars(results, save_path="figures/comparison.png"):
    """Create bar plots comparing different algorithms."""
    logger.info(f"Generating comparison plot: {save_path}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Prepare data
    methods = [
        "Random",
        "Threshold",
        "Reactive",
        "Proactive",
        "Conservative",
        "Q-Learning",
        "SARSA",
    ]

    rewards = [
        results["baselines"]["RandomPolicy"]["mean_reward"],
        results["baselines"]["ThresholdPolicy"]["mean_reward"],
        results["baselines"]["ReactivePolicy"]["mean_reward"],
        results["baselines"]["ProactivePolicy"]["mean_reward"],
        results["baselines"]["ConservativePolicy"]["mean_reward"],
        results["q_learning"]["evaluation"]["mean_reward"],
        results["sarsa"]["evaluation"]["mean_reward"],
    ]

    sla_violations = [
        results["baselines"]["RandomPolicy"]["mean_sla_violations"],
        results["baselines"]["ThresholdPolicy"]["mean_sla_violations"],
        results["baselines"]["ReactivePolicy"]["mean_sla_violations"],
        results["baselines"]["ProactivePolicy"]["mean_sla_violations"],
        results["baselines"]["ConservativePolicy"]["mean_sla_violations"],
        results["q_learning"]["evaluation"]["mean_sla_violations"],
        results["sarsa"]["evaluation"]["mean_sla_violations"],
    ]

    costs = [
        results["baselines"]["RandomPolicy"]["mean_cost"],
        results["baselines"]["ThresholdPolicy"]["mean_cost"],
        results["baselines"]["ReactivePolicy"]["mean_cost"],
        results["baselines"]["ProactivePolicy"]["mean_cost"],
        results["baselines"]["ConservativePolicy"]["mean_cost"],
        results["q_learning"]["evaluation"]["mean_cost"],
        results["sarsa"]["evaluation"]["mean_cost"],
    ]

    # Colors: baselines in blue, RL methods in orange
    colors = ["skyblue"] * 5 + ["coral", "coral"]

    # Plot 1: Rewards
    axes[0].bar(range(len(methods)), rewards, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha="right")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Average Reward Comparison")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Plot 2: SLA Violations
    axes[1].bar(range(len(methods)), sla_violations, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha="right")
    axes[1].set_ylabel("Mean SLA Violations")
    axes[1].set_title("SLA Violations Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Plot 3: Costs
    axes[2].bar(range(len(methods)), costs, color=colors, alpha=0.8)
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels(methods, rotation=45, ha="right")
    axes[2].set_ylabel("Mean Cost")
    axes[2].set_title("Cost Comparison")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison bars saved to {save_path}")
    plt.show()


def main():
    """Main execution function."""
    # Setup logging
    global logger
    log_dir = Path("logs")
    logger = setup_logging(log_dir)

    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("CLOUD AUTOSCALING RL PROJECT")
    logger.info("Comparing SARSA and Q-Learning for Cloud Resource Autoscaling")
    logger.info("=" * 80)
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directories
    logger.info("Creating output directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Generate synthetic workload
    logger.info("Generating synthetic workload data...")
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 1000)
    workload = 50 + 30 * np.sin(t) + 10 * np.sin(t / 7) + np.random.normal(0, 5, 1000)
    workload += np.random.choice([0, 20], size=1000, p=[0.95, 0.05])
    workload = np.clip(workload, 10, 100)
    logger.debug(
        f"Workload stats: min={workload.min():.2f}, max={workload.max():.2f}, mean={workload.mean():.2f}"
    )

    # Run experiments
    logger.info("=" * 80)
    logger.info("RUNNING EXPERIMENTS")
    logger.info("=" * 80)

    # Experiment 1: Basic training
    results1 = experiment_1_basic_training(workload, n_episodes=1000)
    plot_training_curves(results1, "figures/experiment1_training.png")
    plot_comparison_bars(results1, "figures/experiment1_comparison.png")

    # Experiment 2: Exploration rates
    results2 = experiment_2_exploration_rates(workload, n_episodes=500)

    # Experiment 3: Episode lengths
    results3 = experiment_3_episode_lengths(workload, episode_lengths=[500, 1000, 1500])

    # Experiment 4: Learning rates
    results4 = experiment_4_learning_rates(workload, n_episodes=1000)

    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time

    logger.info("=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {elapsed_time}")
    logger.info("Results saved in 'results/' directory")
    logger.info("Figures saved in 'figures/' directory")
    logger.info("Models saved in 'models/' directory")
    logger.info("Logs saved in 'logs/' directory")


if __name__ == "__main__":
    main()
