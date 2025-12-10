#!/usr/bin/env python3
"""
Multi-seed experiment runner with statistical analysis.

Runs each algorithm multiple times with different seeds and computes:
- Mean and standard deviation across seeds
- Confidence intervals
- Statistical significance tests (t-tests, Mann-Whitney U)

Usage:
    python scripts/run_multiseed.py --seeds 5 --episodes 500
    python scripts/run_multiseed.py --seeds 10 --quick
    python scripts/run_multiseed.py --no-wandb  # Disable wandb logging
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.cloud_autoscaling_env import CloudAutoscalingEnv
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SARSAAgent
from agent.deep_rl_agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from wandb_utils import (
    init_wandb,
    finish_run,
    load_wandb_key,
    WANDB_AVAILABLE,
)

# Global flag for wandb logging
USE_WANDB = True

RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
LOGS_DIR = PROJECT_ROOT / "artifacts" / "logs"


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"multiseed_{timestamp}.log"

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


def train_agent_with_seed(
    agent_class,
    agent_name: str,
    seed: int,
    n_episodes: int,
    agent_kwargs: dict,
) -> dict:
    """Train a single agent with a specific seed."""
    # Create environment with seed
    env = CloudAutoscalingEnv(seed=seed)
    env.reset(seed=seed)
    np.random.seed(seed)

    # Create agent
    if agent_class in [QLearningAgent, SARSAAgent]:
        agent = agent_class(
            state_space_shape=env.observation_space.nvec,
            n_actions=env.action_space.n,
            **agent_kwargs,
        )
    else:
        # Deep RL agents use action_dim instead of n_actions
        agent = agent_class(
            state_dim=len(env.observation_space.nvec),
            action_dim=env.action_space.n,
            seed=seed,
            **agent_kwargs,
        )

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if agent_class == SARSAAgent:
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    # Compute final performance (last 10% of episodes)
    final_rewards = episode_rewards[int(len(episode_rewards) * 0.9) :]

    return {
        "seed": seed,
        "episode_rewards": episode_rewards,
        "final_mean": np.mean(final_rewards),
        "final_std": np.std(final_rewards),
    }


def run_multiseed_experiment(
    agent_class,
    agent_name: str,
    seeds: list[int],
    n_episodes: int,
    agent_kwargs: dict,
    logger: logging.Logger,
) -> dict:
    """Run experiment across multiple seeds."""
    logger.info(f"Running {agent_name} with {len(seeds)} seeds...")

    # Initialize wandb run for this agent
    run = None
    if USE_WANDB and WANDB_AVAILABLE:
        run = init_wandb(
            name=f"multiseed_{agent_name}",
            config={
                "agent": agent_name,
                "n_seeds": len(seeds),
                "n_episodes": n_episodes,
                "seeds": seeds,
                **agent_kwargs,
            },
            tags=["multiseed", agent_name],
        )

    results = []
    for seed in tqdm(seeds, desc=agent_name, unit="seed"):
        result = train_agent_with_seed(
            agent_class, agent_name, seed, n_episodes, agent_kwargs
        )
        results.append(result)

        # Log per-seed results
        if USE_WANDB and WANDB_AVAILABLE and run:
            import wandb

            wandb.log(
                {
                    "seed": seed,
                    "final_mean": result["final_mean"],
                    "final_std": result["final_std"],
                }
            )

    # Aggregate results
    final_means = [r["final_mean"] for r in results]

    return {
        "agent": agent_name,
        "n_seeds": len(seeds),
        "n_episodes": n_episodes,
        "seeds": seeds,
        "final_means": final_means,
        "mean": np.mean(final_means),
        "std": np.std(final_means),
        "sem": stats.sem(final_means),
        "ci_95": stats.t.interval(
            0.95,
            len(final_means) - 1,
            loc=np.mean(final_means),
            scale=stats.sem(final_means),
        )
        if len(final_means) > 1
        else (np.mean(final_means), np.mean(final_means)),
        "min": np.min(final_means),
        "max": np.max(final_means),
        "per_seed_results": results,
        "_wandb_run": run,  # Store for later finish
    }


def compute_statistical_tests(results: dict[str, dict], logger: logging.Logger) -> dict:
    """Compute pairwise statistical significance tests."""
    agents = list(results.keys())
    tests = {}

    logger.info("\n" + "=" * 60)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("=" * 60)

    # Find best agent
    best_agent = max(results.keys(), key=lambda x: results[x]["mean"])
    logger.info(f"\nBest agent: {best_agent} (mean={results[best_agent]['mean']:.2f})")

    # Compare best agent to all others
    best_means = results[best_agent]["final_means"]

    for agent in agents:
        if agent == best_agent:
            continue

        other_means = results[agent]["final_means"]

        # T-test (parametric)
        t_stat, t_pvalue = stats.ttest_ind(best_means, other_means)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(
            best_means, other_means, alternative="greater"
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(best_means) + np.var(other_means)) / 2)
        cohens_d = (
            (np.mean(best_means) - np.mean(other_means)) / pooled_std
            if pooled_std > 0
            else 0
        )

        tests[f"{best_agent}_vs_{agent}"] = {
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "u_statistic": u_stat,
            "u_pvalue": u_pvalue,
            "cohens_d": cohens_d,
            "significant_05": t_pvalue < 0.05,
            "significant_01": t_pvalue < 0.01,
        }

        sig_marker = "***" if t_pvalue < 0.01 else "**" if t_pvalue < 0.05 else ""
        logger.info(
            f"\n{best_agent} vs {agent}:"
            f"\n  Mean difference: {np.mean(best_means) - np.mean(other_means):.2f}"
            f"\n  T-test p-value: {t_pvalue:.4f} {sig_marker}"
            f"\n  Mann-Whitney p-value: {u_pvalue:.4f}"
            f"\n  Cohen's d: {cohens_d:.3f}"
        )

    return tests


def print_summary_table(results: dict[str, dict], logger: logging.Logger) -> None:
    """Print formatted summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-SEED EXPERIMENT RESULTS")
    logger.info("=" * 80)

    # Sort by mean reward
    sorted_agents = sorted(
        results.keys(), key=lambda x: results[x]["mean"], reverse=True
    )

    header = (
        f"{'Rank':<6}{'Agent':<20}{'Mean':<15}{'Std':<12}{'95% CI':<25}{'Seeds':<8}"
    )
    logger.info(header)
    logger.info("-" * 80)

    for i, agent in enumerate(sorted_agents, 1):
        r = results[agent]
        ci_low, ci_high = r["ci_95"]
        ci_str = f"[{ci_low:.1f}, {ci_high:.1f}]"

        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        logger.info(
            f"{medal}{i:<4}{agent:<20}{r['mean']:<15.2f}{r['std']:<12.2f}{ci_str:<25}{r['n_seeds']:<8}"
        )


def save_results(results: dict, tests: dict, logger: logging.Logger) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"multiseed_results_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    # Simplify results for JSON
    simplified_results = {}
    for agent_name, agent_data in results.items():
        simplified_results[agent_name] = {
            "n_seeds": agent_data["n_seeds"],
            "n_episodes": agent_data["n_episodes"],
            "seeds": agent_data["seeds"],
            "final_means": [convert(x) for x in agent_data["final_means"]],
            "mean": convert(agent_data["mean"]),
            "std": convert(agent_data["std"]),
            "sem": convert(agent_data["sem"]),
            "ci_95": [convert(x) for x in agent_data["ci_95"]],
            "min": convert(agent_data["min"]),
            "max": convert(agent_data["max"]),
        }

    output = {
        "timestamp": timestamp,
        "results": simplified_results,
        "statistical_tests": {
            k: {kk: convert(vv) for kk, vv in v.items()} for k, v in tests.items()
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    return output_file


def create_comparison_plot(results: dict[str, dict], output_path: Path) -> None:
    """Create bar plot with error bars showing confidence intervals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    agents = sorted(results.keys(), key=lambda x: results[x]["mean"], reverse=True)
    means = [results[a]["mean"] for a in agents]
    [results[a]["std"] for a in agents]

    # Calculate CI half-widths
    ci_errors = []
    for a in agents:
        ci_low, ci_high = results[a]["ci_95"]
        ci_errors.append(
            (results[a]["mean"] - ci_low + ci_high - results[a]["mean"]) / 2
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agents)))
    bars = ax.bar(
        agents, means, yerr=ci_errors, capsize=5, color=colors, edgecolor="black"
    )

    ax.set_ylabel("Mean Reward (Final 10% of Episodes)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title("Multi-Seed Comparison with 95% Confidence Intervals", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Add value labels
    for i, (bar, mean, ci_err) in enumerate(zip(bars, means, ci_errors)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_err + abs(mean) * 0.02,
            f"{mean / 1000:.1f}K",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = PLOTS_DIR / f"multiseed_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {plot_path}")


def main():
    """Main function."""
    global USE_WANDB

    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per seed")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test (3 seeds, 100 episodes)"
    )
    parser.add_argument(
        "--tabular-only", action="store_true", help="Only run tabular methods"
    )
    parser.add_argument(
        "--deep-only", action="store_true", help="Only run deep RL methods"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Setup wandb
    USE_WANDB = not args.no_wandb
    if USE_WANDB and WANDB_AVAILABLE:
        api_key = load_wandb_key()
        if api_key:
            import wandb

            wandb.login(key=api_key)

    if args.quick:
        n_seeds = 3
        n_episodes = 100
    else:
        n_seeds = args.seeds
        n_episodes = args.episodes

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("MULTI-SEED EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Episodes per seed: {n_episodes}")
    logger.info(
        f"Wandb logging: {'enabled' if USE_WANDB and WANDB_AVAILABLE else 'disabled'}"
    )

    # Generate seeds
    base_seed = 42
    seeds = [base_seed + i * 1000 for i in range(n_seeds)]
    logger.info(f"Using seeds: {seeds}")

    # Define agents to test
    agents = []

    if not args.deep_only:
        agents.extend(
            [
                (
                    QLearningAgent,
                    "Q-Learning",
                    {
                        "learning_rate": 0.1,
                        "discount_factor": 0.95,
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    },
                ),
                (
                    SARSAAgent,
                    "SARSA",
                    {
                        "learning_rate": 0.1,
                        "discount_factor": 0.95,
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    },
                ),
            ]
        )

    if not args.tabular_only:
        agents.extend(
            [
                (
                    DQNAgent,
                    "DQN",
                    {
                        "learning_rate": 0.001,
                        "discount_factor": 0.99,
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    },
                ),
                (
                    DoubleDQNAgent,
                    "Double-DQN",
                    {
                        "learning_rate": 0.001,
                        "discount_factor": 0.99,
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    },
                ),
                (
                    DuelingDQNAgent,
                    "Dueling-DQN",
                    {
                        "learning_rate": 0.001,
                        "discount_factor": 0.99,
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    },
                ),
            ]
        )

    # Run experiments
    results = {}
    for agent_class, agent_name, agent_kwargs in agents:
        result = run_multiseed_experiment(
            agent_class, agent_name, seeds, n_episodes, agent_kwargs, logger
        )
        results[agent_name] = result

        # Finish wandb run for this agent with summary
        if USE_WANDB and WANDB_AVAILABLE and result.get("_wandb_run"):
            import wandb

            wandb.log(
                {
                    "summary/mean": result["mean"],
                    "summary/std": result["std"],
                    "summary/sem": result["sem"],
                    "summary/ci_95_low": result["ci_95"][0],
                    "summary/ci_95_high": result["ci_95"][1],
                    "summary/min": result["min"],
                    "summary/max": result["max"],
                }
            )
            finish_run()

    # Print summary
    print_summary_table(results, logger)

    # Statistical tests
    if len(results) > 1:
        tests = compute_statistical_tests(results, logger)
    else:
        tests = {}

    # Save results
    save_results(results, tests, logger)

    # Create plot
    create_comparison_plot(results, PLOTS_DIR)

    logger.info("\n✅ Multi-seed experiment complete!")


if __name__ == "__main__":
    main()
