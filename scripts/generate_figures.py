#!/usr/bin/env python3
"""
Generate publication-ready figures for the research report.

Creates high-quality figures suitable for academic papers:
- Algorithm comparison bar charts with error bars
- Learning curves with smoothing and confidence bands
- Ablation study sensitivity plots
- Policy visualization heatmaps

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --style paper  # For LaTeX papers
    python scripts/generate_figures.py --style poster # For posters
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots" / "publication"


# =============================================================================
# Style Configuration
# =============================================================================

STYLE_CONFIGS = {
    "paper": {
        "figsize_single": (6, 4),
        "figsize_double": (12, 4),
        "figsize_large": (10, 6),
        "fontsize_title": 14,
        "fontsize_label": 12,
        "fontsize_tick": 10,
        "fontsize_legend": 10,
        "dpi": 300,
        "format": "pdf",
        "linewidth": 1.5,
        "markersize": 6,
    },
    "poster": {
        "figsize_single": (8, 6),
        "figsize_double": (16, 6),
        "figsize_large": (14, 8),
        "fontsize_title": 20,
        "fontsize_label": 16,
        "fontsize_tick": 14,
        "fontsize_legend": 14,
        "dpi": 150,
        "format": "png",
        "linewidth": 2.5,
        "markersize": 10,
    },
    "presentation": {
        "figsize_single": (10, 6),
        "figsize_double": (16, 6),
        "figsize_large": (14, 8),
        "fontsize_title": 18,
        "fontsize_label": 14,
        "fontsize_tick": 12,
        "fontsize_legend": 12,
        "dpi": 150,
        "format": "png",
        "linewidth": 2.0,
        "markersize": 8,
    },
}

# Color palette (colorblind-friendly)
COLORS = {
    "q_learning": "#0077BB",  # Blue
    "sarsa": "#33BBEE",  # Cyan
    "dqn": "#009988",  # Teal
    "double_dqn": "#EE7733",  # Orange
    "dueling_dqn": "#CC3311",  # Red
    "random": "#BBBBBB",  # Gray
    "threshold": "#EE3377",  # Magenta
}

AGENT_DISPLAY_NAMES = {
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "random": "Random",
    "threshold": "Threshold",
}


def setup_matplotlib_style(style_config: dict) -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": style_config["fontsize_tick"],
            "axes.titlesize": style_config["fontsize_title"],
            "axes.labelsize": style_config["fontsize_label"],
            "xtick.labelsize": style_config["fontsize_tick"],
            "ytick.labelsize": style_config["fontsize_tick"],
            "legend.fontsize": style_config["fontsize_legend"],
            "figure.dpi": style_config["dpi"],
            "savefig.dpi": style_config["dpi"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# =============================================================================
# Data Loading
# =============================================================================


def convert_string_lists_to_float(data: dict) -> dict:
    """Recursively convert string numbers to floats in results dict."""
    if isinstance(data, dict):
        return {k: convert_string_lists_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        if data and isinstance(data[0], str):
            try:
                return [float(x) for x in data]
            except (ValueError, TypeError):
                return data
        return [convert_string_lists_to_float(x) for x in data]
    elif isinstance(data, str):
        # Try to convert string numbers to float
        try:
            return float(data)
        except (ValueError, TypeError):
            return data
    return data


def load_latest_results() -> Optional[dict]:
    """Load the most recent experiment results."""
    result_files = sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
    if not result_files:
        return None
    with open(result_files[0], encoding="utf-8") as f:
        data = json.load(f)
    # Convert any string lists to float (handles serialization quirks)
    return convert_string_lists_to_float(data)


def load_multiseed_results() -> Optional[dict]:
    """Load multi-seed experiment results."""
    result_files = sorted(RESULTS_DIR.glob("multiseed_results_*.json"), reverse=True)
    if not result_files:
        return None
    with open(result_files[0], encoding="utf-8") as f:
        return json.load(f)


def load_ablation_results() -> dict:
    """Load all ablation study results."""
    ablations = {}
    for ablation_file in RESULTS_DIR.glob("ablation_*.json"):
        with open(ablation_file, encoding="utf-8") as f:
            data = json.load(f)
            ablations[data.get("name", ablation_file.stem)] = data
    return ablations


# =============================================================================
# Figure Generation
# =============================================================================


def create_algorithm_comparison_bar(
    results: dict,
    style_config: dict,
    output_path: Path,
) -> None:
    """Create bar chart comparing algorithm performance."""
    fig, ax = plt.subplots(figsize=style_config["figsize_single"])

    # Extract data
    agents = []
    rewards = []
    colors = []

    # Add baselines
    if "baselines" in results:
        for name, data in results["baselines"].items():
            agents.append(AGENT_DISPLAY_NAMES.get(name, name.title()))
            rewards.append(data.get("mean_reward", 0))
            colors.append(COLORS.get(name, "#999999"))

    # Add RL agents
    for agent_key in ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]:
        if agent_key in results:
            agent_rewards = results[agent_key].get("episode_rewards", [])
            if agent_rewards:
                # Use final 10% performance
                final_rewards = agent_rewards[int(len(agent_rewards) * 0.9) :]
                agents.append(AGENT_DISPLAY_NAMES.get(agent_key, agent_key))
                rewards.append(np.mean(final_rewards))
                colors.append(COLORS.get(agent_key, "#999999"))

    # Sort by reward
    sorted_indices = np.argsort(rewards)[::-1]
    agents = [agents[i] for i in sorted_indices]
    rewards = [rewards[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    # Create bar chart
    x = np.arange(len(agents))
    bars = ax.bar(x, rewards, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        label = f"{reward / 1000:.0f}K" if abs(reward) >= 1000 else f"{reward:.0f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style_config["fontsize_tick"],
        )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Algorithm Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")

    # Add horizontal line at 0
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / f"algorithm_comparison.{style_config['format']}")
    plt.close()
    print(f"  ✓ Algorithm comparison: algorithm_comparison.{style_config['format']}")


def create_learning_curves(
    results: dict,
    style_config: dict,
    output_path: Path,
    smoothing_window: int = 50,
) -> None:
    """Create learning curves with smoothing."""
    fig, ax = plt.subplots(figsize=style_config["figsize_large"])

    agents_to_plot = ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]

    for agent_key in agents_to_plot:
        if agent_key not in results:
            continue

        rewards = results[agent_key].get("episode_rewards", [])
        if not rewards:
            continue

        episodes = np.arange(len(rewards))
        rewards = np.array(rewards)

        # Smooth rewards
        smoothed = uniform_filter1d(rewards, size=smoothing_window, mode="nearest")

        # Plot smoothed curve
        ax.plot(
            episodes,
            smoothed,
            label=AGENT_DISPLAY_NAMES.get(agent_key, agent_key),
            color=COLORS.get(agent_key, "#999999"),
            linewidth=style_config["linewidth"],
        )

        # Add confidence band (rolling std)
        if len(rewards) > smoothing_window:
            rolling_std = np.array(
                [
                    np.std(
                        rewards[
                            max(0, i - smoothing_window // 2) : i
                            + smoothing_window // 2
                        ]
                    )
                    for i in range(len(rewards))
                ]
            )
            ax.fill_between(
                episodes,
                smoothed - rolling_std,
                smoothed + rolling_std,
                alpha=0.15,
                color=COLORS.get(agent_key, "#999999"),
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curves")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path / f"learning_curves.{style_config['format']}")
    plt.close()
    print(f"  ✓ Learning curves: learning_curves.{style_config['format']}")


def create_learning_curves_subplots(
    results: dict,
    style_config: dict,
    output_path: Path,
    smoothing_window: int = 50,
) -> None:
    """Create learning curves with separate subplots for tabular vs deep."""
    fig, axes = plt.subplots(1, 2, figsize=style_config["figsize_double"])

    # Tabular methods
    ax1 = axes[0]
    for agent_key in ["q_learning", "sarsa"]:
        if agent_key not in results:
            continue
        rewards = results[agent_key].get("episode_rewards", [])
        if rewards:
            episodes = np.arange(len(rewards))
            smoothed = uniform_filter1d(
                np.array(rewards), size=smoothing_window, mode="nearest"
            )
            ax1.plot(
                episodes,
                smoothed,
                label=AGENT_DISPLAY_NAMES.get(agent_key, agent_key),
                color=COLORS.get(agent_key),
                linewidth=style_config["linewidth"],
            )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("(a) Tabular Methods")
    ax1.legend(loc="lower right")

    # Deep RL methods
    ax2 = axes[1]
    for agent_key in ["dqn", "double_dqn", "dueling_dqn"]:
        if agent_key not in results:
            continue
        rewards = results[agent_key].get("episode_rewards", [])
        if rewards:
            episodes = np.arange(len(rewards))
            smoothed = uniform_filter1d(
                np.array(rewards), size=smoothing_window, mode="nearest"
            )
            ax2.plot(
                episodes,
                smoothed,
                label=AGENT_DISPLAY_NAMES.get(agent_key, agent_key),
                color=COLORS.get(agent_key),
                linewidth=style_config["linewidth"],
            )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("(b) Deep RL Methods")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path / f"learning_curves_split.{style_config['format']}")
    plt.close()
    print(
        f"  ✓ Learning curves (split): learning_curves_split.{style_config['format']}"
    )


def create_multiseed_comparison(
    results: dict,
    style_config: dict,
    output_path: Path,
) -> None:
    """Create bar chart with confidence intervals from multi-seed results."""
    if not results or "results" not in results:
        print("  ⚠ No multi-seed results found, skipping...")
        return

    fig, ax = plt.subplots(figsize=style_config["figsize_single"])

    data = results["results"]
    agents = []
    means = []
    ci_errors = []
    colors = []

    # Sort by mean reward
    sorted_agents = sorted(data.keys(), key=lambda x: data[x]["mean"], reverse=True)

    for agent_name in sorted_agents:
        agent_data = data[agent_name]
        agents.append(agent_name)
        means.append(agent_data["mean"])

        # Calculate CI half-width
        ci_low, ci_high = agent_data["ci_95"]
        ci_errors.append((ci_high - ci_low) / 2)

        # Get color
        agent_key = agent_name.lower().replace("-", "_").replace(" ", "_")
        colors.append(COLORS.get(agent_key, "#999999"))

    x = np.arange(len(agents))
    ax.bar(
        x,
        means,
        yerr=ci_errors,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        error_kw={"linewidth": 1.5},
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Multi-Seed Comparison (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path / f"multiseed_comparison.{style_config['format']}")
    plt.close()
    print(f"  ✓ Multi-seed comparison: multiseed_comparison.{style_config['format']}")


def create_ablation_heatmap(
    ablations: dict,
    style_config: dict,
    output_path: Path,
) -> None:
    """Create heatmap for hyperparameter sensitivity."""
    # Find grid search ablation
    grid_ablation = None
    for name, data in ablations.items():
        if "grid" in name.lower():
            grid_ablation = data
            break

    if not grid_ablation:
        print("  ⚠ No grid search results found, skipping heatmap...")
        return

    # Extract grid data
    results = grid_ablation.get("results", [])
    if not results:
        return

    # Parse results into grid
    lr_values = set()
    gamma_values = set()
    rewards = {}

    for r in results:
        params = r.get("params", {})
        lr = params.get("learning_rate", 0.1)
        gamma = params.get("discount_factor", 0.95)
        reward = r.get("metrics", {}).get("mean_reward_mean", 0)

        lr_values.add(lr)
        gamma_values.add(gamma)
        rewards[(lr, gamma)] = reward

    lr_values = sorted(lr_values)
    gamma_values = sorted(gamma_values)

    # Create grid
    grid = np.zeros((len(lr_values), len(gamma_values)))
    for i, lr in enumerate(lr_values):
        for j, gamma in enumerate(gamma_values):
            grid[i, j] = rewards.get((lr, gamma), np.nan)

    fig, ax = plt.subplots(figsize=style_config["figsize_single"])

    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(gamma_values)))
    ax.set_yticks(np.arange(len(lr_values)))
    ax.set_xticklabels([f"{g:.2f}" for g in gamma_values])
    ax.set_yticklabels([f"{lr:.3f}" for lr in lr_values])

    ax.set_xlabel("Discount Factor (γ)")
    ax.set_ylabel("Learning Rate (α)")
    ax.set_title("Hyperparameter Sensitivity")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Reward")

    # Add text annotations
    for i in range(len(lr_values)):
        for j in range(len(gamma_values)):
            val = grid[i, j]
            text = f"{val / 1000:.0f}K" if abs(val) >= 1000 else f"{val:.0f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / f"ablation_heatmap.{style_config['format']}")
    plt.close()
    print(f"  ✓ Ablation heatmap: ablation_heatmap.{style_config['format']}")


def create_ablation_line_plots(
    ablations: dict,
    style_config: dict,
    output_path: Path,
) -> None:
    """Create line plots for each ablation study."""
    for ablation_name, ablation_data in ablations.items():
        results = ablation_data.get("results", [])
        if not results:
            continue

        # Determine parameter being varied
        param_name = None
        for key in ["learning_rate", "discount_factor", "epsilon_decay"]:
            if key in ablation_name:
                param_name = key
                break

        if not param_name:
            continue

        # Extract data
        param_values = []
        rewards = []
        stds = []

        for r in sorted(results, key=lambda x: x.get("params", {}).get(param_name, 0)):
            params = r.get("params", {})
            metrics = r.get("metrics", {})

            param_values.append(params.get(param_name, 0))
            rewards.append(metrics.get("mean_reward_mean", 0))
            stds.append(metrics.get("mean_reward_std", 0))

        if not param_values:
            continue

        fig, ax = plt.subplots(figsize=style_config["figsize_single"])

        ax.errorbar(
            param_values,
            rewards,
            yerr=stds,
            marker="o",
            markersize=style_config["markersize"],
            linewidth=style_config["linewidth"],
            capsize=4,
            color=COLORS["dqn"],
        )

        # Highlight best
        best_idx = np.argmax(rewards)
        ax.scatter(
            [param_values[best_idx]],
            [rewards[best_idx]],
            s=style_config["markersize"] * 20,
            marker="*",
            color=COLORS["dueling_dqn"],
            zorder=5,
            label="Best",
        )

        param_display = param_name.replace("_", " ").title()
        ax.set_xlabel(param_display)
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Sensitivity to {param_display}")
        ax.legend()

        # Use log scale for learning rate
        if param_name == "learning_rate":
            ax.set_xscale("log")

        plt.tight_layout()
        safe_name = ablation_name.replace(" ", "_").lower()
        plt.savefig(output_path / f"ablation_{safe_name}.{style_config['format']}")
        plt.close()
        print(f"  ✓ Ablation plot: ablation_{safe_name}.{style_config['format']}")


def create_convergence_comparison(
    results: dict,
    style_config: dict,
    output_path: Path,
) -> None:
    """Create plot showing episodes to convergence for each algorithm."""
    fig, ax = plt.subplots(figsize=style_config["figsize_single"])

    agents = []
    convergence_episodes = []
    colors = []

    threshold = 0.95  # 95% of final performance

    for agent_key in ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]:
        if agent_key not in results:
            continue

        rewards = results[agent_key].get("episode_rewards", [])
        if not rewards:
            continue

        rewards = np.array(rewards)
        final_perf = np.mean(rewards[-50:])
        target = final_perf * threshold

        # Find first episode where rolling average exceeds target
        window = 50
        rolling_avg = uniform_filter1d(rewards, size=window, mode="nearest")

        converged = np.where(rolling_avg >= target)[0]
        conv_episode = converged[0] if len(converged) > 0 else len(rewards)

        agents.append(AGENT_DISPLAY_NAMES.get(agent_key, agent_key))
        convergence_episodes.append(conv_episode)
        colors.append(COLORS.get(agent_key, "#999999"))

    x = np.arange(len(agents))
    ax.bar(
        x, convergence_episodes, color=colors, edgecolor="black", linewidth=0.5
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Episodes to Convergence")
    ax.set_title("Convergence Speed Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path / f"convergence_speed.{style_config['format']}")
    plt.close()
    print(f"  ✓ Convergence speed: convergence_speed.{style_config['format']}")


def create_summary_figure(
    results: dict,
    style_config: dict,
    output_path: Path,
) -> None:
    """Create a summary figure combining multiple plots."""
    fig = plt.figure(figsize=(14, 10))

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Algorithm comparison
    ax1 = fig.add_subplot(gs[0, 0])
    agents, rewards, colors = [], [], []

    for agent_key in [
        "threshold",
        "q_learning",
        "sarsa",
        "dqn",
        "double_dqn",
        "dueling_dqn",
    ]:
        if agent_key == "threshold" and "baselines" in results:
            if "threshold" in results["baselines"]:
                agents.append("Threshold")
                rewards.append(results["baselines"]["threshold"]["mean_reward"])
                colors.append(COLORS["threshold"])
        elif agent_key in results:
            agent_rewards = results[agent_key].get("episode_rewards", [])
            if agent_rewards:
                agents.append(AGENT_DISPLAY_NAMES.get(agent_key, agent_key))
                rewards.append(np.mean(agent_rewards[-100:]))
                colors.append(COLORS.get(agent_key, "#999999"))

    if agents:
        sorted_idx = np.argsort(rewards)[::-1]
        ax1.barh(
            [agents[i] for i in sorted_idx],
            [rewards[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx],
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Mean Reward")
        ax1.set_title("(a) Final Performance")

    # Plot 2: Learning curves
    ax2 = fig.add_subplot(gs[0, 1])
    for agent_key in ["q_learning", "dqn", "dueling_dqn"]:
        if agent_key in results:
            rewards = results[agent_key].get("episode_rewards", [])
            if rewards:
                smoothed = uniform_filter1d(np.array(rewards), size=50, mode="nearest")
                ax2.plot(
                    smoothed,
                    label=AGENT_DISPLAY_NAMES.get(agent_key),
                    color=COLORS.get(agent_key),
                    linewidth=1.5,
                )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("(b) Learning Curves")
    ax2.legend(loc="lower right", fontsize=8)

    # Plot 3: Method type comparison
    ax3 = fig.add_subplot(gs[1, 0])
    method_types = {"Baseline": [], "Tabular": [], "Deep RL": []}

    if "baselines" in results:
        for name, data in results["baselines"].items():
            if name != "random":
                method_types["Baseline"].append(data.get("mean_reward", 0))

    for agent_key in ["q_learning", "sarsa"]:
        if agent_key in results:
            rewards = results[agent_key].get("episode_rewards", [])
            if rewards:
                method_types["Tabular"].append(np.mean(rewards[-100:]))

    for agent_key in ["dqn", "double_dqn", "dueling_dqn"]:
        if agent_key in results:
            rewards = results[agent_key].get("episode_rewards", [])
            if rewards:
                method_types["Deep RL"].append(np.mean(rewards[-100:]))

    types = list(method_types.keys())
    means = [np.mean(v) if v else 0 for v in method_types.values()]
    stds = [np.std(v) if len(v) > 1 else 0 for v in method_types.values()]

    ax3.bar(
        types,
        means,
        yerr=stds,
        color=["#EE3377", "#0077BB", "#009988"],
        edgecolor="black",
        linewidth=0.5,
        capsize=5,
    )
    ax3.set_ylabel("Mean Reward")
    ax3.set_title("(c) Method Type Comparison")

    # Plot 4: Key statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    stats_text = "Key Statistics\n" + "=" * 30 + "\n\n"
    if results:
        best_agent = None
        best_reward = float("-inf")
        for agent_key in ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]:
            if agent_key in results:
                rewards = results[agent_key].get("episode_rewards", [])
                if rewards:
                    mean_reward = np.mean(rewards[-100:])
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_agent = agent_key

        if best_agent:
            stats_text += (
                f"Best Algorithm: {AGENT_DISPLAY_NAMES.get(best_agent, best_agent)}\n"
            )
            stats_text += f"Best Reward: {best_reward / 1000:.1f}K\n\n"

        if "baselines" in results and "threshold" in results["baselines"]:
            threshold_reward = results["baselines"]["threshold"]["mean_reward"]
            improvement = (
                (best_reward - threshold_reward) / abs(threshold_reward)
            ) * 100
            stats_text += f"vs Threshold: {improvement:+.1f}%\n"

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax4.set_title("(d) Summary")

    plt.savefig(output_path / f"summary_figure.{style_config['format']}")
    plt.close()
    print(f"  ✓ Summary figure: summary_figure.{style_config['format']}")


def create_capacity_vs_demand_plot(
    style_config: dict,
    output_path: Path,
    n_steps: int = 200,
) -> None:
    """
    Create capacity vs demand trajectory visualization.

    Runs a sample episode for each trained agent and plots:
    - Demand over time
    - Capacity decisions over time
    - SLA violation threshold
    """
    import pickle
    from agent.cloud_autoscaling_env import CloudAutoscalingEnv

    MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

    # Find available models
    model_files = list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        print("  ⚠ No trained models found, skipping capacity vs demand plot...")
        return

    # Generate workload
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_steps)
    workload = 50 + 30 * np.sin(t) + 5 * np.random.randn(n_steps)
    workload = np.clip(workload, 10, 100)

    # Create environment
    env = CloudAutoscalingEnv(workload_data=workload, seed=42)

    # Collect trajectories for each agent
    trajectories = {}

    for model_file in model_files:
        try:
            with open(model_file, "rb") as f:
                save_data = pickle.load(f)
            agent = save_data.get("agent")
            if agent is None:
                continue

            # Get algorithm name
            algo_name = save_data.get("algorithm", model_file.stem.split("_")[0])

            # Skip if we already have this algorithm (prefer latest)
            if algo_name in trajectories:
                continue

            # Run episode
            state, _ = env.reset()
            demands = []
            capacities = []
            utilizations = []
            sla_violations = []

            for step in range(n_steps - 1):
                # Get action from agent (use training=False for greedy/evaluation mode)
                if hasattr(agent, "select_action"):
                    try:
                        # Try training=False for RL agents
                        action = agent.select_action(state, training=False)
                    except TypeError:
                        try:
                            # Try info=None for baseline policies
                            action = agent.select_action(state, info=None)
                        except TypeError:
                            # Fall back to state-only
                            action = agent.select_action(state)
                elif hasattr(agent, "act"):
                    try:
                        action = agent.act(state, eps=0.0)
                    except TypeError:
                        action = agent.act(state)
                elif hasattr(agent, "get_action"):
                    action = agent.get_action(state)
                else:
                    action = 1  # Default to hold

                next_state, reward, terminated, truncated, info = env.step(action)

                demands.append(info.get("demand", env.current_demand))
                capacities.append(
                    info.get("capacity", env.current_capacity) * env.capacity_unit
                )
                utilizations.append(info.get("utilization", 0))
                sla_violations.append(info.get("sla_violation", 0))

                state = next_state
                if terminated or truncated:
                    break

            trajectories[algo_name] = {
                "demands": demands,
                "capacities": capacities,
                "utilizations": utilizations,
                "sla_violations": sla_violations,
            }

        except Exception as e:
            print(f"  ⚠ Error loading {model_file.name}: {e}")
            continue

    if not trajectories:
        print("  ⚠ No valid trajectories collected, skipping...")
        return

    # Create figure with 2 subplots
    len(trajectories)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Capacity vs Demand for all agents
    ax1 = axes[0]
    steps = np.arange(len(list(trajectories.values())[0]["demands"]))

    # Plot demand (same for all)
    first_traj = list(trajectories.values())[0]
    ax1.fill_between(
        steps, 0, first_traj["demands"], alpha=0.3, color="gray", label="Demand"
    )

    # Plot capacity for each agent
    for agent_name, traj in trajectories.items():
        agent_key = agent_name.lower().replace("-", "_").replace(" ", "_")
        color = COLORS.get(agent_key, "#999999")
        display_name = AGENT_DISPLAY_NAMES.get(agent_key, agent_name)
        ax1.plot(
            steps[: len(traj["capacities"])],
            traj["capacities"],
            label=f"{display_name} Capacity",
            color=color,
            linewidth=style_config["linewidth"],
        )

    ax1.set_ylabel("Demand / Capacity")
    ax1.set_title("(a) Capacity vs Demand Over Time")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Utilization with SLA threshold
    ax2 = axes[1]
    sla_threshold = env.sla_violation_threshold

    for agent_name, traj in trajectories.items():
        agent_key = agent_name.lower().replace("-", "_").replace(" ", "_")
        color = COLORS.get(agent_key, "#999999")
        display_name = AGENT_DISPLAY_NAMES.get(agent_key, agent_name)
        ax2.plot(
            steps[: len(traj["utilizations"])],
            traj["utilizations"],
            label=display_name,
            color=color,
            linewidth=style_config["linewidth"],
            alpha=0.8,
        )

    # Add SLA threshold line
    ax2.axhline(
        sla_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"SLA Threshold ({sla_threshold:.0%})",
    )
    ax2.axhline(
        0.6,  # Target utilization
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="Target (60%)",
    )

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Utilization")
    ax2.set_title("(b) Utilization vs SLA Threshold")
    ax2.legend(loc="upper right", ncol=2)
    ax2.set_ylim(0, 1.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"capacity_vs_demand.{style_config['format']}")
    plt.close()
    print(f"  ✓ Capacity vs demand: capacity_vs_demand.{style_config['format']}")


def create_workload_comparison_plot(
    style_config: dict,
    output_path: Path,
) -> None:
    """
    Create visualization comparing different workload patterns.

    Shows smooth, bursty, and seasonal workload transformations.
    """
    try:
        from env_configs import transform_workload
    except ImportError:
        print("  ⚠ env_configs not available, skipping workload comparison...")
        return

    import pandas as pd

    # Generate base workload
    np.random.seed(42)
    n_steps = 500
    t = np.linspace(0, 4 * np.pi, n_steps)
    base_cpu = 0.5 + 0.2 * np.sin(t) + 0.05 * np.random.randn(n_steps)
    base_cpu = np.clip(base_cpu, 0.1, 0.9)

    df_base = pd.DataFrame({"avg_cpu": base_cpu})

    # Transform to different patterns
    workloads = {
        "Smooth": transform_workload(df_base, "smooth")["avg_cpu"].values,
        "Bursty": transform_workload(df_base, "bursty")["avg_cpu"].values,
        "Seasonal": transform_workload(df_base, "seasonal")["avg_cpu"].values,
    }

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    colors = {"Smooth": "#0077BB", "Bursty": "#EE7733", "Seasonal": "#009988"}

    for idx, (name, cpu) in enumerate(workloads.items()):
        ax = axes[idx]
        steps = np.arange(len(cpu))

        ax.fill_between(steps, 0, cpu, alpha=0.3, color=colors[name])
        ax.plot(steps, cpu, color=colors[name], linewidth=1.5, label=name)

        # Add SLA threshold
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_ylabel("CPU Utilization")
        ax.set_title(f"{name} Workload Pattern")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")

    plt.tight_layout()
    plt.savefig(output_path / f"workload_patterns.{style_config['format']}")
    plt.close()
    print(f"  ✓ Workload patterns: workload_patterns.{style_config['format']}")


def create_sla_violations_by_workload(
    style_config: dict,
    output_path: Path,
) -> None:
    """
    Create chart showing SLA violations by workload type and model over episodes.

    Loads results from all workload types and creates:
    1. Bar chart comparing final SLA violations across workloads and algorithms
    2. Line plots showing SLA violations over episodes for each workload
    """
    import pickle

    # Find all result files
    result_files = sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
    if not result_files:
        print("  ⚠ No results files found, skipping SLA violations chart...")
        return

    # Also check for workload-specific model files to extract SLA data
    model_files = list((PROJECT_ROOT / "artifacts" / "models").glob("*.pkl"))

    # Organize data by workload and algorithm
    workload_data = {"smooth": {}, "bursty": {}, "seasonal": {}}

    # Load data from model files (they contain episode-level SLA data)
    for model_file in model_files:
        try:
            with open(model_file, "rb") as f:
                save_data = pickle.load(f)

            metadata = save_data.get("metadata", {})
            workload_type = save_data.get("workload_type", "smooth")
            algorithm = save_data.get("algorithm", model_file.stem.split("_")[0])

            episode_sla = metadata.get("episode_sla", [])
            mean_sla = metadata.get("mean_sla", 0)

            if episode_sla and workload_type in workload_data:
                workload_data[workload_type][algorithm] = {
                    "episode_sla": episode_sla,
                    "mean_sla": mean_sla,
                }
        except (OSError, pickle.UnpicklingError, KeyError, TypeError):
            continue

    # Also load from the latest results file for baselines
    if result_files:
        with open(result_files[0], encoding="utf-8") as f:
            latest_results = json.load(f)

        # Try to determine workload from filename or content
        for key in [
            "baselines",
            "q_learning",
            "sarsa",
            "dqn",
            "double_dqn",
            "dueling_dqn",
        ]:
            if key in latest_results:
                data = latest_results[key]
                if isinstance(data, dict):
                    if key == "baselines":
                        for baseline_name, baseline_data in data.items():
                            sla = baseline_data.get("mean_sla_violations", 0)
                            # Store in smooth by default (since we can't determine workload from single file)
                            if baseline_name not in workload_data["smooth"]:
                                workload_data["smooth"][baseline_name] = {
                                    "mean_sla": sla,
                                    "episode_sla": [],
                                }
                    else:
                        episode_sla = data.get("episode_sla", [])
                        mean_sla = data.get("mean_sla", 0)
                        if key not in workload_data["smooth"]:
                            workload_data["smooth"][key] = {
                                "episode_sla": episode_sla,
                                "mean_sla": mean_sla,
                            }

    # Check if we have any data
    has_data = any(bool(algos) for algos in workload_data.values())
    if not has_data:
        print("  ⚠ No SLA violation data found in model files, skipping...")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=style_config["figsize_double"])

    # Plot 1: Bar chart comparing mean SLA violations across workloads and algorithms
    ax1 = axes[0]

    workload_names = ["smooth", "bursty", "seasonal"]
    workload_labels = ["Smooth", "Bursty", "Seasonal"]

    # Get all unique algorithms
    all_algos = set()
    for wl_data in workload_data.values():
        all_algos.update(wl_data.keys())
    all_algos = sorted(all_algos)

    if all_algos:
        x = np.arange(len(workload_labels))
        width = 0.8 / max(len(all_algos), 1)

        for i, algo in enumerate(all_algos):
            sla_values = []
            for wl in workload_names:
                if algo in workload_data[wl]:
                    sla_values.append(workload_data[wl][algo].get("mean_sla", 0))
                else:
                    sla_values.append(0)

            offset = (i - len(all_algos) / 2 + 0.5) * width
            color = COLORS.get(algo, "#999999")
            display_name = AGENT_DISPLAY_NAMES.get(algo, algo.replace("_", " ").title())
            ax1.bar(
                x + offset,
                sla_values,
                width * 0.9,
                label=display_name,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

        ax1.set_xlabel("Workload Type")
        ax1.set_ylabel("Mean SLA Violations per Episode")
        ax1.set_title("(a) SLA Violations by Workload & Algorithm")
        ax1.set_xticks(x)
        ax1.set_xticklabels(workload_labels)
        ax1.legend(
            loc="upper right", fontsize=style_config["fontsize_legend"] - 2, ncol=2
        )
        ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Line plot showing SLA violations over episodes (for algorithms with episode data)
    ax2 = axes[1]

    smoothing_window = 50
    plotted_any = False

    for wl_name, wl_label in zip(workload_names, workload_labels):
        for algo, data in workload_data[wl_name].items():
            episode_sla = data.get("episode_sla", [])
            if len(episode_sla) > smoothing_window:
                episodes = np.arange(len(episode_sla))
                smoothed = uniform_filter1d(
                    np.array(episode_sla, dtype=float),
                    size=smoothing_window,
                    mode="nearest",
                )

                color = COLORS.get(algo, "#999999")
                linestyle = (
                    "-"
                    if wl_name == "smooth"
                    else ("--" if wl_name == "bursty" else ":")
                )
                display_name = AGENT_DISPLAY_NAMES.get(
                    algo, algo.replace("_", " ").title()
                )
                label = f"{display_name} ({wl_label})"

                ax2.plot(
                    episodes,
                    smoothed,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=style_config["linewidth"],
                    alpha=0.8,
                )
                plotted_any = True

    if plotted_any:
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("SLA Violations (Smoothed)")
        ax2.set_title("(b) SLA Violations Over Training")
        ax2.legend(
            loc="upper right", fontsize=style_config["fontsize_legend"] - 2, ncol=2
        )
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No episode-level SLA data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=style_config["fontsize_label"],
        )
        ax2.set_title("(b) SLA Violations Over Training")

    plt.tight_layout()
    plt.savefig(output_path / f"sla_violations_by_workload.{style_config['format']}")
    plt.close()
    print(
        f"  ✓ SLA violations by workload: sla_violations_by_workload.{style_config['format']}"
    )


def create_sla_learning_curves(
    results: dict,
    style_config: dict,
    output_path: Path,
    smoothing_window: int = 50,
) -> None:
    """Create learning curves for SLA violations alongside rewards."""
    fig, axes = plt.subplots(1, 2, figsize=style_config["figsize_double"])

    agents_to_plot = ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]

    # Plot rewards
    ax1 = axes[0]
    for agent_key in agents_to_plot:
        if agent_key not in results:
            continue

        rewards = results[agent_key].get("episode_rewards", [])
        if not rewards:
            continue

        episodes = np.arange(len(rewards))
        smoothed = uniform_filter1d(
            np.array(rewards), size=smoothing_window, mode="nearest"
        )

        ax1.plot(
            episodes,
            smoothed,
            label=AGENT_DISPLAY_NAMES.get(agent_key, agent_key),
            color=COLORS.get(agent_key, "#999999"),
            linewidth=style_config["linewidth"],
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("(a) Reward Learning Curves")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot SLA violations
    ax2 = axes[1]
    for agent_key in agents_to_plot:
        if agent_key not in results:
            continue

        sla = results[agent_key].get("episode_sla", [])
        if not sla:
            continue

        episodes = np.arange(len(sla))
        smoothed = uniform_filter1d(
            np.array(sla, dtype=float), size=smoothing_window, mode="nearest"
        )

        ax2.plot(
            episodes,
            smoothed,
            label=AGENT_DISPLAY_NAMES.get(agent_key, agent_key),
            color=COLORS.get(agent_key, "#999999"),
            linewidth=style_config["linewidth"],
        )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("SLA Violations")
    ax2.set_title("(b) SLA Violation Learning Curves")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"reward_sla_learning_curves.{style_config['format']}")
    plt.close()
    print(
        f"  ✓ Reward & SLA learning curves: reward_sla_learning_curves.{style_config['format']}"
    )


def create_sla_summary_table(
    style_config: dict,
    output_path: Path,
) -> None:
    """
    Create a beautiful table visualization showing SLA violations across all models and configs.

    Features:
    - Heatmap-style table with models as rows and workloads as columns
    - Best performer in each column highlighted with green
    - Overall best highlighted with a gold border
    - Clean, publication-ready styling
    """
    import pickle
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap

    # Collect data from model files
    model_files = list((PROJECT_ROOT / "artifacts" / "models").glob("*.pkl"))

    # Organize data by workload and algorithm
    workload_data = {"smooth": {}, "bursty": {}, "seasonal": {}}

    for model_file in model_files:
        try:
            with open(model_file, "rb") as f:
                save_data = pickle.load(f)

            metadata = save_data.get("metadata", {})
            workload_type = save_data.get("workload_type", "smooth")
            algorithm = save_data.get("algorithm", model_file.stem.split("_")[0])

            mean_sla = metadata.get("mean_sla", 0)
            mean_reward = metadata.get("mean_reward", 0)

            if workload_type in workload_data:
                workload_data[workload_type][algorithm] = {
                    "mean_sla": mean_sla,
                    "mean_reward": mean_reward,
                }
        except (OSError, pickle.UnpicklingError, KeyError, TypeError):
            continue

    # Also load from results files
    result_files = sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
    if result_files:
        with open(result_files[0], encoding="utf-8") as f:
            latest_results = json.load(f)

        for key in ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]:
            if key in latest_results:
                data = latest_results[key]
                if isinstance(data, dict) and key not in workload_data["smooth"]:
                    workload_data["smooth"][key] = {
                        "mean_sla": data.get("mean_sla", 0),
                        "mean_reward": data.get("mean_reward", 0),
                    }

    # Check if we have data
    has_data = any(bool(algos) for algos in workload_data.values())
    if not has_data:
        print("  ⚠ No data found for SLA summary table, skipping...")
        return

    # Get all unique algorithms across all workloads
    all_algos = set()
    for wl_data in workload_data.values():
        all_algos.update(wl_data.keys())
    all_algos = sorted(all_algos)

    if not all_algos:
        print("  ⚠ No algorithms found for SLA summary table, skipping...")
        return

    workload_names = ["smooth", "bursty", "seasonal"]
    workload_labels = ["Smooth", "Bursty", "Seasonal"]

    # Build the data matrix
    n_algos = len(all_algos)
    n_workloads = len(workload_names)

    # Create matrix of SLA values (rows = algorithms, cols = workloads)
    sla_matrix = np.full((n_algos, n_workloads), np.nan)
    reward_matrix = np.full((n_algos, n_workloads), np.nan)

    for i, algo in enumerate(all_algos):
        for j, wl in enumerate(workload_names):
            if algo in workload_data[wl]:
                sla_matrix[i, j] = workload_data[wl][algo].get("mean_sla", np.nan)
                reward_matrix[i, j] = workload_data[wl][algo].get("mean_reward", np.nan)

    # Create figure
    fig_height = max(6, 1.5 + n_algos * 0.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Use a nice background
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    # Create custom colormap (green = good/low, red = bad/high)
    colors_cmap = [
        "#2E7D32",
        "#4CAF50",
        "#81C784",
        "#C8E6C9",
        "#FFFFFF",
        "#FFCDD2",
        "#EF9A9A",
        "#E57373",
        "#F44336",
    ]
    cmap = LinearSegmentedColormap.from_list(
        "sla_cmap", colors_cmap[::-1]
    )  # Reverse so low = green

    # Normalize the data for coloring (excluding NaN)
    valid_values = sla_matrix[~np.isnan(sla_matrix)]
    if len(valid_values) > 0:
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        if vmin == vmax:
            vmax = vmin + 1  # Avoid division by zero
    else:
        vmin, vmax = 0, 1

    # Find best (minimum) for each column and overall
    col_bests = []
    for j in range(n_workloads):
        col_vals = sla_matrix[:, j]
        valid_mask = ~np.isnan(col_vals)
        if np.any(valid_mask):
            col_bests.append(np.nanargmin(col_vals))
        else:
            col_bests.append(None)

    # Overall best
    overall_best = None
    overall_best_val = float("inf")
    for i in range(n_algos):
        for j in range(n_workloads):
            if not np.isnan(sla_matrix[i, j]) and sla_matrix[i, j] < overall_best_val:
                overall_best_val = sla_matrix[i, j]
                overall_best = (i, j)

    # Table dimensions
    cell_height = 0.8
    cell_width = 2.5
    header_height = 1.0
    row_label_width = 2.2

    # Starting positions
    start_x = row_label_width
    start_y = n_algos * cell_height

    # Draw header row
    for j, wl_label in enumerate(workload_labels):
        x = start_x + j * cell_width
        y = start_y

        # Header cell
        header_color = (
            "#1565C0"
            if wl_label == "Seasonal"
            else ("#E65100" if wl_label == "Bursty" else "#2E7D32")
        )
        rect = Rectangle(
            (x, y),
            cell_width,
            header_height,
            facecolor=header_color,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_width / 2,
            y + header_height / 2,
            wl_label,
            ha="center",
            va="center",
            fontsize=style_config["fontsize_label"],
            fontweight="bold",
            color="white",
        )

    # Add "Average" column header
    x = start_x + n_workloads * cell_width
    rect = Rectangle(
        (x, start_y),
        cell_width,
        header_height,
        facecolor="#424242",
        edgecolor="white",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + cell_width / 2,
        start_y + header_height / 2,
        "Average",
        ha="center",
        va="center",
        fontsize=style_config["fontsize_label"],
        fontweight="bold",
        color="white",
    )

    # Draw algorithm labels and data cells
    algo_display_names = [
        AGENT_DISPLAY_NAMES.get(a, a.replace("_", " ").title()) for a in all_algos
    ]

    for i, (algo, algo_display) in enumerate(zip(all_algos, algo_display_names)):
        y = start_y - (i + 1) * cell_height

        # Row label (algorithm name)
        algo_color = COLORS.get(algo, "#666666")
        rect = Rectangle(
            (0, y),
            row_label_width,
            cell_height,
            facecolor=algo_color,
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(
            row_label_width / 2,
            y + cell_height / 2,
            algo_display,
            ha="center",
            va="center",
            fontsize=style_config["fontsize_tick"],
            fontweight="bold",
            color="white",
        )

        # Data cells for each workload
        row_values = []
        for j in range(n_workloads):
            x = start_x + j * cell_width
            val = sla_matrix[i, j]

            if np.isnan(val):
                # No data cell
                rect = Rectangle(
                    (x, y),
                    cell_width,
                    cell_height,
                    facecolor="#E0E0E0",
                    edgecolor="white",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(
                    x + cell_width / 2,
                    y + cell_height / 2,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=style_config["fontsize_tick"],
                    color="#9E9E9E",
                    style="italic",
                )
            else:
                row_values.append(val)
                # Normalize value for color
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                cell_color = cmap(norm_val)

                # Check if this is the best in column
                is_col_best = col_bests[j] == i
                is_overall_best = overall_best == (i, j)

                # Draw cell
                rect = Rectangle(
                    (x, y),
                    cell_width,
                    cell_height,
                    facecolor=cell_color,
                    edgecolor="white",
                    linewidth=2,
                )
                ax.add_patch(rect)

                # Add highlight border for best
                if is_overall_best:
                    highlight = Rectangle(
                        (x + 0.05, y + 0.05),
                        cell_width - 0.1,
                        cell_height - 0.1,
                        facecolor="none",
                        edgecolor="#FFD700",
                        linewidth=4,
                    )
                    ax.add_patch(highlight)
                elif is_col_best:
                    highlight = Rectangle(
                        (x + 0.05, y + 0.05),
                        cell_width - 0.1,
                        cell_height - 0.1,
                        facecolor="none",
                        edgecolor="#4CAF50",
                        linewidth=3,
                    )
                    ax.add_patch(highlight)

                # Text color based on background brightness
                text_color = "white" if norm_val > 0.6 or norm_val < 0.3 else "#333333"

                # Add value text
                val_text = f"{val:.1f}"
                fontweight = "bold" if is_col_best or is_overall_best else "normal"
                ax.text(
                    x + cell_width / 2,
                    y + cell_height / 2,
                    val_text,
                    ha="center",
                    va="center",
                    fontsize=style_config["fontsize_label"],
                    fontweight=fontweight,
                    color=text_color,
                )

                # Add star for overall best
                if is_overall_best:
                    ax.text(
                        x + cell_width - 0.3,
                        y + cell_height - 0.2,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color="#FFD700",
                    )

        # Average column
        x = start_x + n_workloads * cell_width
        if row_values:
            avg_val = np.mean(row_values)
            norm_val = (avg_val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            cell_color = cmap(norm_val)

            rect = Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor=cell_color,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rect)

            text_color = "white" if norm_val > 0.6 or norm_val < 0.3 else "#333333"
            ax.text(
                x + cell_width / 2,
                y + cell_height / 2,
                f"{avg_val:.1f}",
                ha="center",
                va="center",
                fontsize=style_config["fontsize_label"],
                color=text_color,
            )
        else:
            rect = Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor="#E0E0E0",
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                x + cell_width / 2,
                y + cell_height / 2,
                "N/A",
                ha="center",
                va="center",
                fontsize=style_config["fontsize_tick"],
                color="#9E9E9E",
                style="italic",
            )

    # Set axis limits and remove axes
    total_width = row_label_width + (n_workloads + 1) * cell_width
    total_height = n_algos * cell_height + header_height
    ax.set_xlim(-0.2, total_width + 0.2)
    ax.set_ylim(-0.5, total_height + 1.5)
    ax.axis("off")

    # Title
    ax.text(
        total_width / 2,
        total_height + 1.0,
        "SLA Violations Summary: Models vs Workload Configurations",
        ha="center",
        va="center",
        fontsize=style_config["fontsize_title"] + 2,
        fontweight="bold",
        color="#1A1A1A",
    )

    # Legend
    legend_y = -0.3
    ax.text(
        0,
        legend_y,
        "Legend:",
        fontsize=style_config["fontsize_tick"],
        fontweight="bold",
        color="#333333",
    )

    # Color scale indicator
    ax.text(
        1.5,
        legend_y,
        "Lower = Better (Green)",
        fontsize=style_config["fontsize_tick"] - 1,
        color="#2E7D32",
    )
    ax.text(
        5.0,
        legend_y,
        "Higher = Worse (Red)",
        fontsize=style_config["fontsize_tick"] - 1,
        color="#F44336",
    )

    # Best indicators
    rect = Rectangle(
        (8.5, legend_y - 0.15),
        0.4,
        0.3,
        facecolor="none",
        edgecolor="#FFD700",
        linewidth=3,
    )
    ax.add_patch(rect)
    ax.text(
        9.1,
        legend_y,
        "= Overall Best",
        fontsize=style_config["fontsize_tick"] - 1,
        color="#333333",
    )

    rect = Rectangle(
        (11.5, legend_y - 0.15),
        0.4,
        0.3,
        facecolor="none",
        edgecolor="#4CAF50",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        12.1,
        legend_y,
        "= Best in Column",
        fontsize=style_config["fontsize_tick"] - 1,
        color="#333333",
    )

    plt.tight_layout()
    plt.savefig(
        output_path / f"sla_summary_table.{style_config['format']}",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close()
    print(f"  ✓ SLA summary table: sla_summary_table.{style_config['format']}")


def create_sla_best_performers_summary(
    style_config: dict,
    output_path: Path,
) -> None:
    """
    Create a beautiful summary chart showing best performing models by workload.

    Features:
    - Elegant grouped visualization by workload type
    - Highlights the best performer (min SLA violations) with a star/crown
    - Clean gradient color scheme
    - Summary statistics panel
    """
    import pickle
    from matplotlib.patches import FancyBboxPatch

    # Collect data from model files
    model_files = list((PROJECT_ROOT / "artifacts" / "models").glob("*.pkl"))

    # Organize data by workload and algorithm
    workload_data = {"smooth": {}, "bursty": {}, "seasonal": {}}

    for model_file in model_files:
        try:
            with open(model_file, "rb") as f:
                save_data = pickle.load(f)

            metadata = save_data.get("metadata", {})
            workload_type = save_data.get("workload_type", "smooth")
            algorithm = save_data.get("algorithm", model_file.stem.split("_")[0])

            mean_sla = metadata.get("mean_sla", 0)
            mean_reward = metadata.get("mean_reward", 0)

            if workload_type in workload_data:
                workload_data[workload_type][algorithm] = {
                    "mean_sla": mean_sla,
                    "mean_reward": mean_reward,
                }
        except (OSError, pickle.UnpicklingError, KeyError, TypeError):
            continue

    # Also load from results files
    result_files = sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
    if result_files:
        with open(result_files[0], encoding="utf-8") as f:
            latest_results = json.load(f)

        for key in ["q_learning", "sarsa", "dqn", "double_dqn", "dueling_dqn"]:
            if key in latest_results:
                data = latest_results[key]
                if isinstance(data, dict) and key not in workload_data["smooth"]:
                    workload_data["smooth"][key] = {
                        "mean_sla": data.get("mean_sla", 0),
                        "mean_reward": data.get("mean_reward", 0),
                    }

    # Check if we have data
    has_data = any(bool(algos) for algos in workload_data.values())
    if not has_data:
        print("  ⚠ No data found for best performers summary, skipping...")
        return

    # Create beautiful figure
    fig = plt.figure(figsize=(14, 10))

    # Use a nice background
    fig.patch.set_facecolor("#FAFAFA")

    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.25)

    # Workload colors (gradient-friendly)
    workload_colors = {
        "smooth": {"primary": "#4CAF50", "light": "#C8E6C9", "dark": "#2E7D32"},
        "bursty": {"primary": "#FF9800", "light": "#FFE0B2", "dark": "#E65100"},
        "seasonal": {"primary": "#2196F3", "light": "#BBDEFB", "dark": "#1565C0"},
    }

    workload_names = ["smooth", "bursty", "seasonal"]
    workload_labels = ["Smooth", "Bursty", "Seasonal"]

    best_performers = {}

    # Create individual workload panels
    for idx, (wl_name, wl_label) in enumerate(zip(workload_names, workload_labels)):
        ax = fig.add_subplot(gs[0, idx])

        algos = workload_data[wl_name]
        if not algos:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=14,
                style="italic",
                color="#666666",
            )
            ax.set_title(
                wl_label,
                fontsize=style_config["fontsize_title"],
                fontweight="bold",
                color=workload_colors[wl_name]["dark"],
            )
            ax.axis("off")
            continue

        # Sort algorithms by SLA violations (ascending - best first)
        sorted_algos = sorted(
            algos.items(), key=lambda x: x[1].get("mean_sla", float("inf"))
        )

        algo_names = [
            AGENT_DISPLAY_NAMES.get(a, a.replace("_", " ").title())
            for a, _ in sorted_algos
        ]
        sla_values = [data.get("mean_sla", 0) for _, data in sorted_algos]

        # Store best performer
        if sorted_algos:
            best_algo, best_data = sorted_algos[0]
            best_performers[wl_name] = {
                "algorithm": AGENT_DISPLAY_NAMES.get(best_algo, best_algo),
                "sla": best_data.get("mean_sla", 0),
                "reward": best_data.get("mean_reward", 0),
            }

        # Create horizontal bar chart
        y_pos = np.arange(len(algo_names))

        # Create gradient colors based on ranking
        colors = []
        for i in range(len(sorted_algos)):
            if i == 0:
                colors.append(workload_colors[wl_name]["primary"])
            else:
                # Fade to lighter color for worse performers
                0.4 + 0.6 * (1 - i / max(len(sorted_algos) - 1, 1))
                colors.append(workload_colors[wl_name]["light"])

        bars = ax.barh(
            y_pos,
            sla_values,
            color=colors,
            edgecolor=workload_colors[wl_name]["dark"],
            linewidth=1.5,
            height=0.6,
        )

        # Highlight the best performer with a star
        if sla_values:
            ax.scatter(
                sla_values[0] + max(sla_values) * 0.08,
                0,
                marker="*",
                s=400,
                color="#FFD700",
                edgecolors=workload_colors[wl_name]["dark"],
                linewidths=1.5,
                zorder=5,
            )

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sla_values)):
            label_x = val + max(sla_values) * 0.02 if max(sla_values) > 0 else 0.1
            fontweight = "bold" if i == 0 else "normal"
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                ha="left",
                fontsize=style_config["fontsize_tick"],
                fontweight=fontweight,
                color=workload_colors[wl_name]["dark"],
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(algo_names, fontsize=style_config["fontsize_tick"])
        ax.set_xlabel("Mean SLA Violations", fontsize=style_config["fontsize_label"])
        ax.set_title(
            wl_label,
            fontsize=style_config["fontsize_title"],
            fontweight="bold",
            color=workload_colors[wl_name]["dark"],
            pad=10,
        )

        # Style the axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(workload_colors[wl_name]["dark"])
        ax.spines["bottom"].set_color(workload_colors[wl_name]["dark"])
        ax.tick_params(colors=workload_colors[wl_name]["dark"])
        ax.set_facecolor("#FFFFFF")

        # Add subtle grid
        ax.xaxis.grid(
            True, linestyle="--", alpha=0.3, color=workload_colors[wl_name]["primary"]
        )
        ax.set_axisbelow(True)

        # Invert y-axis so best is at top
        ax.invert_yaxis()

    # Bottom summary panel
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis("off")
    ax_summary.set_facecolor("#FFFFFF")

    # Create summary table
    if best_performers:
        # Title
        ax_summary.text(
            0.5,
            0.95,
            "BEST PERFORMERS SUMMARY",
            ha="center",
            va="top",
            fontsize=style_config["fontsize_title"] + 2,
            fontweight="bold",
            color="#333333",
            transform=ax_summary.transAxes,
        )

        # Create summary boxes for each workload
        box_width = 0.28
        box_height = 0.6
        start_x = 0.08

        for idx, (wl_name, wl_label) in enumerate(
            zip(workload_names, ["Smooth", "Bursty", "Seasonal"])
        ):
            if wl_name not in best_performers:
                continue

            bp = best_performers[wl_name]
            box_x = start_x + idx * 0.32

            # Draw fancy box
            fancy_box = FancyBboxPatch(
                (box_x, 0.1),
                box_width,
                box_height,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor=workload_colors[wl_name]["light"],
                edgecolor=workload_colors[wl_name]["dark"],
                linewidth=2,
                transform=ax_summary.transAxes,
                zorder=1,
            )
            ax_summary.add_patch(fancy_box)

            # Workload label
            ax_summary.text(
                box_x + box_width / 2,
                0.65,
                wl_label.upper(),
                ha="center",
                va="center",
                fontsize=style_config["fontsize_label"],
                fontweight="bold",
                color=workload_colors[wl_name]["dark"],
                transform=ax_summary.transAxes,
            )

            # Best algorithm (use asterisk instead of star emoji)
            ax_summary.text(
                box_x + box_width / 2,
                0.45,
                f"* {bp['algorithm']}",
                ha="center",
                va="center",
                fontsize=style_config["fontsize_label"] + 1,
                fontweight="bold",
                color="#333333",
                transform=ax_summary.transAxes,
            )

            # SLA violations
            ax_summary.text(
                box_x + box_width / 2,
                0.28,
                f"SLA Violations: {bp['sla']:.1f}",
                ha="center",
                va="center",
                fontsize=style_config["fontsize_tick"],
                color="#555555",
                transform=ax_summary.transAxes,
            )

            # Reward
            reward_str = (
                f"{bp['reward'] / 1000:.1f}K"
                if abs(bp["reward"]) >= 1000
                else f"{bp['reward']:.0f}"
            )
            ax_summary.text(
                box_x + box_width / 2,
                0.15,
                f"Reward: {reward_str}",
                ha="center",
                va="center",
                fontsize=style_config["fontsize_tick"],
                color="#555555",
                transform=ax_summary.transAxes,
            )

    # Add overall title
    fig.suptitle(
        "SLA Violations Analysis: Best Models by Workload Type",
        fontsize=style_config["fontsize_title"] + 4,
        fontweight="bold",
        color="#1A1A1A",
        y=0.98,
    )

    plt.savefig(
        output_path / f"sla_best_performers.{style_config['format']}",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close()
    print(f"  ✓ SLA best performers: sla_best_performers.{style_config['format']}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all publication-ready figures."""
    parser = argparse.ArgumentParser(description="Generate publication-ready figures")
    parser.add_argument(
        "--style",
        choices=["paper", "poster", "presentation"],
        default="paper",
        help="Figure style preset",
    )
    args = parser.parse_args()

    style_config = STYLE_CONFIGS[args.style]
    setup_matplotlib_style(style_config)

    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Generating publication-ready figures (style: {args.style})")
    print("=" * 50)

    # Load data
    results = load_latest_results()
    multiseed_results = load_multiseed_results()
    ablations = load_ablation_results()

    if not results:
        print("❌ No experiment results found. Run: make experiments")
        return

    print(f"\nLoaded results from: {RESULTS_DIR}")
    print(f"Output directory: {PLOTS_DIR}\n")

    # Generate figures
    create_algorithm_comparison_bar(results, style_config, PLOTS_DIR)
    create_learning_curves(results, style_config, PLOTS_DIR)
    create_learning_curves_subplots(results, style_config, PLOTS_DIR)
    create_convergence_comparison(results, style_config, PLOTS_DIR)
    create_summary_figure(results, style_config, PLOTS_DIR)

    # SLA violation charts
    create_sla_learning_curves(results, style_config, PLOTS_DIR)
    create_sla_violations_by_workload(style_config, PLOTS_DIR)
    create_sla_summary_table(style_config, PLOTS_DIR)
    create_sla_best_performers_summary(style_config, PLOTS_DIR)

    # Trajectory visualizations
    create_capacity_vs_demand_plot(style_config, PLOTS_DIR)
    create_workload_comparison_plot(style_config, PLOTS_DIR)

    if multiseed_results:
        create_multiseed_comparison(multiseed_results, style_config, PLOTS_DIR)

    if ablations:
        create_ablation_heatmap(ablations, style_config, PLOTS_DIR)
        create_ablation_line_plots(ablations, style_config, PLOTS_DIR)

    print(f"\n✅ All figures saved to: {PLOTS_DIR}")
    print(f"   Format: {style_config['format'].upper()}, DPI: {style_config['dpi']}")


if __name__ == "__main__":
    main()
