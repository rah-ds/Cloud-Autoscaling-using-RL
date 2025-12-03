"""
Visualization Utilities for Cloud Autoscaling RL Experiments

This module provides publication-quality plotting functions for:
- Learning curves with confidence intervals
- Algorithm comparison charts
- Policy heatmaps
- Hyperparameter sensitivity analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set default style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Publication-quality defaults
FIGSIZE = (10, 6)
DPI = 150
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 12


def smooth_curve(
    values: List[float], window: int = 50, mode: str = "valid"
) -> np.ndarray:
    """
    Smooth a curve using a moving average.

    Args:
        values: Raw values to smooth
        window: Smoothing window size
        mode: Convolution mode ('valid', 'same', 'full')

    Returns:
        Smoothed values
    """
    if len(values) < window:
        window = max(1, len(values) // 5)

    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode=mode)


def plot_learning_curve(
    rewards: List[float],
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    window: int = 50,
    color: str = "blue",
    alpha: float = 0.3,
    show_raw: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = FIGSIZE,
) -> plt.Figure:
    """
    Plot a single learning curve with smoothing.

    Args:
        rewards: List of episode rewards
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        window: Smoothing window
        color: Line color
        alpha: Alpha for raw data
        show_raw: Whether to show raw data behind smoothed curve
        ax: Existing axes to plot on
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    else:
        fig = ax.get_figure()

    episodes = np.arange(len(rewards))

    # Plot raw data
    if show_raw:
        ax.plot(episodes, rewards, alpha=alpha, color=color, linewidth=0.5)

    # Plot smoothed curve
    smoothed = smooth_curve(rewards, window=window)
    smooth_episodes = np.arange(len(smoothed)) + window // 2
    ax.plot(
        smooth_episodes,
        smoothed,
        color=color,
        linewidth=2,
        label=f"Smoothed (w={window})",
    )

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_learning_curves_comparison(
    results: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    window: int = 50,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Compare learning curves from multiple algorithms.

    Args:
        results: Dictionary mapping algorithm name to list of rewards
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        window: Smoothing window
        colors: Optional color mapping for algorithms
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> results = {
        ...     "Q-Learning": q_rewards,
        ...     "SARSA": sarsa_rewards,
        ...     "Random": random_rewards
        ... }
        >>> fig = plot_learning_curves_comparison(results)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (name, rewards) in enumerate(results.items()):
        color = colors.get(name) if colors else default_colors[i % len(default_colors)]

        episodes = np.arange(len(rewards))

        # Plot raw data with transparency
        ax.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.5)

        # Plot smoothed curve
        smoothed = smooth_curve(rewards, window=window)
        smooth_episodes = np.arange(len(smoothed)) + window // 2
        ax.plot(smooth_episodes, smoothed, color=color, linewidth=2, label=name)

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(loc="best", fontsize=FONT_SIZE)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_multi_seed_learning_curve(
    all_rewards: List[List[float]],
    title: str = "Learning Curve (Multiple Seeds)",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    window: int = 50,
    color: str = "#1f77b4",
    figsize: Tuple[int, int] = FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot learning curve with confidence intervals from multiple seeds.

    Args:
        all_rewards: List of reward lists, one per seed
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        window: Smoothing window
        color: Line color
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    # Smooth each run
    smoothed_runs = []
    min_len = min(len(r) for r in all_rewards)

    for rewards in all_rewards:
        smoothed = smooth_curve(rewards[:min_len], window=window, mode="same")
        smoothed_runs.append(smoothed)

    smoothed_runs = np.array(smoothed_runs)

    # Calculate statistics
    mean = np.mean(smoothed_runs, axis=0)
    std = np.std(smoothed_runs, axis=0)
    episodes = np.arange(len(mean))

    # Plot confidence interval
    ax.fill_between(
        episodes, mean - std, mean + std, alpha=0.3, color=color, label="±1 std"
    )

    # Plot mean
    ax.plot(
        episodes, mean, color=color, linewidth=2, label=f"Mean (n={len(all_rewards)})"
    )

    # Plot individual runs faintly
    for rewards in smoothed_runs:
        ax.plot(episodes, rewards, alpha=0.15, color=color, linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(loc="best", fontsize=FONT_SIZE)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_algorithm_comparison_bar(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "mean_reward",
    title: str = "Algorithm Comparison",
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a bar chart comparing algorithms on a specific metric.

    Args:
        metrics: Dict mapping algorithm name to dict of metrics
        metric_name: Which metric to plot
        title: Plot title
        ylabel: Y-axis label (defaults to metric_name)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> metrics = {
        ...     "Q-Learning": {"mean_reward": -500, "std_reward": 50},
        ...     "SARSA": {"mean_reward": -600, "std_reward": 60}
        ... }
        >>> fig = plot_algorithm_comparison_bar(metrics)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    names = list(metrics.keys())
    values = [m.get(metric_name, 0) for m in metrics.values()]
    stds = [
        m.get(f"std_{metric_name.replace('mean_', '')}", 0) for m in metrics.values()
    ]

    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, values, yerr=stds, capsize=5, color=colors, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )

    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_metrics_dashboard(
    metrics: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a dashboard with multiple metrics comparison.

    Args:
        metrics: Dict mapping algorithm name to dict of metrics
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=DPI)

    metric_configs = [
        ("mean_reward", "Mean Reward", axes[0, 0]),
        ("mean_sla_violations", "SLA Violations", axes[0, 1]),
        ("mean_cost", "Mean Cost", axes[1, 0]),
        ("mean_utilization", "Mean Utilization", axes[1, 1]),
    ]

    names = list(metrics.keys())
    colors = sns.color_palette("husl", len(names))

    for metric_name, title, ax in metric_configs:
        values = [m.get(metric_name, 0) for m in metrics.values()]

        bars = ax.bar(names, values, color=colors, edgecolor="black")
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = f"{val:.2%}" if "utilization" in metric_name else f"{val:.1f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.suptitle("Algorithm Performance Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_hyperparameter_heatmap(
    results: Dict[str, float],
    param1_name: str = "learning_rate",
    param2_name: str = "discount_factor",
    metric_name: str = "mean_reward",
    title: str = "Hyperparameter Sensitivity",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a heatmap showing performance across hyperparameter combinations.

    Args:
        results: Dict mapping "param1_param2" to metrics dict
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        metric_name: Metric to visualize
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Parse results into grid
    param1_values = sorted(
        set(r[param1_name] for r in results.values() if param1_name in r)
    )
    param2_values = sorted(
        set(r[param2_name] for r in results.values() if param2_name in r)
    )

    grid = np.zeros((len(param1_values), len(param2_values)))

    for config_name, metrics in results.items():
        if param1_name in metrics and param2_name in metrics:
            i = param1_values.index(metrics[param1_name])
            j = param2_values.index(metrics[param2_name])
            grid[i, j] = metrics.get(metric_name, 0)

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    sns.heatmap(
        grid,
        annot=True,
        fmt=".1f",
        xticklabels=[f"{v:.3f}" for v in param2_values],
        yticklabels=[f"{v:.3f}" for v in param1_values],
        cmap="RdYlGn",
        ax=ax,
    )

    ax.set_xlabel(param2_name.replace("_", " ").title(), fontsize=LABEL_SIZE)
    ax.set_ylabel(param1_name.replace("_", " ").title(), fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_policy_heatmap(
    policy: np.ndarray,
    action_names: List[str] = ["Scale Down", "No Change", "Scale Up"],
    state_labels: Optional[Dict[str, List[str]]] = None,
    title: str = "Learned Policy",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Visualize the learned policy as a heatmap.

    For 3D state space (utilization, capacity, trend), creates
    separate heatmaps for each trend level.

    Args:
        policy: Policy array of shape (util_levels, capacity_levels, trend_levels)
        action_names: Names for each action
        state_labels: Optional labels for state dimensions
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if state_labels is None:
        state_labels = {
            "utilization": ["Low", "Medium", "High"],
            "capacity": [f"Cap {i}" for i in range(policy.shape[1])],
            "trend": ["Decreasing", "Stable", "Increasing"],
        }

    n_trends = policy.shape[2]
    fig, axes = plt.subplots(1, n_trends, figsize=figsize, dpi=DPI)

    if n_trends == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("RdYlGn", len(action_names))

    for i, ax in enumerate(axes):
        trend_name = (
            state_labels["trend"][i] if i < len(state_labels["trend"]) else f"Trend {i}"
        )

        im = ax.imshow(
            policy[:, :, i],
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=len(action_names) - 1,
        )

        ax.set_xticks(range(policy.shape[1]))
        ax.set_xticklabels(state_labels["capacity"][: policy.shape[1]], rotation=45)
        ax.set_yticks(range(policy.shape[0]))
        ax.set_yticklabels(state_labels["utilization"][: policy.shape[0]])

        ax.set_xlabel("Capacity Level")
        ax.set_ylabel("Utilization Level")
        ax.set_title(f"Trend: {trend_name}")

        # Add text annotations
        for y in range(policy.shape[0]):
            for x in range(policy.shape[1]):
                action = int(policy[y, x, i])
                ax.text(
                    x, y, action_names[action][:2], ha="center", va="center", fontsize=8
                )

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, ticks=range(len(action_names)))
    cbar.set_ticklabels(action_names)

    plt.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_training_summary(
    metrics: Dict[str, List[float]],
    title: str = "Training Summary",
    figsize: Tuple[int, int] = (14, 10),
    window: int = 50,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a comprehensive training summary with multiple subplots.

    Args:
        metrics: Dict with keys like 'episode_rewards', 'sla_violations', 'costs'
        title: Plot title
        figsize: Figure size
        window: Smoothing window
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=DPI)

    # Rewards
    ax = axes[0, 0]
    rewards = metrics.get("episode_rewards", [])
    if rewards:
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.3, color="blue")
        smoothed = smooth_curve(rewards, window=window)
        ax.plot(
            np.arange(len(smoothed)) + window // 2, smoothed, color="blue", linewidth=2
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")

    # SLA Violations
    ax = axes[0, 1]
    sla = metrics.get("sla_violations", [])
    if sla:
        episodes = np.arange(len(sla))
        ax.plot(episodes, sla, alpha=0.3, color="red")
        smoothed = smooth_curve(sla, window=window)
        ax.plot(
            np.arange(len(smoothed)) + window // 2, smoothed, color="red", linewidth=2
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("SLA Violations")
    ax.set_title("SLA Violations per Episode")

    # Costs
    ax = axes[1, 0]
    costs = metrics.get("costs", [])
    if costs:
        episodes = np.arange(len(costs))
        ax.plot(episodes, costs, alpha=0.3, color="green")
        smoothed = smooth_curve(costs, window=window)
        ax.plot(
            np.arange(len(smoothed)) + window // 2, smoothed, color="green", linewidth=2
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost")
    ax.set_title("Episode Costs")

    # Episode Lengths
    ax = axes[1, 1]
    lengths = metrics.get("episode_lengths", [])
    if lengths:
        episodes = np.arange(len(lengths))
        ax.plot(episodes, lengths, alpha=0.3, color="purple")
        smoothed = smooth_curve(lengths, window=window)
        ax.plot(
            np.arange(len(smoothed)) + window // 2,
            smoothed,
            color="purple",
            linewidth=2,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length")
    ax.set_title("Episode Lengths")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def create_results_table(
    metrics: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a formatted results table as a string.

    Args:
        metrics: Dict mapping algorithm name to metrics dict
        metric_names: Which metrics to include
        save_path: Optional path to save as text file

    Returns:
        Formatted table string
    """
    if metric_names is None:
        metric_names = ["mean_reward", "std_reward", "mean_sla_violations", "mean_cost"]

    # Create header
    header = ["Algorithm"] + [m.replace("_", " ").title() for m in metric_names]

    # Calculate column widths
    widths = [max(len(h), 15) for h in header]
    widths[0] = max(widths[0], max(len(name) for name in metrics.keys()))

    # Format header
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_row = "|" + "|".join(f" {h:^{w}} " for h, w in zip(header, widths)) + "|"

    lines = [sep, header_row, sep]

    # Format data rows
    for name, m in metrics.items():
        values = [name] + [f"{m.get(mn, 0):.2f}" for mn in metric_names]
        row = "|" + "|".join(f" {v:^{w}} " for v, w in zip(values, widths)) + "|"
        lines.append(row)

    lines.append(sep)

    table = "\n".join(lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"Table saved to {save_path}")

    return table


if __name__ == "__main__":
    # Demo with synthetic data
    print("Generating demo plots...")

    # Create synthetic learning curves
    np.random.seed(42)
    n_episodes = 500

    q_rewards = np.cumsum(np.random.randn(n_episodes) * 10) - 1000
    sarsa_rewards = np.cumsum(np.random.randn(n_episodes) * 10) - 1100
    random_rewards = np.random.randn(n_episodes) * 100 - 1500

    # Test comparison plot
    results = {
        "Q-Learning": q_rewards.tolist(),
        "SARSA": sarsa_rewards.tolist(),
        "Random": random_rewards.tolist(),
    }

    fig = plot_learning_curves_comparison(
        results,
        title="Algorithm Comparison (Demo)",
        save_path="artifacts/plots/demo_comparison.png",
    )
    plt.close(fig)

    # Test multi-seed plot
    multi_seed = [
        (np.cumsum(np.random.randn(n_episodes) * 10) - 1000).tolist() for _ in range(5)
    ]

    fig = plot_multi_seed_learning_curve(
        multi_seed,
        title="Multi-Seed Learning Curve (Demo)",
        save_path="artifacts/plots/demo_multiseed.png",
    )
    plt.close(fig)

    # Test metrics dashboard
    eval_metrics = {
        "Q-Learning": {
            "mean_reward": -500,
            "std_reward": 50,
            "mean_sla_violations": 2.5,
            "mean_cost": 150,
            "mean_utilization": 0.72,
        },
        "SARSA": {
            "mean_reward": -600,
            "std_reward": 60,
            "mean_sla_violations": 3.0,
            "mean_cost": 140,
            "mean_utilization": 0.68,
        },
        "Random": {
            "mean_reward": -1500,
            "std_reward": 100,
            "mean_sla_violations": 10.0,
            "mean_cost": 200,
            "mean_utilization": 0.45,
        },
    }

    fig = plot_metrics_dashboard(
        eval_metrics, save_path="artifacts/plots/demo_dashboard.png"
    )
    plt.close(fig)

    # Test results table
    table = create_results_table(eval_metrics)
    print("\nResults Table:")
    print(table)

    print("\n✓ All demo plots generated in artifacts/plots/")
