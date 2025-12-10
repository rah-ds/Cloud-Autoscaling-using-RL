#!/usr/bin/env python3
"""
Generate comprehensive visualization plots for RL experiment results.

This script creates publication-quality plots from experiment results including:
- Learning curves with confidence intervals
- Algorithm comparison bar charts
- Convergence analysis
- Performance summary tables

Usage:
    python scripts/generate_plots.py                    # Use latest results
    python scripts/generate_plots.py --results FILE     # Use specific results file
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

# Algorithm display names and colors
ALGO_NAMES = {
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "random": "Random",
    "threshold": "Threshold",
}

ALGO_COLORS = {
    "q_learning": "#2ecc71",
    "sarsa": "#3498db",
    "dqn": "#e74c3c",
    "double_dqn": "#9b59b6",
    "dueling_dqn": "#f39c12",
    "random": "#95a5a6",
    "threshold": "#1abc9c",
}


def smooth_curve(data: List[float], window: int = 50) -> np.ndarray:
    """Apply moving average smoothing to data."""
    if len(data) < window:
        window = max(1, len(data) // 5)
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode="valid")
    return smoothed


def compute_confidence_interval(
    data: List[float], window: int = 50, confidence: float = 0.95
) -> tuple:
    """Compute confidence intervals for smoothed data."""
    n = len(data)
    if n < window:
        window = max(1, n // 5)

    means = []
    lower = []
    upper = []

    for i in range(window, n + 1):
        segment = data[i - window : i]
        mean = np.mean(segment)
        std = np.std(segment)
        se = std / np.sqrt(window)
        ci = stats.t.ppf((1 + confidence) / 2, window - 1) * se
        means.append(mean)
        lower.append(mean - ci)
        upper.append(mean + ci)

    return np.array(means), np.array(lower), np.array(upper)


def load_results(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load results from JSON file."""
    if results_path is None:
        # Find latest results file
        result_files = sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
        if not result_files:
            raise FileNotFoundError("No results files found in artifacts/results/")
        results_path = result_files[0]

    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        return json.load(f)


def plot_learning_curves(results: Dict[str, Any], save_path: Path) -> None:
    """Plot learning curves with confidence intervals for all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect all algorithms with episode data
    algos_with_data = []
    for algo, data in results.items():
        if (
            isinstance(data, dict)
            and "episode_rewards" in data
            and len(data["episode_rewards"]) > 0
        ):
            algos_with_data.append((algo, data))

    if not algos_with_data:
        print("No episode data found for learning curves")
        return

    # Plot 1: Raw learning curves
    ax = axes[0, 0]
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)
        ax.plot(rewards, alpha=0.3, color=color, linewidth=0.5)
        smoothed = smooth_curve(rewards, window=50)
        ax.plot(
            range(50, 50 + len(smoothed)),
            smoothed,
            label=name,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves (Raw + Smoothed)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Plot 2: Smoothed curves with confidence intervals
    ax = axes[0, 1]
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)

        mean, lower, upper = compute_confidence_interval(rewards, window=50)
        x = range(50, 50 + len(mean))
        ax.plot(x, mean, label=name, color=color, linewidth=2)
        ax.fill_between(x, lower, upper, alpha=0.2, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves with 95% Confidence Intervals")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative reward
    ax = axes[1, 0]
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Plot 4: Reward improvement rate (derivative of smoothed curve)
    ax = axes[1, 1]
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)

        smoothed = smooth_curve(rewards, window=100)
        improvement = np.diff(smoothed)
        improvement_smoothed = smooth_curve(improvement.tolist(), window=50)
        x = range(150, 150 + len(improvement_smoothed))
        ax.plot(x, improvement_smoothed, label=name, color=color, linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Improvement Rate")
    ax.set_title("Learning Rate (Derivative of Smoothed Rewards)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to: {save_path}")


def plot_algorithm_comparison(results: Dict[str, Any], save_path: Path) -> None:
    """Create bar chart comparing final performance of all algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect performance metrics
    algorithms = []
    mean_rewards = []
    std_rewards = []
    final_rewards = []
    best_rewards = []
    colors = []

    for algo, data in results.items():
        if isinstance(data, dict):
            name = ALGO_NAMES.get(algo, algo)
            algorithms.append(name)
            colors.append(ALGO_COLORS.get(algo, "#333333"))

            if "episode_rewards" in data and len(data["episode_rewards"]) > 0:
                rewards = data["episode_rewards"]
                # Use last 10% as final performance
                final_portion = rewards[int(len(rewards) * 0.9) :]
                mean_rewards.append(np.mean(final_portion))
                std_rewards.append(np.std(final_portion))
                final_rewards.append(data.get("mean_reward", np.mean(final_portion)))
                best_rewards.append(max(rewards))
            elif "mean_reward" in data:
                mean_rewards.append(data["mean_reward"])
                std_rewards.append(data.get("std_reward", 0))
                final_rewards.append(data["mean_reward"])
                best_rewards.append(data["mean_reward"])

    if not algorithms:
        print("No algorithm data found for comparison")
        return

    # Sort by mean reward
    sorted_idx = np.argsort(mean_rewards)[::-1]
    algorithms = [algorithms[i] for i in sorted_idx]
    mean_rewards = [mean_rewards[i] for i in sorted_idx]
    std_rewards = [std_rewards[i] for i in sorted_idx]
    best_rewards = [best_rewards[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    # Plot 1: Mean final reward with error bars
    ax = axes[0]
    x = np.arange(len(algorithms))
    bars = ax.bar(x, mean_rewards, yerr=std_rewards, color=colors, capsize=5, alpha=0.8)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Reward (Final 10%)")
    ax.set_title("Algorithm Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, mean_rewards):
        height = bar.get_height()
        ax.annotate(
            f"{val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Best reward achieved
    ax = axes[1]
    bars = ax.bar(x, best_rewards, color=colors, alpha=0.8)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Best Single Episode Reward")
    ax.set_title("Best Performance Achieved")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, best_rewards):
        height = bar.get_height()
        ax.annotate(
            f"{val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved algorithm comparison to: {save_path}")


def plot_convergence_analysis(results: Dict[str, Any], save_path: Path) -> None:
    """Analyze and plot convergence behavior of algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    algos_with_data = []
    for algo, data in results.items():
        if (
            isinstance(data, dict)
            and "episode_rewards" in data
            and len(data["episode_rewards"]) > 0
        ):
            algos_with_data.append((algo, data))

    if not algos_with_data:
        print("No episode data found for convergence analysis")
        return

    # Plot 1: Variance over time
    ax = axes[0, 0]
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)

        window = 50
        variances = []
        for i in range(window, len(rewards)):
            segment = rewards[i - window : i]
            variances.append(np.var(segment))

        smoothed_var = smooth_curve(variances, 20)
        x = range(window, window + len(smoothed_var))
        ax.plot(x, smoothed_var, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Variance (window=50)")
    ax.set_title("Variance Over Training (Lower = More Stable)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 2: Rolling max reward
    ax = axes[0, 1]
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)

        rolling_max = np.maximum.accumulate(rewards)
        ax.plot(rolling_max, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Best Reward So Far")
    ax.set_title("Running Maximum Reward")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Plot 3: Performance by training phase
    ax = axes[1, 0]
    phases = [
        "Early\n(0-25%)",
        "Mid-Early\n(25-50%)",
        "Mid-Late\n(50-75%)",
        "Late\n(75-100%)",
    ]
    phase_ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    x = np.arange(len(phases))
    width = 0.15

    for i, (algo, data) in enumerate(algos_with_data):
        rewards = data["episode_rewards"]
        n = len(rewards)
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)

        phase_means = []
        for start, end in phase_ranges:
            segment = rewards[int(n * start) : int(n * end)]
            phase_means.append(np.mean(segment))

        offset = (i - len(algos_with_data) / 2) * width
        ax.bar(x + offset, phase_means, width, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Training Phase")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Performance by Training Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Improvement percentage
    ax = axes[1, 1]
    algo_names = []
    improvements = []
    colors_list = []

    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        n = len(rewards)
        early = np.mean(rewards[: int(n * 0.1)])
        late = np.mean(rewards[int(n * 0.9) :])

        if early != 0:
            improvement = ((late - early) / abs(early)) * 100
        else:
            improvement = 0

        algo_names.append(ALGO_NAMES.get(algo, algo))
        improvements.append(improvement)
        colors_list.append(ALGO_COLORS.get(algo, "#333333"))

    bars = ax.barh(algo_names, improvements, color=colors_list, alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Improvement (%)")
    ax.set_title("Reward Improvement: Early (0-10%) vs Late (90-100%)")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, improvements):
        ax.annotate(
            f"{val:.1f}%",
            xy=(val, bar.get_y() + bar.get_height() / 2),
            xytext=(5 if val >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence analysis to: {save_path}")


def plot_summary_dashboard(results: Dict[str, Any], save_path: Path) -> None:
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(18, 12))

    # Collect data
    algos_with_data = []
    for algo, data in results.items():
        if (
            isinstance(data, dict)
            and "episode_rewards" in data
            and len(data["episode_rewards"]) > 0
        ):
            algos_with_data.append((algo, data))

    if not algos_with_data:
        print("No episode data found for dashboard")
        return

    # Layout: 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Learning curves (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)
        smoothed = smooth_curve(rewards, window=50)
        ax1.plot(
            range(50, 50 + len(smoothed)),
            smoothed,
            label=name,
            color=color,
            linewidth=2,
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Progress", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Final performance bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    algo_names = []
    final_rewards = []
    colors_list = []
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        algo_names.append(ALGO_NAMES.get(algo, algo))
        final_rewards.append(np.mean(rewards[int(len(rewards) * 0.9) :]))
        colors_list.append(ALGO_COLORS.get(algo, "#333333"))

    sorted_idx = np.argsort(final_rewards)[::-1]
    ax2.barh(
        [algo_names[i] for i in sorted_idx],
        [final_rewards[i] for i in sorted_idx],
        color=[colors_list[i] for i in sorted_idx],
        alpha=0.8,
    )
    ax2.set_xlabel("Mean Reward")
    ax2.set_title("Final Performance", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. Variance over time
    ax3 = fig.add_subplot(gs[1, 0])
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)
        window = 50
        variances = [
            np.var(rewards[max(0, i - window) : i]) for i in range(window, len(rewards))
        ]
        smoothed_var = smooth_curve(variances, 20)
        x = list(range(window, len(rewards)))
        min_len = min(len(x), len(smoothed_var))
        ax3.plot(
            x[:min_len], smoothed_var[:min_len], label=name, color=color, linewidth=2
        )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Variance")
    ax3.set_title("Stability (Variance)", fontsize=14, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Distribution of rewards (final phase)
    ax4 = fig.add_subplot(gs[1, 1])
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)
        final_rewards = rewards[int(len(rewards) * 0.9) :]
        ax4.hist(
            final_rewards, bins=30, alpha=0.5, label=name, color=color, density=True
        )
    ax4.set_xlabel("Reward")
    ax4.set_ylabel("Density")
    ax4.set_title("Reward Distribution (Final 10%)", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Cumulative reward
    ax5 = fig.add_subplot(gs[1, 2])
    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        color = ALGO_COLORS.get(algo, "#333333")
        name = ALGO_NAMES.get(algo, algo)
        cumulative = np.cumsum(rewards)
        ax5.plot(cumulative, label=name, color=color, linewidth=2)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Cumulative Reward")
    ax5.set_title("Total Reward Accumulated", fontsize=14, fontweight="bold")
    ax5.legend(loc="lower left", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")

    # Create statistics table
    table_data = []
    headers = [
        "Algorithm",
        "Episodes",
        "Mean Reward",
        "Std Dev",
        "Best Reward",
        "Final Reward",
        "Improvement",
    ]

    for algo, data in algos_with_data:
        rewards = data["episode_rewards"]
        n = len(rewards)
        early = np.mean(rewards[: int(n * 0.1)])
        late = np.mean(rewards[int(n * 0.9) :])
        improvement = ((late - early) / abs(early)) * 100 if early != 0 else 0

        table_data.append(
            [
                ALGO_NAMES.get(algo, algo),
                str(n),
                f"{np.mean(rewards):.1f}",
                f"{np.std(rewards):.1f}",
                f"{max(rewards):.1f}",
                f"{late:.1f}",
                f"{improvement:+.1f}%",
            ]
        )

    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Add title
    fig.suptitle(
        "Cloud Autoscaling RL - Training Summary Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary dashboard to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from experiment results"
    )
    parser.add_argument(
        "--results", type=str, default=None, help="Path to specific results JSON file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for plots"
    )
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results) if args.results else None
    results = load_results(results_path)

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)

    # Generate all plots
    plot_learning_curves(results, output_dir / f"learning_curves_{timestamp}.png")
    plot_algorithm_comparison(
        results, output_dir / f"algorithm_comparison_{timestamp}.png"
    )
    plot_convergence_analysis(
        results, output_dir / f"convergence_analysis_{timestamp}.png"
    )
    plot_summary_dashboard(results, output_dir / f"summary_dashboard_{timestamp}.png")

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
