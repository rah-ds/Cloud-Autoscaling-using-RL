"""
Ablation Studies for Cloud Autoscaling RL Project.

This module provides utilities for running systematic ablation studies
to understand the impact of different hyperparameters and components.
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    params: Dict[str, Any]
    description: str = ""

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, AblationConfig):
            return False
        return self.name == other.name


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""

    config: AblationConfig
    metrics: Dict[str, float]
    training_curves: Dict[str, List[float]] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config.name,
            "params": self.config.params,
            "description": self.config.description,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "seed": self.seed,
        }


@dataclass
class AblationStudy:
    """Container for a complete ablation study."""

    name: str
    description: str
    results: List[AblationResult] = field(default_factory=list)
    baseline_config: Optional[AblationConfig] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def add_result(self, result: AblationResult) -> None:
        """Add a result to the study."""
        self.results.append(result)

    def get_best_config(
        self, metric: str = "mean_reward", higher_is_better: bool = True
    ) -> AblationResult:
        """Get the configuration with the best performance."""
        if not self.results:
            raise ValueError("No results in study")

        def key_func(r):
            return r.metrics.get(
                    metric, float("-inf") if higher_is_better else float("inf")
                )
        return (
            max(self.results, key=key_func)
            if higher_is_better
            else min(self.results, key=key_func)
        )

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd

        rows = []
        for result in self.results:
            row = {
                "config_name": result.config.name,
                **result.config.params,
                **result.metrics,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, output_dir: Path) -> Path:
        """Save study results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"ablation_{self.name}_{self.timestamp}.json"

        data = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "baseline": self.baseline_config.name if self.baseline_config else None,
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Ablation study saved to {filepath}")
        return filepath


# =============================================================================
# Ablation Study Runners
# =============================================================================


def run_hyperparameter_ablation(
    env_factory: Callable,
    agent_factory: Callable,
    train_fn: Callable,
    eval_fn: Callable,
    param_name: str,
    param_values: List[Any],
    base_params: Dict[str, Any],
    n_episodes: int = 500,
    n_seeds: int = 3,
    study_name: Optional[str] = None,
) -> AblationStudy:
    """
    Run ablation study varying a single hyperparameter.

    Args:
        env_factory: Function to create environment
        agent_factory: Function to create agent with given params
        train_fn: Training function (env, agent, n_episodes) -> metrics
        eval_fn: Evaluation function (env, agent) -> metrics
        param_name: Name of parameter to vary
        param_values: List of values to test
        base_params: Base parameters for agent
        n_episodes: Training episodes per run
        n_seeds: Number of random seeds
        study_name: Name for the study

    Returns:
        AblationStudy with results
    """
    study_name = study_name or f"{param_name}_ablation"
    study = AblationStudy(
        name=study_name,
        description=f"Ablation study varying {param_name} over values: {param_values}",
    )

    # Set baseline
    baseline_params = base_params.copy()
    study.baseline_config = AblationConfig(
        name="baseline", params=baseline_params, description="Baseline configuration"
    )

    total_runs = len(param_values) * n_seeds

    with tqdm(total=total_runs, desc=f"Ablation: {param_name}") as pbar:
        for value in param_values:
            seed_metrics = []
            seed_curves = []

            for seed in range(n_seeds):
                # Create config
                params = base_params.copy()
                params[param_name] = value
                params["seed"] = seed + 42

                # Create environment and agent
                env = env_factory(seed=seed + 42)
                agent = agent_factory(**params)

                # Train
                start_time = datetime.now()
                agent, train_metrics = train_fn(
                    env, agent, n_episodes=n_episodes, verbose=False
                )
                runtime = (datetime.now() - start_time).total_seconds()

                # Evaluate
                eval_metrics = eval_fn(env, agent, n_episodes=50, verbose=False)

                seed_metrics.append(eval_metrics)
                seed_curves.append(train_metrics.get("episode_rewards", []))

                pbar.update(1)

            # Aggregate across seeds
            aggregated_metrics = {}
            for key in seed_metrics[0].keys():
                values = [m[key] for m in seed_metrics]
                aggregated_metrics[f"{key}_mean"] = float(np.mean(values))
                aggregated_metrics[f"{key}_std"] = float(np.std(values))

            # Create result
            config = AblationConfig(
                name=f"{param_name}={value}",
                params={**base_params, param_name: value},
                description=f"{param_name} set to {value}",
            )

            result = AblationResult(
                config=config,
                metrics=aggregated_metrics,
                training_curves={
                    "episode_rewards": [
                        np.mean([c[i] if i < len(c) else c[-1] for c in seed_curves])
                        for i in range(n_episodes)
                    ]
                },
                runtime_seconds=runtime,
                seed=42,
            )

            study.add_result(result)

    return study


def run_component_ablation(
    env_factory: Callable,
    agent_factory: Callable,
    train_fn: Callable,
    eval_fn: Callable,
    components: Dict[str, Dict[str, Any]],
    base_params: Dict[str, Any],
    n_episodes: int = 500,
    n_seeds: int = 3,
) -> AblationStudy:
    """
    Run ablation study removing/modifying components.

    Args:
        env_factory: Function to create environment
        agent_factory: Function to create agent
        train_fn: Training function
        eval_fn: Evaluation function
        components: Dict of component name -> params to modify
        base_params: Base parameters
        n_episodes: Training episodes
        n_seeds: Number of seeds

    Returns:
        AblationStudy with results
    """
    study = AblationStudy(
        name="component_ablation",
        description="Ablation study removing/modifying components",
    )

    # Add baseline (full model)
    study.baseline_config = AblationConfig(
        name="full_model",
        params=base_params,
        description="Full model with all components",
    )

    # Test each component removal
    all_configs = [("full_model", base_params)] + list(components.items())
    total_runs = len(all_configs) * n_seeds

    with tqdm(total=total_runs, desc="Component Ablation") as pbar:
        for config_name, param_mods in all_configs:
            seed_metrics = []

            for seed in range(n_seeds):
                # Merge params
                if config_name == "full_model":
                    params = base_params.copy()
                else:
                    params = {**base_params, **param_mods}
                params["seed"] = seed + 42

                # Create and train
                env = env_factory(seed=seed + 42)
                agent = agent_factory(**params)
                agent, _ = train_fn(env, agent, n_episodes=n_episodes, verbose=False)
                eval_metrics = eval_fn(env, agent, n_episodes=50, verbose=False)

                seed_metrics.append(eval_metrics)
                pbar.update(1)

            # Aggregate
            aggregated_metrics = {}
            for key in seed_metrics[0].keys():
                values = [m[key] for m in seed_metrics]
                aggregated_metrics[f"{key}_mean"] = float(np.mean(values))
                aggregated_metrics[f"{key}_std"] = float(np.std(values))

            config = AblationConfig(
                name=config_name,
                params=params,
                description=f"Without {config_name}"
                if config_name != "full_model"
                else "Full model",
            )

            study.add_result(AblationResult(config=config, metrics=aggregated_metrics))

    return study


def run_grid_ablation(
    env_factory: Callable,
    agent_factory: Callable,
    train_fn: Callable,
    eval_fn: Callable,
    param_grid: Dict[str, List[Any]],
    base_params: Dict[str, Any],
    n_episodes: int = 500,
    n_seeds: int = 1,
) -> AblationStudy:
    """
    Run full grid search ablation over multiple parameters.

    Args:
        env_factory: Function to create environment
        agent_factory: Function to create agent
        train_fn: Training function
        eval_fn: Evaluation function
        param_grid: Dict of param_name -> list of values
        base_params: Base parameters
        n_episodes: Training episodes
        n_seeds: Number of seeds

    Returns:
        AblationStudy with all combinations
    """
    study = AblationStudy(
        name="grid_ablation", description=f"Grid search over: {list(param_grid.keys())}"
    )

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    total_runs = len(combinations) * n_seeds
    logger.info(
        f"Running grid ablation: {len(combinations)} combinations x {n_seeds} seeds = {total_runs} runs"
    )

    with tqdm(total=total_runs, desc="Grid Ablation") as pbar:
        for combo in combinations:
            params_dict = dict(zip(param_names, combo))
            seed_metrics = []

            for seed in range(n_seeds):
                params = {**base_params, **params_dict, "seed": seed + 42}

                env = env_factory(seed=seed + 42)
                agent = agent_factory(**params)
                agent, _ = train_fn(env, agent, n_episodes=n_episodes, verbose=False)
                eval_metrics = eval_fn(env, agent, n_episodes=50, verbose=False)

                seed_metrics.append(eval_metrics)
                pbar.update(1)

            # Aggregate
            aggregated_metrics = {}
            for key in seed_metrics[0].keys():
                values = [m[key] for m in seed_metrics]
                aggregated_metrics[f"{key}_mean"] = float(np.mean(values))
                aggregated_metrics[f"{key}_std"] = float(np.std(values))

            config_name = "_".join(f"{k}={v}" for k, v in params_dict.items())
            config = AblationConfig(name=config_name, params=params_dict)

            study.add_result(AblationResult(config=config, metrics=aggregated_metrics))

    return study


# =============================================================================
# Visualization
# =============================================================================


def plot_ablation_results(
    study: AblationStudy,
    metric: str = "mean_reward_mean",
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ablation study results as a bar chart.

    Args:
        study: AblationStudy to visualize
        metric: Metric to plot
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = [r.config.name for r in study.results]
    values = [r.metrics.get(metric, 0) for r in study.results]
    errors = [r.metrics.get(metric.replace("_mean", "_std"), 0) for r in study.results]

    # Color baseline differently
    colors = ["#2ecc71" if "baseline" in n or "full" in n else "#3498db" for n in names]

    bars = ax.bar(
        names,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
    )

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Ablation Study: {study.name}", fontsize=14, fontweight="bold")

    # Rotate x labels if many configs
    if len(names) > 5:
        plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Ablation plot saved to {output_path}")

    return fig


def plot_ablation_heatmap(
    study: AblationStudy,
    param1: str,
    param2: str,
    metric: str = "mean_reward_mean",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot 2D heatmap for grid ablation with two parameters.

    Args:
        study: AblationStudy (should be from grid ablation)
        param1: First parameter (x-axis)
        param2: Second parameter (y-axis)
        metric: Metric to visualize
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Extract unique values
    param1_vals = sorted(set(r.config.params.get(param1) for r in study.results))
    param2_vals = sorted(set(r.config.params.get(param2) for r in study.results))

    # Create matrix
    matrix = np.zeros((len(param2_vals), len(param1_vals)))

    for result in study.results:
        p1_val = result.config.params.get(param1)
        p2_val = result.config.params.get(param2)
        if p1_val in param1_vals and p2_val in param2_vals:
            i = param2_vals.index(p2_val)
            j = param1_vals.index(p1_val)
            matrix[i, j] = result.metrics.get(metric, 0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=param1_vals,
        yticklabels=param2_vals,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )

    ax.set_xlabel(param1.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(param2.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Ablation Heatmap: {param1} vs {param2}", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Ablation heatmap saved to {output_path}")

    return fig


def plot_learning_curve_comparison(
    study: AblationStudy,
    figsize: Tuple[int, int] = (12, 6),
    window: int = 10,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot learning curves for all configurations in an ablation study.

    Args:
        study: AblationStudy with training curves
        figsize: Figure size
        window: Smoothing window size
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(study.results)))

    for result, color in zip(study.results, colors):
        rewards = result.training_curves.get("episode_rewards", [])
        if not rewards:
            continue

        # Smooth
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            x = np.arange(window - 1, len(rewards))
        else:
            smoothed = rewards
            x = np.arange(len(rewards))

        ax.plot(x, smoothed, label=result.config.name, color=color, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(f"Learning Curves: {study.name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Learning curve comparison saved to {output_path}")

    return fig


def create_ablation_table(
    study: AblationStudy,
    metrics: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> str:
    """
    Create ASCII table of ablation results.

    Args:
        study: AblationStudy to summarize
        metrics: List of metrics to include (None = all)
        sort_by: Metric to sort by
        ascending: Sort order

    Returns:
        Formatted ASCII table string
    """
    if not study.results:
        return "No results in study"

    # Determine metrics to show
    if metrics is None:
        metrics = [k for k in study.results[0].metrics.keys() if "_mean" in k]

    # Sort results
    results = study.results.copy()
    if sort_by:
        results.sort(key=lambda r: r.metrics.get(sort_by, 0), reverse=not ascending)

    # Build table
    headers = ["Configuration"] + [
        m.replace("_mean", "").replace("_", " ").title() for m in metrics
    ]

    # Calculate column widths
    col_widths = [
        max(len(h), max(len(r.config.name) for r in results)) for h in [headers[0]]
    ]
    col_widths += [max(len(h), 12) for h in headers[1:]]

    # Format header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    lines = [header_line, separator]

    # Format rows
    for result in results:
        row = [result.config.name.ljust(col_widths[0])]
        for i, m in enumerate(metrics):
            val = result.metrics.get(m, 0)
            std = result.metrics.get(m.replace("_mean", "_std"), 0)
            row.append(f"{val:8.2f}±{std:4.2f}".ljust(col_widths[i + 1]))
        lines.append(" | ".join(row))

    return "\n".join(lines)


# =============================================================================
# Pre-built Ablation Configurations
# =============================================================================


def get_learning_rate_ablation_values() -> List[float]:
    """Standard learning rate values to test."""
    return [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]


def get_discount_factor_ablation_values() -> List[float]:
    """Standard discount factor values to test."""
    return [0.8, 0.9, 0.95, 0.99, 0.999]


def get_epsilon_decay_ablation_values() -> List[float]:
    """Standard epsilon decay values to test."""
    return [0.99, 0.995, 0.999, 0.9995]


def get_exploration_ablation_configs() -> Dict[str, Dict[str, Any]]:
    """Component ablation for exploration strategies."""
    return {
        "no_exploration": {"epsilon": 0.0, "epsilon_min": 0.0},
        "constant_exploration": {"epsilon": 0.1, "epsilon_decay": 1.0},
        "slow_decay": {"epsilon_decay": 0.9995},
        "fast_decay": {"epsilon_decay": 0.99},
        "high_min_epsilon": {"epsilon_min": 0.1},
    }


def get_dqn_component_ablation_configs() -> Dict[str, Dict[str, Any]]:
    """Component ablation for DQN architectures."""
    return {
        "no_target_network": {"target_update_freq": 1},  # Update every step = no target
        "small_buffer": {"buffer_size": 1000},
        "large_buffer": {"buffer_size": 100000},
        "small_batch": {"batch_size": 16},
        "large_batch": {"batch_size": 256},
        "shallow_network": {"hidden_dims": (64,)},
        "deep_network": {"hidden_dims": (256, 256, 128)},
        "no_epsilon_decay": {"epsilon_decay": 1.0, "epsilon": 0.1},
    }


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    # Demo with mock data
    print("Ablation Studies Module Demo")
    print("=" * 50)

    # Create mock study
    study = AblationStudy(
        name="learning_rate_demo", description="Demo ablation study for learning rates"
    )

    # Add mock results
    for lr in [0.01, 0.1, 0.3]:
        config = AblationConfig(
            name=f"lr={lr}",
            params={"learning_rate": lr},
            description=f"Learning rate {lr}",
        )
        result = AblationResult(
            config=config,
            metrics={
                "mean_reward_mean": 100 - abs(lr - 0.1) * 200 + np.random.randn() * 5,
                "mean_reward_std": np.random.rand() * 10,
                "mean_sla_violations_mean": np.random.rand() * 5,
                "mean_sla_violations_std": np.random.rand(),
            },
            training_curves={"episode_rewards": list(np.cumsum(np.random.randn(100)))},
        )
        study.add_result(result)

    # Print table
    print("\nAblation Results Table:")
    print(create_ablation_table(study))

    # Find best
    best = study.get_best_config("mean_reward_mean")
    print(f"\nBest configuration: {best.config.name}")
    print(f"  Mean reward: {best.metrics['mean_reward_mean']:.2f}")

    # Save
    output_dir = Path("artifacts/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    study.save(output_dir)

    # Plot
    fig = plot_ablation_results(study, output_path=output_dir / "demo_ablation.png")
    plt.close(fig)

    print(f"\nDemo outputs saved to {output_dir}")
    print("\nAvailable ablation value presets:")
    print(f"  Learning rates: {get_learning_rate_ablation_values()}")
    print(f"  Discount factors: {get_discount_factor_ablation_values()}")
    print(f"  Epsilon decay: {get_epsilon_decay_ablation_values()}")
