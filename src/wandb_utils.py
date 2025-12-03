"""
Weights & Biases (wandb) integration utilities for Cloud Autoscaling RL.

This module provides utilities for:
- Initializing wandb with the project API key
- Logging experiment metrics
- Creating and managing sweeps
- Artifact management
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

import wandb

WANDB_AVAILABLE = True


# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
WANDB_KEY_FILE = PROJECT_ROOT / "keys" / "wandb_key.txt"
DEFAULT_PROJECT = "cloud-autoscaling-rl"
DEFAULT_ENTITY = None  # Will use default entity from wandb account


def load_wandb_key() -> Optional[str]:
    """Load wandb API key from file."""
    if WANDB_KEY_FILE.exists():
        return WANDB_KEY_FILE.read_text().strip()
    return os.environ.get("WANDB_API_KEY")


def init_wandb(
    project: str = DEFAULT_PROJECT,
    entity: Optional[str] = DEFAULT_ENTITY,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    notes: Optional[str] = None,
    mode: str = "online",  # "online", "offline", "disabled"
    reinit: bool = True,
    **kwargs,
) -> Any:
    """
    Initialize a wandb run.

    Args:
        project: Project name on wandb
        entity: Team/user name on wandb
        name: Run name (auto-generated if None)
        config: Hyperparameters and config dict
        tags: List of tags for filtering runs
        group: Group runs together (e.g., by experiment type)
        job_type: Type of job (e.g., "train", "eval", "sweep")
        notes: Notes about the run
        mode: "online", "offline", or "disabled"
        reinit: Allow reinitializing in the same process
        **kwargs: Additional arguments to wandb.init()

    Returns:
        wandb run object or None if wandb is not available
    """
    if not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Install with 'pip install wandb'")
        return None

    # Set API key
    api_key = load_wandb_key()
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    # Initialize wandb
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        group=group,
        job_type=job_type,
        notes=notes,
        mode=mode,
        reinit=reinit,
        **kwargs,
    )

    return run


def log_metrics(
    metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True
) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metric name -> value
        step: Global step number (optional)
        commit: Whether to commit the metrics immediately
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    wandb.log(metrics, step=step, commit=commit)


def log_episode(
    episode: int,
    reward: float,
    sla_violations: int = 0,
    epsilon: Optional[float] = None,
    loss: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log episode-level metrics.

    Args:
        episode: Episode number
        reward: Total episode reward
        sla_violations: Number of SLA violations
        epsilon: Current exploration rate
        loss: Training loss (for deep RL)
        additional_metrics: Any additional metrics to log
    """
    metrics = {
        "episode": episode,
        "reward": reward,
        "sla_violations": sla_violations,
    }

    if epsilon is not None:
        metrics["epsilon"] = epsilon

    if loss is not None:
        metrics["loss"] = loss

    if additional_metrics:
        metrics.update(additional_metrics)

    log_metrics(metrics, step=episode)


def log_summary(
    final_reward: float,
    final_sla: float,
    training_time: Optional[float] = None,
    best_reward: Optional[float] = None,
    additional_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log summary metrics at the end of training.

    Args:
        final_reward: Final (or average over last N episodes) reward
        final_sla: Final SLA violation rate
        training_time: Total training time in seconds
        best_reward: Best reward achieved during training
        additional_summary: Any additional summary metrics
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    wandb.run.summary["final_reward"] = final_reward
    wandb.run.summary["final_sla"] = final_sla

    if training_time is not None:
        wandb.run.summary["training_time"] = training_time

    if best_reward is not None:
        wandb.run.summary["best_reward"] = best_reward

    if additional_summary:
        for key, value in additional_summary.items():
            wandb.run.summary[key] = value


def log_learning_curve(
    rewards: List[float], sla_violations: Optional[List[int]] = None, window: int = 100
) -> None:
    """
    Log smoothed learning curves as wandb tables/plots.

    Args:
        rewards: List of episode rewards
        sla_violations: List of SLA violations per episode (logged if provided)
        window: Smoothing window size
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    # Create smoothed data
    smoothed_rewards = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed_rewards.append(np.mean(rewards[start : i + 1]))

    # Build table data with optional SLA violations
    if sla_violations is not None:
        data = [
            [i, rewards[i], smoothed_rewards[i], sla_violations[i]]
            for i in range(len(rewards))
        ]
        columns = ["episode", "reward", "smoothed_reward", "sla_violations"]
    else:
        data = [[i, rewards[i], smoothed_rewards[i]] for i in range(len(rewards))]
        columns = ["episode", "reward", "smoothed_reward"]

    table = wandb.Table(data=data, columns=columns)
    wandb.log({"learning_curve": table})


def log_model(
    model_path: Union[str, Path],
    name: str = "model",
    artifact_type: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a model as a wandb artifact.

    Args:
        model_path: Path to the model file
        name: Artifact name
        artifact_type: Artifact type (e.g., "model", "checkpoint")
        metadata: Additional metadata for the artifact
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)


def log_q_table(q_table: np.ndarray, name: str = "q_table") -> None:
    """
    Log a Q-table as a wandb artifact.

    Args:
        q_table: The Q-table numpy array
        name: Artifact name
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    # Save Q-table to temp file and log
    artifact = wandb.Artifact(name, type="q_table")
    with artifact.new_file("q_table.npy", mode="wb") as f:
        np.save(f, q_table)
    wandb.log_artifact(artifact)


def finish_run(quiet: bool = False) -> None:
    """Finish the current wandb run."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish(quiet=quiet)


def create_sweep(
    sweep_config: Dict[str, Any],
    project: str = DEFAULT_PROJECT,
    entity: Optional[str] = DEFAULT_ENTITY,
) -> str:
    """
    Create a wandb sweep.

    Args:
        sweep_config: Sweep configuration dictionary
        project: Project name
        entity: Team/user name

    Returns:
        Sweep ID
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb not installed")

    api_key = load_wandb_key()
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    return wandb.sweep(sweep_config, project=project, entity=entity)


def run_sweep_agent(
    sweep_id: str,
    function: callable,
    count: Optional[int] = None,
    project: str = DEFAULT_PROJECT,
    entity: Optional[str] = DEFAULT_ENTITY,
) -> None:
    """
    Run a wandb sweep agent.

    Args:
        sweep_id: Sweep ID from create_sweep()
        function: Training function to run
        count: Number of runs to execute
        project: Project name
        entity: Team/user name
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb not installed")

    api_key = load_wandb_key()
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    wandb.agent(
        sweep_id, function=function, count=count, project=project, entity=entity
    )


# Predefined sweep configuration
SWEEP_CONFIGS = {
    "dqn": {
        "method": "bayes",
        "name": "dqn-hyperparameter-sweep",
        "metric": {"name": "final_reward", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "min": 1e-4,
                "max": 1e-2,
                "distribution": "log_uniform_values",
            },
            "discount_factor": {"values": [0.95, 0.99]},
            "batch_size": {"values": [32, 64, 128]},
            "hidden_dim": {"values": [64, 128, 256]},
            "target_update_freq": {"values": [10, 50, 100]},
        },
    },
}


def get_sweep_config(name: str) -> Dict[str, Any]:
    """Get a predefined sweep configuration by name."""
    if name not in SWEEP_CONFIGS:
        raise ValueError(
            f"Unknown sweep config: {name}. Available: {list(SWEEP_CONFIGS.keys())}"
        )
    return SWEEP_CONFIGS[name]
