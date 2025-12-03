"""
Configuration Management for Cloud Autoscaling RL Experiments

This module provides utilities for loading, merging, and validating
YAML configuration files for experiments.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    config_path: Union[str, Path], config_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load a YAML configuration file with inheritance support.

    Supports `_inherit` key for inheriting from a base config file.

    Args:
        config_path: Path to the config file (can be relative to config_dir)
        config_dir: Directory containing config files

    Returns:
        Loaded and merged configuration dictionary

    Example:
        >>> config = load_config("configs/quick.yaml")
        >>> print(config["experiment"]["n_episodes"])
        100
    """
    config_path = Path(config_path)

    # Determine config directory
    if config_dir is None:
        if config_path.is_absolute():
            config_dir = config_path.parent
        else:
            config_dir = Path(__file__).parent.parent / "configs"
    config_dir = Path(config_dir)

    # Resolve full path
    if not config_path.is_absolute():
        # Check if it's just a filename
        if not config_path.parent.name:
            config_path = config_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Handle inheritance
    if "_inherit" in config:
        parent_name = config.pop("_inherit")
        parent_path = config_dir / parent_name
        parent_config = load_config(parent_path, config_dir)
        config = deep_merge(parent_config, config)

    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.

    Returns:
        Default configuration dictionary
    """
    config_dir = Path(__file__).parent.parent / "configs"
    return load_config(config_dir / "default.yaml")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["experiment", "environment", "q_learning", "sarsa"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate experiment settings
    exp = config["experiment"]
    if exp.get("n_episodes", 0) <= 0:
        raise ValueError("n_episodes must be positive")
    if exp.get("eval_episodes", 0) <= 0:
        raise ValueError("eval_episodes must be positive")

    # Validate agent settings
    for agent in ["q_learning", "sarsa"]:
        agent_config = config[agent]
        if not 0 < agent_config.get("learning_rate", 0) <= 1:
            raise ValueError(f"{agent}.learning_rate must be in (0, 1]")
        if not 0 < agent_config.get("discount_factor", 0) <= 1:
            raise ValueError(f"{agent}.discount_factor must be in (0, 1]")
        if not 0 <= agent_config.get("epsilon_min", 0) <= 1:
            raise ValueError(f"{agent}.epsilon_min must be in [0, 1]")

    return True


class ExperimentConfig:
    """
    Experiment configuration wrapper with attribute access.

    Example:
        >>> cfg = ExperimentConfig.from_yaml("configs/default.yaml")
        >>> print(cfg.experiment.n_episodes)
        1000
        >>> print(cfg.q_learning.learning_rate)
        0.1
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

        # Convert nested dicts to attribute-accessible objects
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, ExperimentConfig(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        config = load_config(path)
        validate_config(config)
        return cls(config)

    @classmethod
    def from_name(cls, name: str) -> "ExperimentConfig":
        """Load configuration by name (e.g., 'default', 'quick')."""
        config_dir = Path(__file__).parent.parent / "configs"
        return cls.from_yaml(config_dir / f"{name}.yaml")

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return self._config

    def __repr__(self) -> str:
        return f"ExperimentConfig({list(self._config.keys())})"


# Convenience function
def get_config(name: str = "default") -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        name: Config name ('default', 'quick', 'full_experiment', 'sweep')

    Returns:
        ExperimentConfig object

    Example:
        >>> cfg = get_config("quick")
        >>> print(cfg.experiment.n_episodes)
        100
    """
    return ExperimentConfig.from_name(name)


if __name__ == "__main__":
    # Test config loading
    print("Testing config loading...")

    # Load default config
    default = load_config("default.yaml")
    print(f"Default config: {default['experiment']['n_episodes']} episodes")

    # Load quick config (inherits from default)
    quick = load_config("quick.yaml")
    print(f"Quick config: {quick['experiment']['n_episodes']} episodes")

    # Test ExperimentConfig class
    cfg = get_config("default")
    print(f"ExperimentConfig: {cfg}")
    print(f"  n_episodes: {cfg.experiment.n_episodes}")
    print(f"  learning_rate: {cfg.q_learning.learning_rate}")

    print("\n✓ All config tests passed!")
