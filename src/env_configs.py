"""
Environment configurations for different workload patterns.

These configurations are used to create diverse training and evaluation
environments to test agent generalization across workload types.

Based on the configurations from the Evaluation_Results notebook.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class AutoScalingEnvConfig:
    """
    Configuration for the AutoScaling environment.
    
    Attributes:
        initial_capacity: Starting number of machines/instances
        target_utilization: Desired utilization level (0-1)
        sla_threshold: Maximum acceptable utilization before SLA violation
        cost_per_machine_per_minute: Cost of running one machine per time step
        min_capacity: Minimum number of machines
        max_capacity: Maximum number of machines
        cooldown_period: Steps to wait between scaling actions
        demand_scale: Multiplier for CPU demand
        cost_weight: Weight for cost in reward function
        util_weight: Weight for utilization penalty in reward
        sla_weight: Weight for SLA violations in reward
    """
    initial_capacity: int = 10
    target_utilization: float = 0.6
    sla_threshold: float = 0.8
    cost_per_machine_per_minute: float = 0.002
    min_capacity: int = 1
    max_capacity: int = 20
    cooldown_period: int = 5
    demand_scale: float = 1000.0
    cost_weight: float = 1.0
    util_weight: float = 1.0
    sla_weight: float = 10.0


# ============================================================================
# Pre-defined configurations for different workload types
# ============================================================================

SMOOTH_CONFIG = AutoScalingEnvConfig(
    initial_capacity=10,
    target_utilization=0.6,
    sla_threshold=0.8,
    cost_per_machine_per_minute=0.002,
    min_capacity=1,
    max_capacity=20,
    cooldown_period=5,
    demand_scale=1000.0,
    cost_weight=1.0,
    util_weight=1.0,
    sla_weight=10.0,
)
"""Smooth environment config - baseline from Google Cloud traces."""


BURSTY_CONFIG = AutoScalingEnvConfig(
    initial_capacity=10,
    target_utilization=0.6,
    sla_threshold=0.8,
    cost_per_machine_per_minute=0.002,
    min_capacity=1,
    max_capacity=20,
    cooldown_period=2,  # Shorter cooldown so agent can react faster
    demand_scale=1000.0,
    cost_weight=1.0,
    util_weight=1.0,
    sla_weight=15.0,  # SLA more important under bursts
)
"""Bursty environment config - high-volatility with abrupt spikes."""


SEASONAL_CONFIG = AutoScalingEnvConfig(
    initial_capacity=10,
    target_utilization=0.6,
    sla_threshold=0.8,
    cost_per_machine_per_minute=0.002,
    min_capacity=1,
    max_capacity=20,
    cooldown_period=3,
    demand_scale=1000.0,
    cost_weight=1.0,
    util_weight=1.0,
    sla_weight=10.0,
)
"""Seasonal environment config - periodic and oscillatory patterns."""


# ============================================================================
# Workload transformation functions
# ============================================================================

def make_bursty_workload(
    df: pd.DataFrame,
    spike_prob: float = 0.02,
    spike_length_range: tuple = (5, 20),
    spike_mult_range: tuple = (3.0, 7.0),
    random_state: int = 0,
    cpu_column: str = "avg_cpu",
) -> pd.DataFrame:
    """
    Transform a smooth workload into a bursty one by injecting random spikes.
    
    Args:
        df: DataFrame with CPU utilization data
        spike_prob: Probability of starting a spike at each timestep (~2%)
        spike_length_range: (min, max) duration of spikes in timesteps
        spike_mult_range: (min, max) multiplier for spike intensity
        random_state: Random seed for reproducibility
        cpu_column: Name of the CPU utilization column
        
    Returns:
        DataFrame with bursty workload pattern
    """
    rng = np.random.RandomState(random_state)
    df_bursty = df.copy()
    n = len(df_bursty)

    cpu = df_bursty[cpu_column].values.copy()

    i = 0
    while i < n:
        if rng.rand() < spike_prob:
            # Create a spike interval
            spike_len = rng.randint(spike_length_range[0], spike_length_range[1] + 1)
            spike_mult = rng.uniform(spike_mult_range[0], spike_mult_range[1])
            j_end = min(i + spike_len, n)
            cpu[i:j_end] = cpu[i:j_end] * spike_mult
            i = j_end
        else:
            i += 1

    # Cap CPU at 1.0
    cpu = np.clip(cpu, 0.0, 1.0)
    df_bursty[cpu_column] = cpu
    return df_bursty


def make_seasonal_workload(
    df: pd.DataFrame,
    period_steps: int = 2000,
    amp_factor: float = 0.5,
    noise_std_factor: float = 0.3,
    random_state: int = 1,
    cpu_column: str = "avg_cpu",
) -> pd.DataFrame:
    """
    Transform a workload into a seasonal pattern with periodic oscillations.
    
    Args:
        df: DataFrame with CPU utilization data
        period_steps: Length of one "day" cycle in timesteps
        amp_factor: How strong the sinusoidal component is
        noise_std_factor: How strong the noise is relative to baseline
        random_state: Random seed for reproducibility
        cpu_column: Name of the CPU utilization column
        
    Returns:
        DataFrame with seasonal workload pattern
    """
    rng = np.random.RandomState(random_state)
    df_seasonal = df.copy()
    n = len(df_seasonal)

    # Use a smoothed version of real CPU as baseline
    baseline = (
        df_seasonal[cpu_column]
        .rolling(window=60, min_periods=1, center=True)
        .mean()
        .values
    )

    # Build sinusoidal seasonal component
    t = np.arange(n)
    season = 1.0 + amp_factor * np.sin(2 * np.pi * t / period_steps)

    # Add multiplicative noise
    noise = 1.0 + noise_std_factor * rng.randn(n)

    cpu = baseline * season * noise

    # Clip to [0, 1]
    cpu = np.clip(cpu, 0.0, 1.0)
    df_seasonal[cpu_column] = cpu
    return df_seasonal


def make_smooth_workload(
    df: pd.DataFrame,
    window: int = 30,
    cpu_column: str = "avg_cpu",
) -> pd.DataFrame:
    """
    Apply smoothing to reduce short-term variance while preserving trends.
    
    Args:
        df: DataFrame with CPU utilization data
        window: Rolling window size for smoothing
        cpu_column: Name of the CPU utilization column
        
    Returns:
        DataFrame with smoothed workload pattern
    """
    df_smooth = df.copy()
    df_smooth[cpu_column] = (
        df_smooth[cpu_column]
        .rolling(window=window, min_periods=1, center=True)
        .mean()
    )
    return df_smooth


# ============================================================================
# Configuration registry for easy access
# ============================================================================

ENV_CONFIGS = {
    "smooth": SMOOTH_CONFIG,
    "bursty": BURSTY_CONFIG,
    "seasonal": SEASONAL_CONFIG,
}

WORKLOAD_TRANSFORMS = {
    "smooth": make_smooth_workload,
    "bursty": make_bursty_workload,
    "seasonal": make_seasonal_workload,
}


def get_config(name: str) -> AutoScalingEnvConfig:
    """
    Get a predefined environment configuration by name.
    
    Args:
        name: One of 'smooth', 'bursty', 'seasonal'
        
    Returns:
        AutoScalingEnvConfig for the specified environment type
    """
    if name.lower() not in ENV_CONFIGS:
        raise ValueError(f"Unknown config name: {name}. Choose from {list(ENV_CONFIGS.keys())}")
    return ENV_CONFIGS[name.lower()]


def transform_workload(df: pd.DataFrame, workload_type: str, **kwargs) -> pd.DataFrame:
    """
    Transform a DataFrame's workload pattern.
    
    Args:
        df: DataFrame with CPU utilization data
        workload_type: One of 'smooth', 'bursty', 'seasonal'
        **kwargs: Additional arguments passed to the transform function
        
    Returns:
        Transformed DataFrame
    """
    if workload_type.lower() not in WORKLOAD_TRANSFORMS:
        raise ValueError(f"Unknown workload type: {workload_type}. Choose from {list(WORKLOAD_TRANSFORMS.keys())}")
    
    transform_fn = WORKLOAD_TRANSFORMS[workload_type.lower()]
    return transform_fn(df, **kwargs)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create sample workload and transform it
    print("Environment Configurations:")
    print("=" * 50)
    
    for name, config in ENV_CONFIGS.items():
        print(f"\n{name.upper()} Config:")
        print(f"  cooldown_period: {config.cooldown_period}")
        print(f"  sla_weight: {config.sla_weight}")
        print(f"  target_utilization: {config.target_utilization}")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    t = np.linspace(0, 10 * np.pi, n_samples)
    base_cpu = 0.4 + 0.2 * np.sin(t) + 0.05 * np.random.randn(n_samples)
    base_cpu = np.clip(base_cpu, 0, 1)
    
    df_base = pd.DataFrame({"avg_cpu": base_cpu})
    
    print("\n\nWorkload Transformations:")
    print("=" * 50)
    
    for name in WORKLOAD_TRANSFORMS.keys():
        df_transformed = transform_workload(df_base, name)
        cpu = df_transformed["avg_cpu"]
        print(f"\n{name.upper()} workload:")
        print(f"  Mean: {cpu.mean():.3f}")
        print(f"  Std:  {cpu.std():.3f}")
        print(f"  Min:  {cpu.min():.3f}")
        print(f"  Max:  {cpu.max():.3f}")
