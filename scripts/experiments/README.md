# Experimental Scripts

This directory contains experimental, deprecated, or work-in-progress scripts that are not part of the main training pipeline.

## Contents

### Legacy Agent Implementations
- `DQN_Agent.py` - Original standalone DQN agent (superseded by `src/agent/`)
- `Double_DQN_Agent.py` - Original standalone Double DQN agent
- `DQN_Utils.py` - Original DQN utilities
- `autoscaling_env.py` - Original environment (superseded by `src/gym_mmpp_env.py`)
- `sarsa_baseline.py` - Original SARSA implementation
- `baseline_expanded.py` - Expanded baseline experiments

### Experimental Training Scripts
- `run_ablations.py` - Ablation study experiments
- `run_all_experiments.py` - Full experiment suite (superseded by `run_baselines.py`)
- `run_multiseed.py` - Multi-seed validation (planned for future work)
- `run_wandb_sweep.py` - W&B hyperparameter sweeps
- `run_with_wandb.py` - W&B integration experiments
- `train_sweep.py` - Training sweep experiments

### Visualization (Deprecated)
- `generate_plots.py` - Original plotting (superseded by `generate_figures.py`)
- `generate_report.py` - Report generation experiments

### Infrastructure
- `setup_google_cluster.sh` - Google Cloud setup script
- `slurm_run_all.slurm` - Rivanna HPC SLURM job script

## Active Scripts

The following scripts in the parent directory (`scripts/`) are actively used:
- `run_baselines.py` - Main training script for all algorithms
- `generate_figures.py` - Publication-ready figure generation
- `show_summary.py` - Experiment summary display
