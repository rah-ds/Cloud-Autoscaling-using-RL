#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH -J "rah5ff-rl"
#SBATCH --mail-user=rah5ff@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=ds5004-apt4c
#SBATCH --output=artifacts/logs/slurm_%j.out
#SBATCH --error=artifacts/logs/slurm_%j.err

echo "Hello $USER, this is node $(hostname)."

# =============================================================================
# Cloud Autoscaling RL - SLURM Job Script
# =============================================================================
# This script runs all experiments on a SLURM cluster (2000 episodes).
#
# Usage:
#   sbatch scripts/slurm_run_all.slurm
#
# Monitor job:
#   squeue -u $USER
#   tail -f artifacts/logs/slurm_<jobid>.out
# =============================================================================

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Cloud Autoscaling RL - SLURM Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Create output directories
mkdir -p artifacts/logs artifacts/plots artifacts/results artifacts/models

# Load modules (adjust for your cluster)
module load anaconda
module load cuda

# Set up Python environment
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    if command -v uv &> /dev/null; then
        uv venv --python=3.12
        uv sync
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e .
    fi
fi

# Activate environment
source .venv/bin/activate

# Set device to CUDA for GPU usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo ""
echo "=============================================="
echo "Training all algorithms on all workloads (2000 episodes)"
echo "=============================================="

# Train all algorithms on all workloads
for workload in smooth bursty seasonal; do
    echo ""
    echo "=== Training on $workload workload ==="
    python scripts/run_baselines.py --algo all --episodes 2000 --workload "$workload"
done

echo ""
echo "=============================================="
echo "Generating summary and figures..."
echo "=============================================="

# Generate summary
python scripts/show_summary.py

# Generate figures
python scripts/generate_figures.py --style paper

echo ""
echo "=============================================="
echo "Job completed successfully!"
echo "End time: $(date)"
echo "=============================================="
