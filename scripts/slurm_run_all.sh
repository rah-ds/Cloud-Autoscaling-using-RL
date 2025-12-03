#!/bin/bash
#SBATCH --job-name=cloud-autoscaling-rl
#SBATCH --output=artifacts/logs/slurm_%j.out
#SBATCH --error=artifacts/logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@university.edu

# =============================================================================
# Cloud Autoscaling RL - SLURM Job Script
# =============================================================================
# This script runs all experiments and ablation studies on a SLURM cluster.
#
# Usage:
#   sbatch scripts/slurm_run_all.sh              # Run everything
#   sbatch scripts/slurm_run_all.sh quick        # Quick test run
#   sbatch scripts/slurm_run_all.sh experiments  # Experiments only
#   sbatch scripts/slurm_run_all.sh ablations    # Ablations only
#   sbatch scripts/slurm_run_all.sh deep         # Deep training (5000 episodes)
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
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Create output directories
mkdir -p artifacts/logs artifacts/plots artifacts/results artifacts/models

# Load modules (adjust for your cluster)
# module load python/3.12
# module load cuda/12.0  # If using GPU

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

# Parse command line argument
MODE="${1:-all}"

echo ""
echo "Running mode: $MODE"
echo "=============================================="

case "$MODE" in
    quick)
        echo "Running quick pipeline..."
        echo ""
        echo "[1/3] Running tests..."
        python -m pytest tests/ -q
        
        echo ""
        echo "[2/3] Running quick experiments..."
        python scripts/run_baselines.py --algo all --quick
        
        echo ""
        echo "[3/3] Running quick ablation..."
        python scripts/run_ablations.py --quick --study lr
        ;;
        
    experiments)
        echo "Running full experiments (1000 episodes)..."
        python scripts/run_baselines.py --algo all --episodes 1000
        ;;
        
    experiments-quick)
        echo "Running quick experiments..."
        python scripts/run_baselines.py --algo all --quick
        ;;
        
    ablations)
        echo "Running all ablation studies..."
        python scripts/run_ablations.py --study all
        ;;
        
    deep)
        echo "Running deep training (5000 episodes)..."
        python scripts/run_baselines.py --algo all --episodes 5000
        ;;
        
    deep-10k)
        echo "Running extended training (10000 episodes)..."
        python scripts/run_baselines.py --algo all --episodes 10000
        ;;
        
    all|*)
        echo "Running complete pipeline..."
        echo ""
        
        echo "[1/4] Running tests..."
        python -m pytest tests/ -v
        
        echo ""
        echo "[2/4] Running experiments (1000 episodes)..."
        python scripts/run_baselines.py --algo all --episodes 1000
        
        echo ""
        echo "[3/4] Running ablation studies..."
        python scripts/run_ablations.py --study all
        
        echo ""
        echo "[4/4] Generating summary..."
        python scripts/show_summary.py
        ;;
esac

echo ""
echo "=============================================="
echo "Job completed successfully!"
echo "End time: $(date)"
echo "=============================================="

# Print summary
echo ""
python scripts/show_summary.py
