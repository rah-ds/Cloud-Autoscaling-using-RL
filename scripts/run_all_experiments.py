#!/usr/bin/env python3
"""
Master script to run all experiments for Cloud Autoscaling RL project.

This script orchestrates the complete experiment pipeline:
1. Baseline comparisons (random, threshold policies)
2. Q-Learning experiments with hyperparameter sweep
3. SARSA experiments with hyperparameter sweep
4. Algorithm comparison and visualization
5. Neural network baselines (optional, requires stable-baselines3)

Usage:
    python scripts/run_all_experiments.py              # Run all experiments
    python scripts/run_all_experiments.py --quick      # Quick test run
    python scripts/run_all_experiments.py --no-nn      # Skip neural network experiments
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all RL experiments")
    parser.add_argument("--quick", action="store_true", 
                        help="Quick test run with fewer episodes")
    parser.add_argument("--no-nn", action="store_true",
                        help="Skip neural network experiments (stable-baselines3)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    python = sys.executable
    episodes = 100 if args.quick else args.episodes
    
    print("="*60)
    print("Cloud Autoscaling RL - Complete Experiment Suite")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick mode: {args.quick}")
    print(f"Episodes: {episodes}")
    print(f"Skip NN: {args.no_nn}")
    
    results = []
    
    # 1. Run tabular RL experiments (Q-Learning and SARSA)
    results.append(run_command(
        [python, "scripts/run_baselines.py", 
         "--episodes", str(episodes),
         "--seed", str(args.seed),
         "--algo", "all"],
        "Tabular RL Experiments (Q-Learning + SARSA + Baselines)"
    ))
    
    # 2. Run SARSA neural network baseline
    if not args.no_nn:
        results.append(run_command(
            [python, "scripts/sarsa_baseline.py",
             "--episodes", str(min(episodes * 2, 2000)),
             "--seed", str(args.seed)],
            "Neural Network SARSA Baseline"
        ))
        
        # 3. Run DQN baseline
        results.append(run_command(
            [python, "scripts/baseline_expanded.py",
             "--algo", "dqn",
             "--timesteps", str(episodes * 100),
             "--seed", str(args.seed)],
            "DQN Baseline (stable-baselines3)"
        ))
        
        # 4. Run PPO baseline
        results.append(run_command(
            [python, "scripts/baseline_expanded.py",
             "--algo", "ppo",
             "--timesteps", str(episodes * 100),
             "--seed", str(args.seed)],
            "PPO Baseline (stable-baselines3)"
        ))
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    # List output files
    artifacts_dir = PROJECT_ROOT / "artifacts"
    if artifacts_dir.exists():
        print(f"\nOutput files in {artifacts_dir}:")
        for f in sorted(artifacts_dir.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(PROJECT_ROOT)}")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
