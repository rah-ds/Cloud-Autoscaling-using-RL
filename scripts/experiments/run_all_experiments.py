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
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_suite_{timestamp}.log"

    logger = logging.getLogger("run_all_experiments")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # File handler
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


# Global logger
logger = logging.getLogger("run_all_experiments")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info("=" * 60)
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True, cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"stdout: {result.stdout[-500:]}")  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr[-500:]}")
        return False
    except FileNotFoundError:
        logger.error(f"✗ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all RL experiments")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test run with fewer episodes"
    )
    parser.add_argument(
        "--no-nn",
        action="store_true",
        help="Skip neural network experiments (stable-baselines3)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Setup logging
    global logger
    log_dir = PROJECT_ROOT / "artifacts" / "logs"
    logger = setup_logging(log_dir, args.log_level)

    python = sys.executable
    episodes = 100 if args.quick else args.episodes

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("Cloud Autoscaling RL - Complete Experiment Suite")
    logger.info("=" * 60)
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Configuration:")
    logger.info(f"  Quick mode: {args.quick}")
    logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Skip NN: {args.no_nn}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Log level: {args.log_level}")

    results = []

    # 1. Run tabular RL experiments (Q-Learning and SARSA)
    logger.info("Starting Experiment 1: Tabular RL Methods")
    results.append(
        run_command(
            [
                python,
                "scripts/run_baselines.py",
                "--episodes",
                str(episodes),
                "--seed",
                str(args.seed),
                "--algo",
                "tabular",
            ],
            "Tabular RL Experiments (Q-Learning + SARSA + Baselines)",
        )
    )

    # 2. Run Deep RL experiments (DQN variants)
    if not args.no_nn:
        logger.info("Starting Experiment 2: Deep RL Methods (DQN variants)")
        results.append(
            run_command(
                [
                    python,
                    "scripts/run_baselines.py",
                    "--episodes",
                    str(episodes),
                    "--seed",
                    str(args.seed),
                    "--algo",
                    "deep",
                ],
                "Deep RL Experiments (DQN + Double DQN + Dueling DQN)",
            )
        )

        # 3. Run SARSA neural network baseline
        logger.info("Starting Experiment 3: Neural Network SARSA")
        results.append(
            run_command(
                [
                    python,
                    "scripts/sarsa_baseline.py",
                    "--episodes",
                    str(min(episodes * 2, 2000)),
                    "--seed",
                    str(args.seed),
                ],
                "Neural Network SARSA Baseline",
            )
        )

        # 4. Run SB3 DQN baseline
        logger.info("Starting Experiment 4: SB3 DQN")
        results.append(
            run_command(
                [
                    python,
                    "scripts/baseline_expanded.py",
                    "--algo",
                    "dqn",
                    "--timesteps",
                    str(episodes * 100),
                    "--seed",
                    str(args.seed),
                ],
                "DQN Baseline (stable-baselines3)",
            )
        )

        # 5. Run SB3 PPO baseline
        logger.info("Starting Experiment 5: SB3 PPO")
        results.append(
            run_command(
                [
                    python,
                    "scripts/baseline_expanded.py",
                    "--algo",
                    "ppo",
                    "--timesteps",
                    str(episodes * 100),
                    "--seed",
                    str(args.seed),
                ],
                "PPO Baseline (stable-baselines3)",
            )
        )
    else:
        logger.info("Skipping neural network experiments (--no-nn flag)")

    # Calculate elapsed time
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {elapsed_time}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {sum(results)}")
    logger.info(f"Failed: {len(results) - sum(results)}")

    # List output files
    artifacts_dir = PROJECT_ROOT / "artifacts"
    if artifacts_dir.exists():
        logger.info(f"Output files in {artifacts_dir}:")
        for f in sorted(artifacts_dir.rglob("*")):
            if f.is_file():
                logger.debug(f"  {f.relative_to(PROJECT_ROOT)}")

    if all(results):
        logger.info("All experiments completed successfully!")
    else:
        logger.warning("Some experiments failed. Check logs for details.")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
