#!/usr/bin/env python3
"""
Train a simple SB3 baseline (DQN or PPO) on the MMPPEnv using MPS/CUDA/CPU device.
Saves model, an evaluation summary and a small plot into the artifacts folder.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from gym_mmpp_env import MMPPEnv  # type: ignore


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sb3_baseline_{timestamp}.log"

    logger = logging.getLogger("run_baselines3")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # File handler
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
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
logger = logging.getLogger("run_baselines3")


# Small wrapper to convert Dict obs -> flat Box (arrivals, state)
import gymnasium as gym
from gymnasium import spaces


class FlattenDictToBox(gym.ObservationWrapper):
    """Convert {'arrivals': array([x]), 'state': int} -> np.array([x, state], dtype=float32)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

    def observation(self, obs):
        if isinstance(obs, tuple) and len(obs) == 2:  # (obs, info)
            obs = obs[0]
        if isinstance(obs, dict):
            a = obs.get("arrivals", 0)
            if isinstance(a, (list, tuple, np.ndarray)):
                a = float(np.asarray(a).ravel()[0])
            s = float(obs.get("state", 0))
            return np.asarray([a, s], dtype=np.float32)
        # fallback: allow explicit array-like inputs (list, tuple, ndarray)
        if isinstance(obs, (np.ndarray, list, tuple)):
            arr = np.asarray(obs).ravel()
            if arr.size >= 2:
                return arr[:2].astype(np.float32)
            out = np.zeros(2, dtype=np.float32)
            out[0] = float(arr.ravel()[0]) if arr.size >= 1 else 0.0
            return out

        # If we get here, obs is an unsupported type (e.g., dict-like keys), raise informative error
        raise TypeError(
            f"Unsupported observation type: {type(obs)}. Expected dict or array-like."
        )


def make_env(seed: int = 0, env_kwargs: dict = None) -> Callable:
    env_kwargs = env_kwargs or {}

    def _init():
        env = MMPPEnv(seed=seed, **env_kwargs)
        env = FlattenDictToBox(env)
        env.action_space.seed(seed)
        return env

    return _init


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    # MPS availability check (torch >= 1.12)
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=os.path.join(os.getcwd(), "artifacts"))
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args()

    # Setup logging
    global logger
    log_dir = Path(args.outdir) / "logs"
    logger = setup_logging(log_dir, args.log_level)

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info(f"Stable-Baselines3 {args.algo.upper()} Training")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Algorithm: {args.algo.upper()}")
    logger.info(f"  Timesteps: {args.timesteps}")
    logger.info(f"  N environments: {args.n_envs}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output dir: {args.outdir}")

    os.makedirs(args.outdir, exist_ok=True)
    device = detect_device()
    logger.info(f"Using device: {device}")

    env_kwargs = {"max_servers": 10, "server_capacity": 10}
    logger.info(f"Environment kwargs: {env_kwargs}")
    vec_env = DummyVecEnv(
        [
            make_env(seed=args.seed + i, env_kwargs=env_kwargs)
            for i in range(args.n_envs)
        ]
    )
    logger.info(f"Created {args.n_envs} vectorized environments")

    policy = "MlpPolicy"  # observation flattened to Box -> MlpPolicy
    logger.info(f"Initializing {args.algo.upper()} with {policy}")
    if args.algo == "ppo":
        model = PPO(policy, vec_env, verbose=1, seed=args.seed, device=device)
    else:
        model = DQN(
            policy,
            vec_env,
            verbose=1,
            seed=args.seed,
            device=device,
            buffer_size=50_000,
        )

    logger.info(f"Training {args.algo.upper()} for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    logger.info("Training complete")

    # save model
    model_path = os.path.join(args.outdir, f"{args.algo}_model_seed{args.seed}.zip")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # evaluate on a fresh single-environment instance
    logger.info("Running evaluation on fresh environment...")
    eval_env = make_env(seed=args.seed + 999, env_kwargs=env_kwargs)()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    logger.info(
        f"Evaluation (20 episodes): mean_reward={mean_reward:.3f} ± {std_reward:.3f}"
    )

    # save evaluation summary + simple plot
    summary = {
        "algo": args.algo,
        "timesteps": args.timesteps,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "seed": args.seed,
        "device": device,
    }
    summary_path = os.path.join(args.outdir, f"eval_{args.algo}_seed{args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation summary saved to {summary_path}")

    plot_path = os.path.join(args.outdir, f"eval_{args.algo}_seed{args.seed}.png")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([args.algo], [mean_reward], yerr=[std_reward])
    ax.set_title(f"Eval: {args.algo.upper()}")
    ax.set_ylabel("Mean reward")
    fig.savefig(plot_path)
    logger.info(f"Evaluation plot saved to {plot_path}")

    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {args.algo.upper()}")
    logger.info(f"Final evaluation: {mean_reward:.3f} ± {std_reward:.3f}")
    logger.info(f"Total runtime: {elapsed_time}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
