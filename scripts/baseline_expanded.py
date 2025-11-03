#!/usr/bin/env python3
"""
Train a simple SB3 baseline (DQN or PPO) on the MMPPEnv using MPS/CUDA/CPU device.
Saves model, an evaluation summary and a small plot into the artifacts folder.
"""
import argparse
import logging
import os
import sys
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

logging.basicConfig(level=logging.INFO)
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
        # fallback: try to coerce from array-like
        arr = np.asarray(obs).ravel()
        if arr.size >= 2:
            return arr[:2].astype(np.float32)
        # pad
        out = np.zeros(2, dtype=np.float32)
        out[0] = float(arr.ravel()[0]) if arr.size >= 1 else 0.0
        return out


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
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = detect_device()
    logger.info("Using device: %s", device)

    env_kwargs = {"max_servers": 10, "server_capacity": 10}
    vec_env = DummyVecEnv([make_env(seed=args.seed + i, env_kwargs=env_kwargs) for i in range(args.n_envs)])

    policy = "MlpPolicy"  # observation flattened to Box -> MlpPolicy
    if args.algo == "ppo":
        model = PPO(policy, vec_env, verbose=1, seed=args.seed, device=device)
    else:
        model = DQN(policy, vec_env, verbose=1, seed=args.seed, device=device, buffer_size=50_000)

    logger.info("Training %s for %d timesteps", args.algo.upper(), args.timesteps)
    model.learn(total_timesteps=args.timesteps)

    # save model
    model_path = os.path.join(args.outdir, f"{args.algo}_model_seed{args.seed}.zip")
    model.save(model_path)
    logger.info("Saved model to %s", model_path)

    # evaluate on a fresh single-environment instance
    eval_env = make_env(seed=args.seed + 999, env_kwargs=env_kwargs)()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    logger.info("Evaluation (20 eps): mean_reward=%.3f +/- %.3f", mean_reward, std_reward)

    # save evaluation summary + simple plot
    summary = {"algo": args.algo, "timesteps": args.timesteps, "mean_reward": float(mean_reward), "std_reward": float(std_reward)}
    with open(os.path.join(args.outdir, f"eval_{args.algo}_seed{args.seed}.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([args.algo], [mean_reward], yerr=[std_reward])
    ax.set_title(f"Eval: {args.algo.upper()}")
    ax.set_ylabel("Mean reward")
    fig.savefig(os.path.join(args.outdir, f"eval_{args.algo}_seed{args.seed}.png"))
    logger.info("Saved eval plot and summary to %s", args.outdir)

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()