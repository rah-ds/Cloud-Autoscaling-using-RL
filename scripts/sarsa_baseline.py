#!/usr/bin/env python3
"""
Train a neural-network SARSA baseline on MMPPEnv. On-policy SARSA (one-step)
with epsilon-greedy exploration and a small MLP Q-approximator.

Saves model (state_dict) and training reward curve to ./artifacts.
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt

# ensure src importable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from gym_mmpp_env import MMPPEnv  # type: ignore


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sarsa_nn_{timestamp}.log"

    logger = logging.getLogger("sarsa_baseline")
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
logger = logging.getLogger("sarsa_baseline")


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def obs_to_feat(obs) -> np.ndarray:
    """Convert env observation (dict or array) -> 1D numpy features [arrivals, state]."""
    if isinstance(obs, tuple) and len(obs) == 2:
        obs = obs[0]
    if isinstance(obs, dict):
        a = obs.get("arrivals", 0)
        if isinstance(a, (list, tuple, np.ndarray)):
            a = float(np.asarray(a).ravel()[0])
        s = float(obs.get("state", 0))
        return np.asarray([a, s], dtype=np.float32)
    arr = np.asarray(obs).ravel()
    out = np.zeros(2, dtype=np.float32)
    if arr.size >= 1:
        out[0] = float(arr[0])
    if arr.size >= 2:
        out[1] = float(arr[1])
    return out


class QNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def epsilon_greedy(q_values: torch.Tensor, eps: float) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(0, q_values.shape[-1]))
    return int(torch.argmax(q_values).item())


def evaluate(
    env: MMPPEnv, policy_fn, episodes: int = 10, max_steps: int = 200
) -> float:
    total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        feat = obs_to_feat(obs)
        done = False
        ep_reward = 0.0
        for t in range(max_steps):
            action = policy_fn(feat)
            next_obs, reward, done, info = env.step(action)
            if isinstance(next_obs, tuple) and len(next_obs) == 2:
                next_obs = next_obs[0]
            feat = obs_to_feat(next_obs)
            ep_reward += float(reward)
            if done:
                break
        total += ep_reward
    return total / episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=1000)
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
    logger.info("Neural Network SARSA Baseline Training")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Gamma: {args.gamma}")
    logger.info(
        f"  Epsilon: {args.eps_start} -> {args.eps_end} (decay: {args.eps_decay_steps})"
    )
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output dir: {args.outdir}")

    os.makedirs(args.outdir, exist_ok=True)
    device = detect_device()
    logger.info(f"Using device: {device}")

    env = MMPPEnv(seed=args.seed, max_servers=10, server_capacity=10)
    obs = env.reset()
    feat = obs_to_feat(obs)
    n_actions = env.action_space.n
    logger.info(f"Environment: n_actions={n_actions}, obs_shape={feat.shape}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    qnet = QNet(input_dim=2, n_actions=n_actions, hidden=64).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    rewards: List[float] = []
    eval_every = max(1, args.episodes // 10)
    eval_scores: List[Tuple[int, float]] = []

    for ep in trange(args.episodes, desc="SARSA"):
        obs = env.reset()
        state_feat = obs_to_feat(obs)
        state_t = torch.tensor(
            state_feat, dtype=torch.float32, device=device
        ).unsqueeze(0)
        qvals = qnet(state_t)
        eps = max(
            args.eps_end,
            args.eps_start
            - (args.eps_start - args.eps_end) * (ep / args.eps_decay_steps),
        )
        action = epsilon_greedy(qvals[0].detach().cpu(), eps)
        ep_reward = 0.0

        for t in range(args.max_steps):
            next_obs, reward, done, info = env.step(action)
            if isinstance(next_obs, tuple) and len(next_obs) == 2:
                next_obs = next_obs[0]
            next_feat = obs_to_feat(next_obs)
            next_t = torch.tensor(
                next_feat, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                q_next = qnet(next_t)
            next_action = epsilon_greedy(q_next[0].detach().cpu(), eps)

            # SARSA target: r + gamma * Q(next, next_action) * (1 - done)
            q_pred = qnet(state_t)[0, action]
            q_next_val = q_next[0, next_action]
            target = reward + args.gamma * q_next_val * (0.0 if done else 1.0)
            target_t = torch.tensor(target, dtype=torch.float32, device=device)

            loss = loss_fn(q_pred, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_reward += float(reward)

            # move to next
            state_t = next_t
            action = next_action

            if done:
                break

        rewards.append(ep_reward)

        # periodic evaluation
        if (ep + 1) % eval_every == 0 or ep == args.episodes - 1:
            mean_eval = evaluate(
                MMPPEnv(seed=args.seed + 1000),
                lambda f: int(
                    torch.argmax(
                        qnet(
                            torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(device)
                        )
                    ).item()
                ),
                episodes=5,
                max_steps=args.max_steps,
            )
            eval_scores.append((ep + 1, mean_eval))
            logger.info(
                f"Episode {ep + 1}/{args.episodes}: reward={ep_reward:.3f}, eval_mean={mean_eval:.3f}, eps={eps:.3f}"
            )

    # save model
    model_path = os.path.join(args.outdir, f"sarsa_qnet_seed{args.seed}.pth")
    torch.save(qnet.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # save reward curve
    plot_path = os.path.join(args.outdir, "sarsa_training_curve.png")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, label="episode reward")
    if eval_scores:
        xs, ys = zip(*eval_scores)
        ax.plot(xs, ys, "-o", label="eval mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.savefig(plot_path)
    logger.info(f"Training plot saved to {plot_path}")

    # final evaluation
    final_eval = evaluate(
        MMPPEnv(seed=args.seed + 999),
        lambda f: int(
            torch.argmax(
                qnet(torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(device))
            ).item()
        ),
        episodes=20,
        max_steps=args.max_steps,
    )

    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final evaluation (20 episodes): mean_reward={final_eval:.3f}")
    logger.info(f"Total runtime: {elapsed_time}")

    env.close()


if __name__ == "__main__":
    main()
