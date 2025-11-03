#!/usr/bin/env python3
"""
Run baseline policies on the MMPPEnv and save summary artifacts.

This script runs simple baseline policies (fixed, reactive, hysteresis, oracle)
against the MMPP environment and produces JSON reports, example traces, and
summary plots. Outputs are written to an artifacts directory by default.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# ensure src is importable when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from gym_mmpp_env import MMPPEnv  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_baselines")


def safe_reset(env):
    """Reset env and return only the observation.

    Many gym/gymnasium environments may return either obs or (obs, info). This
    helper ensures the observation is returned in either case.
    """
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def get_arrivals_from_obs(obs) -> float:
    """Extract scalar arrival count from various observation formats.

    The MMPPEnv may return a dict with an 'arrivals' array, or a flat array.
    This helper normalizes those formats to a single float value.
    """
    if isinstance(obs, dict):
        a = obs.get("arrivals", None)
        # arrivals may be array-like
        if isinstance(a, np.ndarray):
            return float(a.ravel()[0])
        return float(a)
    # if observation is a flat array: arrivals first element
    if isinstance(obs, (list, tuple, np.ndarray)):
        return float(np.asarray(obs).ravel()[0])
    # fallback
    return float(obs)


# Baseline policies ---------------------------------------------------------
def fixed_policy_factory(fixed_servers: int) -> Callable[[float, int, int], int]:
    """Return a policy that always allocates a fixed number of servers."""
    def policy(arrivals: float, state: int, prev: int) -> int:
        return fixed_servers

    return policy


def reactive_policy_factory(server_capacity: int, buffer_servers: int = 0):
    """Return a reactive policy that provisions based on current arrivals.

    The policy computes the needed number of servers to cover current arrivals
    and optionally adds a buffer of extra servers.
    """
    def policy(arrivals: float, state: int, prev: int) -> int:
        needed = int(np.ceil(arrivals / server_capacity))
        return max(0, needed + buffer_servers)

    return policy


def hysteresis_policy_factory(server_capacity: int, up_factor=1.1, down_factor=0.8):
    """Return a hysteresis-based policy to avoid rapid scaling changes.

    The policy increases capacity when arrivals exceed up_factor * current_capacity
    and decreases when below down_factor * current_capacity.
    """
    def policy(arrivals: float, state: int, prev: int) -> int:
        cap = prev * server_capacity
        if arrivals > up_factor * cap:
            # scale up by setting to needed servers
            return int(np.ceil(arrivals / server_capacity))
        if arrivals < down_factor * cap:
            # scale down conservatively
            return max(0, prev - 1)
        return prev

    return policy


def oracle_policy_factory(server_capacity: int):
    """Return an oracle policy that perfectly provisions to match arrivals.

    This baseline is an upper-bound (clairvoyant) that uses current arrivals
    to allocate exactly enough servers.
    """
    def policy(arrivals: float, state: int, prev: int) -> int:
        return int(np.ceil(arrivals / server_capacity))

    return policy


# Experiment runner ---------------------------------------------------------
def run_episode(env: MMPPEnv, policy_fn: Callable[[float, int, int], int], episode_length: int):
    """Run one episode using policy_fn and collect metrics.

    Returns a dict of metrics and traces (arrivals, actions, rewards).
    """
    obs = safe_reset(env)
    prev_action = 0
    total_reward = 0.0
    total_unmet = 0.0
    total_servers = 0.0
    arrivals_trace = []
    actions_trace = []
    rewards_trace = []

    for t in range(episode_length):
        arrivals = get_arrivals_from_obs(obs)
        state = obs.get("state", 0) if isinstance(obs, dict) else 0
        action = int(policy_fn(arrivals, state, prev_action))
        # clip action
        action = max(0, min(env.max_servers, action))

        next_obs, reward, done, info = env.step(action)
        # normalize if env returns (obs, info) tuple from newer gym
        if isinstance(next_obs, tuple) and len(next_obs) == 2:
            next_obs = next_obs[0]

        unmet = max(0.0, arrivals - action * env.server_capacity)
        total_unmet += unmet
        total_servers += action
        total_reward += float(reward)
        arrivals_trace.append(arrivals)
        actions_trace.append(action)
        rewards_trace.append(float(reward))

        prev_action = action
        obs = next_obs
        if done:
            break

    metrics = {
        "total_reward": total_reward,
        "mean_reward": total_reward / max(1, len(rewards_trace)),
        "total_unmet": total_unmet,
        "mean_servers": total_servers / max(1, len(actions_trace)),
        "len": len(actions_trace),
        "arrivals": np.array(arrivals_trace),
        "actions": np.array(actions_trace),
        "rewards": np.array(rewards_trace),
    }
    return metrics


def run_experiment(
    outdir: str,
    policies: Dict[str, Callable[[float, int, int], int]],
    episodes: int = 10,
    episode_length: int = 200,
    seeds: List[int] = (0,),
    env_kwargs: Dict[str, Any] = None,
):
    """Run experiments for each policy across seeds and episodes.

    Saves example traces and a summary plot/JSON into the artifacts directory.
    """
    env_kwargs = env_kwargs or {}
    # ensure artifacts dir is under repo root by default
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(outdir, exist_ok=True)
    summary = defaultdict(list)

    # run each policy across seeds & episodes
    for seed in seeds:
        logger.info("seed=%d", seed)
        # create a fresh env per seed
        env = MMPPEnv(seed=seed, **(env_kwargs or {}))
        for name, policy in policies.items():
            logger.info("Running policy=%s", name)
            for ep in trange(episodes, desc=f"{name}@seed{seed}", leave=False):
                m = run_episode(env, policy, episode_length)
                summary[(name, "total_reward")].append(m["total_reward"])
                summary[(name, "total_unmet")].append(m["total_unmet"])
                summary[(name, "mean_servers")].append(m["mean_servers"])
                # save one example trace per policy+seed (first episode)
                if ep == 0:
                    np.savez(
                        os.path.join(outdir, f"trace_{name}_seed{seed}.npz"),
                        arrivals=m["arrivals"],
                        actions=m["actions"],
                        rewards=m["rewards"],
                    )

    # aggregate & report
    report = {}
    for (name, metric), values in list(summary.items()):
        arr = np.array(values)
        report.setdefault(name, {})[metric] = float(arr.mean())
        report[name].setdefault(metric + "_std", float(arr.std()))

    # print summary
    logger.info("==== Summary ====")
    for name, metrics in report.items():
        logger.info("%s: reward=%.3f±%.3f unmet=%.2f±%.2f mean_servers=%.2f±%.2f",
                    name,
                    metrics.get("total_reward", 0.0),
                    metrics.get("total_reward_std", 0.0),
                    metrics.get("total_unmet", 0.0),
                    metrics.get("total_unmet_std", 0.0),
                    metrics.get("mean_servers", 0.0),
                    metrics.get("mean_servers_std", 0.0))

    # make bar plot of mean total_reward & unmet
    names = list({k[0] for k in summary.keys()})
    names.sort()
    mean_rewards = [np.mean(summary[(n, "total_reward")]) for n in names]
    std_rewards = [np.std(summary[(n, "total_reward")]) for n in names]
    mean_unmet = [np.mean(summary[(n, "total_unmet")]) for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(names, mean_rewards, yerr=std_rewards)
    ax[0].set_title("Mean total reward")
    ax[0].set_ylabel("Total reward")
    ax[1].bar(names, mean_unmet)
    ax[1].set_title("Mean total unmet demand")
    ax[1].set_ylabel("Total unmet")
    plt.tight_layout()
    plot_path = os.path.join(outdir, "baseline_summary.png")
    fig.savefig(plot_path)
    logger.info("Saved summary plot to %s", plot_path)

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join(os.getcwd(), "artifacts"), help="output directory")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--policies", nargs="+", default=["fixed5", "reactive", "hyst", "oracle"])
    args = p.parse_args()

    # define policy instances
    env_kwargs = {"max_servers": 10, "server_capacity": 10}
    policies = {}
    if "fixed5" in args.policies:
        policies["fixed5"] = fixed_policy_factory(5)
    if "fixed2" in args.policies:
        policies["fixed2"] = fixed_policy_factory(2)
    if "reactive" in args.policies:
        policies["reactive"] = reactive_policy_factory(env_kwargs["server_capacity"], buffer_servers=0)
    if "hyst" in args.policies:
        policies["hyst"] = hysteresis_policy_factory(env_kwargs["server_capacity"], up_factor=1.2, down_factor=0.7)
    if "oracle" in args.policies:
        policies["oracle"] = oracle_policy_factory(env_kwargs["server_capacity"])

    report = run_experiment(
        outdir=args.outdir,
        policies=policies,
        episodes=args.episodes,
        episode_length=args.episode_length,
        seeds=args.seeds,
        env_kwargs=env_kwargs,
    )
    # save report
    
    artifacts_dir = args.outdir
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report.json to %s", artifacts_dir)


if __name__ == "__main__":
    main()