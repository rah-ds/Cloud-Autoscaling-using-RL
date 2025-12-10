#!/usr/bin/env python3
"""Generate report with actual experiment results."""

import json
import re
from pathlib import Path


def get_latest_results(results_dir: Path) -> dict | None:
    """Get the most recent experiment results file."""
    result_files = sorted(results_dir.glob("results_*.json"), reverse=True)
    if not result_files:
        return None
    with open(result_files[0], encoding="utf-8") as f:
        return json.load(f)


def get_ablation_results(results_dir: Path) -> dict:
    """Get all ablation study results."""
    ablations = {}
    for ablation_file in results_dir.glob("ablation_*.json"):
        with open(ablation_file, encoding="utf-8") as f:
            data = json.load(f)
            ablations[data.get("name", ablation_file.stem)] = data
    return ablations


def format_reward(reward: float) -> str:
    """Format reward for display."""
    if abs(reward) >= 1_000_000:
        return f"{reward / 1_000_000:.2f}M"
    elif abs(reward) >= 1_000:
        return f"{reward / 1_000:.1f}K"
    return f"{reward:.2f}"


def get_model_scores(results: dict) -> list:
    """Extract model scores from results."""
    scores = []

    # Baselines
    if "baselines" in results:
        for name, data in results["baselines"].items():
            reward = data.get("mean_reward", 0)
            sla = data.get("mean_sla_violations", 0)
            scores.append((name.title(), reward, sla, "Baseline"))

    # Tabular agents
    for agent in ["q_learning", "sarsa"]:
        if agent in results:
            rewards = results[agent].get("episode_rewards", [])
            if rewards:
                final_rewards = rewards[int(len(rewards) * 0.9) :]
                mean_reward = sum(final_rewards) / len(final_rewards)
                sla = results[agent].get("final_sla_violations", 0)
                display_name = agent.replace("_", " ").title()
                scores.append((display_name, mean_reward, sla, "Tabular"))

    # Deep RL agents
    for agent in ["dqn", "double_dqn", "dueling_dqn"]:
        if agent in results:
            rewards = results[agent].get("episode_rewards", [])
            if rewards:
                final_rewards = rewards[int(len(rewards) * 0.9) :]
                mean_reward = sum(final_rewards) / len(final_rewards)
                sla = results[agent].get("final_sla_violations", 0)
                display_name = agent.replace("_", " ").upper()
                scores.append((display_name, mean_reward, sla, "Deep RL"))

    # Sort by reward (higher is better)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def generate_results_table(scores: list) -> str:
    """Generate markdown table of results."""
    lines = ["| Rank | Algorithm | Mean Reward | SLA Violations | Type |"]
    lines.append("|------|-----------|-------------|----------------|------|")

    for i, (name, reward, sla, model_type) in enumerate(scores, 1):
        lines.append(
            f"| {i} | {name} | {format_reward(reward)} | {sla:.2f} | {model_type} |"
        )

    return "\n".join(lines)


def generate_ablation_table(ablation_data: dict, param_name: str) -> str:
    """Generate markdown table for ablation study."""
    results = ablation_data.get("results", [])
    if not results:
        return "No results available"

    # Sort by reward
    sorted_results = sorted(
        results,
        key=lambda x: x.get("metrics", {}).get("mean_reward_mean", float("-inf")),
        reverse=True,
    )

    lines = [f"| {param_name.replace('_', ' ').title()} | Reward | Rank |"]
    lines.append("|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 6 + "|")

    for i, result in enumerate(sorted_results, 1):
        config = result.get("config_name", "")
        reward = result.get("metrics", {}).get("mean_reward_mean", 0)
        best_marker = " ⭐" if i == 1 else ""
        lines.append(f"| {config} | {format_reward(reward)} | {i}{best_marker} |")

    return "\n".join(lines)


def update_report(report_path: Path, results: dict, ablations: dict) -> None:
    """Update report template with actual results."""
    with open(report_path, encoding="utf-8") as f:
        content = f.read()

    scores = get_model_scores(results)

    # Update best model in abstract
    if scores:
        best_model = scores[0][0]
        content = content.replace("[BEST_MODEL]", best_model)

    # Generate and insert results table
    results_table = generate_results_table(scores)

    # Find and replace the placeholder table
    table_pattern = r"\| Rank \| Algorithm \| Mean Reward \| SLA Violations \| Type \|[\s\S]*?\| 7 \| Random \| \[TBD\] \| \[TBD\] \| Baseline \|"
    content = re.sub(table_pattern, results_table, content)

    # Update key results in conclusion
    if scores:
        best_algo = scores[0][0]
        best_reward = format_reward(scores[0][1])
        content = content.replace(
            "[Best algorithm] achieved the highest mean reward of [value]",
            f"{best_algo} achieved the highest mean reward of {best_reward}",
        )

    # Update ablation tables if available
    for ablation_name, ablation_data in ablations.items():
        if "learning_rate" in ablation_name:
            generate_ablation_table(ablation_data, "learning_rate")
            # Find best LR
            results_list = ablation_data.get("results", [])
            if results_list:
                best = max(
                    results_list,
                    key=lambda x: x.get("metrics", {}).get(
                        "mean_reward_mean", float("-inf")
                    ),
                )
                best_lr = best.get("params", {}).get("learning_rate", "N/A")
                content = content.replace(
                    "[Insert finding about optimal learning rate]",
                    f"Optimal learning rate is {best_lr}, achieving the best balance of learning speed and stability.",
                )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Report updated: {report_path}")
    print(f"   Best model: {scores[0][0] if scores else 'N/A'}")
    print(f"   Best reward: {format_reward(scores[0][1]) if scores else 'N/A'}")


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "artifacts" / "results"
    report_path = project_root / "docs" / "report" / "report.md"

    if not report_path.exists():
        print("❌ Report template not found. Run from project root.")
        return

    results = get_latest_results(results_dir)
    ablations = get_ablation_results(results_dir)

    if not results:
        print("❌ No experiment results found. Run: make experiments")
        return

    update_report(report_path, results, ablations)

    print("\n📊 Summary:")
    print("-" * 40)
    scores = get_model_scores(results)
    for i, (name, reward, sla, model_type) in enumerate(scores[:5], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {name}: {format_reward(reward)}")


if __name__ == "__main__":
    main()
