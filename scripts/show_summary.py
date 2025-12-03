#!/usr/bin/env python3
"""Generate a summary of experiment results and ablation studies."""

import json
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


def print_model_performance(results: dict) -> None:
    """Print model performance summary."""
    print("\n🏆 Model Performance (Latest Run)")
    print("=" * 50)

    # Collect all model scores
    scores = []

    # Baselines
    if "baselines" in results:
        for name, data in results["baselines"].items():
            reward = data.get("mean_reward", 0)
            scores.append((name.title(), reward, "baseline"))

    # Tabular agents
    for agent in ["q_learning", "sarsa"]:
        if agent in results:
            rewards = results[agent].get("episode_rewards", [])
            if rewards:
                # Use last 10% of rewards as final performance
                final_rewards = rewards[int(len(rewards) * 0.9) :]
                mean_reward = sum(final_rewards) / len(final_rewards)
                scores.append((agent.replace("_", " ").title(), mean_reward, "tabular"))

    # Deep RL agents
    for agent in ["dqn", "double_dqn", "dueling_dqn"]:
        if agent in results:
            rewards = results[agent].get("episode_rewards", [])
            if rewards:
                final_rewards = rewards[int(len(rewards) * 0.9) :]
                mean_reward = sum(final_rewards) / len(final_rewards)
                display_name = (
                    agent.replace("_", " ").upper() if "dqn" in agent else agent.title()
                )
                scores.append((display_name, mean_reward, "deep"))

    # Sort by reward (higher is better)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Print ranked results
    print(f"\n{'Rank':<6}{'Model':<20}{'Reward':<15}{'Type':<10}")
    print("-" * 50)

    for i, (name, reward, model_type) in enumerate(scores, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<4}{name:<20}{format_reward(reward):<15}{model_type:<10}")

    if scores:
        best_model = scores[0][0]
        print(f"\n✨ Best performing model: {best_model}")


def print_ablation_summary(ablations: dict) -> None:
    """Print ablation study summary."""
    if not ablations:
        print("\n🔬 No ablation studies completed yet")
        return

    print("\n🔬 Ablation Study Insights")
    print("=" * 50)

    for name, data in ablations.items():
        print(f"\n📊 {name.replace('_', ' ').title()}")
        print("-" * 40)

        results = data.get("results", [])
        if not results:
            continue

        # Find best configuration
        best_config = None
        best_reward = float("-inf")

        for result in results:
            reward = result.get("metrics", {}).get("mean_reward_mean", float("-inf"))
            if reward > best_reward:
                best_reward = reward
                best_config = result

        if best_config:
            config_name = best_config.get("config_name", "unknown")
            print(f"  Best config: {config_name}")
            print(f"  Best reward: {format_reward(best_reward)}")

            # Show all tested values with their performance
            print("\n  All configurations:")
            sorted_results = sorted(
                results,
                key=lambda x: x.get("metrics", {}).get(
                    "mean_reward_mean", float("-inf")
                ),
                reverse=True,
            )
            for result in sorted_results[:5]:  # Show top 5
                cfg = result.get("config_name", "")
                reward = result.get("metrics", {}).get("mean_reward_mean", 0)
                is_best = "← best" if result == best_config else ""
                print(f"    {cfg:<25} {format_reward(reward):<12} {is_best}")


def print_recommendations(results: dict, ablations: dict) -> None:
    """Print recommendations based on results."""
    print("\n💡 Recommendations")
    print("=" * 50)

    recommendations = []

    # Check if deep RL is beating tabular
    if results:
        tabular_best = float("-inf")
        deep_best = float("-inf")

        for agent in ["q_learning", "sarsa"]:
            if agent in results:
                rewards = results[agent].get("episode_rewards", [])
                if rewards:
                    final = sum(rewards[-10:]) / 10
                    tabular_best = max(tabular_best, final)

        for agent in ["dqn", "double_dqn", "dueling_dqn"]:
            if agent in results:
                rewards = results[agent].get("episode_rewards", [])
                if rewards:
                    final = sum(rewards[-10:]) / 10
                    deep_best = max(deep_best, final)

        if deep_best > tabular_best and deep_best != float("-inf"):
            recommendations.append(
                "• Deep RL agents outperform tabular methods - consider more training"
            )
        elif tabular_best > deep_best and tabular_best != float("-inf"):
            recommendations.append(
                "• Tabular methods performing well - deep RL may need more episodes"
            )

    # Check ablation insights
    for name, data in ablations.items():
        results_list = data.get("results", [])
        if results_list and "learning_rate" in name:
            best = max(
                results_list,
                key=lambda x: x.get("metrics", {}).get(
                    "mean_reward_mean", float("-inf")
                ),
            )
            lr = best.get("params", {}).get("learning_rate", 0.1)
            if lr < 0.05:
                recommendations.append(
                    f"• Low learning rate ({lr}) works best - patience pays off"
                )
            elif lr > 0.2:
                recommendations.append(
                    f"• High learning rate ({lr}) works best - fast adaptation needed"
                )

    if not recommendations:
        recommendations.append("• Run more experiments to get recommendations")
        recommendations.append("• Try: make ablations to explore hyperparameters")

    for rec in recommendations:
        print(rec)


def main():
    """Main function to generate summary."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "artifacts" / "results"

    # Get latest results
    results = get_latest_results(results_dir)
    ablations = get_ablation_results(results_dir)

    if results:
        print_model_performance(results)
    else:
        print("\n📊 No experiment results found yet")
        print("   Run: make experiments-quick")

    print_ablation_summary(ablations)
    print_recommendations(results or {}, ablations)
    print()


if __name__ == "__main__":
    main()
