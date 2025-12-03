"""Tests for ablation study module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ablations import (
    AblationConfig,
    AblationResult,
    AblationStudy,
    run_hyperparameter_ablation,
    run_component_ablation,
    run_grid_ablation,
    plot_ablation_results,
    create_ablation_table,
    get_learning_rate_ablation_values,
    get_discount_factor_ablation_values,
    get_exploration_ablation_configs,
)


class TestAblationConfig:
    """Tests for AblationConfig dataclass."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = AblationConfig(
            name="test_config",
            params={"lr": 0.1, "gamma": 0.99},
            description="Test configuration",
        )

        assert config.name == "test_config"
        assert config.params["lr"] == 0.1
        assert config.params["gamma"] == 0.99
        assert config.description == "Test configuration"

    def test_config_hashable(self):
        """Test that configs are hashable (for use in sets/dicts)."""
        config1 = AblationConfig(name="a", params={})
        config2 = AblationConfig(name="b", params={})

        # Should be usable in a set
        config_set = {config1, config2}
        assert len(config_set) == 2


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        config = AblationConfig(name="test", params={"lr": 0.1})
        result = AblationResult(
            config=config,
            metrics={"mean_reward": 100.0, "std_reward": 10.0},
            runtime_seconds=5.0,
            seed=42,
        )

        assert result.config.name == "test"
        assert result.metrics["mean_reward"] == 100.0
        assert result.runtime_seconds == 5.0
        assert result.seed == 42

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        config = AblationConfig(name="test", params={"lr": 0.1})
        result = AblationResult(config=config, metrics={"mean_reward": 100.0}, seed=42)

        d = result.to_dict()
        assert d["config_name"] == "test"
        assert d["params"]["lr"] == 0.1
        assert d["metrics"]["mean_reward"] == 100.0


class TestAblationStudy:
    """Tests for AblationStudy class."""

    def test_study_creation(self):
        """Test basic study creation."""
        study = AblationStudy(name="test_study", description="A test ablation study")

        assert study.name == "test_study"
        assert study.description == "A test ablation study"
        assert len(study.results) == 0

    def test_add_result(self):
        """Test adding results to study."""
        study = AblationStudy(name="test", description="")

        config = AblationConfig(name="config1", params={})
        result = AblationResult(config=config, metrics={"reward": 100})

        study.add_result(result)
        assert len(study.results) == 1
        assert study.results[0].config.name == "config1"

    def test_get_best_config(self):
        """Test finding best configuration."""
        study = AblationStudy(name="test", description="")

        # Add results with different rewards
        for i, reward in enumerate([50, 100, 75]):
            config = AblationConfig(name=f"config{i}", params={})
            result = AblationResult(config=config, metrics={"mean_reward": reward})
            study.add_result(result)

        # Best should be config1 with reward=100
        best = study.get_best_config("mean_reward", higher_is_better=True)
        assert best.config.name == "config1"
        assert best.metrics["mean_reward"] == 100

        # Worst (when lower is better) should be config0
        worst = study.get_best_config("mean_reward", higher_is_better=False)
        assert worst.config.name == "config0"

    def test_save_and_load(self):
        """Test saving study to JSON."""
        study = AblationStudy(name="test_save", description="Test saving")

        config = AblationConfig(name="config1", params={"lr": 0.1})
        result = AblationResult(config=config, metrics={"reward": 100})
        study.add_result(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = study.save(Path(tmpdir))
            assert filepath.exists()

            # Check JSON content
            import json

            with open(filepath) as f:
                data = json.load(f)

            assert data["name"] == "test_save"
            assert len(data["results"]) == 1
            assert data["results"][0]["config_name"] == "config1"


class TestAblationPresets:
    """Tests for preset ablation values."""

    def test_learning_rate_values(self):
        """Test learning rate preset values."""
        values = get_learning_rate_ablation_values()

        assert isinstance(values, list)
        assert len(values) > 0
        assert all(0 < v <= 1 for v in values)
        assert 0.1 in values  # Common default

    def test_discount_factor_values(self):
        """Test discount factor preset values."""
        values = get_discount_factor_ablation_values()

        assert isinstance(values, list)
        assert len(values) > 0
        assert all(0 < v <= 1 for v in values)
        assert 0.99 in values  # Common default

    def test_exploration_configs(self):
        """Test exploration ablation configs."""
        configs = get_exploration_ablation_configs()

        assert isinstance(configs, dict)
        assert "no_exploration" in configs
        assert "constant_exploration" in configs
        assert configs["no_exploration"]["epsilon"] == 0.0


class TestAblationTable:
    """Tests for ablation table generation."""

    def test_create_table(self):
        """Test creating ASCII table."""
        study = AblationStudy(name="test", description="")

        for lr in [0.01, 0.1]:
            config = AblationConfig(name=f"lr={lr}", params={"lr": lr})
            result = AblationResult(
                config=config,
                metrics={
                    "mean_reward_mean": 100 * lr,
                    "mean_reward_std": 5.0,
                },
            )
            study.add_result(result)

        table = create_ablation_table(study)

        assert "Configuration" in table
        assert "lr=0.01" in table
        assert "lr=0.1" in table

    def test_empty_study_table(self):
        """Test table with no results."""
        study = AblationStudy(name="empty", description="")
        table = create_ablation_table(study)
        assert "No results" in table


class TestAblationRunners:
    """Tests for ablation study runners with mock components."""

    @pytest.fixture
    def mock_env_factory(self):
        """Create mock environment factory."""

        class MockEnv:
            def __init__(self, seed=42):
                self.seed = seed
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return np.array([0, 0, 0]), {}

            def step(self, action):
                self.step_count += 1
                reward = np.random.randn()
                done = self.step_count >= 10
                return np.array([0, 0, 0]), reward, done, False, {"sla_violations": 0}

        return MockEnv

    @pytest.fixture
    def mock_agent_factory(self):
        """Create mock agent factory."""

        class MockAgent:
            def __init__(self, learning_rate=0.1, **kwargs):
                self.learning_rate = learning_rate
                self.epsilon = 1.0

            def select_action(self, state, training=True):
                return np.random.randint(3)

            def update(self, *args, **kwargs):
                pass

            def decay_epsilon(self):
                self.epsilon *= 0.99

        def factory(**params):
            return MockAgent(**params)

        return factory

    @pytest.fixture
    def mock_train_fn(self):
        """Create mock training function."""

        def train(env, agent, n_episodes=10, verbose=False):
            rewards = [np.random.randn() * 10 for _ in range(n_episodes)]
            metrics = {"episode_rewards": rewards}
            return agent, metrics

        return train

    @pytest.fixture
    def mock_eval_fn(self):
        """Create mock evaluation function."""

        def evaluate(env, agent, n_episodes=5, verbose=False):
            return {
                "mean_reward": np.random.randn() * 10 + agent.learning_rate * 100,
                "std_reward": abs(np.random.randn()) * 5,
            }

        return evaluate

    def test_hyperparameter_ablation(
        self, mock_env_factory, mock_agent_factory, mock_train_fn, mock_eval_fn
    ):
        """Test hyperparameter ablation runner."""
        study = run_hyperparameter_ablation(
            env_factory=mock_env_factory,
            agent_factory=mock_agent_factory,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            param_name="learning_rate",
            param_values=[0.01, 0.1],
            base_params={"learning_rate": 0.1},
            n_episodes=10,
            n_seeds=1,
        )

        assert len(study.results) == 2
        assert study.baseline_config is not None
        assert "learning_rate=0.01" in [r.config.name for r in study.results]

    def test_component_ablation(
        self, mock_env_factory, mock_agent_factory, mock_train_fn, mock_eval_fn
    ):
        """Test component ablation runner."""
        components = {
            "no_learning": {"learning_rate": 0.0},
            "high_lr": {"learning_rate": 0.5},
        }

        study = run_component_ablation(
            env_factory=mock_env_factory,
            agent_factory=mock_agent_factory,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            components=components,
            base_params={"learning_rate": 0.1},
            n_episodes=10,
            n_seeds=1,
        )

        # Should have full_model + 2 components
        assert len(study.results) == 3
        names = [r.config.name for r in study.results]
        assert "full_model" in names

    def test_grid_ablation(
        self, mock_env_factory, mock_agent_factory, mock_train_fn, mock_eval_fn
    ):
        """Test grid ablation runner."""
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "discount_factor": [0.9, 0.99],
        }

        # Mock agent that accepts discount_factor
        def agent_with_gamma(**params):
            class Agent:
                def __init__(self):
                    self.learning_rate = params.get("learning_rate", 0.1)
                    self.epsilon = 1.0

                def select_action(self, state, training=True):
                    return 0

                def update(self, *args):
                    pass

                def decay_epsilon(self):
                    pass

            return Agent()

        study = run_grid_ablation(
            env_factory=mock_env_factory,
            agent_factory=agent_with_gamma,
            train_fn=mock_train_fn,
            eval_fn=mock_eval_fn,
            param_grid=param_grid,
            base_params={},
            n_episodes=10,
            n_seeds=1,
        )

        # 2 x 2 = 4 combinations
        assert len(study.results) == 4


class TestAblationPlots:
    """Tests for ablation visualization functions."""

    def test_plot_ablation_results(self):
        """Test plotting ablation results."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        study = AblationStudy(name="test_plot", description="")

        for i in range(3):
            config = AblationConfig(name=f"config{i}", params={})
            result = AblationResult(
                config=config,
                metrics={"mean_reward_mean": i * 10, "mean_reward_std": 2},
            )
            study.add_result(result)

        fig = plot_ablation_results(study)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_saves_to_file(self):
        """Test that plot can be saved."""
        import matplotlib

        matplotlib.use("Agg")

        study = AblationStudy(name="test_save", description="")
        config = AblationConfig(name="test", params={})
        result = AblationResult(config=config, metrics={"mean_reward_mean": 50})
        study.add_result(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            fig = plot_ablation_results(study, output_path=output_path)

            assert output_path.exists()

            import matplotlib.pyplot as plt

            plt.close(fig)
