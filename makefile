.PHONY: help install sync format lint test clean run-experiments

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install uv package manager (requires brew on macOS or curl on Linux)
	@command -v uv >/dev/null 2>&1 || { \
		if command -v brew >/dev/null 2>&1; then \
			echo "Installing uv via brew..."; \
			brew install uv; \
		else \
			echo "Installing uv via curl..."; \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		fi; \
	}
	@echo "uv is installed"

setup: install ## Set up Python environment with uv
	uv venv --python=3.12
	uv sync
	@echo "Environment setup complete. Activate with: source .venv/bin/activate"

sync: ## Sync dependencies with uv
	uv sync

format: ## Format code with ruff
	uv run ruff format .

lint: ## Lint code with ruff
	uv run ruff check .

lint-fix: ## Lint and fix code with ruff
	uv run ruff check --fix .

test: ## Run all tests with pytest
	uv run pytest tests/ -v

test-quick: ## Run tests quickly (no verbose)
	uv run pytest tests/ -q

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

test-env: ## Run environment tests only
	uv run pytest tests/test_gym_mmpp_env.py tests/test_cloud_autoscaling_env.py -v

test-agents: ## Run agent tests only
	uv run pytest tests/test_agents.py -v

test-policies: ## Run baseline policy tests only
	uv run pytest tests/test_baseline_policies.py -v

clean: ## Remove virtual environment and cache files
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Experiment Targets
# ============================================================================

run-experiments: ## Run all RL experiments (Q-Learning, SARSA, baselines)
	uv run python scripts/run_baselines.py --algo all

run-experiments-quick: ## Quick test run of experiments (fewer episodes)
	uv run python scripts/run_baselines.py --algo all --quick

run-all: ## Run complete experiment suite including neural networks
	uv run python scripts/run_all_experiments.py

run-all-quick: ## Quick test of complete experiment suite
	uv run python scripts/run_all_experiments.py --quick

run-sarsa-nn: ## Train neural network SARSA baseline
	uv run python scripts/sarsa_baseline.py

run-dqn: ## Train DQN baseline (requires stable-baselines3)
	uv run python scripts/baseline_expanded.py --algo dqn

run-ppo: ## Train PPO baseline (requires stable-baselines3)
	uv run python scripts/baseline_expanded.py --algo ppo

# ============================================================================
# Jupyter & Development
# ============================================================================

jupyter: ## Start Jupyter notebook server
	uv run jupyter notebook

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files
