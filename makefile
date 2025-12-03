.PHONY: help setup test lint format clean experiments ablations all quick

help: ## Show available commands
	@echo ""
	@echo "Cloud Autoscaling RL - Available Commands"
	@echo "==========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Setup & Environment
# =============================================================================

setup: ## Install uv and set up Python environment
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python=3.12
	uv sync
	@echo "Environment ready. Run: source .venv/bin/activate"

sync: ## Sync dependencies
	uv sync

# =============================================================================
# Code Quality
# =============================================================================

test: ## Run all tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Check code style
	uv run ruff check .

format: ## Format and fix code
	uv run ruff format .
	uv run ruff check --fix .

# =============================================================================
# Experiments (with tqdm progress bars)
# =============================================================================

experiments: ## Run full experiment suite (~10 min)
	@echo "Running full experiment suite..."
	uv run python scripts/run_baselines.py --algo all --episodes 1000

experiments-quick: ## Quick experiment test (~2 min)
	@echo "Running quick experiments..."
	uv run python scripts/run_baselines.py --algo all --quick

experiments-tabular: ## Run Q-Learning and SARSA only
	uv run python scripts/run_baselines.py --algo tabular

experiments-deep: ## Run DQN variants only
	uv run python scripts/run_baselines.py --algo deep

# =============================================================================
# Ablation Studies (with tqdm progress bars)
# =============================================================================

ablations: ## Run all ablation studies (~15 min)
	@echo "Running ablation studies..."
	uv run python scripts/run_ablations.py --study all

ablations-quick: ## Quick ablation test (~1 min)
	uv run python scripts/run_ablations.py --quick --study lr

ablations-grid: ## Run hyperparameter grid search
	uv run python scripts/run_ablations.py --study grid

# =============================================================================
# Full Pipeline
# =============================================================================

all: test experiments ablations ## Run tests + experiments + ablations
	@echo "All tasks complete!"

quick: ## Quick run of entire pipeline (~5 min)
	@echo "Running quick pipeline..."
	uv run pytest tests/ -q
	uv run python scripts/run_baselines.py --algo tabular --quick
	uv run python scripts/run_ablations.py --quick --study lr
	@echo "Quick pipeline complete!"

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache cleaned"

clean-artifacts: ## Remove generated artifacts
	rm -rf artifacts/logs/* artifacts/plots/* artifacts/results/* artifacts/models/*
	@echo "Artifacts cleaned"

clean-all: clean clean-artifacts ## Full cleanup (cache + artifacts + venv)
	rm -rf .venv htmlcov .coverage
	@echo "Full cleanup complete"

# =============================================================================
# Development
# =============================================================================

jupyter: ## Start Jupyter notebook
	uv run jupyter notebook notebooks/
