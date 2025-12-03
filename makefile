.PHONY: help setup test lint format clean train quick optimize summary figures figures-paper figures-poster

help: ## Show available commands
	@echo ""
	@echo "Cloud Autoscaling RL - Available Commands"
	@echo "==========================================="
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
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

lint: ## Check code style
	uv run ruff check .

format: ## Format and fix code
	uv run ruff format .
	uv run ruff check --fix .

# =============================================================================
# Training & Optimization
# =============================================================================

train: ## Run deep RL training (2000 episodes)
	@echo "Running deep RL training (2000 episodes)..."
	uv run python scripts/run_baselines.py --algo deep --episodes 2000

quick: ## Quick test run (~2 min)
	@echo "Running quick test..."
	uv run python scripts/run_baselines.py --algo deep --quick

quick-all: ## Quick test of all algorithms on all workloads (~10 min)
	@echo "Running quick test of all algorithms on all workloads..."
	@for workload in smooth bursty seasonal; do \
		echo ""; \
		echo "=== Testing on $$workload workload ==="; \
		uv run python scripts/run_baselines.py --algo all --quick --workload $$workload --no-wandb --force; \
	done
	@echo ""
	@echo "All quick tests complete!"

train-all-scenarios: ## Train all algorithms on all workloads (full run)
	@echo "Training all algorithms on all workloads (this will take a while)..."
	@for workload in smooth bursty seasonal; do \
		echo ""; \
		echo "=== Training on $$workload workload ==="; \
		uv run python scripts/run_baselines.py --algo all --episodes 1000 --workload $$workload; \
	done
	@echo ""
	@echo "All training complete!"

train-bursty: ## Train on bursty workload only
	@echo "Training on bursty workload..."
	uv run python scripts/run_baselines.py --algo deep --episodes 1000 --workload bursty

train-seasonal: ## Train on seasonal workload only
	@echo "Training on seasonal workload..."
	uv run python scripts/run_baselines.py --algo deep --episodes 1000 --workload seasonal

optimize: ## Run Optuna Bayesian optimization (50 trials)
	@echo "Running Optuna hyperparameter optimization..."
	uv run python src/optuna_optimization.py --trials 50 --episodes 500

optimize-quick: ## Quick optimization test (10 trials)
	@echo "Running quick optimization..."
	uv run python src/optuna_optimization.py --quick

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf wandb/
	@echo "Cache cleaned"

clean-artifacts: ## Remove generated artifacts
	rm -rf artifacts/logs/* artifacts/plots/* artifacts/results/* artifacts/models/*
	@echo "Artifacts cleaned"

# =============================================================================
# Figures & Reports
# =============================================================================

figures: ## Generate publication-ready figures from latest results
	@echo "Generating publication-ready figures..."
	uv run python scripts/generate_figures.py --style presentation
	@echo ""
	@echo "📊 Figures saved to artifacts/plots/publication/"

figures-paper: ## Generate paper-quality figures (PDF, 300 DPI)
	@echo "Generating paper-quality figures..."
	uv run python scripts/generate_figures.py --style paper
	@echo ""
	@echo "📄 Paper figures saved to artifacts/plots/publication/"

figures-poster: ## Generate poster-quality figures (PNG, high DPI)
	@echo "Generating poster-quality figures..."
	uv run python scripts/generate_figures.py --style poster
	@echo ""
	@echo "🖼️  Poster figures saved to artifacts/plots/publication/"

# =============================================================================
# Utilities
# =============================================================================

summary: ## Show experiment summary
	@echo ""
	@echo "Cloud Autoscaling RL - Experiment Summary"
	@echo "=========================================="
	@echo ""
	@echo "📁 Artifacts:"
	@echo "  Results: $$(ls -1 artifacts/results/*.json 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "  Plots:   $$(ls -1 artifacts/plots/*.png 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "  Models:  $$(ls -1 artifacts/models/*.pth 2>/dev/null | wc -l | tr -d ' ') files"
	@echo ""
	@if ls artifacts/results/results_*.json 1>/dev/null 2>&1; then \
		uv run python scripts/show_summary.py; \
	fi

jupyter: ## Start Jupyter notebook
	uv run jupyter notebook notebooks/
