.PHONY: help setup test lint format clean experiments ablations all quick deep deep-10k summary

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

deep: ## Deep training run with 5000 episodes (~45 min)
	@echo "Running deep training (5000 episodes)..."
	uv run python scripts/run_baselines.py --algo all --episodes 5000

deep-10k: ## Extended training with 10000 episodes (~90 min)
	@echo "Running extended training (10000 episodes)..."
	uv run python scripts/run_baselines.py --algo all --episodes 10000

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
# SLURM Cluster Jobs
# =============================================================================

slurm: ## Submit full pipeline to SLURM cluster
	sbatch scripts/slurm_run_all.sh all

slurm-quick: ## Submit quick test to SLURM
	sbatch scripts/slurm_run_all.sh quick

slurm-deep: ## Submit deep training (5000 ep) to SLURM
	sbatch scripts/slurm_run_all.sh deep

slurm-status: ## Check SLURM job status
	@squeue -u $$USER --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "SLURM not available"

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
# Status & Summary
# =============================================================================

summary: ## Show experiment status and what hasn't been run
	@echo ""
	@echo "Cloud Autoscaling RL - Experiment Summary"
	@echo "=========================================="
	@echo ""
	@echo "📁 Artifacts Directory Status:"
	@echo "  Results: $$(ls -1 artifacts/results/*.json 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "  Plots:   $$(ls -1 artifacts/plots/*.png 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "  Logs:    $$(ls -1 artifacts/logs/*.log 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "  Models:  $$(ls -1 artifacts/models/*.pth 2>/dev/null | wc -l | tr -d ' ') files"
	@echo ""
	@echo "📊 Experiment Results:"
	@if ls artifacts/results/results_*.json 1>/dev/null 2>&1; then \
		echo "  ✅ Baseline experiments completed:"; \
		ls -1t artifacts/results/results_*.json | head -3 | while read f; do \
			echo "     - $$(basename $$f)"; \
		done; \
	else \
		echo "  ❌ No baseline experiments run yet"; \
		echo "     → Run: make experiments"; \
	fi
	@echo ""
	@echo "🔬 Ablation Studies:"
	@if ls artifacts/results/ablation_learning_rate_*.json 1>/dev/null 2>&1; then \
		echo "  Learning Rate:    ✅ Completed"; \
	else \
		echo "  Learning Rate:    ❌ Not run → make ablations-quick"; \
	fi
	@if ls artifacts/results/ablation_discount_factor_*.json 1>/dev/null 2>&1; then \
		echo "  Discount Factor:  ✅ Completed"; \
	else \
		echo "  Discount Factor:  ❌ Not run → make ablations"; \
	fi
	@if ls artifacts/results/ablation_epsilon_decay_*.json 1>/dev/null 2>&1; then \
		echo "  Epsilon Decay:    ✅ Completed"; \
	else \
		echo "  Epsilon Decay:    ❌ Not run → make ablations"; \
	fi
	@if ls artifacts/results/ablation_grid_*.json 1>/dev/null 2>&1; then \
		echo "  Grid Search:      ✅ Completed"; \
	else \
		echo "  Grid Search:      ❌ Not run → make ablations-grid"; \
	fi
	@echo ""
	@echo "📈 Recent Plots:"
	@if ls artifacts/plots/*.png 1>/dev/null 2>&1; then \
		ls -1t artifacts/plots/*.png | head -5 | while read f; do \
			echo "  - $$(basename $$f)"; \
		done; \
	else \
		echo "  No plots generated yet"; \
	fi
	@echo ""
	@echo "💡 Quick Start:"
	@echo "  make experiments-quick   Run quick baseline experiments (~2 min)"
	@echo "  make ablations-quick     Run quick ablation study (~1 min)"
	@echo "  make quick               Run entire quick pipeline (~5 min)"
	@echo ""
	@# Run detailed analysis if results exist
	@if ls artifacts/results/results_*.json 1>/dev/null 2>&1; then \
		uv run python scripts/show_summary.py; \
	fi

# =============================================================================
# Development
# =============================================================================

jupyter: ## Start Jupyter notebook
	uv run jupyter notebook notebooks/
