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

test: ## Run tests (placeholder for when tests are added)
	@echo "No tests configured yet"

clean: ## Remove virtual environment and cache files
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

run-experiments: ## Run main experiments
	uv run python agent/main_experiments.py

jupyter: ## Start Jupyter notebook server
	uv run jupyter notebook

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files
