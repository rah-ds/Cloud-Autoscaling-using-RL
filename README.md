# Cloud Autoscaling using Reinforcement Learning

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python 3.12](https://img.shields.io/badge/Python-3.12-black?logo=python&logoColor=blue)

## Objective

This project aims to solve for when to scale and will explore whether reinforcement learning can make smarter decisions about cloud resource auto-scaling than today's simple threshold rules.

We aim to build a small simulator where cloud workloads vary over time and then have an RL agent decide when to add or remove capacity. The goal is to see if reinforcement learning methods like SARSA and Q-learning can keep performance high while avoiding unnecessary cost.

### Expected Impact: RL vs. Traditional Methods

Traditional auto-scaling approaches rely on simple threshold-based rules (e.g., "add a server when CPU > 80%"). While straightforward, these methods often lead to:
- **Reactive scaling**: Resources are added only after thresholds are breached, causing performance degradation
- **Over-provisioning**: Conservative thresholds result in wasted resources and higher costs
- **Poor adaptation**: Static rules don't learn from workload patterns or adjust to changing conditions

In contrast, reinforcement learning approaches offer several potential advantages:
- **Proactive scaling**: RL agents can learn to anticipate demand patterns and scale preemptively
- **Adaptive policies**: Agents continuously learn and optimize based on observed rewards (balancing SLA compliance and cost)
- **Better cost-performance trade-offs**: RL can find nuanced policies that traditional rules cannot express
- **Generalization**: Trained agents may adapt to new workload patterns without manual rule tuning

This project investigates whether these theoretical benefits translate to measurable improvements in simulated cloud environments.

## Prerequisites

Before getting started with this project, ensure you have the following:

### Required
- **Python 3.12 or higher**: The project is built and tested with Python 3.12
- **uv package manager**: Fast Python package installer and resolver ([installation instructions](#getting-started-with-uv))
- **Git**: For version control and repository management

### Optional
- **Make**: For using the provided Makefile commands (available by default on macOS/Linux)
- **Jupyter**: For running interactive notebooks (installed automatically with dependencies)
- **Homebrew** (macOS): Simplifies installation of uv and other tools

### System Requirements
- Operating System: macOS, Linux, or Windows (with WSL recommended)
- RAM: At least 4GB recommended for running experiments
- Disk Space: ~500MB for dependencies and datasets

## Plan

We plan to explore the use of both simulated and real-world datasets to
drive the cloud auto-scaling environment. This approach lets us
prototype quickly with lightweight data while leaving open the
possibility of testing against more realistic traces.

## Tentative Schedule

| Task Description                                                                 | Target Date   |
|----------------------------------------------------------------------------------|---------------|
| Begin exploring the Kaggle dataset, normalize CPU utilization, and experiment with simple demand traces | October 11    |
| Build an initial version of the simulator (states, actions, rewards) and test different ways of including the trend feature | October 18    |
| Try out simple baseline policies; compare how well they track demand; refine reward design if needed | October 25    |
| Start implementing RL agents (SARSA, Q-learning); experiment with different exploration rates and episode lengths | November 1    |
| Run initial experiments with RL policies; evaluate early results and adjust simulator design or state representation as needed | November 8    |
| Explore feasibility of incorporating one of the real-world traces from the GitHub dataset collection; test integration if time permits | November 15   |
| Continue refining experiments, focusing on SLA vs. cost trade-offs and the effect of the trend feature | November 22   |
| Consolidate results, generate plots and visualizations, and begin drafting the report | November 29   |
| Finalize report and prepare presentation                                          | Dec 6 - 9     |

## Data

* [Kaggle - Cloud Computing Performance
  Metrics](https://www.kaggle.com/datasets/abdurraziq01/cloud-computing-performance-metrics)

  * Simulated CPU utilization and other system metrics

  * Normalized CPU values will provide workload demand traces

  * Used to build utilization buckets and compute trend features

  * Lightweight, easy to use for prototyping and debugging

<!-- -->

* [GitHub -- Awesome Cloud Computing
  Datasets](https://github.com/ACAT-SCUT/Awesome-CloudComputing-Datasets)

  * Curated list of large-scale, real-world traces

  * Includes Google Cluster Data, Alibaba Cluster Traces, and others

  * Candidate for adding realistic workload patterns

  * May be used to test how well the RL agent generalizes beyond
    synthetic data

## Getting Started with uv

### Quick Start

The fastest way to get started is using the provided Makefile:

```bash
# Install uv and set up the environment
make setup

# Activate the virtual environment
source .venv/bin/activate
```

### Manual Setup

If you prefer to set up manually or don't have Make installed:

```bash
# Install uv (macOS/Linux with Homebrew)
brew install uv

# Or install with curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.12
uv venv --python=3.12

# Sync dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### Available Make Commands

The project includes a Makefile with convenient commands for common tasks:

```bash
make help              # Show all available commands
make setup             # Install uv and set up environment
make sync              # Sync dependencies
make format            # Format code with ruff
make lint              # Lint code with ruff
make lint-fix          # Lint and auto-fix issues
make clean             # Remove virtual environment and cache files
make run-experiments   # Run main experiments
make jupyter           # Start Jupyter notebook server
make pre-commit-install # Install pre-commit hooks
make pre-commit-run    # Run pre-commit hooks on all files
```

## Development Workflow

### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality. The hooks automatically:
- Remove trailing whitespace
- Ensure files end with a newline
- Validate YAML and TOML files
- Check for merge conflicts
- Run Ruff for linting and formatting

To set up pre-commit hooks:

```bash
# Install pre-commit hooks
make pre-commit-install

# Or manually
uv run pre-commit install
```

Once installed, the hooks will run automatically on `git commit`. You can also run them manually:

```bash
# Run on all files
make pre-commit-run

# Or manually
uv run pre-commit run --all-files
```

### Branch Protection Rules

To maintain code quality and collaboration standards, we recommend the following branch protection rules for the `main` branch:

#### Required Settings:
- **Require pull request reviews before merging**: At least 1 approval required
- **Require status checks to pass before merging**: 
  - Ruff linting checks
  - Pre-commit hook validation
- **Require branches to be up to date before merging**: Ensures changes are tested against latest code
- **Require conversation resolution before merging**: All review comments must be addressed

#### Recommended Settings:
- **Dismiss stale pull request approvals when new commits are pushed**: Ensures reviews reflect current state
- **Restrict who can push to matching branches**: Limit to maintainers only
- **Allow force pushes**: Disabled
- **Allow deletions**: Disabled

These rules help ensure:
- Code is reviewed before integration
- Automated checks pass consistently
- The main branch remains stable
- All team members follow consistent practices

## Important Links

* The [class Canvas](https://canvas.its.virginia.edu/courses/159418/modules)
* The class repo can be found [here](https://github.com/UVADS/reinforcement_learning_online_msds/commits/main/)
* **Rivanna HPC Resources** (for long-running experiments):
  * [Rivanna User Guide](https://www.rc.virginia.edu/userinfo/rivanna/overview/)
  * [Getting Started with Rivanna](https://www.rc.virginia.edu/userinfo/rivanna/login/)
  * [Job Submission on Rivanna](https://www.rc.virginia.edu/userinfo/rivanna/slurm/)
  * [Python on Rivanna](https://www.rc.virginia.edu/userinfo/rivanna/software/python/)
  * To run experiments on Rivanna:
    1. SSH into Rivanna: `ssh <your_id>@rivanna.hpc.virginia.edu`
    2. Load Python module: `module load anaconda`
    3. Create a SLURM job script for long-running experiments
    4. Submit with: `sbatch your_script.sh`
    5. Monitor with: `squeue -u <your_id>`

## Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository** and create a new branch for your feature or bugfix
2. **Make your changes** following the code style guidelines (see below)
3. **Test your changes** to ensure nothing breaks
4. **Run pre-commit hooks** to validate code quality
5. **Submit a pull request** with a clear description of your changes

### Code Style Guidelines

- Follow PEP 8 style guidelines for Python code
- Use Ruff for linting and formatting (runs automatically via pre-commit)
- Write clear, descriptive commit messages
- Add comments for complex logic
- Update documentation when adding new features

### Reporting Issues

Found a bug or have a feature request? Please open an issue on GitHub with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment details (OS, Python version, etc.)

### Development Setup

1. Follow the [Getting Started](#getting-started-with-uv) guide
2. Install pre-commit hooks: `make pre-commit-install`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make changes and commit with descriptive messages
5. Run `make lint` and `make format` before submitting

## Contributors

### Project Team

* **Balasubramanyam, Srivatsa** - Core contributor, RL agent implementation and experiments
* **Healy, Ryan** - Core contributor, simulator development and data processing
* **McGregor, Bruce** - Core contributor, baseline policies and evaluation metrics

### Contributing

This project is part of a graduate-level reinforcement learning course. We appreciate any feedback, suggestions, or contributions from the community. Please see the [Contributing](#contributing) section above for guidelines.

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

* University of Virginia - Master of Science in Data Science program
* Course instructors and TAs for guidance and support
* Kaggle and GitHub dataset contributors for providing valuable data resources
