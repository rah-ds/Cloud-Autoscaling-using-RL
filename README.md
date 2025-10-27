# Objective

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python 3.12](https://img.shields.io/badge/Python-3.12-black?logo=python&logoColor=blue)

This project aims to solve for when to scale and will explore whether reinforcement learning can make smarter decisions about cloud resource auto-scaling than today’s simple threshold rules.

We aim to build a small simulator where cloud workloads vary over time and then have an RL agent decide when to add or remove capacity. The goal is to see if Reinforcement methods like SARSA and Q-learning can keep performance high while avoiding unnecessary cost.

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

## Important Links

* The [class Canvas](https://canvas.its.virginia.edu/courses/159418/modules)
* The class repo can be found [here](https://github.com/UVADS/reinforcement_learning_online_msds/commits/main/)
* TODO: add in Rivanna start up info (could be needed for long running experiments)

## Authors

* Balasubramanyam, Srivatsa
* Healy, Ryan
* McGregor, Bruce

## Getting Started with uv


```bash
# brew install uv
uv venv --python=3.12
uv sync
```


TODO: add in makefile
TODO: add in pre commit hooks
TODO: add in branch protection rules

## TODO:

try and make the env an open gym function?