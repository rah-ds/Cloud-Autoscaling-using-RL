# Plan

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
