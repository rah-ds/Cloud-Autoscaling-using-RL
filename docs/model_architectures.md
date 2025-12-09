# Model Architectures & Evaluation Metrics

This document provides detailed descriptions of all algorithms implemented in this project, their neural network architectures, and the evaluation metrics used.

## Algorithms Overview

We implement **7 algorithms** across **4 categories**:

| Category | Algorithm | Type | Key Characteristics |
|----------|-----------|------|---------------------|
| Baseline | Random | Non-learning | Uniform random action selection |
| Baseline | Threshold | Rule-based | Scale up >80%, scale down <40% |
| Tabular RL | SARSA | On-policy | Conservative, learns from actual actions |
| Tabular RL | Q-Learning | Off-policy | Learns optimal policy, more aggressive |
| Deep RL | DQN | Value-based | Neural network Q-function approximation |
| Deep RL | Double DQN | Value-based | Reduces overestimation bias |
| Deep RL | Dueling DQN | Value-based | Separates state value and action advantage |
| Policy Gradient | REINFORCE | Policy-based | Direct policy optimization with baseline |

## Neural Network Architectures

### DQN / Double DQN
```
Input (state_dim=3) → FC(128) → ReLU → FC(128) → ReLU → FC(action_dim=3)
```
- **Parameters**: ~17,000
- **Output**: Q-values for each action

### Dueling DQN
```
Input (state_dim=3) → FC(128) → ReLU → FC(128) → ReLU
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                           ↓
            Value Stream                                Advantage Stream
            FC(128) → V(s)                              FC(128) → A(s,a)
                    ↓                                           ↓
                    └─────────────── Q(s,a) = V(s) + (A(s,a) - mean(A)) ──────┘
```
- **Parameters**: ~25,000
- **Output**: Q-values via value-advantage decomposition

### REINFORCE (Policy Network)
```
Input (state_dim=3) → FC(128) → ReLU → FC(128) → ReLU → FC(action_dim=3) → Softmax
```
- **Parameters**: ~17,000
- **Output**: Action probabilities π(a|s)

## Hyperparameters

### Tabular Methods (SARSA, Q-Learning)
| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.99 |
| Initial ε | 1.0 |
| Final ε | 0.01 |
| ε decay | 0.995 |
| Training episodes | 1,000 |

### Deep RL Methods (DQN, Double DQN, Dueling DQN, REINFORCE)
| Parameter | Value |
|-----------|-------|
| Learning rate | 5×10⁻⁴ |
| Discount factor (γ) | 0.99 |
| Initial ε | 1.0 |
| Final ε | 0.05 |
| ε decay | 0.999 |
| Batch size | 64 |
| Replay buffer size | 100,000 |
| Target update (τ) | 10⁻³ |
| Hidden layers | [128, 128] |
| Training episodes | 1,000 |

## State Space

Each state is a 3-tuple:
```
s_t = (utilization, capacity, trend)
```

| Component | Values | Description |
|-----------|--------|-------------|
| Utilization | {0, 1, 2} | Low (<40%), Medium (40-80%), High (>80%) |
| Capacity | {1, ..., C_max} | Current number of active capacity units |
| Trend | {-1, 0, +1} | Falling, Flat, Rising demand |

**Total state space size**: 3 × 5 × 3 = **45 states**

## Action Space

| Action | Value | Description |
|--------|-------|-------------|
| Scale Down | -1 | Remove one capacity unit |
| Hold | 0 | Maintain current capacity |
| Scale Up | +1 | Add one capacity unit |

## Reward Function

The multi-component reward function balances SLA compliance, cost, and stability:

```
r_t = r_opt + r_eff + r_SLA + r_waste + r_cost + r_churn
```

| Component | Value | Condition |
|-----------|-------|-----------|
| Optimal utilization | +10 | 40% ≤ u ≤ 80% |
| Efficiency bonus | +5 | 60% ≤ u ≤ 70% |
| SLA penalty | -50 × (1 + (u - 0.9)) | u ≥ 90% |
| Waste penalty | -5 × (0.2 - u) | u < 20% |
| Cost penalty | -0.5 × C | Always |
| Churn penalty | -2 | Action ≠ hold |

## Evaluation Metrics

### Primary Metrics
- **Mean Reward**: Average cumulative reward over final 100 episodes
- **SLA Violations**: Count of timesteps where utilization ≥ 90%
- **Improvement**: Percentage improvement over random baseline

### Convergence Metrics
- **Convergence Speed**: Episodes to reach 90% of final performance
- **Stability**: Variance in rewards during final 100 episodes
- **Learning Trend**: Slope of smoothed reward curve

## Key Results

| Algorithm | Mean Reward | SLA Violations | Improvement |
|-----------|-------------|----------------|-------------|
| **REINFORCE** | -16,938 | **329** | +77.1% |
| SARSA | -16,893 | 337 | +77.2% |
| Q-Learning | -17,215 | 341 | +76.8% |
| Threshold | -18,158 | 365 | +75.5% |
| Double DQN | -21,596 | 397 | +70.8% |
| DQN | -21,563 | 400 | +70.9% |
| Dueling DQN | -27,586 | 474 | +62.8% |
| Random | -74,095 | 676 | — |

**Key Finding**: REINFORCE achieves the fewest SLA violations (329), making it the best choice when service reliability is paramount. Tabular methods achieve the best cumulative rewards while remaining interpretable.
