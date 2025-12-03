# Model Architectures and Metrics

This document describes the reinforcement learning algorithms implemented for cloud autoscaling, their architectures, and the metrics used to evaluate them.

## Table of Contents

- [Overview](#overview)
- [Environment](#environment)
- [Baseline Policies](#baseline-policies)
- [Tabular Methods](#tabular-methods)
- [Deep RL Methods](#deep-rl-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameters](#hyperparameters)

---

## Overview

Our autoscaling system uses reinforcement learning to decide when to scale cloud resources up or down. The agent observes the current system state and takes actions to balance performance (avoiding SLA violations) with cost (minimizing over-provisioning).

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Autoscaling Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐      Action       ┌──────────────────┐           │
│   │          │ ───────────────▶  │                  │           │
│   │  Agent   │                   │   Environment    │           │
│   │          │ ◀───────────────  │  (Cloud Sim)     │           │
│   └──────────┘   State, Reward   └──────────────────┘           │
│                                                                  │
│   Actions: {Scale Down, Hold, Scale Up}                         │
│   State: (utilization, capacity, trend, ...)                    │
│   Reward: f(SLA compliance, cost efficiency)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment

### State Space

The environment provides a continuous state representation:

| Feature | Description | Range |
|---------|-------------|-------|
| `utilization` | Current CPU/memory utilization | [0, 1] |
| `capacity` | Number of active instances | [1, max_capacity] |
| `demand_trend` | Recent demand trajectory | [-1, 1] |
| `time_features` | Time-of-day encoding | [0, 1] |

### Action Space

Discrete action space with 3 actions:

```
Action Space = {0: Scale Down, 1: Hold, 2: Scale Up}

┌─────────────────────────────────────────────────────────────┐
│                        Action Effects                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Scale Down (0)          Hold (1)           Scale Up (2)   │
│   ┌─────────┐            ┌─────────┐         ┌─────────┐    │
│   │ █ █ █   │            │ █ █ █ █ │         │ █ █ █ █ █│   │
│   │ █ █ █   │            │ █ █ █ █ │         │ █ █ █ █ █│   │
│   │   ↓     │            │    =    │         │     ↑    │   │
│   │ █ █     │            │ █ █ █ █ │         │ █ █ █ █ █│   │
│   └─────────┘            └─────────┘         │ █ █ █ █ █│   │
│   -1 instance            No change           └─────────┘    │
│                                              +1 instance    │
└─────────────────────────────────────────────────────────────┘
```

### Reward Function

The reward balances SLA compliance with cost efficiency:

```
reward = -α × SLA_violations - β × over_provisioning_cost + γ × efficiency_bonus

Where:
  • α = penalty weight for SLA violations (high utilization)
  • β = penalty weight for over-provisioning (wasted resources)
  • γ = bonus for optimal utilization range
```

---

## Baseline Policies

### Random Policy

Selects actions uniformly at random. Used as a lower bound for comparison.

```
┌────────────────────────────────────┐
│         Random Policy              │
├────────────────────────────────────┤
│                                    │
│   P(Scale Down) = 1/3              │
│   P(Hold)       = 1/3              │
│   P(Scale Up)   = 1/3              │
│                                    │
│   No learning, pure exploration    │
│                                    │
└────────────────────────────────────┘
```

### Threshold Policy

Traditional rule-based autoscaling:

```
┌────────────────────────────────────────────────────────────┐
│                    Threshold Policy                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   if utilization > HIGH_THRESHOLD (e.g., 0.8):             │
│       action = Scale Up                                     │
│   elif utilization < LOW_THRESHOLD (e.g., 0.3):            │
│       action = Scale Down                                   │
│   else:                                                     │
│       action = Hold                                         │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  0%        30%              80%              100%   │  │
│   │  ├──────────┼───────────────┼──────────────────┤    │  │
│   │  │Scale Down│     Hold      │    Scale Up     │    │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Tabular Methods

### Q-Learning

Off-policy temporal difference learning with ε-greedy exploration.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Q-Learning                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Q-Table Structure:                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  State    │ Scale Down │   Hold   │  Scale Up  │        │   │
│   ├───────────┼────────────┼──────────┼────────────┤        │   │
│   │  s₁       │   -2.3     │   1.5    │    0.8     │        │   │
│   │  s₂       │    0.5     │   2.1    │   -1.2     │        │   │
│   │  s₃       │   -0.8     │   0.3    │    3.2     │        │   │
│   │  ...      │    ...     │   ...    │    ...     │        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Update Rule:                                                   │
│   Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]          │
│                                                                  │
│   Where:                                                         │
│   • α = learning rate                                            │
│   • γ = discount factor                                          │
│   • max_a' Q(s',a') = best future value (off-policy)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### SARSA

On-policy temporal difference learning.

```
┌─────────────────────────────────────────────────────────────────┐
│                          SARSA                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Update Rule:                                                   │
│   Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]                 │
│                                                                  │
│   Key Difference from Q-Learning:                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Q-Learning:  Uses max_a' Q(s',a') — greedy next action│   │
│   │   SARSA:       Uses Q(s',a') — actual next action taken │   │
│   │                                                          │   │
│   │   SARSA follows the policy it's learning (on-policy)    │   │
│   │   More conservative, accounts for exploration           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   SARSA = State-Action-Reward-State-Action                      │
│   (s, a, r, s', a')                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deep RL Methods

### DQN (Deep Q-Network)

Uses a neural network to approximate the Q-function for continuous state spaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                      DQN Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   State Input                                                    │
│   [util, capacity, trend, ...]                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │  Dense Layer    │  128 units, ReLU                          │
│   │  (FC1)          │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Dense Layer    │  128 units, ReLU                          │
│   │  (FC2)          │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Output Layer   │  3 units (Q-values for each action)       │
│   │  (FC3)          │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   Q(s, Scale Down), Q(s, Hold), Q(s, Scale Up)                  │
│                                                                  │
│   Key Innovations:                                               │
│   • Experience Replay Buffer                                     │
│   • Target Network (updated every N steps)                       │
│   • ε-greedy exploration with decay                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Experience Replay:
┌─────────────────────────────────────────────────────────────────┐
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐      │
│   │ t-7 │ t-6 │ t-5 │ t-4 │ t-3 │ t-2 │ t-1 │  t  │     │      │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘      │
│          │           │                 │                         │
│          └───────────┼─────────────────┘                         │
│                      ▼                                           │
│              Random Sampling                                     │
│              (breaks correlation)                                │
└─────────────────────────────────────────────────────────────────┘
```

### Double DQN

Addresses overestimation bias in standard DQN by decoupling action selection from evaluation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Double DQN Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌──────────────┐                             │
│          ┌────────▶│ Online Net   │────────┐                    │
│          │         │   (θ)        │        │                    │
│          │         └──────────────┘        │                    │
│   State ─┤                                 ├──▶ Update θ        │
│          │         ┌──────────────┐        │                    │
│          └────────▶│ Target Net   │────────┘                    │
│                    │   (θ⁻)       │                             │
│                    └──────────────┘                             │
│                                                                  │
│   Standard DQN:                                                  │
│   y = r + γ · max_a' Q(s', a'; θ⁻)                             │
│                ↑ Same network selects AND evaluates             │
│                                                                  │
│   Double DQN:                                                    │
│   a* = argmax_a' Q(s', a'; θ)    ← Online net selects          │
│   y = r + γ · Q(s', a*; θ⁻)      ← Target net evaluates        │
│                                                                  │
│   Benefit: Reduces overestimation of Q-values                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dueling DQN

Separates state value from action advantages for better learning.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Dueling DQN Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   State Input                                                    │
│   [util, capacity, trend, ...]                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │  Shared Layers  │  Feature extraction                       │
│   │  (FC1, FC2)     │  128 → 128 units                          │
│   └────────┬────────┘                                           │
│            │                                                     │
│      ┌─────┴─────┐                                              │
│      │           │                                               │
│      ▼           ▼                                               │
│   ┌──────┐   ┌────────┐                                         │
│   │Value │   │Advantage│                                        │
│   │Stream│   │ Stream  │                                        │
│   │ V(s) │   │ A(s,a)  │                                        │
│   │      │   │         │                                        │
│   │ [1]  │   │  [3]    │                                        │
│   └───┬──┘   └────┬───┘                                         │
│       │           │                                              │
│       └─────┬─────┘                                              │
│             │                                                    │
│             ▼                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))               │   │
│   └─────────────────────────────────────────────────────────┘   │
│             │                                                    │
│             ▼                                                    │
│   Q(s, Scale Down), Q(s, Hold), Q(s, Scale Up)                  │
│                                                                  │
│   Why This Works:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • V(s): "How good is this state?"                      │   │
│   │  • A(s,a): "How much better is action a than average?"  │   │
│   │  • Learns state value even when action doesn't matter   │   │
│   │  • Better credit assignment in sparse reward settings   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### REINFORCE (Policy Gradient)

Unlike value-based methods (DQN), REINFORCE learns a policy directly.

```
┌─────────────────────────────────────────────────────────────────┐
│                   REINFORCE Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Key Difference from DQN:                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  DQN:       Learns Q(s,a) → derives policy from values  │   │
│   │  REINFORCE: Learns π(a|s) → policy directly             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   State Input                                                    │
│   [util, capacity, trend, ...]                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │  Dense Layer    │  128 units, ReLU                          │
│   │  (FC1)          │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Dense Layer    │  128 units, ReLU                          │
│   │  (FC2)          │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Output Layer   │  3 units + Softmax                        │
│   │  (Probabilities)│                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   P(Scale Down), P(Hold), P(Scale Up)  ← sum to 1.0            │
│                                                                  │
│   Training:                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. Run full episode, collect (s, a, r) trajectory     │   │
│   │  2. Compute discounted returns: G_t = Σ γ^k r_{t+k}     │   │
│   │  3. Policy gradient: ∇J = Σ ∇log π(a|s) · G_t          │   │
│   │  4. Update: θ ← θ + α · ∇J                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Variance Reduction (Baseline):                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Instead of G_t, use advantage: A_t = G_t - V(s)        │   │
│   │  This reduces variance while keeping gradient unbiased  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Algorithm Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Algorithm Comparison                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Algorithm      │ State Space │ Off/On-Policy │ Key Feature            │
│   ───────────────┼─────────────┼───────────────┼────────────────────────│
│   Q-Learning     │ Discrete    │ Off-policy    │ Simple, tabular        │
│   SARSA          │ Discrete    │ On-policy     │ Conservative updates   │
│   DQN            │ Continuous  │ Off-policy    │ Experience replay      │
│   Double DQN     │ Continuous  │ Off-policy    │ Reduced overestimation │
│   Dueling DQN    │ Continuous  │ Off-policy    │ Value/advantage split  │
│   REINFORCE      │ Continuous  │ On-policy     │ Policy gradient        │
│                                                                          │
│   Method Categories:                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  VALUE-BASED          vs          POLICY-BASED                  │   │
│   │  ─────────────────────────────────────────────────────────────  │   │
│   │  Q-Learning, SARSA                REINFORCE                     │   │
│   │  DQN, Double DQN, Dueling DQN                                   │   │
│   │                                                                  │   │
│   │  Learn Q(s,a) → derive policy     Learn π(a|s) directly        │   │
│   │  Deterministic output             Stochastic output             │   │
│   │  ε-greedy exploration             Natural exploration           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Complexity Hierarchy:                                                  │
│                                                                          │
│   Simple ◀────────────────────────────────────────────────▶ Complex     │
│                                                                          │
│   Random → Threshold → Q-Learning → SARSA → DQN → REINFORCE → Dueling  │
│     │         │            │          │       │        │         │      │
│   No        Rule-       Tabular    Tabular  Neural  Policy   Separate  │
│   Learning  Based       Q-table    Q-table  Net     Gradient Streams   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics

### Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Mean Reward** | Σr / n | Average reward per episode |
| **Cumulative Reward** | Σr | Total reward over training |
| **SLA Violation Rate** | violations / total_steps | % of steps with high utilization |
| **Over-provisioning Rate** | underutilized / total_steps | % of steps with low utilization |

### Learning Metrics

| Metric | What It Measures | Evidence of Learning |
|--------|------------------|---------------------|
| **Reward Improvement** | (final - initial) / initial | > 50% improvement |
| **Variance Reduction** | (var_early - var_final) / var_early | > 50% reduction |
| **Trend Slope** | Linear regression slope | Positive slope |
| **Convergence Time** | Episodes to stable policy | Faster = better |

### Performance Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│              Metrics Visualization                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Reward Improvement:                                            │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ DQN        ████████████████████████████████████░░ +99.2%  │ │
│   │ Double DQN ██████████████████████████████████████░ +101.1%│ │
│   │ Dueling    ████████████████████████████████████████ +104% │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│   Variance Reduction (Stability):                                │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ DQN        ████████████████████████████████░░░░░░░ -81.2% │ │
│   │ Double DQN ██████████████████████████████████████░ -85.7% │ │
│   │ Dueling    ████████████████████████████████░░░░░░░ -81.1% │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│   Training Trend (slope per episode):                            │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ DQN        ▁▂▃▄▅▆▆▇▇█  +5.6/ep                            │ │
│   │ Double DQN ▁▂▃▄▅▆▇▇██  +5.9/ep                            │ │
│   │ Dueling    ▁▂▃▄▅▆▆▇▇█  +5.7/ep                            │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

### Deep RL Hyperparameters

| Parameter | DQN | Double DQN | Dueling DQN |
|-----------|-----|------------|-------------|
| Learning Rate | 1e-3 | 1e-3 | 1e-3 |
| Discount Factor (γ) | 0.99 | 0.99 | 0.99 |
| Epsilon Start | 1.0 | 1.0 | 1.0 |
| Epsilon End | 0.01 | 0.01 | 0.01 |
| Epsilon Decay | 0.995 | 0.995 | 0.995 |
| Batch Size | 64 | 64 | 64 |
| Replay Buffer Size | 10,000 | 10,000 | 10,000 |
| Target Update Freq | 100 | 100 | 100 |
| Hidden Layers | [128, 128] | [128, 128] | [128, 128] |

### Policy Gradient Hyperparameters

| Parameter | REINFORCE |
|-----------|-----------|
| Learning Rate | 1e-3 |
| Discount Factor (γ) | 0.99 |
| Use Baseline | Yes |
| Hidden Layers | [128, 128] |
| Gradient Clipping | 1.0 |
| Update Frequency | End of episode (Monte Carlo) |

### Tabular Hyperparameters

| Parameter | Q-Learning | SARSA |
|-----------|------------|-------|
| Learning Rate (α) | 0.1 | 0.1 |
| Discount Factor (γ) | 0.99 | 0.99 |
| Epsilon Start | 1.0 | 1.0 |
| Epsilon End | 0.01 | 0.01 |
| Epsilon Decay | 0.995 | 0.995 |

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Initialize                                                          │
│   ┌─────────────────┐                                                   │
│   │ • Create env    │                                                   │
│   │ • Init agent    │                                                   │
│   │ • Set ε = 1.0   │                                                   │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   2. Episode Loop (2000 episodes)                                       │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  ┌─────────────────────────────────────────────────────────┐    │  │
│   │  │  Step Loop (until done)                                  │    │  │
│   │  │  ┌─────────────────────────────────────────────────────┐│    │  │
│   │  │  │ a) Observe state s                                  ││    │  │
│   │  │  │ b) Select action a (ε-greedy)                       ││    │  │
│   │  │  │ c) Execute a, get r, s'                             ││    │  │
│   │  │  │ d) Store (s, a, r, s', done) in replay buffer       ││    │  │
│   │  │  │ e) Sample batch, compute loss, update network       ││    │  │
│   │  │  │ f) Every N steps: update target network             ││    │  │
│   │  │  └─────────────────────────────────────────────────────┘│    │  │
│   │  └─────────────────────────────────────────────────────────┘    │  │
│   │                                                                  │  │
│   │  • Decay epsilon                                                 │  │
│   │  • Log episode reward                                            │  │
│   │  • Save checkpoint if best                                       │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│            │                                                             │
│            ▼                                                             │
│   3. Evaluation                                                          │
│   ┌─────────────────┐                                                   │
│   │ • Generate plots│                                                   │
│   │ • Save results  │                                                   │
│   │ • Export model  │                                                   │
│   └─────────────────┘                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*
2. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*
3. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*
4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
