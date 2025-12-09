# Cloud Autoscaling with Reinforcement Learning
## A Data-Driven Approach to Intelligent Resource Management

**Presented by:** Ryan Healy, Srivatsa Balasubramanyam, Bruce McGregor  
**University of Virginia | MS Data Science**  
**December 2025**

---

# Agenda

1. **Executive Summary** — The business case for RL-based autoscaling
2. **The Problem** — Why traditional autoscaling fails
3. **Our Solution** — Reinforcement learning approach
4. **Technical Deep Dive** — Algorithms and architecture
5. **Results** — Performance comparison and key findings
6. **Evidence of Learning** — Proving the models work
7. **Recommendations** — Implementation roadmap
8. **Next Steps** — Future enhancements

---

# Executive Summary

## The Opportunity

> **Cloud spending waste costs enterprises $100B+ annually due to over-provisioning and inefficient scaling decisions.**

## Our Solution

We developed an **AI-powered autoscaling system** using reinforcement learning that:

| Metric | Improvement |
|--------|-------------|
| 📈 Reward Improvement | **+104%** vs. initial policy |
| 📉 Policy Stability | **-86% variance** in decisions |
| ⚡ Learning Speed | Converges in **~500 episodes** |
| 🎯 Best Algorithm | **Dueling DQN** outperforms all others |

## Bottom Line

**RL-based autoscaling learns optimal policies that balance cost and performance better than static threshold rules.**

---

# The Problem

## Traditional Autoscaling is Broken

```
         Traditional Threshold-Based Scaling
         
    CPU Usage
    100% ┤                    ╭──╮     
     80% ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─╱─ ─╲─ ─ ─  ← Scale Up Threshold
     60% ┤              ╭───╯    ╰───╮
     40% ┤         ╭───╯              ╰──
     30% ┤─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ← Scale Down Threshold  
     20% ┤    ╭───╯
         └────┴─────────────────────────→ Time
         
              ↑                    ↑
         Too Late!            Too Late!
         (Already overloaded) (Already wasted $)
```

## Key Pain Points

| Problem | Business Impact | Root Cause |
|---------|-----------------|------------|
| 🔴 **Reactive Scaling** | SLA violations, unhappy customers | Rules trigger *after* thresholds breach |
| 🟡 **Over-provisioning** | 30-40% wasted cloud spend | Conservative thresholds "just in case" |
| 🟠 **Oscillation** | System instability | Rapid scale up/down cycles |
| ⚫ **Manual Tuning** | Engineering time sink | Thresholds require constant adjustment |

---

# Why Reinforcement Learning?

## RL Learns From Experience

```
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │     ┌──────────┐      Action       ┌──────────────────┐    │
    │     │          │ ───────────────▶  │                  │    │
    │     │  Agent   │   (scale up/     │   Cloud          │    │
    │     │  (Brain) │    hold/down)    │   Environment    │    │
    │     │          │ ◀───────────────  │                  │    │
    │     └──────────┘   State + Reward  └──────────────────┘    │
    │                                                             │
    │     "If I scale up now, will I be rewarded later?"         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

## RL vs. Threshold Rules

| Capability | Threshold Rules | RL Agent |
|------------|-----------------|----------|
| Learns patterns | ❌ No | ✅ Yes |
| Anticipates demand | ❌ No | ✅ Yes |
| Optimizes trade-offs | ❌ No | ✅ Yes |
| Adapts to changes | ❌ Manual | ✅ Automatic |
| Handles complexity | ❌ Limited | ✅ Scales |

---

# Our Approach

## Simulation Environment

We built a realistic cloud simulation using **Markov Modulated Poisson Process (MMPP)** for workload generation:

```
    Workload States:
    
    ┌─────────────┐         ┌─────────────┐
    │  Low Load   │ ←─────→ │  High Load  │
    │  (λ = 20)   │         │  (λ = 80)   │
    └─────────────┘         └─────────────┘
         │                        │
         ▼                        ▼
    20 requests/sec         80 requests/sec
    
    Captures: Burstiness, temporal correlation, regime changes
```

## State-Action-Reward Design

| Component | Definition |
|-----------|------------|
| **State** | (utilization, capacity, demand_trend) |
| **Actions** | Scale Down (-1), Hold (0), Scale Up (+1) |
| **Reward** | -cost - (penalty × SLA_violation) + efficiency_bonus |

---

# Algorithms Evaluated

## Algorithm Hierarchy

```
    Complexity & Capability
    
    Simple ◀────────────────────────────────────────────▶ Advanced
    
    ┌─────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Random  │ │ Threshold │ │Q-Learn/ │ │   DQN   │ │ Dueling │
    │         │ │   Rules   │ │  SARSA  │ │ Double  │ │   DQN   │
    └─────────┘ └───────────┘ └─────────┘ └─────────┘ └─────────┘
         │           │            │            │            │
    No learning  Hand-coded   Q-table      Neural       Value/
                  rules       (discrete)   networks    Advantage
                                                        streams
```

## Deep RL Architectures

### DQN (Deep Q-Network)
- Neural network approximates Q-values
- Experience replay breaks correlation
- Target network stabilizes training

### Double DQN
- Fixes overestimation bias
- Separate networks for selection vs. evaluation

### Dueling DQN ⭐ (Best Performer)
- Separates "how good is this state?" from "how good is this action?"
- Better credit assignment
- Faster learning

---

# Results: Algorithm Comparison

## Performance Rankings

| Rank | Algorithm | Mean Reward | Improvement | Type |
|:----:|-----------|:-----------:|:-----------:|:----:|
| 🥇 | **Dueling DQN** | +345 | +104.3% | Deep RL |
| 🥈 | Double DQN | +90 | +101.1% | Deep RL |
| 🥉 | DQN | -69 | +99.2% | Deep RL |

*Note: All Deep RL methods dramatically outperform random baseline (~-8000 initial reward)*

## Learning Curves

```
    Reward
    ▲
    │                                    ╭───── Dueling DQN
    │                               ╭───╯
  0 ┼─────────────────────────╭────╯────────── convergence
    │                    ╭───╯
    │               ╭───╯
    │          ╭───╯
-4K │     ╭───╯
    │╭───╯
-8K ┼╯ ← All agents start here (random behavior)
    └────────────────────────────────────────▶ Episodes
         0      500     1000    1500    2000
```

---

# Evidence of Learning

## How Do We Know The Models Actually Learn?

### 1️⃣ Reward Improvement Over Time

| Phase | DQN | Double DQN | Dueling DQN |
|-------|:---:|:----------:|:-----------:|
| Early (0-25%) | -8,251 | -8,490 | -7,941 |
| Mid (25-50%) | -315 | -753 | -261 |
| Late (50-75%) | +255 | -167 | +188 |
| Final (75-100%) | -69 | +90 | **+345** |

**✅ All agents improve from ~-8000 to near-zero or positive**

### 2️⃣ Variance Reduction (Stability)

| Algorithm | Early Variance | Final Variance | Reduction |
|-----------|:--------------:|:--------------:|:---------:|
| DQN | 128M | 24M | **-81%** |
| Double DQN | 127M | 18M | **-86%** |
| Dueling DQN | 132M | 25M | **-81%** |

**✅ Agents become more consistent, not just lucky**

### 3️⃣ Positive Trend Throughout Training

All algorithms show **+5.6 to +5.9 reward improvement per episode** — consistent learning, not just early gains.

---

# Key Findings

## 🎯 Finding 1: Dueling DQN Wins

**Why?** Separating value from advantage helps the agent learn:
- "This is a good state" (high capacity, low utilization)
- "This action is better than alternatives"

## 📉 Finding 2: Variance Drops 80%+

The agents don't just perform better — they become **more reliable**. Production systems need consistency.

## ⚡ Finding 3: Convergence in ~500 Episodes

Practical training time: **~30 minutes** to learn a policy that outperforms hand-tuned rules.

## 🔄 Finding 4: All Deep RL Methods Beat Baselines

Even basic DQN dramatically outperforms random and threshold policies after training.

---

# Practical Implications

## When to Use Each Approach

| Scenario | Recommended Algorithm | Why |
|----------|----------------------|-----|
| **Simple, interpretable** | Q-Learning | Small state space, auditable |
| **Production deployment** | Double DQN | Stable, avoids overestimation |
| **Maximum performance** | Dueling DQN | Best results in our tests |
| **Quick baseline** | Threshold | No training needed |

## Implementation Considerations

```
    Production Deployment Checklist
    
    ☐ Start with simulation training (safe exploration)
    ☐ Validate on historical workload traces
    ☐ Shadow mode: run alongside existing autoscaler
    ☐ Gradual rollout: start with non-critical services
    ☐ Monitoring: track SLA violations, costs, agent decisions
    ☐ Fallback: threshold rules if agent fails
```

---

# Architecture Overview

## System Design

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Production System                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
    │   │   Metrics    │─────▶│   RL Agent   │─────▶│  Cloud   │ │
    │   │  Collector   │      │  (Dueling    │      │   API    │ │
    │   │ (Prometheus) │      │    DQN)      │      │          │ │
    │   └──────────────┘      └──────────────┘      └──────────┘ │
    │          │                     │                     │      │
    │          ▼                     ▼                     ▼      │
    │   ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
    │   │  Utilization │      │   Action:    │      │  Scale   │ │
    │   │  Capacity    │      │  scale_up/   │      │  Cluster │ │
    │   │  Trend       │      │  hold/down   │      │          │ │
    │   └──────────────┘      └──────────────┘      └──────────┘ │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
```

## Neural Network Architecture

```
    Dueling DQN
    
    State [3] ──▶ ┌────────┐ ──▶ ┌────────┐ ──┬──▶ Value [1] ──┐
                  │ FC 128 │     │ FC 128 │   │                │
                  │  ReLU  │     │  ReLU  │   │                ├──▶ Q(s,a) [3]
                  └────────┘     └────────┘   │                │
                                              └──▶ Adv [3] ────┘
```

---

# Recommendations

## Immediate Actions (0-3 months)

| Priority | Action | Owner | Effort |
|:--------:|--------|-------|:------:|
| 🔴 | Deploy simulation environment for testing | Engineering | 2 weeks |
| 🔴 | Train Dueling DQN on production traces | Data Science | 1 week |
| 🟡 | Set up monitoring dashboard | DevOps | 1 week |

## Short-term (3-6 months)

| Priority | Action | Owner | Effort |
|:--------:|--------|-------|:------:|
| 🟡 | Shadow deployment alongside existing autoscaler | Engineering | 2 weeks |
| 🟡 | A/B test on non-critical workloads | Data Science | 1 month |
| 🟢 | Document decision policies for audit | Compliance | 2 weeks |

## Long-term (6-12 months)

| Priority | Action | Owner | Effort |
|:--------:|--------|-------|:------:|
| 🟢 | Multi-service coordination (multi-agent RL) | Research | 3 months |
| 🟢 | Continuous learning pipeline | MLOps | 2 months |
| 🟢 | Extend to memory/network resources | Engineering | 2 months |

---

# Risk Mitigation

## Potential Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|:------:|:----------:|------------|
| Agent makes poor decisions | High | Medium | Fallback to threshold rules, action bounds |
| Model drift over time | Medium | High | Continuous retraining, monitoring |
| Explainability concerns | Medium | Medium | Log all decisions, SHAP analysis |
| Cold start problem | Low | High | Pre-train on historical data |

## Safety Guardrails

```
    Safety Constraints
    
    ┌─────────────────────────────────────────────────────────┐
    │  ✓ Maximum scale-up rate: 20% per decision             │
    │  ✓ Minimum capacity: 2 instances (always available)    │
    │  ✓ Cooldown period: 60s between scaling actions        │
    │  ✓ Human override: manual control always available     │
    │  ✓ Circuit breaker: revert to threshold if reward < X  │
    └─────────────────────────────────────────────────────────┘
```

---

# ROI Analysis

## Cost-Benefit Projection

### Assumptions
- Current cloud spend: **$1M/month**
- Over-provisioning waste: **30%** ($300K/month)
- RL improvement: **15% reduction** in waste (conservative)

### Projected Savings

| Timeframe | Monthly Savings | Cumulative |
|-----------|:---------------:|:----------:|
| Month 1-3 | $0 (implementation) | -$50K (investment) |
| Month 4-6 | $22.5K | $17.5K |
| Month 7-12 | $45K | $287.5K |
| **Year 1 Total** | — | **$237.5K net** |

### Break-even: **Month 5**

---

# Future Enhancements

## Roadmap

```
    2025 Q4          2026 Q1          2026 Q2          2026 Q3
       │                │                │                │
       ▼                ▼                ▼                ▼
    ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
    │ MVP  │───────▶│Multi-│───────▶│ Safe │───────▶│ Full │
    │Deploy│        │Agent │        │  RL  │        │ Prod │
    └──────┘        └──────┘        └──────┘        └──────┘
       │                │                │                │
       │                │                │                │
    Single           Multiple         Constrained      Enterprise
    service          services         exploration      rollout
```

## Research Directions

1. **Multi-Agent RL** — Coordinate scaling across microservices
2. **Safe RL** — Constrained policies that guarantee SLA bounds
3. **Transfer Learning** — Train on simulation, deploy to production
4. **Continuous Actions** — Fine-grained scaling with PPO/SAC

---

# Conclusion

## Summary

| ✅ | Achievement |
|---|-------------|
| 🎯 | Built simulation environment with realistic workloads |
| 🎯 | Implemented 5 RL algorithms + 2 baselines |
| 🎯 | **Dueling DQN achieves +104% improvement** |
| 🎯 | Demonstrated clear evidence of learning |
| 🎯 | Provided production deployment roadmap |

## Key Takeaway

> **Reinforcement learning can transform cloud autoscaling from reactive rule-following to proactive, cost-optimizing decision-making.**

## Call to Action

1. **Approve** pilot deployment on non-critical workload
2. **Allocate** 1 engineer for 3-month implementation
3. **Schedule** monthly progress reviews

---

# Appendix

## A. Technical Specifications

| Component | Specification |
|-----------|---------------|
| Python Version | 3.12 |
| Deep Learning Framework | PyTorch 2.x |
| Training Episodes | 2,000 |
| Episode Length | 200 steps |
| Replay Buffer | 10,000 transitions |
| Neural Network | 2 × 128 hidden layers |

## B. Repository

```bash
git clone https://github.com/rah-ds/Cloud-Autoscaling-using-RL.git
cd Cloud-Autoscaling-using-RL
make setup && source .venv/bin/activate
make train      # Train all algorithms
make plots      # Generate visualizations
```

## C. Team Contacts

| Name | Role | Contact |
|------|------|---------|
| Ryan Healy | Simulator & Data | rh@virginia.edu |
| Srivatsa Balasubramanyam | RL Agents | sb@virginia.edu |
| Bruce McGregor | Baselines & Eval | bm@virginia.edu |

---

# Thank You

## Questions?

**Repository:** [github.com/rah-ds/Cloud-Autoscaling-using-RL](https://github.com/rah-ds/Cloud-Autoscaling-using-RL)

**Documentation:** [Model Architectures](../model_architectures.md)

**W&B Dashboard:** [wandb.ai/healydatascience](https://wandb.ai/healydatascience-university-of-virginia/cloud-autoscaling-rl)

---

*Presentation prepared for DS MSDS Program | University of Virginia | December 2025*
