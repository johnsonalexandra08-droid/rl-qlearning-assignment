# rl-qlearning-assignment

**AAI 201 — M9: Q-Learning with OpenAI Gymnasium**  
**Student:** Hannah Johnson

---

## Overview

This project implements the **Q-Learning** reinforcement learning algorithm using the [Gymnasium](https://gymnasium.farama.org/) library (formerly OpenAI Gym). The agent learns to navigate the **FrozenLake-v1** environment — a 4×4 grid where it must travel from the start (top-left) to the goal (bottom-right) while avoiding holes.

---

## Environment: FrozenLake-v1

```
S F F F       S = Start
F H F H       F = Frozen (safe)
F F F H       H = Hole (episode ends, reward = 0)
H F F G       G = Goal (reward = 1.0)
```

- **State space:** 16 discrete positions (0–15)
- **Action space:** 4 actions (LEFT=0, DOWN=1, RIGHT=2, UP=3)
- **Reward:** 1.0 only upon reaching the Goal, 0.0 otherwise
- **Configuration:** `is_slippery=False` (deterministic transitions)

---

## Algorithm: Q-Learning

Q-Learning is a model-free, off-policy reinforcement learning algorithm. The agent updates a Q-table using the **Bellman equation**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Hyperparameters used:**

| Parameter | Value | Description |
|---|---|---|
| Alpha (α) | 0.8 | Learning rate |
| Gamma (γ) | 0.95 | Discount factor |
| Epsilon start | 1.0 | Initial exploration rate |
| Epsilon decay | 0.995 | Decay per episode |
| Epsilon min | 0.01 | Minimum exploration rate |
| Episodes | 3,000 | Total training episodes |

---

## Results

| Metric | Value |
|---|---|
| Training Episodes | 3,000 |
| Test Episodes | 100 |
| **Success Rate** | **~95–98%** |
| Average Test Reward | ~0.95+ |

The agent converges to a near-optimal policy, successfully navigating to the goal in the vast majority of test runs.

---

## Visualizations

- **Learning Curve:** Total reward per episode and 100-episode moving average showing convergence
- **Q-Table Heatmap:** Q-values for all 16 states × 4 actions
- **Policy Map:** Best action per grid cell (arrows showing learned navigation path)

---

## Key Findings

1. **Sparse rewards** are the biggest challenge — reward only arrives at the goal, making early exploration critical
2. **Epsilon decay** is essential — starting with full exploration and gradually shifting to exploitation drives convergence
3. **High gamma (0.95)** is important because the reward is delayed — the agent must value future states to learn correctly
4. **Real-world applications** include industrial process control, robotics navigation, inventory optimization, and healthcare treatment planning

---

## Files

| File | Description |
|---|---|
| `rl_qlearning_assignment.ipynb` | Main Jupyter notebook with all code, outputs, and reflection |
| `learning_curve.png` | Plot of training rewards and moving average |
| `qtable_heatmap.png` | Q-table and policy map visualization |
| `README.md` | This file |

---

## How to Run

**In Google Colab:**
1. Upload `rl_qlearning_assignment.ipynb` to Colab
2. Run all cells (Runtime → Run all)
3. The first cell installs Gymnasium automatically

**Locally:**
```bash
pip install gymnasium numpy matplotlib seaborn
jupyter notebook rl_qlearning_assignment.ipynb
```

---

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Ch. 18). O'Reilly.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
