# CAAM+ Optimizer

**CAAM+** (Cosine-Aligned Adaptive Momentum) is a novel first-order optimizer that extends Adam with two adaptive mechanisms that respond to the local geometry of the loss landscape at each step.

---

## How It Works

Standard Adam uses fixed hyperparameters for momentum (`β₁`) and a constant base learning rate. CAAM+ makes both of these dynamic:

**1. Curvature-Aware Learning-Rate Scaling**

The effective step size `α_t = lr × s_t` is modulated by the relative change in gradient norm between consecutive steps:

```
S_t  = ‖g_t‖₂
δS   = (S_t − S_{t-1}) / (S_{t-1} + ε_S)
s_t  = clip(1 − k·δS,  s_min, s_max)
α_t  = lr × s_t
```

When the gradient norm shrinks (smooth descent), the step is enlarged. When it grows (high curvature), the step is reduced.

**2. Cosine-Aligned β₁ Scheduling**

The momentum coefficient `β₁` is set dynamically based on the cosine similarity between the current gradient and the running momentum vector:

```
cos  = ⟨g_t, m_{t-1}⟩ / (‖g_t‖ · ‖m_{t-1}‖)
β₁_t = β₁_min + (β₁_max − β₁_min) × max(0, cos)
```

High alignment → high `β₁` (trust the momentum direction). Low or negative alignment → low `β₁` (reset toward the raw gradient).

**Full update rule:**

```
m_t = β₁_t · m_{t-1} + (1 − β₁_t) · g_t
v_t = β₂   · v_{t-1} + (1 − β₂)   · g_t²
x_t = x_{t-1} − α_t · m_t / (√v_t + ε)
```

---

## Benchmarks

Comparison against Adam and SGD+Momentum across four tasks:

| Benchmark | CAAM+ | Adam | SGD+Momentum |
|-----------|:-----:|:----:|:------------:|
| Well-conditioned quadratic (κ=10) | **94 iters** | 276 iters | 156 iters |
| Ill-conditioned quadratic (κ=50)  | **575 iters** | 1,819 iters | 176 iters |
| Rosenbrock (f < 1e-4)             | 6,129 iters | 7,888 iters | **837 iters** |
| MNIST test accuracy (20 epochs)   | ~97.6% | ~97.7% | ~98.0% |

CAAM+ shows the strongest advantage in high-curvature settings (ill-conditioned quadratics), where its adaptive LR scaling directly suppresses oscillation.

---

## Repository Structure

```
CAAM_Optimizer.ipynb   # Full benchmark notebook with explanations
README.md
```

The notebook is organized into 7 sections:

- **§0** — Imports & utility functions (SPD matrix factory, Rosenbrock)
- **§1** — Optimizer implementations: CAAM+, Adam, SGD+Momentum
- **§2** — Benchmark 1: Well-conditioned quadratic (κ ≤ 10)
- **§3** — Benchmark 2: Ill-conditioned quadratic (κ ≥ 50)
- **§4** — Benchmark 3: Rosenbrock function
- **§5** — Full-model wrappers for `nn.Module` training
- **§6** — Benchmark 4: MNIST with a 784→128→64→10 MLP
- **§7** — Summary of results and key takeaways

---

## Quickstart

```bash
git clone https://github.com/<your-username>/caam-optimizer
cd caam-optimizer
pip install torch torchvision matplotlib
jupyter notebook CAAM_Optimizer.ipynb
```

Or install dependencies inline at the top of the notebook:

```python
import sys
!{sys.executable} -m pip install torch torchvision matplotlib
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | task-specific | Base learning rate |
| `beta1_min` | 0.80 | Lower bound for adaptive β₁ |
| `beta1_max` | 0.99 | Upper bound for adaptive β₁ |
| `beta2` | 0.999 | Second-moment decay |
| `k` | 0.5 | Curvature sensitivity |
| `s_min` | 0.5 | Minimum LR scale factor |
| `s_max` | 1.5 | Maximum LR scale factor |
| `eps` | 1e-8 | Denominator stability constant |

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- matplotlib
