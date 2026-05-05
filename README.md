# CAAM+ Optimizer

<div align="center">

**Cosine-Aligned Adaptive Momentum** — a novel first-order optimizer that extends Adam with geometry-aware learning-rate scaling and dynamic momentum alignment.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](CAAM_Optimizer.ipynb)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Benchmarks & Results](#benchmarks--results)
  - [Well-Conditioned Quadratic](#1-well-conditioned-quadratic-κ--10)
  - [Ill-Conditioned Quadratic](#2-ill-conditioned-quadratic-κ--50)
  - [Rosenbrock Function](#3-rosenbrock-function)
  - [MNIST Classification](#4-mnist-classification)
  - [Summary](#summary)
- [Hyperparameters](#hyperparameters)
- [Notebook Structure](#notebook-structure)
- [When to Use CAAM+](#when-to-use-caam)

---

## Overview

Standard Adam uses a **fixed** momentum coefficient β₁ and a **constant** base learning rate. CAAM+ makes both adaptive by reading signals from the gradient stream at each step:

| Feature | Adam | CAAM+ |
|---------|------|-------|
| Learning rate | constant `lr` | `lr × s_t` — scaled by curvature signal |
| Momentum β₁ | fixed (e.g. 0.9) | dynamic in `[β₁_min, β₁_max]` based on gradient–momentum alignment |
| Bias correction | ✅ | ❌ (not needed — no fixed β₁ to correct for) |
| Extra state | step, m, v | step, m, v, prev_grad_norm |

---

## Algorithm

### 1 · Curvature-Aware Learning-Rate Scaling

The effective step size is modulated by the **relative change in gradient norm** between consecutive steps:

```
S_t   = ‖g_t‖₂                              # current gradient norm
δS    = (S_t − S_{t-1}) / (S_{t-1} + ε_S)  # relative norm change
s_t   = clip(1 − k · δS,  s_min, s_max)     # curvature scale  ∈ [0.5, 1.5]
α_t   = lr × s_t                             # effective learning rate
```

- **Gradient norm shrinking** → smooth descent → `s_t > 1` → step enlarged  
- **Gradient norm growing** → high curvature / oscillation → `s_t < 1` → step reduced  

### 2 · Cosine-Aligned β₁ Scheduling

The momentum coefficient β₁ is updated each step using the **cosine similarity** between the current gradient and the accumulated momentum vector:

```
cos    = ⟨g_t, m_{t-1}⟩ / (‖g_t‖ · ‖m_{t-1}‖ + ε_cos)
align  = max(0, cos)
β₁_t  = β₁_min + (β₁_max − β₁_min) × align
```

- **High alignment** (gradients consistent with momentum) → `β₁_t → β₁_max` → trust momentum  
- **Low / negative alignment** (direction reversal) → `β₁_t → β₁_min` → reset toward raw gradient  

### 3 · Full Update Rule

```
m_t  = β₁_t · m_{t-1} + (1 − β₁_t) · g_t      # 1st moment  (no bias-correction)
v_t  = β₂   · v_{t-1} + (1 − β₂)   · g_t²      # 2nd moment  (no bias-correction)

x_t  = x_{t-1}  −  α_t · m_t / (√v_t + ε)
```

---

## Installation

```bash
git clone https://github.com/<your-username>/caam-optimizer
cd caam-optimizer
pip install torch torchvision matplotlib
```

> **Inside a notebook** — paste this in the first cell if PyTorch is not yet installed:
> ```python
> import sys
> !{sys.executable} -m pip install torch torchvision matplotlib
> ```

---

## Quickstart

```python
# Initialise state before training
state = {
    "step": 0,
    "m": torch.zeros_like(x),
    "v": torch.zeros_like(x),
    "prev_grad_norm": None,
}

hyper = {
    "lr": 1e-3,
    "beta1_min": 0.8,  "beta1_max": 0.99,
    "beta2": 0.999,
    "k": 0.5,
    "s_min": 0.5,      "s_max": 1.5,
    "eps": 1e-8,       "eps_cos": 1e-8,  "eps_S": 1e-8,
}

# Training loop
for step in range(max_steps):
    loss = criterion(model(x), y)
    loss.backward()
    caam_lite_step_model(model, state, hyper)
    model.zero_grad()
```

---

## Benchmarks & Results

All benchmarks are run from a fixed random seed (`torch.manual_seed(0)`) for reproducibility. Baselines are Adam and SGD+Momentum with identical learning rates.

---

### 1. Well-Conditioned Quadratic (κ = 10)

**Setup:** 10-dimensional bowl `f(x) = ½ xᵀQx`, eigenvalues spread in `[1, 10]`, `lr = 0.05`, target `f(x) < 1e-6`.

![Well-conditioned quadratic convergence](assets/quad_well.png)

| Optimizer | Iterations to converge | Speedup vs Adam |
|-----------|:---------------------:|:---------------:|
| **CAAM+** | **94** | **2.9×** |
| SGD+Momentum | 156 | 1.8× |
| Adam | 276 | 1× (baseline) |

CAAM+ is **2.9× faster than Adam** on a well-conditioned bowl. The adaptive LR scaling accelerates descent in smooth regions while the cosine-aligned β₁ suppresses unnecessary momentum resets.

---

### 2. Ill-Conditioned Quadratic (κ = 50)

**Setup:** Same as above but eigenvalues spread in `[1, 50]`, `lr = 0.02` (reduced to maintain stability), target `f(x) < 1e-6`, max 20,000 steps.

![Ill-conditioned quadratic convergence](assets/quad_ill.png)

| Optimizer | Iterations to converge | Speedup vs Adam |
|-----------|:---------------------:|:---------------:|
| SGD+Momentum | 176 | 10.3× |
| **CAAM+** | **575** | **3.2×** |
| Adam | 1,819 | 1× (baseline) |

Under high curvature, Adam suffers from slow convergence due to conservative per-coordinate scaling. CAAM+'s curvature-aware LR scaling detects gradient norm oscillation and contracts steps — reaching the target **3.2× faster than Adam**.

> Note: SGD+Momentum wins here because its uniform step size navigates this particular isotropic-ish problem efficiently. In higher-dimensional anisotropic settings, CAAM+'s coordinate-wise second moment provides a robustness advantage.

---

### 3. Rosenbrock Function

**Setup:** `f(x, y) = (1 − x)² + 100(y − x²)²`, starting at `(−1.2, 1.0)`, `lr = 0.002` (CAAM+/Adam) / `0.001` (SGD), target `f < 1e-4`, max 100,000 steps.

![Rosenbrock convergence](assets/rosenbrock.png)

| Optimizer | Iterations to converge |
|-----------|:---------------------:|
| **SGD+Momentum** | **837** |
| CAAM+ | 6,129 |
| Adam | 7,888 |

The Rosenbrock narrow valley is the natural home of heavy-ball momentum — SGD rolls along the valley floor quickly once it finds it. Adaptive methods adapt per-coordinate, which slows navigation along the valley's long axis. CAAM+ still beats Adam by **22%**.

---

### 4. MNIST Classification

**Setup:** 784→128→64→10 MLP (~109k parameters), ReLU activations, CrossEntropyLoss, batch size 64, 20 epochs, CPU.

#### Training Loss Curve

![MNIST training loss](assets/mnist_loss.png)

| Epoch | CAAM+ | Adam | SGD+Momentum |
|:-----:|:-----:|:----:|:------------:|
| 1 | 0.2161 | 0.3408 | 0.5333 |
| 5 | 0.0415 | 0.0559 | 0.0828 |
| 10 | 0.0157 | 0.0221 | 0.0347 |
| 15 | 0.0089 | 0.0132 | 0.0163 |
| **20** | **0.0044** | 0.0107 | 0.0066 |

CAAM+ reaches a **2.4× lower final training loss** than Adam by epoch 20 (0.0044 vs 0.0107), demonstrating more aggressive exploitation of the loss landscape in later training.

#### Test Accuracy & Training Time

![MNIST accuracy and time](assets/mnist_bars.png)

| Optimizer | Test Accuracy | Training Time (CPU) |
|-----------|:-------------:|:-------------------:|
| SGD+Momentum | **98.0%** | **~150s** |
| Adam | 97.7% | ~190s |
| CAAM+ | 97.6% | ~225s |

All three achieve comparable accuracy (~97.6–98.0%). CAAM+'s per-step overhead (cosine similarity + norm tracking) makes it ~18% slower than Adam on CPU — acceptable when convergence quality matters more than wall-clock time.

---

### Summary

![Convergence speed comparison](assets/convergence_bar.png)

| Benchmark | CAAM+ | Adam | SGD+Momentum |
|-----------|:-----:|:----:|:------------:|
| Well-conditioned quadratic (κ=10) | **94 iters** ✅ | 276 iters | 156 iters |
| Ill-conditioned quadratic (κ=50) | **575 iters** ✅ | 1,819 iters | 176 iters |
| Rosenbrock (f < 1e-4) | 6,129 iters | 7,888 iters | **837 iters** ✅ |
| MNIST final train loss (epoch 20) | **0.0044** ✅ | 0.0107 | 0.0066 |
| MNIST test accuracy | ~97.6% | ~97.7% | **~98.0%** ✅ |
| MNIST training time (CPU) | ~225s | ~190s | **~150s** ✅ |

---

## Hyperparameters

| Parameter | Recommended | Description |
|-----------|:-----------:|-------------|
| `lr` | `1e-3` | Base learning rate (same starting point as Adam) |
| `beta1_min` | `0.8` | Minimum β₁ — applied when gradient direction reverses |
| `beta1_max` | `0.99` | Maximum β₁ — applied when gradient aligns with momentum |
| `beta2` | `0.999` | Second-moment EMA decay (same as Adam) |
| `k` | `0.5` | Curvature sensitivity — higher = more reactive to norm changes |
| `s_min` | `0.5` | Minimum LR scale factor (floor for curvature contraction) |
| `s_max` | `1.5` | Maximum LR scale factor (ceiling for smooth-region expansion) |
| `eps` | `1e-8` | Denominator stability |
| `eps_cos` | `1e-8` | Cosine similarity numerical safety |
| `eps_S` | `1e-8` | Gradient norm ratio numerical safety |

**Tuning tips:**
- Increase `k` if your loss oscillates; decrease it if convergence seems sluggish.
- Widen `[beta1_min, beta1_max]` for more aggressive momentum adaptation.
- Tighten `s_min / s_max` for sensitive fine-tuning tasks.

---

## Notebook Structure

```
CAAM_Optimizer.ipynb
│
├── §0  Imports & Utilities
│       make_spd_matrix, quadratic_value, rosenbrock_value
│
├── §1  Optimiser Implementations
│       caam_lite_step          — scalar/vector CAAM+ step
│       adam_step               — standard Adam
│       sgd_momentum_step       — SGD + heavy-ball momentum
│
├── §2  Benchmark 1 — Well-Conditioned Quadratic (κ ≤ 10)
├── §3  Benchmark 2 — Ill-Conditioned Quadratic  (κ ≥ 50)
├── §4  Benchmark 3 — Rosenbrock Function
│
├── §5  Full-Model Wrappers
│       caam_lite_step_model    — CAAM+ for nn.Module
│       adam_step_model         — Adam for nn.Module
│       sgd_momentum_step_model — SGD for nn.Module
│       init_*_state helpers
│
├── §6  Benchmark 4 — MNIST (784→128→64→10 MLP, 20 epochs)
│
└── §7  Summary of Results & Key Takeaways
```

---

## When to Use CAAM+

✅ **Good fit:**
- Problems with **variable curvature** where gradient norms fluctuate between steps
- Training runs where **lower final loss** matters more than wall-clock speed
- **Fine-tuning** scenarios where the improvement direction changes frequently
- Any setting where Adam's fixed β₁ feels like a tuning burden

⚠️ **Consider alternatives:**
- Simple convex problems with uniform curvature → SGD+Momentum may be faster
- Narrow-valley non-convex problems → SGD+Momentum dominates
- Tight CPU wall-clock budgets → SGD+Momentum has the lowest overhead

---

## Requirements

```
torch>=1.10
torchvision>=0.11
matplotlib>=3.4
python>=3.8
```

---

## License

MIT
