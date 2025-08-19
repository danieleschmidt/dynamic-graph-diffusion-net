# Mathematical Formulations for DGDN Research

## Overview

This document provides comprehensive mathematical formulations for the Dynamic Graph Diffusion Network (DGDN) architecture and its novel research extensions. These formulations establish the theoretical foundation for reproducible research and peer review.

## Core DGDN Architecture

### 1. Temporal Graph Representation

A temporal graph is defined as:
```
G_T = (V, E_T, X, A, T)
```

Where:
- `V = {v_1, v_2, ..., v_n}`: Set of nodes
- `E_T ⊆ V × V × ℝ⁺`: Set of temporal edges with timestamps
- `X ∈ ℝⁿˣᵈ`: Node feature matrix
- `A_t ∈ ℝⁿˣⁿ`: Adjacency matrix at time t
- `T = {t_1, t_2, ..., t_m}`: Set of timestamps

### 2. Edge-Time Encoding

The continuous time encoding uses Fourier features:

```
φ(t) = [sin(ω₁t + b₁), cos(ω₁t + b₁), ..., sin(ω_kt + b_k), cos(ω_kt + b_k)]
```

Where:
- `ω_i ∼ N(0, σ²)`: Learnable frequency parameters
- `b_i ∼ U(0, 2π)`: Phase parameters
- `k = d_time/2`: Number of frequency components

**Theoretical Properties:**
- **Universal Approximation**: Fourier features can approximate any continuous periodic function
- **Scale Invariance**: `φ(αt) = f(α)φ(t)` for appropriate scaling
- **Differentiability**: Enables gradient-based optimization

### 3. Variational Diffusion Process

The forward diffusion process adds structured noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

The reverse process learns denoising:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Variational Lower Bound:**
```
L_VLB = E_q[log p_θ(x_0 | x_1)] - Σ_{t=2}^T E_q[KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t))]
     - KL(q(x_T | x_0) || p(x_T))
```

### 4. Multi-Head Temporal Attention

The attention mechanism incorporates both spatial and temporal information:

```
Attention(Q, K, V, φ) = softmax(QK^T / √d_k + φ_temporal)V
```

Where:
- `Q, K, V ∈ ℝⁿˣᵈ`: Query, key, value matrices
- `φ_temporal`: Temporal encoding bias
- Multi-head extension: `MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O`

**Temporal Attention Weights:**
```
α_{ij}^(t) = exp((q_i^T k_j + φ(t_{ij})) / √d_k) / Σ_k exp((q_i^T k_k + φ(t_{ik})) / √d_k)
```

## Novel Research Extensions

### 1. Adaptive Diffusion Networks

**Complexity-Based Step Adaptation:**

Graph structural entropy:
```
H(G_t) = -Σᵢ (d_i / 2|E|) log(d_i / 2|E|)
```

Temporal volatility:
```
σ_t = √(Σᵢ (Δt_i - μ_Δt)² / |E|)
```

Adaptive step count:
```
n_steps(G_t) = max(n_min, min(n_max, ⌊α·H(G_t) + β·σ_t + γ⌋))
```

**Information-Theoretic Stopping Criterion:**
```
I(X_{t-1}, X_t) = ∫∫ p(x_{t-1}, x_t) log(p(x_{t-1}, x_t) / (p(x_{t-1})p(x_t))) dx_{t-1}dx_t
```

Stop when `I(X_{t-1}, X_t) < ε_stop`

### 2. Temporal Causal Discovery

**Granger Causality for Graphs:**

For nodes `i` and `j`, the causal influence is:
```
GC(i → j) = log(σ²(ε_{j,restricted}) / σ²(ε_{j,full}))
```

Where:
- `σ²(ε_{j,restricted})`: Prediction error variance using only node j's history
- `σ²(ε_{j,full})`: Prediction error variance using both nodes' histories

**Temporal Causal Strength:**
```
CS(i → j, τ) = ∫ KL(P(X_j^{t+τ} | do(X_i^t = x)) || P(X_j^{t+τ})) dx
```

**Intervention Effect Size:**
```
ATE(i → j, τ) = E[Y_j^{t+τ} | do(X_i^t = 1)] - E[Y_j^{t+τ} | do(X_i^t = 0)]
```

### 3. Uncertainty Quantification

**Epistemic Uncertainty (Model Uncertainty):**
```
Var_epistemic[f(x)] = E_θ[(f_θ(x) - E_θ[f_θ(x)])²]
```

**Aleatoric Uncertainty (Data Uncertainty):**
```
Var_aleatoric[f(x)] = E_θ[Var_data[f_θ(x)]]
```

**Total Predictive Uncertainty:**
```
Var_total[f(x)] = Var_epistemic[f(x)] + E_θ[Var_aleatoric[f(x)]]
```

## Loss Functions

### 1. Multi-Component DGDN Loss

```
L_total = L_recon + β₁L_KL + β₂L_temporal + β₃L_diffusion
```

**Reconstruction Loss:**
```
L_recon = -E_{(i,j,t)∈E_T}[y_{ij} log σ(e_{ij}^t) + (1-y_{ij}) log(1-σ(e_{ij}^t))]
```

**KL Divergence Loss:**
```
L_KL = KL(q_φ(z|x) || p(z)) = -½ Σᵢ (1 + log σᵢ² - μᵢ² - σᵢ²)
```

**Temporal Regularization:**
```
L_temporal = λ Σₜ ||h_t - h_{t-1}||² + γ Σₜ ||∇_t h_t||²
```

**Diffusion Loss:**
```
L_diffusion = E_{t,ε}[||ε - ε_θ(√α̅_t x_0 + √(1-α̅_t)ε, t)||²]
```

### 2. Causal Discovery Loss

**Causal Structure Learning:**
```
L_causal = L_likelihood + λ₁L_sparsity + λ₂L_acyclicity
```

Where:
```
L_sparsity = ||A||₁  (L1 regularization on adjacency matrix)
L_acyclicity = tr(e^(A⊙A)) - d  (Differentiable acyclicity constraint)
```

## Optimization and Training

### 1. Gradient Computation

For the variational objective, gradients are computed using the reparameterization trick:

```
∇_φ E_q[f(z)] = ∇_φ E_ε[f(μ_φ(x) + σ_φ(x)ε)]
                = E_ε[∇_φ f(μ_φ(x) + σ_φ(x)ε)]
```

### 2. Adaptive Learning Rate Schedule

**Cosine Annealing with Warm Restarts:**
```
η_t = η_min + ½(η_max - η_min)(1 + cos(πT_cur/T_i))
```

**Learning Rate Adaptation for Causal Discovery:**
```
η_causal = η_base × (1 - |DAG_violation|/n²)
```

### 3. Memory-Efficient Training

**Gradient Checkpointing:**
Save memory by trading computation for storage:
```
Memory_saved = O(L) → O(√L)
Computation_overhead = 33%
```

Where L is the number of layers.

## Statistical Analysis

### 1. Significance Testing

**Paired t-test for Model Comparison:**
```
t = (x̄_D - μ₀) / (s_D / √n)
```

Where:
- `x̄_D`: Mean difference between models
- `s_D`: Standard deviation of differences
- `n`: Number of test samples

**Wilcoxon Signed-Rank Test (Non-parametric):**
```
W = Σᵢ rank(|x_i|) × sign(x_i)
```

### 2. Effect Size Measures

**Cohen's d:**
```
d = (μ₁ - μ₂) / σ_pooled
```

**Cliff's Delta (Non-parametric):**
```
δ = (P(X₁ > X₂) - P(X₁ < X₂)) / (P(X₁ > X₂) + P(X₁ < X₂) + P(X₁ = X₂))
```

## Convergence Analysis

### 1. Diffusion Process Convergence

The diffusion process converges when:
```
||x_t - x_{t-1}||₂ < ε_conv
```

**Convergence Rate:**
For the reverse diffusion process:
```
||x_t - x*||₂ ≤ C·ρᵗ||x₀ - x*||₂
```

Where `ρ < 1` is the contraction factor.

### 2. Causal Discovery Convergence

**Score Function Convergence:**
```
lim_{n→∞} S(G_n) = S(G*)
```

Where `S(G)` is the causal score function and `G*` is the true causal graph.

## Complexity Analysis

### 1. Time Complexity

**DGDN Forward Pass:**
- Edge-time encoding: `O(|E| × d_time)`
- Multi-head attention: `O(n² × d_hidden × h)`
- Diffusion steps: `O(T × n × d_hidden²)`
- **Total**: `O(T × n² × d_hidden × h)`

**Causal Discovery:**
- Granger causality computation: `O(n² × L × d_hidden)`
- Causal structure optimization: `O(n³)` per iteration
- **Total**: `O(I × n³)` where I is iterations

### 2. Space Complexity

**Memory Requirements:**
- Node embeddings: `O(n × d_hidden)`
- Attention matrices: `O(h × n²)`
- Diffusion history: `O(T × n × d_hidden)`
- **Total**: `O(T × n × d_hidden + h × n²)`

## Theoretical Guarantees

### 1. Universal Approximation

**Theorem**: DGDN with sufficient width and depth can approximate any continuous function on temporal graphs.

**Proof Sketch**: Follows from universal approximation properties of:
1. Fourier features for time encoding
2. Multi-layer perceptrons for spatial processing
3. Attention mechanisms for relationship modeling

### 2. Convergence Guarantees

**Theorem**: Under Lipschitz continuity and bounded gradients, DGDN training converges to a local minimum.

**Conditions**:
- `||∇L(θ)|| ≤ M` for all θ
- `L(θ)` is L-smooth: `||∇L(θ₁) - ∇L(θ₂)|| ≤ L||θ₁ - θ₂||`

**Convergence Rate**: `O(1/√T)` for SGD, `O(1/T)` for adaptive methods.

## Implementation Considerations

### 1. Numerical Stability

**Log-Sum-Exp Trick for Softmax:**
```
softmax(x_i) = exp(x_i - max(x)) / Σⱼ exp(x_j - max(x))
```

**Gradient Clipping:**
```
g ← g × min(1, θ_clip / ||g||)
```

### 2. Hyperparameter Sensitivity

**Critical Hyperparameters:**
- `β₁, β₂, β₃`: Loss component weights
- `T`: Number of diffusion steps  
- `d_hidden`: Hidden dimension
- `h`: Number of attention heads

**Recommended Ranges:**
- `β₁ ∈ [0.01, 0.1]`: KL weight
- `β₂ ∈ [0.001, 0.01]`: Temporal regularization
- `T ∈ [3, 15]`: Diffusion steps
- `h ∈ [4, 16]`: Attention heads

## Future Mathematical Extensions

### 1. Continuous Normalizing Flows

Replace discrete diffusion with continuous flows:
```
dx/dt = f_θ(x(t), t)
```

**Log-likelihood:**
```
log p(x₀) = log p(x₁) - ∫₀¹ tr(∂f/∂x(t)) dt
```

### 2. Graph Neural ODEs

Extend to continuous graph dynamics:
```
dh/dt = GNN_θ(h(t), A(t))
```

### 3. Quantum Graph Networks

Incorporate quantum computational elements:
```
|ψ⟩ = Σᵢ αᵢ|node_i⟩
U_quantum = exp(-iĤt)
```

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
2. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
3. Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. NIPS Workshop.
4. Granger, C. W. J. (1969). Investigating causal relations by econometric models. Econometrica.
5. Chen, R. T. Q., et al. (2018). Neural ordinary differential equations. NeurIPS.

---

*This document provides the mathematical foundation for reproducible research with DGDN. All formulations are designed to be implementation-ready and scientifically rigorous.*