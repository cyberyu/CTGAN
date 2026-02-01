# CTGAN Codebase Analysis: Model Variants & Configuration Options

## Summary
The CTGAN codebase provides **2 main model variants** for synthetic data generation:

1. **CTGAN** (Conditional Tabular GAN) - GAN-based approach
2. **TVAE** (Tabular Variational AutoEncoder) - VAE-based approach

---

## 1. CTGAN (Conditional Tabular GAN)

### Conceptual Architecture Diagram:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CTGAN ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Real Tabular Data (Mixed: Continuous + Categorical)
  │
  ├─► DataTransformer (Preprocessing)
  │     ├─► Continuous → Normalize + Mode-specific normalization
  │     └─► Categorical → One-hot encoding
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PHASE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────┐         ┌──────────────────────────┐            │
│  │   Random Noise (z)     │         │  Conditional Vector (c)  │            │
│  │  [embedding_dim=128]   │         │  (sampled from training) │            │
│  └───────────┬────────────┘         └────────────┬─────────────┘            │
│              │                                    │                          │
│              └────────────────┬───────────────────┘                          │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                   │
│                    │     GENERATOR       │                                   │
│                    ├─────────────────────┤                                   │
│                    │  Residual Block 1   │ ◄─ generator_dim[0]=256          │
│                    │  (Linear+BN+ReLU)   │                                   │
│                    ├─────────────────────┤                                   │
│                    │  Residual Block 2   │ ◄─ generator_dim[1]=256          │
│                    │  (Linear+BN+ReLU)   │                                   │
│                    ├─────────────────────┤                                   │
│                    │    Output Layer     │                                   │
│                    │  (Linear → data_dim)│                                   │
│                    └──────────┬──────────┘                                   │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                   │
│                    │  Activation Layer   │                                   │
│                    │  • tanh (continuous)│                                   │
│                    │  • Gumbel-Softmax   │                                   │
│                    │    (categorical)    │                                   │
│                    └──────────┬──────────┘                                   │
│                               │                                              │
│                               ▼                                              │
│                       Fake Data (x_fake)                                     │
│                               │                                              │
│              ┌────────────────┴────────────────┐                             │
│              │                                 │                             │
│              ▼                                 ▼                             │
│    ┌──────────────────┐              ┌──────────────────┐                   │
│    │  Real Data Batch │              │  Fake Data Batch │                   │
│    │    (x_real)      │              │    (x_fake)      │                   │
│    └────────┬─────────┘              └────────┬─────────┘                   │
│             │                                  │                             │
│             └──────────────┬───────────────────┘                             │
│                            │                                                 │
│                            ▼                                                 │
│              ┌──────────────────────────────┐                                │
│              │      DISCRIMINATOR (PacGAN)  │                                │
│              ├──────────────────────────────┤                                │
│              │ Group samples (pac=10)       │                                │
│              ├──────────────────────────────┤                                │
│              │  Hidden Layer 1              │ ◄─ discriminator_dim[0]=256   │
│              │  (Linear+LeakyReLU+Dropout)  │                                │
│              ├──────────────────────────────┤                                │
│              │  Hidden Layer 2              │ ◄─ discriminator_dim[1]=256   │
│              │  (Linear+LeakyReLU+Dropout)  │                                │
│              ├──────────────────────────────┤                                │
│              │  Output Layer (Linear → 1)   │                                │
│              └────────────┬─────────────────┘                                │
│                           │                                                  │
│                           ▼                                                  │
│              ┌──────────────────────────────┐                                │
│              │   Wasserstein Loss + GP      │                                │
│              │  • Discriminator Loss        │                                │
│              │  • Generator Loss            │                                │
│              │  • Gradient Penalty          │                                │
│              │  • Conditional Loss          │                                │
│              └──────────────────────────────┘                                │
│                           │                                                  │
│                           ▼                                                  │
## 2. TVAE (Tabular Variational AutoEncoder)

### Conceptual Architecture Diagram:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TVAE ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Real Tabular Data (Mixed: Continuous + Categorical)
  │
  ├─► DataTransformer (Preprocessing)
  │     ├─► Continuous → Normalize + Mode-specific normalization
  │     └─► Categorical → One-hot encoding
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PHASE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│                    Real Data Batch (x)                                       │
│                    [data_dim dimensions]                                     │
│                            │                                                 │
│                            ▼                                                 │
│              ┌──────────────────────────────┐                                │
│              │         ENCODER              │                                │
│              ├──────────────────────────────┤                                │
│              │   Hidden Layer 1             │ ◄─ compress_dims[0]=128       │
│              │   (Linear + ReLU)            │                                │
│              ├──────────────────────────────┤                                │
│              │   Hidden Layer 2             │ ◄─ compress_dims[1]=128       │
│              │   (Linear + ReLU)            │                                │
│              └──────────┬───────────────────┘                                │
│                         │                                                    │
│                         ├─────────────────┬──────────────────┐               │
│                         ▼                 ▼                  ▼               │
│              ┌──────────────┐  ┌──────────────┐   ┌──────────────┐          │
│              │  Mean (μ)    │  │ Log-Var (σ²) │   │  Std Dev     │          │
│              │  FC Layer    │  │  FC Layer    │   │  exp(0.5*σ²) │          │
│              └──────┬───────┘  └──────┬───────┘   └──────┬───────┘          │
│                     │                 │                  │                   │
│                     └────────┬────────┴──────────────────┘                   │
│                              │                                               │
│                              ▼                                               │
│              ┌────────────────────────────────────┐                          │
│              │   REPARAMETERIZATION TRICK         │                          │
│              │   z = μ + ε * σ                    │                          │
│              │   where ε ~ N(0,1)                 │                          │
│              └──────────────┬─────────────────────┘                          │
│                             │                                                │
│                   Latent Vector (z)                                          │
│                   [embedding_dim=128]                                        │
│                             │                                                │
│                             ▼                                                │
│              ┌──────────────────────────────┐                                │
│              │         DECODER              │                                │
│              ├──────────────────────────────┤                                │
│              │   Hidden Layer 1             │ ◄─ decompress_dims[0]=128     │
│              │   (Linear + ReLU)            │                                │
│              ├──────────────────────────────┤                                │
│              │   Hidden Layer 2             │ ◄─ decompress_dims[1]=128     │
│              │   (Linear + ReLU)            │                                │
│              ├──────────────────────────────┤                                │
│              │   Output Layer               │                                │
│              │   (Linear → data_dim)        │                                │
│              └──────────┬───────────────────┘                                │
│                         │                                                    │
│                         ├─────────────────┬─────────────────┐                │
│                         ▼                 ▼                 ▼                │
│              Reconstructed Data      Sigma (σ)      Apply Activation         │
│                   (x_recon)         (noise param)    • tanh (continuous)     │
│                         │                 │          • softmax (categorical) │
│                         └────────┬────────┘                                  │
│                                  │                                           │
│                                  ▼                                           │
│              ┌───────────────────────────────────────┐                       │
│              │         LOSS CALCULATION              │                       │
│              ├───────────────────────────────────────┤                       │
│              │  1. Reconstruction Loss:              │                       │
│              │     • Continuous: (x - x_recon)²/σ²   │                       │
│              │     • Categorical: Cross-Entropy      │                       │
│              │                                       │                       │
│              │  2. KL Divergence:                    │                       │
│              │     KL(q(z|x) || p(z))                │                       │
│              │     = -0.5 * Σ(1 + log(σ²) - μ² - σ²)│                       │
│              │                                       │                       │
│              │  Total Loss = loss_factor * Recon +   │                       │
│              │               KL                      │                       │
│              └─────────────────┬─────────────────────┘                       │
│                                │                                             │
│                                ▼                                             │
│                   Backpropagation & Optimization                             │
│                   • Adam optimizer with L2 regularization (l2scale=1e-5)    │
│                   • Update Encoder & Decoder jointly                         │
│                   • Clamp sigma ∈ [0.01, 1.0]                                │
│                   • Repeat for epochs=300                                    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          SAMPLING PHASE (Generation)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Random Latent Vector: z ~ N(0, I)                                           │
│            [embedding_dim=128]                                               │
│                    │                                                         │
│                    ▼                                                         │
│            Trained Decoder                                                   │
│                    │                                                         │
│                    ▼                                                         │
│         Generated Data + Sigma                                               │
│                    │                                                         │
│                    ▼                                                         │
│          Apply tanh activation                                               │
│                    │                                                         │
│                    ▼                                                         │
│      DataTransformer.inverse_transform()                                     │
│                    │                                                         │
│                    ▼                                                         │
│  OUTPUT: Synthetic Tabular Data (Same format as input)                       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Key Data Flow:
• Real Data → Encoder → Latent Distribution (μ, σ²)
• Sample z from N(μ, σ²) using reparameterization trick
• z → Decoder → Reconstructed Data
• Loss = Reconstruction Error + KL Divergence (regularization)
• KL term encourages latent space to be N(0, I) for easy sampling
• Decoder learns smooth manifold for generating new samples
```

### Architecture Components:
- **Encoder**: Multi-layer perceptron that outputs mean (μ) and variance (σ²)
- **Decoder**: Multi-layer perceptron that reconstructs data
- **Training Method**: Variational lower bound optimization (ELBO)────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          SAMPLING PHASE (Generation)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Random Noise (z) + Conditional Vector (c)                                   │
│              │                                                                │
│              ▼                                                                │
│       Trained Generator                                                      │
│              │                                                                │
│              ▼                                                                │
│    Synthetic Data (Transformed)                                              │
│              │                                                                │
│              ▼                                                                │
│    DataTransformer.inverse_transform()                                       │
│              │                                                                │
│              ▼                                                                │
│  OUTPUT: Synthetic Tabular Data (Same format as input)                       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Key Data Flow:
• Real Data → Transform → Train Discriminator (classify real vs fake)
• Noise + Condition → Generator → Fake Data → Train via Discriminator feedback
• Conditional Vector ensures generated data respects categorical distributions
• PacGAN groups samples to prevent mode collapse
• Gradient Penalty stabilizes training (WGAN-GP)
```

### Architecture Components:
- **Generator**: Uses Residual blocks with BatchNorm and ReLU
- **Discriminator**: Multi-layer perceptron with LeakyReLU and Dropout
- **Training Method**: WGAN-GP (Wasserstein GAN with Gradient Penalty)

### Configurable Parameters:

#### Network Architecture:
- `embedding_dim` (int, default=128): Size of random noise input to generator
- `generator_dim` (tuple, default=(256, 256)): Hidden layer sizes for generator
- `discriminator_dim` (tuple, default=(256, 256)): Hidden layer sizes for discriminator

#### Training Hyperparameters:
- `generator_lr` (float, default=2e-4): Learning rate for generator
- `generator_decay` (float, default=1e-6): Weight decay for generator optimizer
- `discriminator_lr` (float, default=2e-4): Learning rate for discriminator
- `discriminator_decay` (float, default=1e-6): Weight decay for discriminator optimizer
- `batch_size` (int, default=500): Training batch size (must be even)
- `epochs` (int, default=300): Number of training epochs
- `discriminator_steps` (int, default=1): Discriminator updates per generator update

#### Advanced Options:
- `pac` (int, default=10): PacGAN - number of samples grouped together in discriminator
- `log_frequency` (bool, default=True): Use log frequency for categorical sampling
- `verbose` (bool, default=False): Show training progress
- `enable_gpu` (bool, default=True): Use GPU if available
- `cuda` (bool, deprecated): Legacy GPU option

### Model Variants via Configuration:

#### 1. **Shallow CTGAN** (Fast, less capacity)
```python
CTGAN(
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    embedding_dim=64,
    epochs=100
)
```

#### 2. **Deep CTGAN** (More capacity, slower)
```python
CTGAN(
    generator_dim=(512, 512, 512),
    discriminator_dim=(512, 512, 512),
    embedding_dim=256,
    epochs=500
)
```

#### 3. **Fast Training CTGAN**
```python
CTGAN(
    discriminator_steps=5,  # More discriminator updates
    batch_size=1000,
    epochs=100
)
```

#### 4. **High Quality CTGAN**
```python
CTGAN(
    generator_dim=(512, 512),
    discriminator_dim=(512, 512),
    embedding_dim=256,
    epochs=500,
    pac=10,
    log_frequency=True
)
```

#### 5. **Small Dataset CTGAN**
```python
CTGAN(
    batch_size=100,
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    epochs=500,  # More epochs for small data
    generator_decay=1e-5,  # Higher regularization
    discriminator_decay=1e-5
)
```

---

## 2. TVAE (Tabular Variational AutoEncoder)

### Architecture Components:
- **Encoder**: Multi-layer perceptron that outputs mean (μ) and variance (σ²)
- **Decoder**: Multi-layer perceptron that reconstructs data
- **Training Method**: Variational lower bound optimization (ELBO)

### Configurable Parameters:

#### Network Architecture:
- `embedding_dim` (int, default=128): Latent space dimension
- `compress_dims` (tuple, default=(128, 128)): Encoder hidden layer sizes
- `decompress_dims` (tuple, default=(128, 128)): Decoder hidden layer sizes

#### Training Hyperparameters:
- `l2scale` (float, default=1e-5): L2 regularization weight
- `batch_size` (int, default=500): Training batch size
- `epochs` (int, default=300): Number of training epochs
- `loss_factor` (float, default=2): Weight for reconstruction loss vs KL divergence

#### Advanced Options:
- `verbose` (bool, default=False): Show training progress
- `enable_gpu` (bool, default=True): Use GPU if available
- `cuda` (bool, deprecated): Legacy GPU option

### Model Variants via Configuration:

#### 1. **Shallow TVAE** (Fast, simple data)
```python
TVAE(
    embedding_dim=64,
    compress_dims=(64, 64),
    decompress_dims=(64, 64),
    epochs=100
)
```

#### 2. **Deep TVAE** (Complex data)
```python
TVAE(
    embedding_dim=256,
    compress_dims=(512, 256, 128),
    decompress_dims=(128, 256, 512),
    epochs=500
)
```

#### 3. **High Regularization TVAE** (Prevent overfitting)
```python
TVAE(
    l2scale=1e-4,
    loss_factor=3,
    epochs=300
)
```

#### 4. **Large Latent Space TVAE** (Capture more variation)
```python
TVAE(
    embedding_dim=512,
    compress_dims=(256, 256),
    decompress_dims=(256, 256),
    loss_factor=2
)
```

---

## Model Selection Guide

### Use **CTGAN** when:
- ✅ You have **medium to large datasets** (1000+ samples)
- ✅ Data has **complex multi-modal distributions**
- ✅ Need **high quality** synthetic samples
- ✅ Can afford **longer training time**
- ✅ Dataset has **many categorical variables**

### Use **TVAE** when:
- ✅ You have **small datasets** (<1000 samples)
- ✅ Need **faster training**
- ✅ Data has **continuous/numerical features** primarily
- ✅ Want **more stable training** (VAE vs GAN)
- ✅ Need **deterministic latent representations**

---

## Architecture Customization Matrix

### CTGAN Parameter Impact:

| Parameter | Small ↓ | Large ↑ | Impact |
|-----------|---------|---------|--------|
| `embedding_dim` | Faster | Better quality | Noise input diversity |
| `generator_dim` | Faster | More capacity | Model expressiveness |
| `discriminator_dim` | Faster | Better discrimination | Training quality |
| `epochs` | Faster | Better convergence | Training duration |
| `batch_size` | Stable gradients | Faster epochs | Memory vs speed |
| `pac` | Simpler | Better mode capture | Computation cost |
| `discriminator_steps` | Faster | Better discriminator | Generator/discriminator balance |

### TVAE Parameter Impact:

| Parameter | Small ↓ | Large ↑ | Impact |
|-----------|---------|---------|--------|
| `embedding_dim` | Simpler | More variation | Latent space richness |
| `compress_dims` | Faster | Better encoding | Encoder capacity |
| `decompress_dims` | Faster | Better decoding | Decoder capacity |
| `l2scale` | Less regularization | More regularization | Overfitting control |
| `loss_factor` | Prioritize KL | Prioritize reconstruction | VAE objective balance |

---

## Practical Configuration Examples

### Example 1: Adult Income Dataset (Your Use Case)
```python
# Recommended for Adult dataset (32k samples, mixed types)
CTGAN(
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    epochs=300,
    batch_size=500,
    pac=10,
    log_frequency=True,
    verbose=True
)
```

### Example 2: Small Medical Dataset
```python
# Better for <1000 samples
TVAE(
    embedding_dim=128,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    epochs=500,
    batch_size=50,
    l2scale=1e-4,
    verbose=True
)
```

### Example 3: Large Financial Dataset
```python
# For 100k+ samples with complex patterns
CTGAN(
    embedding_dim=256,
    generator_dim=(512, 512, 256),
    discriminator_dim=(512, 512, 256),
    epochs=500,
    batch_size=1000,
    pac=20,
    discriminator_steps=3,
    verbose=True
)
```

---

## Key Findings

1. **Only 2 base architectures** (CTGAN and TVAE), but **highly configurable**
2. **CTGAN is more flexible** with 13 tunable parameters
3. **TVAE is simpler** with 9 tunable parameters
4. **No pre-defined model variants** - users must configure manually
5. Both models share the same **data preprocessing pipeline** (DataTransformer)
6. Both support **GPU acceleration** via PyTorch
7. **PacGAN (pac parameter)** is unique to CTGAN for better mode coverage

---

## Recommendations for Your Project

Based on the Adult Income dataset characteristics:
- **Dataset size**: 32,561 samples → Good for CTGAN
- **Feature types**: Mixed (8 categorical, 6 numerical) → CTGAN handles better
- **Target**: Binary classification → Need high-quality class balance

**Optimal Configuration**:
```python
CTGAN(
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    batch_size=500,
    epochs=300,  # You used 10 - that's why quality was poor!
    pac=10,
    log_frequency=True,
    discriminator_steps=1,
    verbose=True,
    enable_gpu=True
)
```

This configuration balances training time and quality for your dataset size.
