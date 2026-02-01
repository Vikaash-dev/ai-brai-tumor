# PHOENIX-v3.1 Implementation Analysis & Optimization Plan

**Self-Review Date:** February 2026  
**Status:** Critical Analysis Phase

---

## 1. Gap Analysis: Current vs. Required Implementation

### 1.1 What PHOENIX-v3.1 Paper Requires

| Component | Paper Requirement | Current Status | Gap |
|-----------|------------------|----------------|-----|
| **SpatialMixer** | Depthwise-separable before flattening | ✅ Implemented | Minor optimizations needed |
| **Hybrid Pyramid** | Conv2D (high-res) + SSM (low-res) | ❌ Not implemented | Need full architecture |
| **Liquid-S6-KAN** | Combined SSM + KAN backbone | ⚠️ Partial (separate) | Need integration |
| **Multi-Spectral Concordance Gating** | Frequency-domain fusion | ❌ Not implemented | Critical gap |
| **Symbolic Mirror TTT** | Test-time spline adaptation | ⚠️ Partial | Need entropy minimization |
| **True 2.5D Loading** | [z-1, z, z+1] volumetric context | ❌ Not implemented | Critical gap |
| **Priority Scout** | ROI-aware time constants | ⚠️ Basic | Need full integration |

### 1.2 Critical Issues Identified

#### Issue 1: B-Spline Basis Computation (KAN Layer)
**Problem**: Recursive B-spline computation is inefficient and may have numerical issues.

**Current Code** (`models/kan_layer.py` line 76-109):
```python
def _basis_function(self, x, i, k, knots):
    if k == 0:
        return tf.cast((knots[i] <= x) & (x < knots[i + 1]), tf.float32)
    # Recursive formula - INEFFICIENT!
```

**Fix**: Use Cox-de Boor iterative algorithm instead of recursion.

#### Issue 2: SSM Selective Scan Not Parallelized
**Problem**: Using `tf.scan` is sequential and slow.

**Current Code** (`models/liquid_ssm.py` line 142-193):
```python
# Sequential scan (could be parallelized with associative scan)
x = tf.scan(scan_fn, (deltaA_t, deltaB_u_t), initializer=x0)
```

**Fix**: Implement parallel associative scan for O(log N) depth.

#### Issue 3: Missing Multi-Spectral Concordance Gating
**Problem**: No frequency-domain fusion for multi-modal MRI.

**Required**: FFT-based concordance detection across T1/T2/FLAIR modalities.

#### Issue 4: No True 2.5D Data Loading
**Problem**: Current loader uses 2D slices without volumetric context.

**Required**: Input tensors as [z-1, z, z+1] for temporal/volumetric context.

---

## 2. Research-Based Optimizations

### 2.1 From Official Mamba Paper (arXiv:2312.00752)

**Key Insight**: The selective scan should use **parallel associative scan** for efficiency.

```
Parallel Scan Algorithm:
- Divide sequence into chunks
- Compute partial results in parallel
- Combine using associative property
- Complexity: O(N) work, O(log N) depth
```

### 2.2 From Vision Mamba (Vim) Paper (arXiv:2401.09417)

**Key Insight**: Bidirectional SSM processing improves vision tasks.

```
Forward SSM: processes left-to-right
Backward SSM: processes right-to-left
Fusion: Concatenate or add both directions
```

### 2.3 From U-KAN Paper (arXiv:2406.02918)

**Key Insight**: Efficient KAN uses radial basis functions (RBF) instead of B-splines.

```
RBF-KAN: φ(x) = exp(-||x - c||² / 2σ²)
- Faster computation
- Better extrapolation
- Smoother gradients
```

### 2.4 From nnU-Net (Nature Methods 2021)

**Key Insight**: Patch-based training with overlapping inference.

```
Training: Random 3D patches
Inference: Sliding window with Gaussian weighting
Benefit: Handles arbitrary input sizes
```

---

## 3. Optimized Implementation Plan

### Phase 1: Fix Core Components

1. **Optimize B-Spline Computation**
   - Implement Cox-de Boor iterative algorithm
   - Add caching for repeated evaluations
   - Consider RBF as alternative

2. **Parallelize SSM Scan**
   - Implement parallel associative scan
   - Add bidirectional processing option
   - Use chunked processing for memory efficiency

3. **Add Multi-Spectral Concordance Gating**
   - Implement 2D FFT for frequency analysis
   - Create concordance detection mechanism
   - Fuse modalities in frequency domain

### Phase 2: Build Hybrid Architecture

1. **Create Hybrid Pyramid**
   - Stage 1-2: ResBlock with Conv2D (224×224, 112×112)
   - Stage 3-4: Liquid-S6-KAN (56×56, 28×28)
   - Adaptive routing based on resolution

2. **Implement True 2.5D Loading**
   - Create NIfTI/DICOM volume loader
   - Extract [z-1, z, z+1] triplets
   - Handle boundary conditions

### Phase 3: Test-Time Adaptation

1. **Complete Symbolic Mirror TTT**
   - Entropy computation on predictions
   - Gradient descent on KAN spline weights only
   - Early stopping based on entropy convergence

---

## 4. Performance Targets

| Metric | Current (Est.) | Target | Mechanism |
|--------|----------------|--------|-----------|
| **Dice (WT)** | ~92% | >93% | Multi-Spectral Gating |
| **Dice (ET)** | ~84% | >87% | Liquid-SSM Semantics |
| **Hausdorff95** | ~5mm | <3.5mm | SpatialMixer Topology |
| **Parameters** | ~2.2M | ~1.18M | Hybrid KAN Efficiency |
| **Inference** | ~50ms | ~12ms | Parallel SSM + Conv2D |

---

## 5. Implementation Priority

### Critical (Must Have)
1. ✅ SpatialMixer (implemented, needs testing)
2. 🔄 Hybrid Pyramid Architecture (in progress)
3. ❌ Multi-Spectral Concordance Gating
4. ❌ True 2.5D Data Loading

### Important (Should Have)
1. ⚠️ Parallel Associative Scan
2. ⚠️ RBF-KAN Alternative
3. ⚠️ Bidirectional SSM

### Nice to Have
1. Symbolic Mirror TTT (full implementation)
2. ONNX/TFLite Export
3. Multi-GPU Training

---

## 6. Next Steps

1. **Create `models/model_v3_1.py`** - Full Hybrid Pyramid architecture
2. **Create `src/data/loader_true_2_5d.py`** - Volumetric data pipeline
3. **Create `models/spectral_gating.py`** - MSCG module
4. **Update tests and documentation**

---

**Analysis Complete. Proceeding with optimized implementation.**
