# Cross-Analysis: Research & Repos for Brain Tumor MRI AI

## Goal
Map foundational components (data → model → evaluation) to key research papers and open-source implementations relevant to brain tumor MRI detection/segmentation.

## Core Tasks
- **Segmentation (primary)**: Whole/enhancing/necrotic tumor regions.
- **Classification (secondary)**: Tumor presence/type.

## Data Foundations
- **BraTS (2018–2023)**: Multimodal MRI (T1, T1ce, T2, FLAIR), standard benchmark for glioma segmentation.
- **Preprocessing essentials**: skull stripping, bias-field (N4) correction, intensity normalization per modality, consistent resampling/cropping, data leakage prevention (patient-level splits).

## Research Papers (Representative)
- **U-Net & Variants**: U-Net (Ronneberger et al.), nnU-Net (Isensee et al.) for strong, self-configuring baselines.
- **Transformers / SSMs**: Swin-UNETR (Hatamizadeh et al.), VM-UNet (Vision Mamba), SegMamba; focus on long-range context with linear-time attention/SSM.
- **KAN / Hybrid**: KAN-based medical models for spline-based expressivity with fewer parameters.
- **Optimization & Stability**: Adan / Lion optimizers; mixed precision (AMP); cosine/OneCycle schedulers; gradient clipping.
- **Robustness & TTA**: Test-time augmentation; entropy minimization; batch-stat updates; uncertainty via MC-dropout.

## Open-Source Repos (Representative)
- **nnU-Net**: End-to-end baseline with preprocessing, training, and evaluation pipelines; strong default for medical segmentation.
- **MONAI**: Toolkit with medical-specific transforms, networks (UNet/UNETR/SwinUNETR), metrics (Dice/HD95), and handlers.
- **Swin-UNETR / VM-UNet / SegMamba**: Reference implementations for transformer/SSM-based medical segmentation.
- **TorchIO / Rising / batchgenerators**: Data I/O and augmentation utilities for medical imaging.

## Fundamental Building Blocks (What to Bring In)
1. **Data Pipeline**
   - Deterministic patient-level splits; modality-consistent normalization; N4 + skull stripping; patch or 2.5D slicing; leakage checks.
2. **Model Baselines**
   - Start with **2D/3D U-Net** (or nnU-Net config) as a reproducible anchor.
   - Add **Transformer/SSM** variant (e.g., Swin-UNETR or VM-UNet) for long-range context.
3. **Training Loop**
   - Loss: Dice + Cross-Entropy/Focal; optional boundary loss.
   - Schedulers: cosine or OneCycle; AMP; gradient clipping; early stopping.
4. **Evaluation**
   - Metrics: Dice (per class + mean), HD95; report mean ± std over folds.
   - Validation: k-fold or held-out; ensure patient-level non-leakage.
5. **Robustness & Adaptation**
   - Test-time augmentation; optional entropy-min TTA; uncertainty (MC-dropout).
6. **Reproducibility**
   - Seeds; config files; `requirements.txt`/`pyproject`; log runs (CSV/TensorBoard).
7. **Governance**
   - Dataset licensing and usage notes; privacy considerations for clinical MRI.

## Suggested Minimal Path for This Repo
1. Define task scope (segmentation focus on BraTS) and add data usage notes.
2. Add structure: `src/{models,training,data,utils}`, `configs/`, `requirements.txt`.
3. Implement nnU-Net-inspired baseline (or MONAI UNet) with a small training/eval script.
4. Add metrics logging (Dice/HD95) and deterministic splits.
5. Incrementally prototype a transformer/SSM variant (e.g., Swin-UNETR or VM-UNet) once baseline is stable.

This cross-analysis is descriptive only; no external code is added. It aligns the clean-slate repo with proven research and open-source implementations.
