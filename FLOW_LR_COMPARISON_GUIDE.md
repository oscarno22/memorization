# Flow Learning Rate Comparison Guide

## Overview

The Makefile has been modified to support learning rate comparison experiments specifically for normalizing flows (SimpleRealNVP). This allows you to study how different learning rates affect memorization patterns in flow models.

## New Make Targets

### Individual Learning Rate Experiments

#### Learning Rate 1e-3 (Higher)
```bash
# Cross-validation only (5 repeats × 10 folds = 50 models)
make mem_flow_bmnist_lr3_cv

# Full training only (1 model)
make mem_flow_bmnist_lr3_full

# Both CV and full training
make mem_flow_bmnist_lr3
```

#### Learning Rate 1e-4 (Lower)
```bash
# Cross-validation only (5 repeats × 10 folds = 50 models)
make mem_flow_bmnist_lr4_cv

# Full training only (1 model)
make mem_flow_bmnist_lr4_full

# Both CV and full training
make mem_flow_bmnist_lr4
```

### Combined Targets

```bash
# Run both learning rate experiments (UPDATED)
make mem_flow_light

# Generate summary files for learning rate comparison
make flow_lr_analysis
```

## Complete Workflow

### Step 1: Run the learning rate comparison experiments
```bash
make mem_flow_light
```
This will run both:
- SimpleRealNVP with lr=1e-3 (stored in `results/flow_bmnist_lr3/`)
- SimpleRealNVP with lr=1e-4 (stored in `results/flow_bmnist_lr4/`)

### Step 2: Generate summary files
```bash
make flow_lr_analysis
```
This creates:
- `summaries/mem_flow_bmnist_lr3.npz`
- `summaries/mem_flow_bmnist_lr4.npz`

### Step 3: Run memorization analysis
```bash
python scripts/analysis_figure_mem_histograms.py \
    --lr3-file summaries/mem_flow_bmnist_lr3.npz \
    --lr4-file summaries/mem_flow_bmnist_lr4.npz \
    -o output/flow_memorization_comparison.tex
```

### Step 4: Generate other comparisons
```bash
# Loss evolution comparison
python scripts/analysis_figure_losses.py \
    --lr3-file summaries/mem_flow_bmnist_lr3.npz \
    --lr4-file summaries/mem_flow_bmnist_lr4.npz \
    -o output/flow_losses_comparison.tex

# Memorization quantiles
python scripts/analysis_figure_mem_quantiles.py \
    summaries/mem_flow_bmnist_lr3.npz \
    -o output/flow_lr3_quantiles.tex

python scripts/analysis_figure_mem_quantiles.py \
    summaries/mem_flow_bmnist_lr4.npz \
    -o output/flow_lr4_quantiles.tex
```

## File Structure

After running the experiments, you'll have:
```
results/
├── flow_bmnist_lr3/
│   ├── results/        # CV results for lr=1e-3
│   └── checkpoints/    # Model checkpoints for lr=1e-3
└── flow_bmnist_lr4/
    ├── results/        # CV results for lr=1e-4
    └── checkpoints/    # Model checkpoints for lr=1e-4

summaries/
├── mem_flow_bmnist_lr3.npz    # Aggregated results for lr=1e-3
└── mem_flow_bmnist_lr4.npz    # Aggregated results for lr=1e-4
```

## Experiment Details

**Model**: SimpleRealNVP (4 layers, 256 hidden units)
**Dataset**: BinarizedMNIST
**Architecture**: L4-H256 (4 coupling layers, 256 hidden dimensions)

### Learning Rate 1e-3:
- Cross-validation: 5 repeats × 10 folds = 50 models
- Training: 20 epochs each
- Full model: 50 epochs

### Learning Rate 1e-4:
- Cross-validation: 5 repeats × 10 folds = 50 models  
- Training: 20 epochs each
- Full model: 50 epochs

## Key Research Questions

This setup allows you to investigate:

1. **How does learning rate affect memorization in normalizing flows?**
2. **Do flows show similar memorization patterns to VAEs when trained with different learning rates?**
3. **What is the relationship between learning rate and likelihood overfitting in flows?**
4. **How do the exact likelihood estimates from flows compare to VAE approximations?**

## Next Steps

1. Run `make mem_flow_light` (will take several hours)
2. Run `make flow_lr_analysis` 
3. Use existing analysis scripts to compare memorization patterns
4. Generate publication-quality figures showing flow memorization behavior
