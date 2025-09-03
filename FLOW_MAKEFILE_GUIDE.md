# SimpleRealNVP Makefile Usage Guide

## New Make Targets for SimpleRealNVP

You can now run your normalizing flow model independently with these new targets:

### Individual Experiments

#### BinarizedMNIST with SimpleRealNVP
```bash
# Cross-validation only (5 repeats × 10 folds = 50 models, ~2-3 hours)
make mem_flow_bmnist_cv

# Full training only (1 model, ~30 minutes)  
make mem_flow_bmnist_full

# Both CV and full (recommended)
make mem_flow_bmnist
```

#### CIFAR-10 with SimpleRealNVP
```bash
# Cross-validation only (3 repeats × 10 folds = 30 models, ~4-6 hours)
make mem_flow_cifar_cv

# Full training only (1 model, ~1.5 hours)
make mem_flow_cifar_full

# Both CV and full
make mem_flow_cifar
```

### Combined Targets

```bash
# Run both BinarizedMNIST and CIFAR-10 flow experiments
make mem_flow_all

# Just BinarizedMNIST (lightweight)
make mem_flow_light

# Ultra-quick test (just BinarizedMNIST CV)
make mem_flow_quick
```

### Analysis Targets

```bash
# Generate summary files for flow experiments
make flow_analysis

# Just BinarizedMNIST summary
make flow_quick_analysis
```

## Optimized Parameters for M4 MacBook

### BinarizedMNIST SimpleRealNVP
- **Architecture**: 4 layers, 256 hidden units
- **Training**: 20 epochs (CV), 50 epochs (full)
- **Batch size**: 64
- **Learning rate**: 1e-3
- **Repeats**: 5 (instead of 10 for faster completion)

### CIFAR-10 SimpleRealNVP  
- **Architecture**: 3 layers, 128 hidden units (lighter)
- **Training**: 30 epochs (CV), 75 epochs (full)
- **Batch size**: 32 (smaller for memory efficiency)
- **Learning rate**: 5e-4 (more conservative)
- **Repeats**: 3 (lighter workload)

## Running Individual Experiments by Model Type

### Existing VAE Models
```bash
# BinarizedMNIST with different learning rates
make mem_mnist_lr3     # 1e-3 learning rate
make mem_mnist_lr4     # 1e-4 learning rate

# CIFAR-10 with DiagonalGaussianDCVAE
make mem_cifar10

# CelebA with ConstantGaussianDCVAE  
make mem_celeba
```

### Your New Flow Model
```bash
# All flow experiments
make mem_flow_all

# Individual datasets
make mem_flow_bmnist
make mem_flow_cifar
```

## Recommended Workflow for M4 MacBook

1. **Start Small** (15 minutes):
   ```bash
   make mem_flow_quick
   ```

2. **Light Experiment** (2-3 hours):
   ```bash
   make mem_flow_light
   ```

3. **Full Flow Analysis** (6-8 hours):
   ```bash
   make mem_flow_all
   make flow_analysis
   ```

4. **Compare with VAEs** (add original experiments):
   ```bash
   make mem_mnist_lr3  # This will take much longer (10+ hours)
   ```

## File Organization

Results will be stored in:
- `results/flow_bmnist/results/` - BinarizedMNIST flow results
- `results/flow_cifar10/results/` - CIFAR-10 flow results
- `summaries/mem_flow_*.npz` - Processed memorization scores

## Limitations & Notes

1. **No CelebA Flow**: Not included yet (would need color image handling)
2. **Reduced Repeats**: Flow experiments use fewer repeats for faster completion
3. **Conservative Parameters**: Optimized for M4 MacBook, not maximum performance
4. **Sequential Execution**: Make runs one experiment at a time (no parallelization)

## Troubleshooting

If you get memory errors, try:
- Reducing batch size: `--batch-size 32` → `--batch-size 16`
- Fewer layers: Edit Makefile to use `L2-H128` instead of `L4-H256`
- Shorter training: Reduce `--epochs`

## Comparison with Paper

- **Paper**: 10 repeats × 10 folds × 100 epochs = massive computational load
- **Your Setup**: 3-5 repeats × 10 folds × 20-50 epochs = manageable on laptop
- **Trade-off**: Less statistical power but same methodology
