# Greek BERT Pretraining Guide

## Quick Start

```bash
cd /cluster/tufts/perseuslab/pnadel01/greek-bert

# Run single GPU
python scripts/train.py --config-path scripts/train_config.yaml

# Run multi-GPU (example with 4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --config-path scripts/train_config.yaml
```

## RoBERTa Optimizations Implemented

### 1. Dynamic Masking ✅
- Masks are regenerated on every forward pass
- Prevents overfitting to specific mask patterns
- ~0.5% GLUE improvement (RoBERTa paper)

### 2. Higher Learning Rate ✅
- Base LR: 6e-4 (vs 1e-4 before)
- 5% linear warmup
- Faster convergence

### 3. Adam β₂ = 0.98 ✅
- More stable with large batches
- Key RoBERTa optimization

### 4. 30% Masking Rate ✅
- ModernBERT-style masking
- More learning signal per step

## Configuration

### Key Hyperparameters (`scripts/train_config.yaml`)

```yaml
# Learning rate
lr: 6e-4              # Base learning rate
max_lr: 6e-4          # Peak after warmup
pct_start: 0.05       # 5% warmup

# MLM masking
mask_prob: 0.30       # 30% masking (ModernBERT)

# Batch size
batch_size: 16        # Per-GPU
gradient_accumulation_steps: 1

# Optimizer
weight_decay: 0.01    # AdamW weight decay
# betas=(0.9, 0.98)  # Fixed in train.py

# Training
num_epochs: 20        # Number of epochs
use_mixed_precision: True
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=scripts/runs
```

### Checkpoints
- Saved every 5 epochs
- Location: `scripts/checkpoints/`
- Format: `checkpoint_epoch_N.pt`

## Expected Training Time

With current ~3.5GB dataset:
- 1 epoch: ~10-20 minutes (single GPU)
- 20 epochs: ~4-8 hours (single GPU)
- Adjust `num_epochs` based on compute budget

## Troubleshooting

### Loss spikes at start
```yaml
# Reduce learning rate temporarily
max_lr: 3e-4
```

### OOM errors
```yaml
# Reduce batch size
batch_size: 8
```

### Slow convergence
```yaml
# Increase training steps
num_epochs: 30
```

## Next Steps (Medium-Term)

1. **Add more data** - Current ~3.5GB vs RoBERTa's 160GB
2. **Sequence packing** - Reduce padding waste
3. **Gradient accumulation** - Simulate larger batches
4. **Longer training** - 100k+ steps

## References

- RoBERTa Paper: https://arxiv.org/abs/1907.11692
- ModernBERT: https://huggingface.co/blog/modernbert
- Original GreekBERT: https://github.com/nlpaueb/GreekBERT
