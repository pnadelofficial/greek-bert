# Greek BERT Training Optimizations - Short-Term Changes

## Overview
This document summarizes the RoBERTa-inspired optimizations applied to the Greek BERT training pipeline. These changes are based on best practices from:
- **RoBERTa**: "A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
- **ModernBERT**: HuggingFace's state-of-the-art encoder architecture

## Changes Made

### 1. Dynamic Masking ✅
**File**: `scripts/utils.py` - `mlm_masking()` function

**Before**: Static masking - masks were generated once per batch using numpy random number generator.

**After**: Dynamic masking - fresh masks are generated on every forward pass using PyTorch's `torch.rand()`.

**Why**: RoBERTa found this simple change alone improved performance by ~0.5% on GLUE benchmarks. Each training step sees different mask patterns, preventing overfitting to specific masks.

**Code change**:
```python
# Old: numpy-based static masking
rand = np_rng.random(seq.shape)
mask_mask = rand < mask_prob * 0.8

# New: PyTorch-based dynamic masking
mask = torch.rand_like(input_ids, dtype=torch.float32) < mask_prob
```

---

### 2. Increased Learning Rate ✅
**File**: `scripts/train_config.yaml`

**Before**: `lr: 1e-5`, `max_lr: 1e-4`

**After**: `lr: 6e-4`, `max_lr: 6e-4`

**Why**: RoBERTa used 6e-4 for base models. The higher learning rate (with proper warmup) leads to faster convergence and better final performance.

**Note**: The OneCycleLR scheduler will linearly warm up from 0 to 6e-4 over 5% of training steps.

---

### 3. Adam β₂ = 0.98 ✅
**File**: `scripts/train.py` - optimizer initialization

**Before**: Default AdamW `betas=(0.9, 0.999)`

**After**: `betas=(0.9, 0.98)`

**Why**: RoBERTa found that β₂=0.98 (instead of 0.999) improves stability when training with large batch sizes. This is a key RoBERTa optimization.

**Code change**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=float(hp_config.lr),
    weight_decay=float(hp_config.weight_decay),
    betas=(0.9, 0.98)  # RoBERTa: β₂=0.98 for stability
)
```

---

### 4. Increased Masking Rate ✅
**File**: `scripts/train_config.yaml`

**Before**: `mask_prob: 0.15` (15%)

**After**: `mask_prob: 0.30` (30%)

**Why**: ModernBERT uses 30% masking, which works well with dynamic masking. Higher masking rate means more learning signal per step.

**Note**: You can experiment with 0.15 (RoBERTa) vs 0.30 (ModernBERT) to see which works better for Greek.

---

## Expected Impact

| Change | Expected Improvement | Difficulty |
|--------|---------------------|------------|
| Dynamic masking | ~0.5% GLUE improvement | Easy ✅ |
| Higher LR | Faster convergence | Easy ✅ |
| β₂=0.98 | Better stability | Easy ✅ |
| Higher masking | More learning signal | Easy ✅ |

**Combined**: These changes should provide noticeable improvements in both training loss convergence and downstream task performance.

---

## Files Modified

1. `scripts/utils.py` - Dynamic masking implementation
2. `scripts/train.py` - Optimizer configuration
3. `scripts/train_config.yaml` - Hyperparameter updates

---

## Next Steps

### Immediate (run these changes):
```bash
cd /cluster/tufts/perseuslab/pnadel01/greek-bert
python scripts/train.py --config-path scripts/train_config.yaml
```

### Monitor:
1. **Training loss** - Should converge faster with higher LR
2. **Validation loss** - Compare with previous runs
3. **Check for instability** - With higher LR, watch for loss spikes

### Medium-term improvements (not yet implemented):
1. **More training data** - Current ~3.5GB vs RoBERTa's 160GB
2. **More training steps** - RoBERTa trained for 500k steps
3. **Sequence packing** - Reduce padding waste
4. **Longer sequences** - RoBERTa trained on 512-token sequences (you're already doing this)

### Long-term improvements:
1. **ModernBERT architecture** - RoPE, GeGLU, alternating attention
2. **Multi-phase training** - Short context → long context
3. **Data diversity** - Add more Greek corpora

---

## Hyperparameter Reference

| Parameter | Your Setup | RoBERTa | ModernBERT |
|-----------|-----------|---------|------------|
| Learning Rate | 6e-4 | 6e-4 | 1e-4 |
| Warmup | 5% | 5% | N/A |
| Batch Size | 16 | 8000 | Large |
| Weight Decay | 0.01 | 0.01 | N/A |
| Adam β₂ | 0.98 | 0.98 | N/A |
| Mask Prob | 0.30 | 0.15 | 0.30 |
| Masking | Dynamic | Dynamic | Dynamic |
| Gradient Clip | 1.0 | 0.0 | N/A |

---

## References

1. **RoBERTa Paper**: https://arxiv.org/abs/1907.11692
2. **ModernBERT Blog**: https://huggingface.co/blog/modernbert
3. **RoBERTa GitHub**: https://github.com/pytorch/fairseq

---

## Troubleshooting

If you see training instability:
1. Reduce `max_lr` to 3e-4 temporarily
2. Enable gradient clipping: `max_norm=0.1` in `clip_grad_norm_`
3. Reduce `mask_prob` to 0.15

If training is too slow:
1. Increase `batch_size` if GPU memory allows
2. Enable gradient accumulation: `gradient_accumulation_steps: 2`
3. Consider reducing `num_epochs` and increasing training steps instead
