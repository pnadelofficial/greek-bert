# RoBERTa vs. Greek BERT - Configuration Comparison

## Summary of Changes

| Component | Before (Original) | After (RoBERTa-Optimized) | Impact |
|-----------|------------------|---------------------------|--------|
| **Masking Strategy** | Static (numpy RNG) | Dynamic (PyTorch) | ~0.5% GLUE |
| **Learning Rate** | 1e-5 base, 1e-4 max | 6e-4 base & max | Faster convergence |
| **Adam β₂** | 0.999 (default) | 0.98 | Better stability |
| **Masking Rate** | 15% | 30% | More signal |
| **Weight Decay** | 0.01 | 0.01 | ✅ Unchanged |
| **Gradient Clip** | 1.0 | 1.0 | ⚠️ Could remove |

## Detailed Changes

### 1. Dynamic Masking

**Before** (`scripts/utils.py` - old version):
```python
def mlm_masking(input_ids, np_rng, mask_prob=0.3, ...):
    seq = input_ids.cpu().numpy()
    rand = np_rng.random(seq.shape)  # Static mask per batch
    mask_mask = rand < mask_prob * 0.8
    # ...
```

**After** (current version):
```python
def mlm_masking(input_ids, mask_token_id, mask_prob=0.15, ...):
    mask = torch.rand_like(input_ids, dtype=torch.float32) < mask_prob
    # Fresh mask every forward pass
```

**Why it matters**: RoBERTa paper showed this simple change alone improved performance by ~0.5% on GLUE benchmarks. Each training step sees different mask patterns, preventing the model from "memorizing" specific masks.

---

### 2. Learning Rate

**Before** (`scripts/train_config.yaml` - old version):
```yaml
lr: 1e-5
max_lr: 1e-4
pct_start: 0.05
```

**After** (current version):
```yaml
lr: 6e-4
max_lr: 6e-4
pct_start: 0.05
```

**Why it matters**: RoBERTa used 6e-4 for base models. The higher learning rate (with proper warmup) leads to:
- Faster convergence
- Better final performance
- More efficient use of training steps

**Note**: OneCycleLR will linearly warm up from 0 to 6e-4 over 5% of training steps.

---

### 3. Adam β₂ Parameter

**Before** (`scripts/train.py` - old version):
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=float(hp_config.lr),
    weight_decay=float(hp_config.weight_decay)
    # Default betas=(0.9, 0.999)
)
```

**After** (current version):
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=float(hp_config.lr),
    weight_decay=float.h