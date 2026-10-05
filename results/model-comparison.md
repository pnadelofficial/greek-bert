# 4-model comparison (compiled 2026-10-05 from logs_from_cluster/)

Source: `logs_from_cluster/{greekbert-continuation,real-aristoberto-continuation,fresh-bert-control,fresh-modernbert}/{metrics.json,eval_results.json}`.

## ⚠️ Data quality caveat — read before using any WSD number below

The WSD "baseline" comparison is only correct for **model 2**. Models 1, 3,
and 4's WSD stage recorded its baseline model as `external-aristoBERTo` (the
old, mislabeled path -- actually GreekBERT, not aristoBERTo; see
`wsd/wsd.py`'s `DEFAULT_BASELINE` comment and `configs/train.yaml`'s IDENTITY
CORRECTION note). Model 2's WSD stage correctly used
`external-aristoberto-real`. **Rerun WSD for models 1, 3, and 4 before using
any "vs. aristoBERTo" number in the paper** -- `git pull` first to make sure
the fix is actually present on the cluster, then:

```bash
RUN_NAME=arm1-aristo-continuation STAGES=wsd sbatch slurm/posttrain.sh
RUN_NAME=arm2-bert-fresh          STAGES=wsd sbatch slurm/posttrain.sh
RUN_NAME=arm4-modernbert-scratch  STAGES=wsd sbatch slurm/posttrain.sh
```

Pretraining and spaCy numbers below are unaffected (neither stage touches the
baseline model) and are final as reported.

## Pretraining

| Model | RUN_NAME | Architecture | Weights | best_epoch | best_val_loss | best_val_ppl |
|---|---|---|---|---|---|---|
| 1 — GreekBERT continuation | `arm1-aristo-continuation` | BERT-base | GreekBERT continued | 40 | 1.5818 | 4.864 |
| 2 — real aristoBERTo continuation | `model2-aristo-continuation` | BERT-base | aristoBERTo continued | 40 | 1.6082 | 4.994 |
| 3 — fresh BERT (control) | `arm2-bert-fresh` | BERT-base | random | 39 | 5.5875 | 267.07 |
| 4 — fresh ModernBERT | `arm4-modernbert-scratch` | ModernBERT | random | 39 | 1.8140 | 6.135 |

Model 3's much higher loss/PPL is expected, not a bug -- it's the random-init
control at the same token budget as models 1/2/4, and the gap is exactly the
"how much does prior knowledge buy you" signal the control exists to produce.
(We previously found and fixed a real LR-divergence bug for this run; the
full 40-epoch curve now decreases monotonically with no recurrence of the
climb-then-plateau pattern -- it plateaus early at a much higher loss than
the others, which is a capacity/budget story, not instability.)

## spaCy downstream (PROIEL+Perseus combined eval)

| Model | TAG | POS | MORPH | LEMMA | UAS | LAS | SENT_F |
|---|---|---|---|---|---|---|---|
| 1 — GreekBERT continuation | 98.55 | 98.54 | 94.72 | 88.21 | 87.31 | 83.76 | 75.51 |
| 2 — real aristoBERTo continuation | 98.51 | 98.48 | 94.76 | 88.22 | 86.59 | 82.85 | 70.82 |
| 3 — fresh BERT (control) | 92.05 | 91.63 | 82.49 | 81.12 | 64.53 | 57.81 | 20.56 |
| 4 — fresh ModernBERT | 97.63 | 97.36 | 92.38 | 88.64 | 80.64 | 75.77 | 59.37 |

All percentages. Lemma numbers here are WITHOUT the odyCy-style lexicon layer
(`custom_components/lemmatizer.py` / `build_lemma_lexicon.py`) -- that's
still the planned final pass once this table is otherwise locked.


## WSD (glaux harmonia/kosmos, 5-seed mean, best-epoch)

| Model | "ours" harmonia | "ours" kosmos | baseline harmonia | baseline kosmos | baseline model used |
|---|---|---|---|---|---|
| 1 — GreekBERT continuation | 0.8481 | 0.8960 | 0.8241 | 0.8705 | ⚠️ `external-aristoBERTo` (WRONG -- GreekBERT) |
| 2 — real aristoBERTo continuation | 0.8407 | 0.8982 | 0.8296 | 0.8749 | ✅ `external-aristoberto-real` |
| 3 — fresh BERT (control) | 0.8074 | 0.8291 | 0.8093 | 0.8742 | ⚠️ `external-aristoBERTo` (WRONG -- GreekBERT) |
| 4 — fresh ModernBERT | 0.7778 | 0.8742 | 0.8278 | 0.8691 | ⚠️ `external-aristoBERTo` (WRONG -- GreekBERT) |

Don't draw "beats aristoBERTo" conclusions from the ⚠️ rows yet -- rerun those
three first.
