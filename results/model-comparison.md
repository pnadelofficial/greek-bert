# 5-model comparison (compiled 2026-10-13 from logs_from_cluster/)

Source: `logs_from_cluster/{greekbert-continuation,real-aristoberto-continuation,fresh-bert-control,fresh-modernbert,fresh-greekbert-continuation}/{metrics.json,eval_results.json}`.
(Despite its directory name, `fresh-greekbert-continuation` is model 5 --
`RUN_NAME=modernbert-stage3-final`, confirmed via its `manifest.json`: three-
stage ModernBERT continuation, `pretrained_model` pointing at
`modernbert-stage2-ancient/hf_format`.)

## Data quality note (resolved)

All 5 models' WSD stages now correctly use `external-aristoberto-real` as the
baseline (confirmed via each run's `manifest.json`). Earlier runs for models
1, 3, and 4 had recorded a mislabeled baseline (`external-aristoBERTo`,
actually GreekBERT, not aristoBERTo -- see `wsd/wsd.py`'s `DEFAULT_BASELINE`
comment and `configs/train.yaml`'s IDENTITY CORRECTION note); all three have
since been rerun. The tables below are final.

## Pretraining

| Model | RUN_NAME | Architecture | Weights | best_epoch | best_val_loss | best_val_ppl |
|---|---|---|---|---|---|---|
| 1 — GreekBERT continuation | `arm1-aristo-continuation` | BERT-base | GreekBERT continued | 40 | 1.5818 | 4.864 |
| 2 — real aristoBERTo continuation | `model2-aristo-continuation` | BERT-base | aristoBERTo continued | 40 | 1.6082 | 4.994 |
| 3 — fresh BERT (control) | `arm2-bert-fresh` | BERT-base | random | 39 | 5.5875 | 267.07 |
| 4 — fresh ModernBERT | `arm4-modernbert-scratch` | ModernBERT | random | 39 | 1.8140 | 6.135 |
| 5 — ModernBERT, aristoBERTo-shaped history | `modernbert-stage3-final` | ModernBERT | modern Greek → First1KGreek → new data | 38 | 1.9469 | 7.007 |

Model 3's much higher loss/PPL is expected, not a bug -- it's the random-init
control at the same token budget as models 1/2/4, and the gap is exactly the
"how much does prior knowledge buy you" signal the control exists to produce.
(We previously found and fixed a real LR-divergence bug for this run; the
full 40-epoch curve now decreases monotonically with no recurrence of the
climb-then-plateau pattern -- it plateaus early at a much higher loss than
the others, which is a capacity/budget story, not instability.)

Model 5's pretrain loss/PPL (this is stage 3 only -- its own final
continuation step) is somewhat higher than model 4's single-stage fresh
ModernBERT (7.01 vs 6.14 PPL) and model 2's (4.99). Worth keeping in mind
when interpreting its downstream numbers below: more pretraining *history*
(three stages) didn't translate into a lower final MLM loss than model 4's
simpler one-stage run on this metric alone -- the real comparison is in the
downstream suite, which is what model 5 was built to answer.

## spaCy downstream (PROIEL+Perseus combined eval)

| Model | TAG | POS | MORPH | LEMMA | UAS | LAS | SENT_F |
|---|---|---|---|---|---|---|---|
| 1 — GreekBERT continuation | 98.55 | 98.54 | 94.72 | 88.21 | 87.31 | 83.76 | 75.51 |
| 2 — real aristoBERTo continuation | 98.51 | 98.48 | 94.76 | 88.22 | 86.59 | 82.85 | 70.82 |
| 3 — fresh BERT (control) | 92.05 | 91.63 | 82.49 | 81.12 | 64.53 | 57.81 | 20.56 |
| 4 — fresh ModernBERT | 97.63 | 97.36 | 92.38 | 88.64 | 80.64 | 75.77 | 59.37 |
| 5 — ModernBERT, aristoBERTo-shaped history | 98.06 | 97.82 | 92.94 | 88.83 | 82.94 | 78.67 | 63.89 |

All percentages. Lemma numbers here are WITHOUT the odyCy-style lexicon layer
(`custom_components/lemmatizer.py` / `build_lemma_lexicon.py`) -- that's
still the planned final pass once this table is otherwise locked.

Model 5 beats model 4 on every spaCy metric (e.g. SENT_F 63.89 vs 59.37, LAS
78.67 vs 75.77) despite a slightly worse raw pretrain PPL -- consistent with
the architecture-comparison story model 5 exists to test: matching model 2's
training-history shape helps ModernBERT's downstream performance even though
it doesn't necessarily win on the pretraining objective itself. Still behind
models 1/2 (the BERT continuations) on UAS/LAS/SENT_F specifically, though
closer than model 4 was.

## WSD (glaux harmonia/kosmos, 5-seed mean, best-epoch)

| Model | "ours" harmonia | "ours" kosmos | baseline harmonia | baseline kosmos | baseline model used |
|---|---|---|---|---|---|
| 1 — GreekBERT continuation | 0.8296 | 0.8916 | 0.8204 | 0.8756 | ✅ `external-aristoberto-real` |
| 2 — real aristoBERTo continuation | 0.8407 | 0.8982 | 0.8296 | 0.8749 | ✅ `external-aristoberto-real` |
| 3 — fresh BERT (control) | 0.7963 | 0.8298 | 0.8148 | 0.8684 | ✅ `external-aristoberto-real` |
| 4 — fresh ModernBERT | 0.7870 | 0.8778 | 0.8148 | 0.8705 | ✅ `external-aristoberto-real` |
| 5 — ModernBERT, aristoBERTo-shaped history | 0.7944 | 0.8865 | 0.8259 | 0.8727 | ✅ `external-aristoberto-real` |

Model 5 vs. model 4 (now both on the corrected baseline): a small win on
both words (harmonia 0.794 vs 0.787, kosmos 0.887 vs 0.878) -- directionally
consistent with the spaCy results, though WSD's small eval sets (n=27/31)
mean this shouldn't be over-read on its own.
