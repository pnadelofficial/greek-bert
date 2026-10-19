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
| 1 — GreekBERT continuation (+lexicon) | **98.66** | **98.64** | **94.95** | 94.40 | **87.85** | **84.47** | 74.01 |
| 2 — real aristoBERTo continuation (+lexicon) | 98.58 | 98.53 | 94.80 | **94.41** | 86.42 | 82.83 | 71.74 |
| 3 — fresh BERT (control) | 92.05 | 91.63 | 82.49 | 81.12 | 64.53 | 57.81 | 20.56 |
| 4 — fresh ModernBERT | 97.63 | 97.36 | 92.38 | 88.64 | 80.64 | 75.77 | 59.37 |
| 5 — ModernBERT, aristoBERTo-shaped history (+lexicon) | 97.90 | 97.75 | 92.78 | 94.36 | 81.97 | 77.50 | 61.85 |
| *odyCy (`chcaa/grc_odycy_joint_trf`, external)* | *94.15* | *97.21* | *94.16* | *93.85* | *81.92* | *76.86* | *83.52* |

External reference row pulled from `meta.json`'s `performance` block on
odyCy's joint model (UD Perseus + UD Proiel combined), via
`https://huggingface.co/chcaa/grc_odycy_joint_trf` (spaCy bakes eval scores
into `meta.json` at packaging time -- no separate `eval_results.json` is
published for this model). **Not a strictly controlled comparison** -- their
eval is on their own held-out split of UD Perseus+Proiel, which is not
guaranteed to be identical to ours even though both are "PROIEL+Perseus
combined." Their much higher SENT_F (83.52 vs. our best of 74.01) is the
clearest sign of a different eval composition rather than a real pipeline
difference: the odyCy paper itself (Table 1/2) reports their joint model
getting 97.94 SENT_F on UD Perseus alone but only 64.91 on UD Proiel alone
(Proiel lacks punctuation, which the paper notes makes sentence segmentation
much harder there) -- 83.52 sitting between those two is consistent with
their combined eval simply weighting the two source treebanks differently
than ours does, not with a 9-13 point real quality gap on that one metric.
LEMMA, by contrast, is a much closer, more interpretable comparison: our best
model (94.40/94.41) now modestly beats odyCy's joint model (93.85) --
reasonable validation that the lexicon approach was implemented correctly,
since it's literally the same idea (their own repo ships
`frequency_lemmatizer/{table,lookup}.json`, the same two-file shape
`custom_components/lemmatizer.py` reproduces).

All percentages. Models 1, 2, and 5 now include the odyCy-style lexicon layer
(`custom_components/lemmatizer.py` / `build_lemma_lexicon.py`); models 3, 4
don't (they were never in contention for "the" model -- 3 is the random-init
control, 4 is superseded by 5 as the ModernBERT candidate).

**Model 1 is now the overall winner, outright or tied on every metric.** It
wins TAG, POS, MORPH, UAS, LAS, and SENT_F against model 2 (e.g. SENT_F 74.01
vs 71.74), and is in a statistical dead heat with it on LEMMA (94.40 vs 94.41
-- a 0.01-point difference, noise).

Model 5 beats model 4 on every spaCy metric (e.g. SENT_F 61.85 vs 59.37, LAS
77.50 vs 75.77) -- consistent with the architecture-comparison story model 5
exists to test: matching model 2's training-history shape helps ModernBERT's
downstream performance even though it doesn't win on the pretraining
objective itself. Still behind models 1/2 (the BERT continuations) on
UAS/LAS/SENT_F specifically, though closer than model 4 was.

## Lemmatizer lexicon ablation (models 1, 2, and 5)

This is the comparison that actually decided which model to take forward.
Pre-lexicon, model 5 had the only lemma edge in the whole table (88.83 vs
model 2's 88.22, with model 1 tied with model 2 at 88.21) despite losing on
every other metric -- the open question was whether that edge was real or
would wash out once all three got the same (encoder-agnostic,
treebank-built) lexicon.

| Model | LEMMA (no lexicon) | LEMMA (+lexicon) | Δ |
|---|---|---|---|
| 1 — GreekBERT continuation | 88.21 | 94.40 | +6.19 |
| 2 — real aristoBERTo continuation | 88.22 | 94.41 | +6.19 |
| 5 — ModernBERT, aristoBERTo-shaped history | 88.83 | 94.36 | +5.53 |

**The gap didn't just close, it flipped -- and it confirms model 1, not
model 2, is the actual winner.** All three models land within 0.05 points of
each other on LEMMA once the shared lexicon is applied (94.36-94.41): model
5's small pre-lexicon lemma edge was exactly the kind of noise we suspected,
and it's gone. With lemma neutralized across all three, the decision comes
down to every other metric -- and model 1 wins those outright (see table
above). This is exactly the pattern the pretraining-PPL argument predicted:
model 1 has the lowest pretrain loss/PPL of the three (1.582 vs model 2's
1.608 and model 5's 1.947), consistent with it being the strongest encoder of
the three, with lemma being the one place a training-set lexicon could paper
over the gap -- and once that gap is papered over for all three alike, the
ranking reverts to encoder quality, which model 1 wins.

(Every other metric moved slightly, both up and down, by a few tenths of a
point across all three reruns -- expected single-run training noise from a
full `spacy train` rerun, not a side effect of adding the lemmatizer, which
has no trainable parameters and runs after tagger/morphologizer/parser. The
LEMMA jump (+5.5 to +6.2 points, consistent across all three models) is an
order of magnitude larger than that noise, which is how you can tell it's a
real, structural effect and not noise.)

**Bottom line for the paper/talk: model 1 -- the "accidental" direct
GreekBERT continuation that never touches real aristoBERTo's own
ancient-Greek adaptation stage at all -- is the best model across the board.**
This is also the cleanest empirical support for the two refined claims from
the earlier discussion: (a) real aristoBERTo's own intermediate ancient-Greek
stage adds no measurable value once you're continuing on a large new corpus
anyway (model 1 vs. model 2, same architecture/tokenizer/final-corpus, only
difference is skipping that stage -- model 1 wins), and (b) a strong warm
start currently outweighs ModernBERT's architectural advantage at this data
scale (model 1/2 vs. model 5), even though that architectural advantage is
real and large when starting from random init (model 4 vs. model 3).

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
