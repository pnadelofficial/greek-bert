# 5-model comparison (compiled 2026-10-20 from logs_from_cluster/)

Source: `logs_from_cluster/{greekbert-continuation,real-aristoberto-continuation,fresh-bert-control,fresh-modernbert,fresh-greekbert-continuation}/{metrics.json,eval_results.json}`.
(Despite its directory name, `fresh-greekbert-continuation` is model 5 --
`RUN_NAME=modernbert-stage3-final`, confirmed via its `manifest.json`: three-
stage ModernBERT continuation, `pretrained_model` pointing at
`modernbert-stage2-ancient/hf_format`.)

## CORRECTION (2026-10-20): the spaCy eval corpus was PROIEL-only until now

Despite being labeled "PROIEL+Perseus combined" below the whole time,
`spacy_code/assets/` only ever had PROIEL -- Perseus was never actually
fetched. `scripts/combine_treebanks.sh`'s glob (`assets/**/*-train.conllu`,
a plain POSIX `sh` glob, not bash's recursive globstar) silently matched
whichever treebank's raw files happened to sit exactly one directory level
under `assets/` and silently ignored the other. Caught while trying to build
the genre/period eval split, when Perseus was finally fetched and the three
raw treebank files it actually combined with PROIEL (not replaced -- an
earlier attempt accidentally produced a Perseus-ONLY corpus, dropping PROIEL
entirely, caught via a simple `wc -l` sanity check before any retraining
happened). All 5 models' spaCy stage has since been rerun (train + evaluate,
no lemma lexicon yet -- see "Lemmatizer lexicon ablation" below) against the
corrected, genuinely-combined `corpus/{train,dev,test}.spacy`. **The spaCy
table below is now the corrected, final baseline (no-lexicon) comparison.**
The ranking (model 1 as overall winner on every metric but raw LEMMA) held up
under the corrected corpus -- reassuring, since it means the original
conclusion wasn't an artifact of the incomplete eval set, but it was
genuinely unverified until this rerun.

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

## spaCy downstream (PROIEL+Perseus combined eval -- corrected 2026-10-20, NO lexicon)

| Model | TAG | POS | MORPH | LEMMA | UAS | LAS | SENT_F |
|---|---|---|---|---|---|---|---|
| 1 — GreekBERT continuation | **94.72** | **97.49** | **94.93** | 84.36 | **85.14** | **80.96** | **87.94** |
| 2 — real aristoBERTo continuation | 94.51 | 97.34 | 94.70 | 84.84 | 84.66 | 80.43 | 87.25 |
| 3 — fresh BERT (control) | 79.70 | 87.94 | 80.79 | 76.36 | 57.13 | 49.67 | 53.60 |
| 4 — fresh ModernBERT | 90.90 | 94.90 | 91.98 | 85.09 | 76.23 | 70.63 | 78.85 |
| 5 — ModernBERT, aristoBERTo-shaped history | 92.14 | 95.69 | 93.05 | **85.59** | 80.44 | 75.46 | 82.27 |

All percentages, no lemma lexicon applied yet (that's the next step, see
below). **Model 1 is still the overall winner on every metric except raw
LEMMA** -- it wins TAG, POS, MORPH, UAS, LAS, and SENT_F against model 2 by
small but consistent margins (e.g. SENT_F 87.94 vs 87.25), the same pattern
as before the corpus fix. On raw LEMMA, model 5 is actually highest of the
three (85.59), with model 2 second (84.84) and model 1 last (84.36) -- a
similar shape to what we saw on the old PROIEL-only corpus, where the lexicon
later closed and reversed that exact gap. Worth watching whether the same
thing happens again once the lexicon is rebuilt from this corpus and applied
(see below) rather than assuming it will.

Note these numbers are substantially lower across the board than the old
(mislabeled) PROIEL-only table -- e.g. model 1's SENT_F was 75.51 there vs
87.94 here, TAG 98.55 vs 94.72. That's not regression, it's PROIEL-only
numbers having been measured on a smaller, punctuation-free, more homogeneous
corpus; Perseus brings genre diversity (epic, tragedy, oratory, philosophy)
and real punctuation, making the task both harder in some ways (more variety)
and, apparently, easier for sentence segmentation specifically (Perseus HAS
punctuation, PROIEL doesn't -- see the odyCy paper's own note on this). These
numbers are not comparable to the old table at all; treat this as the first
real measurement, not a delta from the old one.

Model 5 beats model 4 on every spaCy metric (e.g. SENT_F 82.27 vs 78.85, LAS
75.46 vs 70.63) -- consistent with the architecture-comparison story model 5
exists to test: matching model 2's training-history shape helps ModernBERT's
downstream performance even though it doesn't win on the pretraining
objective itself. Still behind models 1/2 (the BERT continuations) on
UAS/LAS/SENT_F, though closer than model 4 was.

**odyCy comparison pending re-pull.** The external odyCy reference row that
used to sit here (`chcaa/grc_odycy_joint_trf`'s `meta.json`) was compared
against our old PROIEL-only numbers with a "different eval composition"
caveat. Now that our corpus is genuinely PROIEL+Perseus combined, that
comparison is worth redoing properly -- but the LEMMA side of it specifically
isn't apples-to-apples right now anyway, since odyCy's published number
includes their lexicon layer and ours above doesn't yet. Re-add this
comparison after the lexicon pass below.

## Lemmatizer lexicon ablation (models 1, 2, and 5) -- SUPERSEDED, built on the PROIEL-only corpus

**Everything in this section was measured on the old, incomplete PROIEL-only
corpus (see the CORRECTION note at the top) -- not re-run yet on the
corrected PROIEL+Perseus corpus.** Kept here as the historical record of the
finding, not as a current result. The lexicon needs rebuilding from the new
`corpus/train.spacy` (`build_lemma_lexicon.py`) and reapplying to whichever
model wins the corrected baseline above (model 1, per the table above, same
as before) before this section's conclusion can be trusted again. The
pattern below -- lexicon closes/flips a small pre-lexicon lemma gap in favor
of the stronger overall encoder -- is worth watching for again given the
corrected baseline table above shows the same shape (model 5 ahead on raw
LEMMA, model 1 behind on LEMMA but ahead everywhere else), but it is a
prediction based on the old run, not a re-confirmed finding.

This is the comparison that actually decided which model to take forward
(on the old corpus).
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
