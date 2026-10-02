# The model comparison (formerly "the 2x2")

**2026-10 reframe:** this started as a 2x2 grid (architecture x tokenizer).
Along the way we found `configs/train.yaml`'s `pretrained_model` default
(`nlpaueb/bert-base-greek-uncased-v1`) is GreekBERT (modern Greek), not
aristoBERTo -- aristoBERTo is GreekBERT continued on ancient Greek by its own
authors (see the IDENTITY CORRECTION note at the top of `configs/train.yaml`).
Rather than treat that as a bug to erase, it turned out to be one of three
candidate approaches already under consideration for "what's the best way to
build an Ancient Greek BERT": (1) fresh ModernBERT on new data, (2) continue
real aristoBERTo on new data, (3) continue GreekBERT directly on new data. The
grid is now: those three candidates, plus one random-init control that makes
the "does prior knowledge help" question answerable instead of just reporting
a leaderboard. A fifth model (ModernBERT put through aristoBERTo's own
two-stage history) was scoped and **deferred** -- see "Deferred: model 5" at
the bottom.

| Model | What it is | Architecture | Tokenizer | Weights | Config | Corpus | RUN_NAME | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | GreekBERT fine-tuned on new data | BERT-base | nlpaueb 35k WP | GreekBERT continued | `configs/train.yaml` (default) | `nlpaueb_tokenized_open_greek_dataset_v2` | `arm1-aristo-continuation` | done |
| 2 | aristoBERTo fine-tuned on new data | BERT-base | nlpaueb 35k WP | real aristoBERTo continued | `configs/train.yaml` + `PRETRAINED_MODEL=../models/external-aristoberto-real` | same as model 1 | `model2-aristo-continuation` | **not yet run** |
| 3 | Fresh BERT on new data (control) | BERT-base | nlpaueb 35k WP | random | `configs/train-bert-fresh.yaml` | same as model 1 | `arm2-bert-fresh` | done |
| 4 | Fresh ModernBERT on new data | ModernBERT | Greek 50k BPE | random | `configs/train-modernbert.yaml` | `greekbpe_tokenized_open_greek_dataset_v2` | `arm4-modernbert-scratch` | done |

Reading the table:
- **(1) vs (2)** -- does fine-tuning the *actual* ancient-Greek-adapted
  aristoBERTo beat fine-tuning its modern-Greek starting point directly? Tests
  whether aristoBERTo's own ancient-Greek adaptation stage adds value beyond
  what your (larger/different) new corpus already provides.
- **(1) vs (3), (2) vs (3)** -- does starting from prior Greek knowledge (of
  either kind) actually help over random init, holding architecture,
  tokenizer, and corpus fixed? This is the causal control: without model 3,
  you can't tell "prior knowledge helped" apart from "more effective total
  training compute."
- **(3) vs (4)** -- architecture effect (BERT vs ModernBERT) at a matched,
  controlled data budget (both random-init, same corpus-size class, different
  tokenizer because each architecture uses its natural one).
- **Best model for the paper** is whichever of (1), (2), (4) wins on the
  downstream suite (WSD + spaCy); model 3 is a control, not a release
  candidate.

## Shared measurement (required for the comparison to mean anything)

- **Same split.** All models use the document-level 20% holdout from
  `scripts/prepare_data.py` with the SAME seed (default `TrainingConfig.seed`),
  so held-out documents are identical across models and val loss/PPL are
  directly comparable. (The two corpora differ in tokenizer, so token counts
  differ, but the *documents* held out are the same.)
- **Same recipe within a weight-initialization class.** Models 1 and 2 share
  `configs/train.yaml`'s continuation recipe (`mask_prob 0.30`, dynamic
  80/10/10 + special-token guard, `lr 2e-4`, AdamW beta2=0.98, bf16,
  best-checkpoint by all-reduced val loss) -- they should differ ONLY in which
  checkpoint they continue. Models 3 and 4 are random-init and may need their
  own tuned LR/mask-rate (see the arm-2 LR postmortem below) since a
  continued-pretraining-tuned recipe is not necessarily safe for cold-start
  training.
- **Same downstream protocol.** Each model's export goes through
  `slurm/posttrain.sh` (convert -> WSD -> spaCy) with identical WSD seeds and
  the same spaCy config; results land in `results/table.md` keyed by
  (run, stage, model, metric, seed, config_hash, git_sha).
- **Report best-checkpoint numbers**, not final-epoch: `best_model.pt` is the
  val-loss-selected export and `convert_to_hf.py` prefers it.
- **Lemmatizer lexicon.** Once all final models are posttrained, rerun the
  spaCy stage for each with `SPACY_CONFIG=configs/gpu_default_freq_lemma.cfg`
  (after `cd spacy_code && uv run python build_lemma_lexicon.py`, once --
  arm-agnostic). Only the LEMMA number should move; see
  `custom_components/lemmatizer.py`.

## Random-init LR postmortem (applies to models 3 and 4)

`configs/train-bert-fresh.yaml` and `configs/train-modernbert.yaml` both
originally copied `configs/train.yaml`'s continuation-tuned
`lr=max_lr=2e-4, pct_start=0.05`. Model 3 (arm 2) showed loss improving for
~10 epochs then climbing and plateauing -- the classic signature of peak LR
being too hot for a random-init model (a warm-started model tolerates a given
LR much better than one starting from a rough, unconverged loss landscape).
Fix applied to `train-bert-fresh.yaml`: lowered peak LR (linearly scaled from
original BERT-base's `1e-4` at batch 256 down to this repo's effective batch
128, i.e. toward `~1e-4`). Model 3's pretrain run is reported complete;
worth a final look at its loss/accuracy curve to confirm the climb-then-
plateau pattern didn't recur before treating it as final.
Warmup fraction (`pct_start=0.05`) was NOT the issue -- at ~215k total steps
that's already a more generous warmup than original BERT's own 1% recipe.
Model 4 (arm 4) converged well under the original un-tuned recipe and needed
no change; if you rerun it, there's no evidence it needs the same fix model 3
did.

## Commands (submit from the project root)

```bash
# Corpus prep (one-off, CPU job) -- needed for models 1-3 (WordPiece) and
# model 4 (BPE). See slurm/prepare_bpe_corpus.sh for the BPE side.
python scripts/prepare_data.py \
    --tokenizer nlpaueb/bert-base-greek-uncased-v1 \
    --out data/nlpaueb_tokenized_open_greek_dataset_v2

# Model 1 (GreekBERT continuation -- done)
RUN_NAME=arm1-aristo-continuation \
  TOKENIZED_DATASET_PATH=data/nlpaueb_tokenized_open_greek_dataset_v2 \
  sbatch slurm/pretrain.sh

# Model 2 (real aristoBERTo continuation -- NOT YET RUN)
# models/external-aristoBERTo (no suffix) was CONFIRMED to be GreekBERT, not
# aristoBERTo -- the real one is models/external-aristoberto-real. Also see
# wsd/wsd.py's DEFAULT_BASELINE, fixed the same way: every WSD row in
# results/table.md labeled "external-aristoBERTo" before this fix was scored
# against GreekBERT and needs rerunning against the real model.
RUN_NAME=model2-aristo-continuation \
  PRETRAINED_MODEL=../models/external-aristoberto-real \
  TOKENIZED_DATASET_PATH=data/nlpaueb_tokenized_open_greek_dataset_v2 \
  CONFIG_PATH=configs/train.yaml sbatch slurm/pretrain.sh

# Model 3 (fresh BERT, control -- done)
RUN_NAME=arm2-bert-fresh \
  TOKENIZED_DATASET_PATH=data/nlpaueb_tokenized_open_greek_dataset_v2 \
  CONFIG_PATH=configs/train-bert-fresh.yaml sbatch slurm/pretrain.sh

# Model 4 (fresh ModernBERT -- done)
RUN_NAME=arm4-modernbert-scratch \
  TOKENIZED_DATASET_PATH=data/greekbpe_tokenized_open_greek_dataset_v2 \
  CONFIG_PATH=configs/train-modernbert.yaml sbatch slurm/pretrain.sh

# Downstream for each model (after its pretrain finishes):
RUN_NAME=<run-name> sbatch slurm/posttrain.sh
```

## BPE tokenizer (resolved)

`tokenizers/modernbert-greek-tokenizer` (v1) was corrupted (mojibake'd
double-encoded vocab, see git history for the full diagnosis). Fixed by
`scripts/train_bpe.py` -> `tokenizers/modernbert-greek-tokenizer-v2/`
(committed, already trained). `configs/train-modernbert.yaml` and
`slurm/prepare_bpe_corpus.sh` already point at v2. Nothing further needed
here unless the tokenizer itself needs retraining.

## Deferred: model 5 (ModernBERT through aristoBERTo's own two-stage history)

The original ask: a ModernBERT counterpart that goes through the same
two-stage history as real aristoBERTo -- pretrain from scratch on GreekBERT's
own modern-Greek corpus (~30GB: Greek Wikipedia + Europarl Greek + OSCAR
Greek, all public), then continue on the existing ancient-Greek corpus, same
as model 2 does to real aristoBERTo. Not doable as a config change: aristoBERTo's
own ancient-Greek scrape (~900MB) was never published, and the modern-Greek
corpus-assembly pipeline doesn't exist in this repo yet. Reusing real
aristoBERTo's *weights* as a shortcut doesn't work either -- different
architecture (BERT vs ModernBERT) and different vocab (35k WordPiece vs 50k
BPE) mean nothing transfers directly. A faithful version requires: (a) a new
modern-Greek corpus-assembly pipeline (download + clean + dedup Wikipedia /
Europarl / OSCAR), (b) a full from-scratch ModernBERT pretrain on it
(~model-4 cost), (c) a continuation run on the existing corpus (~model-1/2
cost). Scoped, understood, deliberately out of scope for the current paper
pass -- revisit later if there's appetite for it.
