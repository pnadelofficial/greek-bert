# The 2×2 comparison

The audit's core finding (item 3): continuing aristoBERTo on aristoBERTo's own
uncased 35k WordPiece vocab has a **structurally low ceiling** — the tokenizer
is the dominant factor for a morphologically rich, diacritic-heavy language,
and uncase merges the accent forms the morphologizer/lemmatizer need. This
page defines the four arms that isolate the two axes that can actually move
the number: **architecture** and **tokenizer**.

```
                       tokenizer
                  aristo 35k WP        Greek 50k BPE
                  (uncased)            (cased, diacritic-preserving)
  BERT         (1) aristo-continuation (2) fresh BERT, random init
  arch         (3) —                   —
  ModernBERT   —                       (3) warm trunk + re-init emb/head
                                        (4) fully from scratch
```

| Arm | Arch | Tokenizer | Weights | Config | Corpus | Isolates |
|---|---|---|---|---|---|---|
| 1 | BERT-base | nlpaueb 35k WP | aristoBERTo continued | `configs/train.yaml` | `nlpaueb_tokenized_open_greek_dataset_v2` | baseline (running) |
| 2 | BERT-base | nlpaueb 35k WP | random | `configs/train-bert-fresh.yaml` | same as arm 1 | corpus vs prior weights |
| 3 | ModernBERT | Greek 50k BPE | trunk warm, emb+head re-init | `configs/train-modernbert.yaml` + `PRETRAINED_MODEL=<arm-4 export>` | `greekbpe_tokenized_open_greek_dataset_v2` | tokenizer+arch package (audit's headline arm) |
| 4 | ModernBERT | Greek 50k BPE | fully random | `configs/train-modernbert.yaml` (as checked in) | same as arm 3 | architecture at fixed tokenizer |

Reading the table:
- **(1) vs (3)/(4)** — does the cased-50k-BPE + ModernBERT package beat
  continuing the baseline? This is the audit's "real second arm".
- **(2) vs (3)/(4)** — same fresh-init regime, BERT vs ModernBERT trunk:
  isolates the architecture.
- **(3) vs (4)** — same arch+tokenizer, warm trunk vs cold trunk: how much of
  ModernBERT's pretraining transfers to Ancient Greek.
- **(1) vs (2)** — continued vs from-scratch at the SAME tokenizer: is
  aristoBERTo's prior knowledge worth keeping?

## Shared measurement (required for the comparison to mean anything)

- **Same split.** All arms use the document-level 20% holdout from
  `scripts/prepare_data.py` with the SAME seed (default `TrainingConfig.seed`),
  so the held-out documents are identical across arms and val loss/PPL are
  directly comparable. (The two corpora differ in tokenizer, so token counts
  differ, but the *documents* held out are the same.)
- **Same recipe.** `mask_prob 0.30`, dynamic 80/10/10 + special-token guard,
  `lr 2e-4`, AdamW β₂=0.98, bf16, same batch/epoch budget. The configs
  differ ONLY in the (architecture, tokenizer, weights) triple.
- **Same downstream protocol.** Each arm's export goes through
  `slurm/posttrain.sh` (convert → WSD → spaCy) with identical WSD seeds and
  the same spaCy config; results land in `results/table.md` keyed by
  (run, stage, model, metric, seed, config_hash, git_sha).
- **Report best-checkpoint numbers**, not final-epoch: `best_model.pt` is the
  val-loss-selected export and `convert_to_hf.py` prefers it.

## Cost / priority

ModernBERT-base is ~2.5× BERT-base (22 layers, 768 hidden, gated MLP), so
arms 3/4 cost ~2.5× per token. Suggested order: **1 → 3 → 2 → 4** (3 is the
headline, 2 is cheap, 4 is most expensive and most "explained" by 3). Arm 4
may need more than the 40-epoch token budget to converge (random trunk) —
watch the `NOT converged` flag; arms 1–3 share the budget fairly because
their trunks are already trained.

## Commands (submit from the project root)

```bash
# Corpus for the BPE arms (one-off, CPU job) — same split/seed as arm 1:
python scripts/prepare_data.py \
    --tokenizer tokenizers/modernbert-greek-tokenizer \
    --out data/greekbpe_tokenized_open_greek_dataset_v2

# Arm 1 (in progress)
RUN_NAME=arm1-aristo-continuation \
  TOKENIZED_DATASET_PATH=data/nlpaueb_tokenized_open_greek_dataset_v2 \
  sbatch slurm/pretrain.sh

# Arm 2
RUN_NAME=arm2-bert-fresh \
  TOKENIZED_DATASET_PATH=data/nlpaueb_tokenized_open_greek_dataset_v2 \
  CONFIG_PATH=configs/train-bert-fresh.yaml sbatch slurm/pretrain.sh

# Arm 4 (from-scratch ModernBERT)
RUN_NAME=arm4-modernbert-scratch \
  TOKENIZED_DATASET_PATH=data/greekbpe_tokenized_open_greek_dataset_v2 \
  CONFIG_PATH=configs/train-modernbert.yaml sbatch slurm/pretrain.sh

# Arm 3 (warm-start from arm 4's export; run AFTER arm 4)
RUN_NAME=arm3-modernbert-warm \
  PRETRAINED_MODEL=runs/arm4-modernbert-scratch/hf_format \
  TOKENIZED_DATASET_PATH=data/greekbpe_tokenized_open_greek_dataset_v2 \
  CONFIG_PATH=configs/train-modernbert.yaml sbatch slurm/pretrain.sh

# Downstream for each arm (after its pretrain finishes):
RUN_NAME=<arm-run> sbatch slurm/posttrain.sh
```

## ⚠️ BLOCKER: the 50k BPE tokenizer is corrupted (arms 3 & 4 cannot run as-is)

`tokenizers/modernbert-greek-tokenizer` was **trained on mojibake'd text**. Its
vocab keys are double-encoded UTF-8 (e.g. the key `Î»Î¿Î³Î¿Ï‚` is `λόγος`
encoded UTF-8→bytes→decoded-as-latin1). Verified empirically:

- 1345/2000 sampled vocab keys are double-encoded Greek (`k.encode('latin-1').decode('utf-8')` yields real Greek).
- Feeding REAL Greek (`ἀρμονία`) gives 5 tokens incl. byte-fragments like `á¼ĢÏģ` — not the single word-token a 50k BPE should produce.
- The `pre_tokenizer` is `ByteLevel`, which re-encodes to bytes **at runtime**, so it can never match the (already-double-encoded) vocab keys. Real Greek fragments to ~1.3–4.4 tokens/word instead of ~1.

**Consequence:** arm 3/4 pretraining on this tokenizer would train a model whose
input is byte-noise, not Ancient Greek words. The "cased 50k BPE" advantage
the audit predicts (diacritic preservation, ~1 token/word) does not exist in
this artifact.

**Fix — implemented in `scripts/train_bpe.py`** (HF `tokenizers`, not a custom
BPE). It reuses the same clean-UTF-8 corpus loaders as `prepare_data.py`, so the
retrain is a drop-in:

```bash
# 1. Train the clean BPE (one-off CPU job; ~30-60 min for the full corpus):
python scripts/train_bpe.py
#    -> tokenizers/modernbert-greek-tokenizer-v2/

# 2. Tokenize the corpus with it (same split/seed as arm 1):
python scripts/prepare_data.py \
    --tokenizer tokenizers/modernbert-greek-tokenizer-v2 \
    --out data/greekbpe_tokenized_open_greek_dataset_v2

# 3. Point configs/train-modernbert.yaml at the v2 tokenizer and launch 3/4.
```

Design choices baked into `train_bpe.py` (all verified by a self-check that
FAILS LOUDLY on the exact bugs the old artifact had):
- **WhitespaceSplit pre-tokenizer, NOT ByteLevel.** Whole-word BPE tokens; no
  runtime byte re-encoding (the root cause of the old artifact's corruption).
  Unseen words fall back to `[UNK]` (rare at 50k over a 700M-token corpus).
- **Specials added AFTER training** (`add_special_tokens`), because every
current `tokenizers` version drops/renumbers specials passed via
  `BpeTrainer(special_tokens=...)`. On reload the specials land at the END of
  the vocab (content tokens keep ids 0+). `train.py`/`mlm_masking` resolve all
  ids from the tokenizer at runtime, so the exact numbers don't matter — the
  only invariant is that specials are distinct from content, which the
  self-check asserts.
- **No post_processor:** `tokenize_and_prepare_mlm` prepends `[CLS]`/appends
  `[SEP]` itself, so the trainer must not double-wrap.
- **Streams the corpus from disk** (`_corpus_text_iter`), reading the source
  parquet/txt files in bounded batches (`--sample-batch-chars`, default 20M
  chars ≈ 40–60 MB resident). The first version loaded the whole ~13 GB corpus
  into a Python dict (tens of GB resident from per-string overhead) and was
  OOM-killed by the SLURM cgroup at the first pre-processing chunk. Now only
  one small batch is ever resident, so peak corpus memory is a few MB
  regardless of corpus size.
- Self-checks: (a) tokens/word ≤ 2.0 (old artifact gave 1.3–4.4), (b) no vocab
  key is double-encoded, (c) all five specials present, distinct, and not
  colliding with a content id.
- **Liveness:** unbuffered stdout + a `--heartbeat <path>` file (also
  `BPE_HEARTBEAT` env) that records `{pid, phase, updated}` as it moves through
  `scan → bpe-train-start → bpe-train-done → self-check → saving`. The BPE
  trainer's own progress bar is TTY-gated (silent under sbatch), so the
  heartbeat is how you confirm a multi-hour train is alive:
  `watch cat <hb_path>`.

**Unblocks:** nothing in arms 1/2. Arms 3/4 are blocked until this is redone.
The aristo-continuation arm (1) and fresh-BERT arm (2) are unaffected and can
proceed now.

## Open design questions (decide before arm 4)

1. **Arm 4 schedule.** 2e-4/40-epoch is a *continued*-pretraining recipe. A
   genuinely from-scratch model may want a higher peak LR and/or more tokens.
   Decision: keep the shared recipe for comparability (recommended for the
   2×2) and treat a tuned from-scratch run as a separate follow-up.
2. **Arm 3 "warm trunk" source.** Using arm 4's own export is the cleanest
   (same code, same tokenizer). The alternative — loading upstream
   `answerdotai/ModernBERT-base` English weights — is a *different*
   experiment (cross-lingual transfer), not the audit's re-init idea.
3. **WSD target piece.** The BPE arm reads the token whose span ends the
   target word (the BPE analogue of WordPiece "last piece"). `wsd.py` records
   the mode in results; if arm 1/2 and arm 3/4 disagree systematically, run
   both modes once.
