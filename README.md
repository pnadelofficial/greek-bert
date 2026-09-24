<<<<<<< Updated upstream
# greek-bert

This repository contains code written for Beyond Translation: Opening up the Human Record, a Schmidt Sciences Humanities and AI Virtual Institute (HAVI) grant. This project seeks to develop a cutting-edge Ancient Greek Bi-directional Encoder Representations from Transformers (BERT) model, as well as a set of associated downstream tools or models built from it. 

## Parts
This repository is dynamic and growing. It is composed of several sections enumerated below:
* *scripts*: contains the BERT training script, as well as SBATCH .sh scripts for running jobs on HPC Clusters. You can find the completed BERT model here: https://huggingface.co/pnadel/ancient-greek-bert.
* *lemma-pos*: contains code for training and evaluating a morphological tagger for Ancient Greek, which outputs lemmata and morphological tags given a surface form. You can find the completed tagger here: https://huggingface.co/pnadel/ancient-greek-morph-tagger.
* *wsd*: contains code for training and evaluating a word sense disambiguator, as a way to quickly evaulate BERT models on a complex downstream task. A completed model has not yet been pushed to HuggingFace but is available upon request.
* *spacy_code* contains code for training and evaluating a `spaCy` syntactic parser for Ancient Greek. This model attains the best UAS and LAS scores for any Ancient Greek syntactic parser. Evaluation metrics and a complete model can be found here: https://huggingface.co/pnadel/ancient-greek-parser.
* *sbert* contains code for training and evaluating a `SentenceTranformer` SBERT model. This is not yet complete.


=======
# GreekBERT

Pretraining and evaluation of an Ancient Greek BERT model, built on
[nlpaueb/bert-base-greek-uncased-v1](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1).

Pipeline: **pretrain (MLM)** → **convert to HuggingFace format** → **downstream
evaluation** (word sense disambiguation, spaCy parsing/morphology/lemmatization,
and contrastive sentence embeddings).

---

## Quick start

All jobs are submitted from the **project root** (so SLURM logs land in `logs/`).
Every job needs a `RUN_NAME`, which names its artifact directory
`runs/<RUN_NAME>/` so runs never overwrite each other.

```bash
cd /cluster/tufts/perseuslab/pnadel01/greek-bert

# 1. Pretrain (8x H200, 4 days)
RUN_NAME=myrun sbatch slurm/pretrain.sh

# 2. Convert + fine-tune WSD + train/eval spaCy
RUN_NAME=myrun sbatch slurm/posttrain.sh

# Or just some stages:
RUN_NAME=myrun STAGES=convert        sbatch slurm/posttrain.sh
RUN_NAME=myrun STAGES=wsd,spacy      sbatch slurm/posttrain.sh

# Monitor
tensorboard --logdir=runs/myrun/tensorboard
```

Other jobs:

```bash
RUN_NAME=sbert-v1                    sbatch slurm/sbert.sh    # contrastive SBERT
RUN_NAME=spacy-v1                    sbatch slurm/spacy.sh    # spaCy alone
RUN_NAME=spacy-v1 SPACY_CONFIG=configs/gpu_default_freq_lemma.cfg sbatch slurm/spacy.sh
```

### Job scripts

| Script | Does | Replaces |
|---|---|---|
| `slurm/env.sh` | Sourced by all jobs: paths, Python env, GPU profiles | — |
| `slurm/pretrain.sh` | MLM pretraining | `scripts/train.sh` |
| `slurm/posttrain.sh` | convert → WSD → spaCy, selectable via `STAGES` | `scripts/train_and_eval_{both,spacy,wsd}.sh` |
| `slurm/spacy.sh` | spaCy train + eval alone | `spacy_code/{train_from_hf,train_lemma_from_hf}.sh`, `spacy_code/scripts/train.sh` |
| `slurm/sbert.sh` | SBERT contrastive training | `scripts/train_sbert_contrastive.sh`, `sbert/train-contrastive.sh` |

### Environment variables

Set before `sbatch`, or edit the defaults in `slurm/env.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `RUN_NAME` | — (**required**) | Names `runs/<RUN_NAME>/` |
| `PROFILE` | `h200x8` | GPU/partition profile (see below) |
| `PYTHON_ENV` | `uv` | `uv` (current) or `conda` (legacy envs) |
| `STAGES` | `convert,wsd,spacy` | Which post-training stages to run |
| `MODEL_DIR` | per-run export | HF model to fine-tune from |
| `CONFIG_PATH` | `configs/train.yaml` | Training config |
| `SPACY_CONFIG` | `configs/gpu_default.cfg` | spaCy config |

**Profiles** (the old scripts each hardcoded a different partition/GPU with no
explanation — pick what's actually free):

| Profile | Partition | GPUs | CPUs | Mem | Time |
|---|---|---|---|---|---|
| `h200x8` | gpu | 8x H200 | 64 | 128g | 4d |
| `b200x8` | gpu | 8x B200 (`new_gpu` res.) | 16 | 32g | 2d |
| `h200x1` | gpu | 1x H200 | 16 | 32g | 2d |
| `l40sx4` | gpu | 4x L40S | 16 | 32g | 2d |
| `h100x1` | tuftsai | 1x H100 | 16 | 32g | 2d |

---

## Choosing which model downstream jobs use

Downstream jobs (WSD, spaCy) never hardcode a model path. They resolve:

1. `$MODEL_DIR`, if set
2. `models/current` — a symlink you control

```bash
scripts/set_current_model.sh --list                    # what's available
scripts/set_current_model.sh models/greekbert-2025-09-18
```

This replaces the old `hf_format918` / `hf_format120` directories, which were
referenced by name in `wsd.py` and the spaCy configs.

---

## Layout

```
greek-bert/
├── configs/train.yaml       # hyperparameters (single source of truth)
├── slurm/                   # job scripts + env.sh
├── scripts/                 # train.py, convert_to_hf.py, utils.py, paths.py
├── tokenizers/              # modernbert-greek-tokenizer
├── wsd/                     # word sense disambiguation (glaux)
├── sbert/                   # contrastive sentence embeddings
├── spacy_code/              # spaCy parser/morph/lemmatizer  (separate uv project)
├── lemma-pos/               # lemma+POS tagger (treebank-trained)
├── notebooks/               # exploratory only
├── docs/archive/            # historical change notes
│
├── data/                    # gitignored — training corpora (~5 GB)
├── models/                  # gitignored — HF exports + `current` symlink
├── runs/<RUN_NAME>/         # gitignored — checkpoints, tensorboard, exports
└── logs/                    # gitignored — SLURM .out/.err
```

### Config paths

Relative paths in `configs/train.yaml` resolve against **the config file's own
directory**, so a run behaves identically from any working directory. Under
SLURM, `CHECKPOINT_DIR` / `TENSORBOARD_DIR` / `HF_EXPORT_DIR` are injected per
`RUN_NAME` and override the file's values.

---

## Python environments

There are **two** uv projects and they cannot be merged — spaCy needs Python
3.12 with `cupy<14`, the root project needs 3.13 with transformers 5.x:

| Project | Python | Used for |
|---|---|---|
| `.` (root) | 3.13 | pretraining, WSD, SBERT |
| `spacy_code/` | 3.12 | spaCy parser / morphologizer / lemmatizer |

`PYTHON_ENV=uv` (default) uses `uv run` in the right project directory.
`PYTHON_ENV=conda` restores the legacy `torchrun` / `general_purpose_textgen` /
`spacy_gpu` conda envs.

> The legacy `spacy_gpu` conda env is **broken** — its `cupy` predates NumPy 2.0
> (`AttributeError: np.float_ was removed`). Use `PYTHON_ENV=uv` for spaCy work.

---

## Data

Training data lives in `data/` (~5 GB, gitignored):

- `nlpaueb_tokenized_open_greek_dataset/` — current pretraining corpus
- `modernbert_tokenized_open_greek_dataset/`, `jacobo_tokenized_open_greek_dataset/`
- `ancient-greek-datasets/` — SBERT translation pairs
- `europarl_el/` — raw Greek Europarl

Pre-tokenized with `scripts/utils.py::tokenize_and_prepare_mlm`. Masks are
**not** precomputed — they're generated dynamically per step (RoBERTa-style).

---

## Training notes

RoBERTa-inspired settings, based on [arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692):

- **Dynamic masking** — fresh masks every forward pass (~0.5% GLUE gain)
- **30% masking rate** — ModernBERT-style, more signal per step
- **LR 2e-4** with 5% warmup (OneCycle). RoBERTa's 6e-4 assumes an 8k-sequence
  batch and a from-scratch model; at effective batch 128 it is too hot for
  continuing a converged aristoBERTo (audit item 4).
- **Adam β₂ = 0.98** — stability at large batch

Historical change notes from that work are in `docs/archive/`.

---

## Reference models

Pulled into `models/` for comparison (gitignored):

- `models/external-aristoBERTo` — Jacobo's aristoBERTo
- `models/external-shlm-grc-en` — `kevinkrahn/shlm-grc-en` (bilingual GRC↔EN sentence encoder)
- `models/greekbert-jan20` — Jan-20 export (base for the SBERT run)
- `models/greekbert-2025-09-18` — Sep-18 export (was `hf_format918`)
>>>>>>> Stashed changes
