#!/usr/bin/env bash
# =============================================================================
# One-off CPU job: tokenize the corpus with the clean v2 Greek BPE tokenizer,
# producing the shared input for arms 3/4 of the 2x2 comparison
# (docs/EXPERIMENTS-2X2.md). Uses the SAME split/seed as arm 1's corpus (the
# document-level 20% holdout defaults from TrainingConfig), so held-out
# documents are identical across arms even though the two corpora differ in
# tokenizer/token counts.
#
# No GPU needed -- this only runs an already-trained tokenizer over the
# corpus, it does not train one (that's scripts/train_bpe.py, a separate
# one-off job, already run: tokenizers/modernbert-greek-tokenizer-v2/ is
# committed).
#
#   sbatch slurm/prepare_bpe_corpus.sh
#   TOKENIZER_PATH=tokenizers/some-other-tokenizer \
#     OUT_DIR=data/some-other-dataset sbatch slurm/prepare_bpe_corpus.sh
#
# Extra args pass straight through to prepare_data.py, e.g.:
#   sbatch slurm/prepare_bpe_corpus.sh --include-wikidata
#
# Submit from the project root so logs land in logs/.
# =============================================================================
#SBATCH -J GreekBERT-PrepBPE
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/GreekBERT-PrepBPE.%j.%N.out
#SBATCH --error=logs/GreekBERT-PrepBPE.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

# This isn't tied to a single pretraining run (arms 3 AND 4 share this one
# corpus), so give env.sh's mandatory RUN_NAME a default instead of requiring
# the caller to invent one. Override it if you want the runs/<name>/ dir
# (unused here, but created by env.sh) called something else.
export RUN_NAME="${RUN_NAME:-corpus-prep-greekbpe-v2}"
source "slurm/env.sh"

TOKENIZER_PATH="${TOKENIZER_PATH:-tokenizers/modernbert-greek-tokenizer-v2}"
OUT_DIR="${OUT_DIR:-data/greekbpe_tokenized_open_greek_dataset_v2}"

if [[ ! -f "$PROJECT_ROOT/$TOKENIZER_PATH/tokenizer.json" ]]; then
  echo "ERROR: no tokenizer.json at $TOKENIZER_PATH" >&2
  echo "       Do NOT point this at tokenizers/modernbert-greek-tokenizer (v1)" >&2
  echo "       -- it is corrupted (mojibake vocab, see docs/EXPERIMENTS-2X2.md)." >&2
  echo "       If you need to (re)build v2: python scripts/train_bpe.py" >&2
  exit 1
fi

echo "Host      : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job       : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Python    : $PYTHON_ENV"
echo "Tokenizer : $TOKENIZER_PATH"
echo "Output    : $OUT_DIR"

load_env pretrain

stage "tokenize corpus ($TOKENIZER_PATH -> $OUT_DIR)"
py_run pretrain "$PROJECT_ROOT" \
  python scripts/prepare_data.py \
    --tokenizer "$TOKENIZER_PATH" \
    --out "$OUT_DIR" \
    "$@"
stage_done "prepare_data -> $OUT_DIR"

echo ""
echo "Next: arm 4 pretrain, e.g."
echo "  RUN_NAME=arm4-modernbert-scratch \\"
echo "    TOKENIZED_DATASET_PATH=$OUT_DIR \\"
echo "    CONFIG_PATH=configs/train-modernbert.yaml sbatch slurm/pretrain.sh"
