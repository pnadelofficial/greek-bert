#!/usr/bin/env bash
# =============================================================================
# One-off CPU job: tokenize this project's own corpus (data/*.parquet, Harvard
# institutional-books Ancient Greek subset + Europarl -- same default mix as
# data/nlpaueb_tokenized_open_greek_dataset_v2) with bowphs/GreBerta's OWN
# tokenizer (RoBERTa-style, 52k BPE vocab -- unrelated to nlpaueb's 35k
# WordPiece vocab arms 1-3 use). Produces the input both
# scripts/eval_pretrained_mlm.py (zero-shot baseline eval) and a GreBerta
# continuation-pretraining run need.
#
# Uses prepare_data.py's DEFAULT corpus flags/seed -- the same command that
# built data/nlpaueb_tokenized_open_greek_dataset_v2 (see
# docs/EXPERIMENTS-2X2.md), just a different --tokenizer/--out -- so the
# document-level 20% holdout is IDENTICAL to every other arm's, which is what
# makes the resulting val numbers directly comparable.
#
#   sbatch slurm/prepare_greberta_corpus.sh
#
# Extra args pass straight through to prepare_data.py, e.g.:
#   sbatch slurm/prepare_greberta_corpus.sh --include-wikidata
#
# Submit from the project root so logs land in logs/.
# =============================================================================
#SBATCH -J GreekBERT-PrepGreBerta
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/GreekBERT-PrepGreBerta.%j.%N.out
#SBATCH --error=logs/GreekBERT-PrepGreBerta.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

# Not tied to a single pretraining run (the eval AND the continuation-training
# arm both read this same corpus), so give env.sh's mandatory RUN_NAME a
# default rather than requiring the caller to invent one.
export RUN_NAME="${RUN_NAME:-corpus-prep-greberta-v2}"
source "slurm/env.sh"

TOKENIZER_PATH="${TOKENIZER_PATH:-bowphs/GreBerta}"
OUT_DIR="${OUT_DIR:-data/greberta_tokenized_open_greek_dataset_v2}"

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
echo "Next (zero-shot baseline eval, no training):"
echo "  TOKENIZED_DATASET_PATH=$OUT_DIR sbatch slurm/eval_greberta_baseline.sh"
echo ""
echo "Next (continue pretraining GreBerta on this corpus -- 'model 6'):"
echo "  RUN_NAME=model6-greberta-continuation \\"
echo "    PRETRAINED_MODEL=bowphs/GreBerta \\"
echo "    TOKENIZED_DATASET_PATH=$OUT_DIR \\"
echo "    CONFIG_PATH=configs/train.yaml sbatch slurm/pretrain.sh"
