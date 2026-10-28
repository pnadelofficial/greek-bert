#!/usr/bin/env bash
# =============================================================================
# Zero-shot MLM validation eval for bowphs/GreBerta against THIS PROJECT's own
# held-out validation split -- the real, controlled counterpart to GreBerta's
# own self-reported (Perseus-only) numbers. Same spirit as
# slurm/eval_odycy_baseline.sh / slurm/eval_latincy_baseline.sh on the spaCy
# side, but this one scores a language model's MLM loss, not a parser, so it
# runs scripts/eval_pretrained_mlm.py in the project's OWN root uv env (no
# isolated venv needed -- GreBerta is a plain RobertaForMaskedLM, nothing
# exotic). See that script's docstring for the exact metrics (same
# val_loss/val_ppl/masked-token-acc train.py's own validation loop computes,
# so this is directly comparable to the 5 pretraining arms' best_val_loss).
#
# REQUIRES data/greberta_tokenized_open_greek_dataset_v2 to already exist --
# build it first:
#   sbatch slurm/prepare_greberta_corpus.sh
#
#   sbatch slurm/eval_greberta_baseline.sh
#   MODEL=bowphs/GreBerta TOKENIZED_DATASET_PATH=data/some-other-dataset \
#     sbatch slurm/eval_greberta_baseline.sh
#
# Submit from the project root so logs land in logs/.
# =============================================================================
#SBATCH -J GreekBERT-EvalGreBerta
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --output=logs/GreekBERT-EvalGreBerta.%j.%N.out
#SBATCH --error=logs/GreekBERT-EvalGreBerta.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

export RUN_NAME="${RUN_NAME:-baseline-eval-greberta}"
source "slurm/env.sh"

MODEL="${MODEL:-bowphs/GreBerta}"
TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-$PROJECT_ROOT/data/greberta_tokenized_open_greek_dataset_v2}"
OUT="${OUT:-results/greberta_mlm_eval.json}"

if [[ ! -d "$TOKENIZED_DATASET_PATH" ]]; then
  echo "ERROR: $TOKENIZED_DATASET_PATH does not exist yet." >&2
  echo "       Build it first: sbatch slurm/prepare_greberta_corpus.sh" >&2
  exit 1
fi

echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Python   : $PYTHON_ENV"
echo "Model    : $MODEL"
echo "Dataset  : $TOKENIZED_DATASET_PATH"

load_env pretrain

stage "zero-shot MLM eval ($MODEL)"
py_run pretrain "$PROJECT_ROOT" \
  python scripts/eval_pretrained_mlm.py \
    --model "$MODEL" \
    --tokenized-dataset-path "$TOKENIZED_DATASET_PATH" \
    --output "$OUT"
stage_done "eval -> $OUT"

echo ""
echo "Done. Fold $OUT into results/model-comparison.md as a new external"
echo "baseline row (same pattern as the odyCy/LatinCy spaCy rows)."
