#!/usr/bin/env bash
# =============================================================================
# spaCy stage on its own (iterate on parser/lemmatizer without re-pretraining).
# Replaces:
#   spacy_code/train_from_hf.sh
#   spacy_code/train_lemma_from_hf.sh
#   spacy_code/scripts/train.sh
#
#   RUN_NAME=spacy-v1 sbatch slurm/spacy.sh
#   RUN_NAME=spacy-v1 SPACY_CONFIG=configs/gpu_default_freq_lemma.cfg sbatch slurm/spacy.sh
#   RUN_NAME=spacy-v1 MODEL_DIR=/path/to/hf_model sbatch slurm/spacy.sh
#
# Submit from the project root so logs land in logs/.
# =============================================================================
#SBATCH -J GreekBERT-SpaCy
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32g
#SBATCH --reservation=new_gpu
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/GreekBERT-SpaCy.%j.%N.out
#SBATCH --error=logs/GreekBERT-SpaCy.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

source "slurm/env.sh"

SPACY_CONFIG="${SPACY_CONFIG:-configs/gpu_default.cfg}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/current}"
export MODEL_DIR
SPACY_OUT="${SPACY_OUT:-$RUN_DIR/spacy}"

SPACY_CODE_ARG=()
if [[ "$SPACY_CONFIG" == *freq_lemma* ]]; then
  SPACY_CODE_ARG=(--code "custom_components/lemmatizer.py")
fi

# Reports the REAL node identity: SLURM_JOB_ID can be inherited by a shell that
# never ran under sbatch, so printing it alone was misleading.
echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Profile   : $PROFILE ($SBATCH_PARTITION / $SBATCH_GRES)"
echo "spaCy cfg : $SPACY_CONFIG"
echo "Base model: $MODEL_DIR"
echo "Output    : $SPACY_OUT"

load_env spacy

stage "spaCy train ($SPACY_CONFIG)"
py_run spacy "$SPACY_DIR" \
  spacy train "$SPACY_CONFIG" \
    --paths.train corpus/train.spacy \
    --paths.dev corpus/dev.spacy \
    --gpu-id 0 \
    --output "$SPACY_OUT" \
    "${SPACY_CODE_ARG[@]}"
stage_done "spaCy train"

stage "spaCy evaluate"
py_run spacy "$SPACY_DIR" \
  spacy evaluate "$SPACY_OUT/model-best" corpus/test.spacy \
    --gpu-id 0 \
    --output "$SPACY_OUT/eval_results.json"
stage_done "spaCy evaluate -> $SPACY_OUT/eval_results.json"
