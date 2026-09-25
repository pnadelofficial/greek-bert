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
#SBATCH -n 8
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
# Same visibility fix as posttrain.sh: models/current is a manually maintained
# symlink, and printing the LINK (not the target) hid a stale pointer — the
# arm1 incident fine-tuned models/greekbert-2025-09-18 for hours before the
# resolved path surfaced. Resolve it, print the target, and fail fast if the
# target is gone (previously spaCy crashed deep inside `spacy train`).
MODEL_DIR_REAL="$(readlink -f "$MODEL_DIR" 2>/dev/null \
  || python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$MODEL_DIR" 2>/dev/null \
  || echo "$MODEL_DIR")"
if [[ ! -f "$MODEL_DIR_REAL/config.json" ]]; then
  echo "ERROR: no HuggingFace model at MODEL_DIR=$MODEL_DIR (resolved: $MODEL_DIR_REAL)" >&2
  echo "       Point MODEL_DIR at an export, or pick one with:" >&2
  echo "         scripts/set_current_model.sh --list" >&2
  exit 1
fi
SPACY_OUT="${SPACY_OUT:-$RUN_DIR/spacy}"

SPACY_CODE_ARG=()
if [[ "$SPACY_CONFIG" == *freq_lemma* ]]; then
  SPACY_CODE_ARG=(--code "custom_components/lemmatizer.py")
fi

# The checked-in cfg hardcodes components.transformer.model.name="../models/current"
# (see the comment at that key). That is a static string in a versioned file, so it
# never saw $MODEL_DIR -- spacy train always fine-tuned whatever models/current
# happened to point to, even when MODEL_DIR was explicitly resolved and validated
# above to something else. Override it on the CLI with the resolved path so the
# encoder spaCy actually trains from matches what this script printed as "Base model".
SPACY_CODE_ARG+=(--components.transformer.model.name "$MODEL_DIR_REAL")

# Reports the REAL node identity: SLURM_JOB_ID can be inherited by a shell that
# never ran under sbatch, so printing it alone was misleading.
echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Profile   : $PROFILE ($SBATCH_PARTITION / $SBATCH_GRES)"
echo "spaCy cfg : $SPACY_CONFIG"
if [[ "$MODEL_DIR_REAL" != "$MODEL_DIR" ]]; then
  echo "Base model: $MODEL_DIR -> $MODEL_DIR_REAL"
else
  echo "Base model: $MODEL_DIR"
fi
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
