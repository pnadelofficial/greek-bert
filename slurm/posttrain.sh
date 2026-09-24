#!/usr/bin/env bash
# =============================================================================
# Post-training pipeline. Replaces the three ~90%-identical scripts:
#   scripts/train_and_eval_both.sh
#   scripts/train_and_eval_spacy.sh
#   scripts/train_and_eval_wsd.sh
#
# Stages are selected with STAGES instead of forked files:
#   convert  torch checkpoint -> HuggingFace format
#   wsd      word sense disambiguation fine-tune + eval
#   spacy    spaCy parser/morph/lemmatizer train + eval
#
#   RUN_NAME=myrun sbatch slurm/posttrain.sh                  # convert,wsd,spacy
#   RUN_NAME=myrun STAGES=convert sbatch slurm/posttrain.sh   # convert only
#   RUN_NAME=myrun STAGES=wsd,spacy sbatch slurm/posttrain.sh # skip convert
#
# Submit from the project root so logs land in logs/.
# =============================================================================
#SBATCH -J GreekBERT-Post
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=32g
#SBATCH --reservation=new_gpu
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/GreekBERT-Post.%j.%N.out
#SBATCH --error=logs/GreekBERT-Post.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

source "slurm/env.sh"

STAGES="${STAGES:-convert,wsd,spacy}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/train.yaml}"

# The HF-format model that WSD + spaCy fine-tune from. Defaults to this run's
# export, so `convert -> wsd -> spacy` chains automatically within one RUN_NAME.
MODEL_DIR="${MODEL_DIR:-$HF_EXPORT_DIR}"
export MODEL_DIR

# spaCy config: gpu_default.cfg (no custom lemmatizer) or gpu_default_freq_lemma.cfg
SPACY_CONFIG="${SPACY_CONFIG:-configs/gpu_default.cfg}"
SPACY_CODE_ARG=()
if [[ "$SPACY_CONFIG" == *freq_lemma* ]]; then
  SPACY_CODE_ARG=(--code "custom_components/lemmatizer.py")
fi

run_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# Fail fast if a downstream stage was asked to run without a model to fine-tune
# from. This happens when STAGES skips `convert` and the run has no export yet.
if { run_stage wsd || run_stage spacy; } && [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: no HuggingFace model at MODEL_DIR=$MODEL_DIR" >&2
  echo "       Either include the convert stage (STAGES=convert,wsd,spacy)," >&2
  echo "       or point MODEL_DIR at an existing export, e.g.:" >&2
  echo "         MODEL_DIR=$MODELS_DIR/current STAGES=wsd,spacy sbatch slurm/posttrain.sh" >&2
  echo "       Available exports:" >&2
  "$PROJECT_ROOT/scripts/set_current_model.sh" --list >&2 || true
  exit 1
fi

# Reports the REAL node identity: SLURM_JOB_ID can be inherited by a shell that
# never ran under sbatch, so printing it alone was misleading.
echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Profile  : $PROFILE ($SBATCH_PARTITION / $SBATCH_GRES)"
echo "Run name : $RUN_NAME"
echo "Python   : $PYTHON_ENV"
echo "Stages   : $STAGES"
echo "Model    : $MODEL_DIR"
echo "spaCy cfg: $SPACY_CONFIG"

# --- 1. convert -------------------------------------------------------------
if run_stage convert; then
  stage "convert checkpoint -> HuggingFace format"
  load_env pretrain
  py_run pretrain "$PROJECT_ROOT" \
    python scripts/convert_to_hf.py --config-path="$CONFIG_PATH"
  stage_done "convert -> $HF_EXPORT_DIR"
fi

if ! run_stage wsd && ! run_stage spacy; then
  echo ""; echo "Done. (no downstream stages selected)"
  exit 0
fi

# --- 2. WSD -----------------------------------------------------------------
if run_stage wsd; then
  stage "post-train + evaluate WSD (glaux)"
  # WSD lives in the ROOT uv project (same deps as pretraining).
  load_env wsd
  py_run wsd "$PROJECT_ROOT" python wsd/wsd.py
  stage_done "WSD"
fi

# --- 3. spaCy ---------------------------------------------------------------
if run_stage spacy; then
  stage "post-train spaCy ($SPACY_CONFIG)"
  # spaCy is a SEPARATE uv project pinned to python 3.12 + cupy<14.
  # This used to be `cd ../spacy`, which does not exist — the directory is
  # spacy_code/. That silently killed the spaCy stage on every run.
  load_env spacy
  py_run spacy "$SPACY_DIR" \
    spacy train "$SPACY_CONFIG" \
      --paths.train corpus/train.spacy \
      --paths.dev corpus/dev.spacy \
      --gpu-id 0 \
      --output "$RUN_DIR/spacy" \
      "${SPACY_CODE_ARG[@]}"

  stage_done "spaCy train"

  stage "evaluate spaCy model"
  py_run spacy "$SPACY_DIR" \
    spacy evaluate "$RUN_DIR/spacy/model-best" corpus/test.spacy \
      --gpu-id 0 \
      --output "$RUN_DIR/spacy/eval_results.json"
  stage_done "spaCy eval -> $RUN_DIR/spacy/eval_results.json"
fi

echo ""
echo "Pipeline complete. Artifacts under: $RUN_DIR"
