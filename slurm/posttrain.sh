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

# Resolve MODEL_DIR to its real path BEFORE anything else. models/current is a
# manually maintained symlink, and a header that prints the LINK (not the
# target) hid a stale pointer for one whole WSD+spaCy job: "post-train arm1"
# actually fine-tuned models/greekbert-2025-09-18, and the mismatch only
# surfaced ~20 minutes in, on wsd.py's "Encoder:" line, AFTER compute started.
_realpath() {
  readlink -f "$1" 2>/dev/null \
    || python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$1" 2>/dev/null \
    || echo "$1"
}
MODEL_DIR_REAL="$(_realpath "$MODEL_DIR")"
# Canonicalize the run's export too, so a symlinked project root cannot make
# the mismatch check below fire on the DEFAULT (correct) MODEL_DIR.
HF_EXPORT_DIR_REAL="$(_realpath "$HF_EXPORT_DIR")"

# spaCy config: gpu_default.cfg (no custom lemmatizer) or gpu_default_freq_lemma.cfg
SPACY_CONFIG="${SPACY_CONFIG:-configs/gpu_default.cfg}"
SPACY_CODE_ARG=()
if [[ "$SPACY_CONFIG" == *freq_lemma* ]]; then
  SPACY_CODE_ARG=(--code "custom_components/lemmatizer.py")
fi

# Same fix as slurm/spacy.sh: the checked-in cfg hardcodes
# components.transformer.model.name="../models/current", a static string that never
# saw $MODEL_DIR. Without this override, spaCy trained from whatever models/current
# pointed to instead of this run's own export ($HF_EXPORT_DIR) -- the same class of
# bug as the WSD mismatch this file already guards against above.
SPACY_CODE_ARG+=(--components.transformer.model.name "$MODEL_DIR_REAL")

run_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# Fail fast if a downstream stage was asked to run without a model to fine-tune
# from. This happens when STAGES skips `convert` and the run has no export yet.
if { run_stage wsd || run_stage spacy; } && [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: no HuggingFace model at MODEL_DIR=$MODEL_DIR" >&2
  echo "       Either include the convert stage (STAGES=convert,wsd,spacy)," >&2
  echo "       or point MODEL_DIR at an existing export — normally THIS run's" >&2
  echo "       own export, which the convert stage produces:" >&2
  echo "         RUN_NAME=$RUN_NAME STAGES=convert,wsd,spacy sbatch slurm/posttrain.sh" >&2
  echo "       Do NOT default to models/current: it points at the last manually" >&2
  echo "       promoted model, which is often NOT the run being post-trained" >&2
  echo "       (that is exactly how arm1's WSD run silently evaluated an old" >&2
  echo "       Sep-2025 export)." >&2
  echo "       Available exports:" >&2
  "$PROJECT_ROOT/scripts/set_current_model.sh" --list >&2 || true
  exit 1
fi

# Warn before burning compute when the job's model is not this run's own
# export. The default MODEL_DIR is $HF_EXPORT_DIR (this run), so this only
# fires when the caller explicitly pointed elsewhere — e.g. a stale
# models/current. It is a warning, not an error: re-evaluating an older model
# under a new RUN_NAME is a legitimate ablation.
if { run_stage wsd || run_stage spacy; } && [[ -f "$HF_EXPORT_DIR/config.json" ]] \
   && [[ "$MODEL_DIR_REAL" != "$HF_EXPORT_DIR_REAL" ]]; then
  echo "WARNING: MODEL_DIR is not this run's own export."
  echo "         RUN_NAME     : $RUN_NAME"
  echo "         run's export : $HF_EXPORT_DIR"
  echo "         MODEL_DIR    : $MODEL_DIR -> $MODEL_DIR_REAL"
  echo "         wsd/spacy will fine-tune from $MODEL_DIR_REAL, NOT from"
  echo "         runs/$RUN_NAME/hf_format. If that is not intentional (e.g."
  echo "         models/current still points at an old export), re-point it:"
  echo "           scripts/set_current_model.sh $HF_EXPORT_DIR"
  echo ""
fi

# Reports the REAL node identity: SLURM_JOB_ID can be inherited by a shell that
# never ran under sbatch, so printing it alone was misleading.
echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Profile  : $PROFILE ($SBATCH_PARTITION / $SBATCH_GRES)"
echo "Run name : $RUN_NAME"
echo "Python   : $PYTHON_ENV"
echo "Stages   : $STAGES"
if [[ "$MODEL_DIR_REAL" != "$MODEL_DIR" ]]; then
  echo "Model    : $MODEL_DIR -> $MODEL_DIR_REAL"
else
  echo "Model    : $MODEL_DIR"
fi
echo "spaCy cfg: $SPACY_CONFIG"

# --- 1. convert -------------------------------------------------------------
if run_stage convert; then
  stage "convert checkpoint -> HuggingFace format"
  load_env pretrain
  py_run pretrain "$PROJECT_ROOT" \
    python scripts/convert_to_hf.py --config-path="$CONFIG_PATH"
  stage_done "convert -> $HF_EXPORT_DIR"
  # models/current is deliberately NOT auto-updated: parallel posttrain jobs
  # would race on one symlink. Standalone WSD/spaCy/sbert jobs that read
  # models/current must be re-pointed explicitly.
  echo "Tip: standalone jobs that read models/current still point at their"
  echo "     previous target. To promote this export, run:"
  echo "       scripts/set_current_model.sh $HF_EXPORT_DIR"
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
  check_spacy_gpu
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
