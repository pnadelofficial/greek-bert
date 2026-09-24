#!/usr/bin/env bash
# ============================================================================
# Shared environment + path setup for every SLURM job in this repo.
#
# Source this from every job script:   source "$(dirname "$0")/env.sh"
#
# It defines the project paths, the two Python environments, and the
# GPU/partition profiles. Nothing here runs a job.
#
# OVERRIDES (export before sbatch, or edit the defaults below):
#   RUN_NAME     names this run's artifacts under runs/<RUN_NAME>/   (REQUIRED)
#   PROFILE      gpu profile name from the case statement below
#   PYTHON_ENV   uv | conda                          (default: uv)
#   MODEL_DIR    HF-format model dir used by WSD + spaCy post-training
# ============================================================================

# ---------------------------------------------------------------------------
# Paths — no absolute paths anywhere else in the repo
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

export SPACY_DIR="$PROJECT_ROOT/spacy_code"   # NOT "../spacy" — that path does not exist
export DATA_DIR="$PROJECT_ROOT/data"
export MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$PROJECT_ROOT/tokenizers/modernbert-greek-tokenizer}"

# Where pretraining writes checkpoints / tensorboard logs.
# Derived from RUN_NAME so runs never overwrite each other.
export RUN_NAME="${RUN_NAME:?RUN_NAME must be set (names runs/<RUN_NAME>/ artifacts)}"
export RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
export CHECKPOINT_DIR="$RUN_DIR/checkpoints"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-$RUN_DIR/hf_format}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# ---------------------------------------------------------------------------
# Python environment
#
# There are genuinely TWO projects here and they cannot be merged:
#   root        python 3.13, transformers 5.x   -> pretraining, WSD, SBERT
#   spacy_code  python 3.12, spacy + cupy<14    -> spaCy parser/lemmatizer
#
# PYTHON_ENV=uv     uses the checked-in uv.lock per project (current setup)
# PYTHON_ENV=conda  uses the legacy conda envs (kept for old job arrays)
# ---------------------------------------------------------------------------
PYTHON_ENV="${PYTHON_ENV:-uv}"

# Legacy conda env names, only used when PYTHON_ENV=conda
CONDA_ENV_PRETRAIN="${CONDA_ENV_PRETRAIN:-torchrun}"
CONDA_ENV_WSD="${CONDA_ENV_WSD:-general_purpose_textgen}"
CONDA_ENV_SPACY="${CONDA_ENV_SPACY:-spacy_gpu}"

load_env() {
  # load_env <pretrain|wsd|spacy>
  local which_env="$1"
  case "$PYTHON_ENV" in
    uv)
      module load uv
      ;;
    conda)
      module load modtree/deprecated
      module load anaconda/2023.07.tuftsai
      local env_name
      case "$which_env" in
        pretrain) env_name="$CONDA_ENV_PRETRAIN" ;;
        wsd)      env_name="$CONDA_ENV_WSD" ;;
        spacy)    env_name="$CONDA_ENV_SPACY" ;;
        *) echo "load_env: unknown env '$which_env'" >&2; return 1 ;;
      esac
      echo "conda activate $env_name"
      source activate "$env_name"
      ;;
    *)
      echo "PYTHON_ENV must be 'uv' or 'conda', got '$PYTHON_ENV'" >&2
      return 1
      ;;
  esac
}

# Wraps a command in the right runner for the chosen env.
# Usage: py_run <pretrain|wsd|spacy> <project_dir> <cmd...>
py_run() {
  local which_env="$1"; shift
  local project_dir="$1"; shift
  case "$PYTHON_ENV" in
    uv)   (cd "$project_dir" && uv run "$@") ;;
    conda) (cd "$project_dir" && "$@") ;;
  esac
}

# ---------------------------------------------------------------------------
# GPU / partition profiles.
# The old scripts each hardcoded a different partition + GPU type with no
# explanation. Pick the profile that matches what's actually free.
# ---------------------------------------------------------------------------
PROFILE="${PROFILE:-h200x8}"

case "$PROFILE" in
  # 8x H200 — full pretraining (was scripts/train.sh)
  h200x8)  SBATCH_PARTITION=gpu  SBATCH_GRES="gpu:h200:8" SBATCH_CPUS=64 SBATCH_MEM=128g SBATCH_TIME=4-00:00:00 ;;
  # 8x B200, reserved — the pretrain + convert + WSD + spaCy pipeline
  b200x8)  SBATCH_PARTITION=gpu  SBATCH_GRES="gpu:b200:8" SBATCH_CPUS=16 SBATCH_MEM=32g  SBATCH_TIME=2-00:00:00 SBATCH_RESERVATION=new_gpu ;;
  # 1x H200 — post-training stages (WSD, spaCy)
  h200x1)  SBATCH_PARTITION=gpu  SBATCH_GRES="gpu:h200:1" SBATCH_CPUS=16 SBATCH_MEM=32g  SBATCH_TIME=2-00:00:00 ;;
  # 4x L40S — SBERT contrastive (was sbert/train-contrastive.sh)
  l40sx4)  SBATCH_PARTITION=gpu  SBATCH_GRES="gpu:l40s:4" SBATCH_CPUS=16 SBATCH_MEM=32g  SBATCH_TIME=2-00:00:00 ;;
  # 1x H100 on the tuftsai partition (was scripts/train_sbert_contrastive.sh)
  h100x1)  SBATCH_PARTITION=tuftsai SBATCH_GRES="gpu:h100:1" SBATCH_CPUS=16 SBATCH_MEM=32g SBATCH_TIME=2-00:00:00 ;;
  custom)  : "${SBATCH_PARTITION:?}" "${SBATCH_GRES:?}" ;;
  *) echo "Unknown PROFILE '$PROFILE' (see slurm/env.sh)" >&2; return 1 2>/dev/null || exit 1 ;;
esac

export SBATCH_PARTITION SBATCH_GRES SBATCH_CPUS SBATCH_MEM SBATCH_TIME
export SBATCH_RESERVATION

# ---------------------------------------------------------------------------
# Logging helper — every stage announces itself and its exit status, so a
# failed stage can never masquerade as success the way bare `echo`s did.
# ---------------------------------------------------------------------------
stage() {
  echo ""
  echo "=============================================================="
  echo "[$(date '+%F %T')] STAGE START: $*"
  echo "=============================================================="
}

stage_done() {
  echo "[$(date '+%F %T')] STAGE OK: $*"
}
