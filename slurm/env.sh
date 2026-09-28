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

# sbatch inherits the SUBMITTING shell's exported environment by default (no
# script here uses --export=NONE). This repo runs TWO independent uv projects
# from one script (root for pretrain/convert/WSD, spacy_code/ for spaCy), each
# selected purely by `cd`-ing into it before `uv run` (see py_run() below). If
# the submitting shell has VIRTUAL_ENV / UV_PROJECT_ENVIRONMENT / UV_PROJECT
# set -- e.g. from a `source .venv/bin/activate` or a convenience export left
# in .bashrc -- `uv run` can honor that override regardless of cwd:
# UV_PROJECT_ENVIRONMENT in particular forces ALL uv projects in the shell to
# share one .venv path, ignoring which directory you cd'd into. That silently
# makes the spaCy stage run against the ROOT project's venv (no cupy<14, no
# spacy-transformers) instead of spacy_code/.venv, surfacing as
# "ValueError: Cannot use GPU, CuPy is not installed" even when a fresh manual
# `cd spacy_code && uv run ...` test in the same shell looked fine. Strip
# these so every stage's project selection is decided ONLY by py_run's cd,
# never by whatever the submitting shell happened to have exported.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT UV_PROJECT

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

# Fail fast, before burning GPU allocation on `spacy train`, if thinc's own
# GPU-availability check (thinc.compat) would fail. This does NOT just test
# `import cupy` -- thinc/compat.py sets has_cupy = True only if `import cupy`,
# `import cupy.cublas`, AND `import cupyx` ALL succeed, and thinc.util.require_gpu
# raises "Cannot use GPU, CuPy is not installed" whenever has_cupy is False,
# regardless of WHICH of the three failed. A plain `import cupy` can succeed
# while `import cupy.cublas` fails (it dynamically loads libcublas.so at
# import time, a different failure mode than the top-level package missing),
# so testing only the top-level import can pass here and still hit this exact
# error inside `spacy train`. Reproduce thinc's exact check so a failure here
# pinpoints which specific import breaks and why.
#
# Also note: root pyproject.toml depends on `cupy-cuda12x[ctk]>=14.2.0` -- the
# `ctk` extra (only on cupy>=14, confirmed via PyPI metadata) pulls in
# pip-installable CUDA toolkit libs (cuda-toolkit[cublas,cudart,...]) so cupy
# does not need a system CUDA module. spacy_code/pyproject.toml pins
# `cupy-cuda12x<14` (required for spaCy/thinc compat) and that major version
# has NO ctk extra at all, so its cuBLAS/etc. discovery depends on a system
# CUDA module being loaded, or on auto-detecting the nvidia-*-cu12 packages
# pulled in transitively by torch -- a plausible reason cupy.cublas
# specifically fails to load even when the cupy package itself is installed.
check_spacy_gpu() {
  echo "Checking cupy/GPU availability in the spacy_code environment (thinc's exact check)..."
  if ! py_run spacy "$SPACY_DIR" python -c '
import sys
print("  interpreter :", sys.executable)
print("  sys.prefix  :", sys.prefix)

import cupy
print("  cupy        :", cupy.__file__, cupy.__version__)

import cupy.cublas
print("  cupy.cublas :", cupy.cublas.__file__)

import cupyx
print("  cupyx       :", cupyx.__file__)

print("  gpu count   :", cupy.cuda.runtime.getDeviceCount())

import thinc.compat as c
print("  thinc has_cupy     :", c.has_cupy)
print("  thinc has_cupy_gpu :", c.has_cupy_gpu)
print("  thinc has_gpu      :", c.has_gpu)
'; then
    echo "ERROR: thinc cannot see a usable GPU in the spacy_code environment" >&2
    echo "       py_run resolved above. This reproduces thinc/compat.py's own" >&2
    echo "       check (import cupy, cupy.cublas, cupyx, then getDeviceCount())" >&2
    echo "       BEFORE 'spacy train' runs, to avoid burning a GPU allocation" >&2
    echo "       on a doomed run. Whichever import/line printed above last is" >&2
    echo "       the one that failed -- if it's cupy.cublas or cupyx specifically" >&2
    echo "       (not the top-level cupy import), this is a missing/incompatible" >&2
    echo "       cuBLAS shared library, not a venv-selection problem: spacy_code's" >&2
    echo "       cupy-cuda12x<14 has no [ctk] extra (unlike the root project's" >&2
    echo "       cupy-cuda12x[ctk]>=14.2.0), so it needs cuBLAS/cuSPARSE/etc" >&2
    echo "       discoverable via a loaded system CUDA module or via nvidia-*-cu12" >&2
    echo "       pip packages already present in spacy_code/.venv (pulled in by" >&2
    echo "       torch). Check 'module avail cuda' / whatever CUDA module the" >&2
    echo "       cluster provides, and whether it needs to be loaded for this job." >&2
    exit 1
  fi
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
