#!/usr/bin/env bash
# =============================================================================
# SBERT contrastive training (Greek->English, Krahn et al.).
# Replaces BOTH of:
#   scripts/train_sbert_contrastive.sh  (h100:1 on tuftsai, `python ../sbert/...`)
#   sbert/train-contrastive.sh          (l40s:4, `python contrastive.py`)
#
# The old scripts/train_sbert_contrastive.sh reached into a sibling directory
# (`python ../sbert/contrastive.py`) — that cross-directory coupling is what let
# the `cd ../spacy` bug survive unnoticed. Each job now cds explicitly.
#
#   RUN_NAME=sbert-v1 sbatch slurm/sbert.sh              # l40s:4
#   RUN_NAME=sbert-v1 PROFILE=h100x1 sbatch slurm/sbert.sh
#
# Submit from the project root so logs land in logs/.
# =============================================================================
#SBATCH -J GreekSBERT
#SBATCH --output=logs/GreekSBERT.%j.%N.out
#SBATCH --error=logs/GreekSBERT.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Base model for the sentence encoder. This is the Jan-20 export that SBERT was
# originally trained from — NOT models/current, which now points at hf_format918.
export BASE_MODEL="${BASE_MODEL:-$MODELS_DIR/greekbert-jan20}"
export SBERT_OUTPUT="${SBERT_OUTPUT:-$RUN_DIR/sbert}"

# Reports the REAL node identity: SLURM_JOB_ID can be inherited by a shell that
# never ran under sbatch, so printing it alone was misleading.
echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Profile   : $PROFILE ($SBATCH_PARTITION / $SBATCH_GRES)"
echo "Run name  : $RUN_NAME"
echo "Base model: $BASE_MODEL"
echo "Output    : $SBERT_OUTPUT"

load_env wsd   # SBERT used the general_purpose_textgen env under conda

stage "SBERT contrastive training"
py_run wsd "$PROJECT_ROOT/sbert" python contrastive.py
stage_done "SBERT -> $SBERT_OUTPUT"
