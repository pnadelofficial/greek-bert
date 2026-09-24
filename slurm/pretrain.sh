#!/usr/bin/env bash
# =============================================================================
# Pretraining only (MLM, RoBERTa-style). Replaces scripts/train.sh.
#
#   RUN_NAME=myrun sbatch slurm/pretrain.sh
#   RUN_NAME=myrun PROFILE=h200x8 sbatch slurm/pretrain.sh
#   RUN_NAME=myrun PYTHON_ENV=conda sbatch slurm/pretrain.sh
#
# Pass the number of GPUs as the first sbatch argument (default 8):
#   RUN_NAME=myrun sbatch slurm/pretrain.sh 4
# =============================================================================
#SBATCH -J GreekBERT
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=32g
#SBATCH --reservation=new_gpu
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/GreekBERT.%j.%N.out
#SBATCH --error=logs/GreekBERT.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

# NOTE: --output/--error resolve relative to the SUBMITTING directory, so submit
# from the project root:   RUN_NAME=myrun sbatch slurm/pretrain.sh
# (logs/ is gitignored; old runs dropped these .out/.err files in scripts/).

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

NUM_GPUS="${1:-8}"

# Reports the REAL node identity: SLURM_JOB_ID can be inherited by a shell that
# never ran under sbatch, so printing it alone was misleading.
echo "Host     : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job      : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "Profile  : $PROFILE ($SBATCH_PARTITION / $SBATCH_GRES)"
echo "Run name : $RUN_NAME"
echo "Python   : $PYTHON_ENV"
echo "GPUs     : $NUM_GPUS"

# train.py resolves relative paths against the config file's directory, so the
# config uses ../../runs/<RUN_NAME>/... and we submit from the project root.
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/train.yaml}"

load_env pretrain

stage "pretraining (${NUM_GPUS} GPUs)"
py_run pretrain "$PROJECT_ROOT" \
  torchrun --standalone --nnodes 1 --nproc_per_node="$NUM_GPUS" \
  scripts/train.py --config-path="$CONFIG_PATH"

stage_done "pretraining"
echo "Checkpoints : $CHECKPOINT_DIR"
echo "TensorBoard : $TENSORBOARD_DIR"
echo "Next        : RUN_NAME=$RUN_NAME sbatch slurm/posttrain.sh"
