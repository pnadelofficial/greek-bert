#!/usr/bin/env bash
# =============================================================================
# Install odyCy (chcaa/grc_odycy_joint_trf) into its OWN isolated venv --
# deliberately NOT spacy_code's uv-managed one -- and evaluate it against
# THIS PROJECT's own corpus/test.spacy, the exact same held-out set every one
# of our 5 models was just scored on. This replaces the earlier comparison
# against odyCy's self-reported meta.json numbers with a real, controlled,
# apples-to-apples number.
#
# Why a separate venv: odyCy's own meta.json pins
# spacy_version:">=3.7.4,<3.8.0" -- a hard conflict with whatever newer spaCy
# spacy_code's own venv needs. Trying to make both coexist is the same class
# of problem that broke `spacy-huggingface-hub` in Colab (it forced a typer
# downgrade that broke spaCy's own CLI). An isolated venv sidesteps the
# conflict entirely instead of fighting it.
#
# Why CPU-only, deliberately: odyCy's pinned spacy<3.8 almost certainly
# resolves to a ~2023-era torch that predates CUDA kernel support for this
# cluster's B200s (Blackwell/sm_100) -- the same compatibility class of
# problem already hit with scripts/train_bpe.py. This script never calls
# spacy.require_gpu()/prefer_gpu(), so it stays on CPU regardless of what's
# installed; cupy is deliberately not installed into this venv at all. Costs
# wall-clock time for a one-off comparison, not correctness.
#
# NEEDS OUTBOUND INTERNET ACCESS -- same caveat as the other fetch-style
# scripts in this directory; adjust -p below if 'batch' isn't the right
# partition.
#
#   sbatch slurm/eval_odycy_baseline.sh
#
# Submit from the project root so logs land in logs/ and corpus paths resolve.
# =============================================================================
#SBATCH -J GreekBERT-EvalOdyCy
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=03:00:00
#SBATCH --output=logs/GreekBERT-EvalOdyCy.%j.%N.out
#SBATCH --error=logs/GreekBERT-EvalOdyCy.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Host : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job  : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"

ENV_DIR="${ODYCY_ENV_DIR:-$PROJECT_ROOT/.odycy_eval_venv}"
WHEEL_URL="${ODYCY_WHEEL_URL:-https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl}"
TEST_SPACY="${TEST_SPACY:-spacy_code/corpus/test.spacy}"
OUT="${OUT:-results/odycy_eval_results.json}"

module load uv 2>/dev/null || true

echo ""
echo "=== Creating isolated venv for odyCy (NOT spacy_code's) ==="
# Python 3.11: spacy<3.8 (odyCy's own pin) has prebuilt wheels through about
# this era. A newer default (e.g. what forced a source build in Colab) risks
# the exact "Dataset scripts are no longer supported"-style build failure we
# already hit once with a different old-pinned package.
if [[ -d "$ENV_DIR" ]]; then
  echo "Already present at $ENV_DIR -- skipping creation. Delete it first to rebuild."
else
  uv venv "$ENV_DIR" --python 3.11
  # Let uv resolve odyCy's full dependency tree (spacy, spacy-transformers,
  # thinc, torch, ...) itself rather than specifying it by hand -- pin only
  # the top-level constraint odyCy's own meta.json declares.
  uv pip install --python "$ENV_DIR/bin/python" "spacy>=3.7.4,<3.8.0"
  uv pip install --python "$ENV_DIR/bin/python" "$WHEEL_URL"
fi

echo ""
echo "=== Running evaluation against $TEST_SPACY ==="
mkdir -p "$(dirname "$OUT")"
"$ENV_DIR/bin/python" scripts/eval_external_spacy_model.py \
    --pipeline grc_odycy_joint_trf \
    --test-spacy "$TEST_SPACY" \
    --output "$OUT"

echo ""
echo "Done. Results in $OUT -- fold these into results/model-comparison.md"
echo "as the real (not self-reported) odyCy comparison row."
