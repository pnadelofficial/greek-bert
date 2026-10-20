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

# NOTE: deliberately NOT using `${BASH_SOURCE[0]}`-based self-location here
# (the `cd "$(dirname "${BASH_SOURCE[0]}")/.."` pattern slurm/env.sh uses when
# SOURCED). This script is run directly by sbatch, and on this cluster sbatch
# stages/copies the submitted script into /var/spool/slurm/... before
# executing it -- so for a directly-submitted script, ${BASH_SOURCE[0]}
# resolves to that spool path, not the original slurm/eval_odycy_baseline.sh
# location (confirmed: this exact bug created .odycy_eval_venv under
# /var/spool/slurm the first time). Every other script in this project avoids
# this because they `source "slurm/env.sh"` (a literal relative-path string,
# which only needs the CWD to already be the project root -- true because
# every job here is submitted from there by convention; sbatch preserves the
# submission directory as the job's CWD even though it relocates the script
# file itself). Just rely on that same convention directly instead.
if [[ ! -d slurm || ! -d spacy_code ]]; then
  echo "ERROR: this doesn't look like the project root (no slurm/ or" >&2
  echo "       spacy_code/ in the current directory). Submit from the" >&2
  echo "       project root: sbatch slurm/eval_odycy_baseline.sh" >&2
  exit 1
fi

echo "Host : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job  : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "CWD  : $(pwd)"

ENV_DIR="${ODYCY_ENV_DIR:-.odycy_eval_venv}"
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
  # ONE install call for everything together -- letting uv resolve odyCy's
  # full dependency tree (spacy, spacy-transformers, thinc, torch, ...) in a
  # single pass rather than two separate calls that could each resolve
  # somewhat independently.
  #
  # typer/click pinned explicitly -- confirmed root cause (not a guess):
  # spaCy 3.7.5's own code does `from click import NoSuchOption` directly in
  # spacy/cli/_util.py, assuming click is always pulled in transitively via
  # typer (true for the typer versions spaCy 3.7.x actually shipped against).
  # Very recent typer releases (0.2x+) moved their real implementation into a
  # separate `typer-slim` package that does NOT pull in click by default --
  # uv's resolver, given only spaCy's own loose/unconstrained typer
  # requirement, picked the newest typer (confirmed via the install log:
  # typer==0.27.3 + typer-slim==0.24.0, no click at all), breaking spaCy's
  # unstated assumption. Pinning typer back into the era spaCy 3.7.x was
  # actually built against restores the old (click-bundling) behavior; the
  # explicit click pin is a belt-and-suspenders safety net on top of that.
  uv pip install --python "$ENV_DIR/bin/python" \
    "spacy>=3.7.4,<3.8.0" "typer>=0.3,<0.10" "click>=8.0,<9.0" "$WHEEL_URL"
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
