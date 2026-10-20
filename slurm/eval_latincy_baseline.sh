#!/usr/bin/env bash
# =============================================================================
# Install LatinCy's grc_dep_web_trf (https://huggingface.co/latincy/grc_dep_web_trf)
# into its OWN isolated venv and evaluate it against THIS PROJECT's own
# corpus/test.spacy, same methodology as slurm/eval_odycy_baseline.sh (see
# that script for the full rationale on isolated venvs / CPU-only / gold-
# tokenized evaluation via scripts/eval_external_spacy_model.py, which this
# reuses unchanged).
#
# Known methodological caveat specific to this model, not odyCy: its
# packaged `functions.py` (bundled into the wheel) registers a custom
# tokenizer that normalizes 4 different Unicode encodings of the elision
# apostrophe to a canonical form BEFORE its `lookup_lemmatizer` component
# ever sees a token. Our gold-tokenized eval bypasses every pipeline's own
# tokenizer entirely (feeds pre-split gold words directly, for fair alignment
# against UD's standard tokenization, same as odyCy) -- which means this
# model's lookup-lemmatizer does NOT get the normalized input it expects.
# Its LEMMA number here is likely an underestimate of this model's true
# capability for that reason specifically; every other metric is unaffected
# (they don't depend on that normalization step).
#
# Unlike odyCy, this model targets a NEWER spaCy (>=3.8.11,<3.9.0, per its own
# meta.json) and its own declared requirements already include `click>=8.0`
# explicitly -- the LatinCy authors appear to already be aware of the
# typer/click packaging issue odyCy hit. Deliberately NOT copying the
# numpy<2.0/typer pins from eval_odycy_baseline.sh here without evidence
# they're needed for this newer spacy release -- if uv's resolver hits the
# same class of old-package-vs-too-new-transitive-dependency error we saw
# twice with odyCy, add the matching pin the same way (see that script's
# comments for the diagnostic pattern).
#
# NEEDS OUTBOUND INTERNET ACCESS -- same caveat as the other fetch/eval
# scripts in this directory; adjust -p below if 'batch' isn't the right
# partition.
#
#   sbatch slurm/eval_latincy_baseline.sh
#
# Submit from the project root so logs land in logs/ and corpus paths resolve.
# =============================================================================
#SBATCH -J GreekBERT-EvalLatinCy
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=03:00:00
#SBATCH --output=logs/GreekBERT-EvalLatinCy.%j.%N.out
#SBATCH --error=logs/GreekBERT-EvalLatinCy.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

# Same convention as every other script here: submit from the project root,
# don't self-locate via ${BASH_SOURCE[0]} (sbatch stages the submitted script
# into /var/spool/slurm/... on this cluster before running it, so that
# pattern resolves to the wrong place for a directly-submitted script -- see
# the comment history in slurm/eval_odycy_baseline.sh for how this was found).
if [[ ! -d slurm || ! -d spacy_code ]]; then
  echo "ERROR: this doesn't look like the project root (no slurm/ or" >&2
  echo "       spacy_code/ in the current directory). Submit from the" >&2
  echo "       project root: sbatch slurm/eval_latincy_baseline.sh" >&2
  exit 1
fi

echo "Host : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job  : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"
echo "CWD  : $(pwd)"

ENV_DIR="${LATINCY_ENV_DIR:-.latincy_eval_venv}"
WHEEL_URL="${LATINCY_WHEEL_URL:-https://huggingface.co/latincy/grc_dep_web_trf/resolve/main/grc_dep_web_trf-3.8.4-py3-none-any.whl}"
TEST_SPACY="${TEST_SPACY:-spacy_code/corpus/test.spacy}"
OUT="${OUT:-results/latincy_eval_results.json}"

module load uv 2>/dev/null || true

echo ""
echo "=== Creating isolated venv for LatinCy (NOT spacy_code's, NOT odyCy's) ==="
if [[ -d "$ENV_DIR" ]]; then
  echo "Already present at $ENV_DIR -- skipping creation. Delete it first to rebuild."
else
  # Python 3.11, same reasoning as the odyCy script: a well-supported era for
  # prebuilt wheels across this whole dependency stack, avoiding a forced
  # source build.
  uv venv "$ENV_DIR" --python 3.11
  # Pin exactly what grc_dep_web_trf's own meta.json declares as its
  # requirements (spacy, spacy-transformers, latincy-preprocess), plus click
  # as a harmless explicit safety net (their own requirements already
  # include it) -- one combined install call so uv resolves the whole graph
  # together rather than in separate passes.
  uv pip install --python "$ENV_DIR/bin/python" \
    "spacy>=3.8.11,<3.9.0" "spacy-transformers>=1.3.9,<1.4.0" \
    "latincy-preprocess[grc]>=0.5.1" "click>=8.0" "$WHEEL_URL"
fi

echo ""
echo "=== Running evaluation against $TEST_SPACY ==="
mkdir -p "$(dirname "$OUT")"
"$ENV_DIR/bin/python" scripts/eval_external_spacy_model.py \
    --pipeline grc_dep_web_trf \
    --test-spacy "$TEST_SPACY" \
    --output "$OUT"

echo ""
echo "Done. Results in $OUT -- fold these into results/model-comparison.md"
echo "as a second external reference row. Remember: LEMMA here likely"
echo "underestimates this model's true capability (see the normalization"
echo "caveat at the top of this script) -- flag that alongside the number."
