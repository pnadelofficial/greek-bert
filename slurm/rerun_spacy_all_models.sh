#!/usr/bin/env bash
# =============================================================================
# Rerun the spaCy stage (train + evaluate, NO lemma lexicon) for all 5 models
# against the corrected corpus/{train,dev,test}.spacy (PROIEL+Perseus
# combined). Run slurm/fetch_and_convert_perseus.sh FIRST and wait for it to
# finish -- this script does not check that for you beyond the file-existence
# guard below.
#
# This is the baseline pass only, matching the original (pre-lexicon)
# methodology -- it intentionally does NOT touch the lemma lexicon or
# configs/gpu_default_freq_lemma.cfg. Once this finishes and you've rebuilt
# the lexicon yourself against the new corpus (build_lemma_lexicon.py reads
# corpus/train.spacy, so it needs rebuilding too), retrain the lexicon config
# on whichever model wins this pass -- the same way models 1/2/5 got it
# before, just pointed at a possibly different winner this time.
#
# Submits 5 INDEPENDENT sbatch jobs (one per model) rather than running them
# sequentially in one job, so they run in parallel across separate GPU
# allocations -- each only needs the single GPU spaCy actually uses (see the
# -n 1 --gres=gpu:b200:1 override, same as every other spaCy-only run in this
# project).
#
#   bash slurm/rerun_spacy_all_models.sh
#
# Run directly from the project root (this is a plain launcher script, NOT
# itself meant to be submitted via sbatch -- it just calls sbatch 5 times).
# =============================================================================
set -euo pipefail

CORPUS_DIR="spacy_code/corpus"
for split in train dev test; do
  if [[ ! -f "$CORPUS_DIR/$split.spacy" ]]; then
    echo "ERROR: $CORPUS_DIR/$split.spacy not found." >&2
    echo "       Run this from the project root, and make sure" >&2
    echo "       slurm/fetch_and_convert_perseus.sh has already finished." >&2
    exit 1
  fi
done

RUN_NAMES=(
  arm1-aristo-continuation
  model2-aristo-continuation
  arm2-bert-fresh
  arm4-modernbert-scratch
  modernbert-stage3-final
)

for name in "${RUN_NAMES[@]}"; do
  echo "Submitting spaCy rerun for $name ..."
  RUN_NAME="$name" STAGES=spacy SPACY_CONFIG=configs/gpu_default.cfg \
    sbatch -n 1 --gres=blackwell:b200:1 slurm/posttrain.sh
done

echo ""
echo "All 5 submitted -- check squeue for progress."
echo "These numbers supersede the 'PROIEL+Perseus combined' spaCy table in"
echo "results/model-comparison.md, which was actually PROIEL-only. The model"
echo "ranking (model 1 as overall winner) was based on that incomplete corpus"
echo "and may change once these land."
