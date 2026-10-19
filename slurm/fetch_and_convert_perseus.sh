#!/usr/bin/env bash
# =============================================================================
# One-off CPU job: fetch the official UD Ancient Greek Perseus treebank,
# combine it with the existing PROIEL treebank (scripts/combine_treebanks.sh),
# and convert both to spaCy's binary .spacy format.
#
# Background: spacy_code/assets/ only ever had PROIEL
# (grc_proiel-ud-{train,dev,test}.conllu) -- Perseus was never added, despite
# the whole eval methodology (and results/model-comparison.md's own header)
# assuming "PROIEL+Perseus combined." Every spaCy number recorded so far was
# actually scored on PROIEL alone. This fixes the corpus; it does NOT rerun
# any training -- see slurm/rerun_spacy_all_models.sh for that, run AFTER
# this finishes.
#
# Known, already-documented consequence of combining these two treebanks
# (odyCy's own paper flags the same things): proper nouns are PROPN in
# PROIEL but plain NOUN in Perseus; PROIEL has no punctuation, Perseus does;
# PROIEL's morph feature set is richer; lemma conventions differ (Perseus
# dash-compounds vs PROIEL parenthetical optional letters); Perseus's
# official release has a known beta-code->Unicode conversion bug on some
# characters. Not bugs in this script -- don't chase them as one.
#
# NEEDS OUTBOUND INTERNET ACCESS -- same caveat as slurm/fetch_external_corpora.sh:
# submit to whichever partition actually has network access; adjust -p below
# if 'batch' isn't it.
#
#   sbatch slurm/fetch_and_convert_perseus.sh
#
# Submit from the project root so logs land in logs/ and asset paths resolve.
# =============================================================================
#SBATCH -J GreekBERT-FetchPerseus
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=logs/GreekBERT-FetchPerseus.%j.%N.out
#SBATCH --error=logs/GreekBERT-FetchPerseus.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

export RUN_NAME="${RUN_NAME:-fetch-perseus}"
source "slurm/env.sh"

echo "Host : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job  : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"

# ADJUST THIS if your existing PROIEL files live under a differently-named
# subdirectory. scripts/combine_treebanks.sh's `assets/**/*-train.conllu` is
# a plain POSIX sh glob (the script is #!/usr/bin/env sh, not bash) -- `**`
# has no bash-globstar recursive meaning here, it just matches ONE level of
# subdirectory. So Perseus needs its own assets/<name>/ dir, same shape as
# whatever PROIEL's directory is already called. Check first:
#   ls spacy_code/assets/
# and fix PERSEUS_DIR_NAME below if your PROIEL layout differs from this
# guess (the UD repo's own directory name).
PERSEUS_DIR_NAME="${PERSEUS_DIR_NAME:-UD_Ancient_Greek-Perseus}"
ASSETS_DIR="$SPACY_DIR/assets"
PERSEUS_DIR="$ASSETS_DIR/$PERSEUS_DIR_NAME"

stage "fetch UD Ancient Greek Perseus treebank"
if [[ -d "$PERSEUS_DIR" ]]; then
  echo "Already present at $PERSEUS_DIR -- skipping clone."
  echo "Delete it first if you want to re-fetch a different release."
else
  git clone --depth 1 \
    https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus.git \
    "$PERSEUS_DIR"
fi

# Fail loud rather than silently combining an incomplete set if the UD repo's
# file naming doesn't match what we expect.
for split in train dev test; do
  f="$(find "$PERSEUS_DIR" -maxdepth 1 -name "*-ud-$split.conllu" | head -1)"
  if [[ -z "$f" ]]; then
    echo "ERROR: no *-ud-$split.conllu found in $PERSEUS_DIR" >&2
    echo "       Check the clone actually succeeded and the UD repo's file" >&2
    echo "       naming matches this script's assumption." >&2
    exit 1
  fi
  echo "  found: $f"
done
stage_done "Perseus -> $PERSEUS_DIR"

# --- combine (same script that already handles PROIEL) ----------------------
stage "combine treebanks (assets/**/*-train.conllu etc -> assets/{train,dev,test}.conllu)"
(cd "$SPACY_DIR" && sh scripts/combine_treebanks.sh)
wc -l "$ASSETS_DIR"/train.conllu "$ASSETS_DIR"/dev.conllu "$ASSETS_DIR"/test.conllu
stage_done "combined conllu written to $ASSETS_DIR/{train,dev,test}.conllu"

# --- convert to spaCy's binary format ----------------------------------------
# -n/--n-sents groups N treebank sentences into one Doc (needed for sents_f to
# be a meaningful metric at all -- a single-sentence Doc can't test sentence-
# boundary detection). Defaulting to 10. ADJUST N_SENTS (env override) if you
# know this doesn't match whatever was used for the original PROIEL-only
# conversion -- an inconsistent value between treebanks would itself be a
# confound, the same class of problem this whole fix is trying to remove.
N_SENTS="${N_SENTS:-10}"

load_env spacy
for split in train dev test; do
  stage "spacy convert ($split, -n $N_SENTS)"
  py_run spacy "$SPACY_DIR" \
    python -m spacy convert "assets/$split.conllu" corpus/ \
      --converter conllu -n "$N_SENTS"
  stage_done "corpus/$split.spacy"
done

echo ""
echo "Done. spacy_code/corpus/{train,dev,test}.spacy now contain combined"
echo "PROIEL+Perseus data."
echo "Next: bash slurm/rerun_spacy_all_models.sh"
