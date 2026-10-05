#!/usr/bin/env bash
# =============================================================================
# One-off CPU job: fetch the external corpora needed for the ModernBERT
# two-stage comparison (docs/EXPERIMENTS-2X2.md, "model 5"):
#   stage 1 (modern Greek)  -- Greek Wikipedia (+ mC4 Greek, streamed later)
#   stage 2 (ancient Greek) -- First1KGreek
#
# NEEDS OUTBOUND INTERNET ACCESS. If this cluster's normal compute/GPU nodes
# are firewalled (common on HPC), submit this to whichever partition actually
# has network access (login node, or a "batch"-style partition) -- adjust
# -p below if 'batch' isn't it. The actual pretrain jobs then read from the
# local copies this script produces, so THEY don't need network access.
#
#   sbatch slurm/fetch_external_corpora.sh
#
# Submit from the project root so logs land in logs/ and data/ paths resolve.
# =============================================================================
#SBATCH -J GreekBERT-FetchCorpora
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --output=logs/GreekBERT-FetchCorpora.%j.%N.out
#SBATCH --error=logs/GreekBERT-FetchCorpora.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peter.nadel@tufts.edu

set -euo pipefail

export RUN_NAME="${RUN_NAME:-fetch-external-corpora}"
source "slurm/env.sh"

echo "Host : $(hostname) (SLURMD_NODENAME=${SLURMD_NODENAME:-n/a})"
echo "Job  : ${SLURM_JOB_ID:-<not under SLURM; running interactively>}"

load_env pretrain

# --- First1KGreek (stage 2: ancient Greek) -----------------------------------
# Shallow clone -- only the latest commit, not the repo's full git history.
stage "fetch First1KGreek"
FIRST1K_DIR="$PROJECT_ROOT/data/first1kgreek"
if [[ -d "$FIRST1K_DIR/data" ]]; then
  echo "Already present at $FIRST1K_DIR -- skipping clone."
  echo "Delete it first if you want to re-fetch."
else
  git clone --depth 1 https://github.com/OpenGreekAndLatin/First1KGreek.git "$FIRST1K_DIR"
fi
N_XML=$(find "$FIRST1K_DIR/data" -name '*.xml' ! -name '__cts__.xml' | wc -l | tr -d ' ')
echo "First1KGreek: $N_XML work XML files"
stage_done "First1KGreek -> $FIRST1K_DIR"

# --- Greek Wikipedia (stage 1: modern Greek) ---------------------------------
# Pre-download into data/wikidata_cache/ so prepare_data.py's --include-wikidata
# can load it later WITHOUT needing network (matches the existing pattern in
# load_wikipedia()'s "offline cluster: the cached copy may be all we have").
stage "fetch Greek Wikipedia (wikimedia/wikipedia, 20231101.el)"
py_run pretrain "$PROJECT_ROOT" python -c "
from datasets import load_dataset
ds = load_dataset('wikimedia/wikipedia', '20231101.el', cache_dir='data/wikidata_cache')['train']
print(f'Cached {len(ds)} Greek Wikipedia articles')
"
stage_done "Greek Wikipedia cached -> data/wikidata_cache/"

# --- mC4 Greek connectivity smoke-test (stage 1: modern Greek, OSCAR stand-in) ---
# NOT fully downloaded here -- prepare_data.py's --include-mc4 streams it with
# a byte cap at tokenize time (mC4's Greek config is far larger than needed).
# This just confirms the stream is reachable from THIS node/partition before
# you find out the hard way during a multi-hour tokenization job.
stage "mC4 Greek connectivity check (legacy-datasets/mc4, config 'el')"
py_run pretrain "$PROJECT_ROOT" python -c "
from datasets import load_dataset
ds = load_dataset('legacy-datasets/mc4', 'el', split='train', streaming=True, trust_remote_code=True)
row = next(iter(ds))
print('mC4 stream OK, first doc has', len(row.get('text') or ''), 'chars')
"
stage_done "mC4 Greek reachable"

echo ""
echo "All external corpora ready. Next:"
echo "  python scripts/prepare_data.py --exclude-open-greek --exclude-europarl \\"
echo "      --include-wikidata --include-mc4 \\"
echo "      --tokenizer tokenizers/modernbert-greek-tokenizer-v2 \\"
echo "      --out data/modernbert_stage1_modern_greek"
echo "  python scripts/prepare_data.py --exclude-open-greek --exclude-europarl \\"
echo "      --include-first1k \\"
echo "      --tokenizer tokenizers/modernbert-greek-tokenizer-v2 \\"
echo "      --out data/modernbert_stage2_ancient_greek"
