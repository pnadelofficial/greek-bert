#!/usr/bin/env bash
# =============================================================================
# Point models/current at a HuggingFace-format export.
#
# This replaces hardcoding date-stamped dirs like `hf_format918` in wsd.py and
# spacy_code/configs/gpu_default.cfg. Those now read models/current (or $MODEL_DIR).
#
#   scripts/set_current_model.sh models/Sep18Run
#   scripts/set_current_model.sh /abs/path/to/any/hf/model
#   scripts/set_current_model.sh --list
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
CURRENT="$MODELS_DIR/current"

if [[ "${1:-}" == "--list" || -z "${1:-}" ]]; then
  echo "Available exports in $MODELS_DIR:"
  find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d \
    -exec sh -c 'test -f "$1/config.json" && echo "  $(basename "$1")"' _ {} \; 2>/dev/null | sort
  # Run-scoped exports (posttrain's convert stage) live in
  # runs/<RUN_NAME>/hf_format, NOT under models/. List them too: they were
  # previously invisible here, which made the stale models/current pointer
  # hard to notice (the export a run just produced never appeared in this
  # list, so "current -> some-old-model" looked normal).
  for d in "$REPO_ROOT"/runs/*/hf_format; do
    [[ -f "$d/config.json" ]] || continue
    run_name="$(basename "$(dirname "$d")")"
    echo "  runs/$run_name/hf_format   (run-scoped export)"
  done
  echo ""
  if [[ -e "$CURRENT" ]]; then
    echo "current -> $(readlink -f "$CURRENT")"
  else
    echo "current -> (not set)"
  fi
  exit 0
fi

TARGET="$(cd "$1" 2>/dev/null && pwd || true)"
if [[ -z "$TARGET" ]]; then
  echo "error: '$1' is not an existing directory" >&2
  exit 1
fi
if [[ ! -f "$TARGET/config.json" ]]; then
  echo "error: '$TARGET' has no config.json -- not a HuggingFace model dir" >&2
  exit 1
fi

mkdir -p "$MODELS_DIR"
ln -sfn "$TARGET" "$CURRENT"
echo "models/current -> $TARGET"
