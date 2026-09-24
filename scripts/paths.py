"""Shared path helpers for the GreekBERT repo.

Single source of truth for locating the project root and the current
HuggingFace-format model, so that no script or config hardcodes an absolute
path or a date-stamped directory name (the old `hf_format918` problem).

Resolution order for the current model:
  1. $MODEL_DIR              (explicit override; exported by slurm/env.sh)
  2. <repo>/models/current   (symlink maintained by scripts/set_current_model.sh)
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/ or wsd/ or sbert/ -> repo root is one level up
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def model_dir(default: str | None = None) -> str:
    """Return the HF-format model directory to fine-tune from."""
    override = os.environ.get("MODEL_DIR")
    if override:
        return str(Path(override).expanduser().resolve())

    current = MODELS_DIR / "current"
    if current.exists():
        return str(current.resolve())

    if default is not None:
        return default

    raise FileNotFoundError(
        f"No model to load: $MODEL_DIR is unset and {current} does not exist.\n"
        f"Run: scripts/set_current_model.sh <models/SOME_EXPORT>  (or export MODEL_DIR=...)"
    )


def repo_path(*parts: str) -> Path:
    """Absolute path to something inside the repo."""
    return REPO_ROOT.joinpath(*parts)
