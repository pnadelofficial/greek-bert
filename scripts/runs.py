"""Run manifest + results table plumbing (audit item 12).

Every stage (pretrain, convert, WSD, spaCy, Morpheus) writes a small JSON into
`runs/<RUN_NAME>/` and upserts a row into the repo-level `results/table.md`.
Before this, results lived in `docs/archive/`, loose `.json` files and notebook
outputs, and no comparison in the repo recorded which code SHA and which config
produced it — including the baseline comparison that motivated the work.

Two artifacts:

  runs/<RUN_NAME>/manifest.json   provenance: git SHA, config hash, config dict,
                                  environment, timestamps. Written once by
                                  `record_run_provenance`.
  runs/<RUN_NAME>/metrics.json    accumulated per-stage metrics. Written by
                                  `write_metrics`, which merges rather than
                                  overwrites, so stages can run independently.
  results/table.md                one row per (run, stage, model, seed). Upserted
                                  by `record_result`.

Usage from a stage:

    from runs import record_result, write_metrics

    write_metrics("pretrain", {"val_loss": 1.98, "val_ppl": 7.2})
    record_result(stage="pretrain", model="greekbert", metric="val_ppl",
                  value=7.2, split="proiel-test")
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from paths import REPO_ROOT, repo_path

RESULTS_TABLE = repo_path("results", "table.md")

_TABLE_HEADER = (
    "| run | stage | model | metric | value | split | seed | config_hash | git_sha | updated |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def run_dir(run_name: str | None = None) -> Path:
    """Directory holding this run's artifacts.

    Resolution order mirrors slurm/env.sh: explicit argument, then $RUN_DIR,
    then runs/$RUN_NAME, then runs/_adhoc (so an un-named local run still
    produces a record instead of crashing).
    """
    if run_name:
        return repo_path("runs", run_name)
    env_dir = os.environ.get("RUN_DIR")
    if env_dir:
        return Path(env_dir)
    env_name = os.environ.get("RUN_NAME")
    if env_name:
        return repo_path("runs", env_name)
    return repo_path("runs", "_adhoc")


def git_sha() -> str:
    """Current commit, with a `-dirty` suffix if the tree has changes."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        sha = out.stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return f"{sha}-dirty" if status.stdout.strip() else sha
    except Exception:
        return "unknown"


def record_run_provenance(
    *,
    stage: str,
    config_dict: dict | None = None,
    config_hash: str | None = None,
    extra: dict | None = None,
    run_name: str | None = None,
) -> Path:
    """Write/extend runs/<RUN_NAME>/manifest.json with provenance.

    Safe to call from several stages in one run: the manifest accumulates a
    `stages` entry per stage rather than being rewritten wholesale.
    """
    directory = run_dir(run_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"

    manifest = {}
    if path.is_file():
        try:
            manifest = json.loads(path.read_text())
        except Exception:
            manifest = {}

    manifest.setdefault("run", directory.name)
    manifest.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    manifest["git_sha"] = git_sha()
    manifest["hostname"] = platform.node()
    if config_hash:
        manifest["config_hash"] = config_hash
    if config_dict is not None:
        manifest["config"] = config_dict

    stages = manifest.setdefault("stages", {})
    stages[stage] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra:
        stages[stage].update(extra)

    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path


def write_metrics(metrics: dict, *, stage: str | None = None, run_name: str | None = None) -> Path:
    """Merge `metrics` into runs/<RUN_NAME>/metrics.json.

    Merging (not overwriting) matters: convert/wsd/spaCy run as separate SLURM
    stages and each must contribute to the same file.
    """
    directory = run_dir(run_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics.json"

    existing = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}

    existing["run"] = directory.name
    existing["git_sha"] = git_sha()
    if stage:
        block = existing.setdefault("stages", {})
        block[stage] = {**block.get(stage, {}), **metrics}
    else:
        existing.update(metrics)
    existing["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str))
    return path


def record_result(
    *,
    stage: str,
    model: str,
    metric: str,
    value: float | str,
    split: str = "",
    seed: int | str = "",
    config_hash: str = "",
    run_name: str | None = None,
    table_path: Path | None = None,
) -> None:
    """Upsert one row into results/table.md, keyed by the run/stage/model/metric/split/seed tuple."""
    directory = run_dir(run_name)
    table_path = table_path or RESULTS_TABLE
    table_path.parent.mkdir(parents=True, exist_ok=True)

    key = (directory.name, stage, model, metric, split, str(seed))
    rows: dict[tuple, list[str]] = {}
    order: list[tuple] = []
    if table_path.is_file():
        for line in table_path.read_text().splitlines():
            if not line.startswith("|") or line.startswith("|---") or line.startswith("| run "):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 10:
                continue
            row_key = (cells[0], cells[1], cells[2], cells[3], cells[5], cells[6])
            rows[row_key] = cells
            order.append(row_key)

    cells = [
        directory.name,
        stage,
        model,
        metric,
        f"{value:.6g}" if isinstance(value, float) else str(value),
        split,
        str(seed),
        config_hash,
        git_sha(),
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ]
    if key not in rows:
        order.append(key)
    rows[key] = cells

    lines = ["# Results table", "", "Auto-maintained by `scripts/runs.py::record_result`.",
             "One row per (run, stage, model, metric, split, seed). Re-running a stage updates its row.", ""]
    lines.append(_TABLE_HEADER.rstrip("\n"))
    for row_key in order:
        lines.append("| " + " | ".join(rows[row_key]) + " |")
    table_path.write_text("\n".join(lines) + "\n")
