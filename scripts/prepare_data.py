#!/usr/bin/env python3
"""Build the pretraining datasets with a genre-disjoint, reproducible split.

Audit item 2: the previous split (a random 20% of *rows* in
notebooks/testing.ipynb) cut contiguous 512-token chunks from the same
documents, so train and validation shared authors, dialects, and
near-verbatim repeated passages within a document. Validation loss was
optimistic, and it was the ONLY signal used to select checkpoints. This is
the single biggest measurement problem in the repo.

CORRECTION (2026-10): an earlier version of this note claimed "Open Greek
duplicates Perseus texts across editions." That's wrong -- data/*.parquet
("Open Greek") is the Ancient Greek subset of Harvard's institutional-books
dataset (https://huggingface.co/datasets/institutional/institutional-books-hl),
not Perseus-derived at all, per direct confirmation from whoever built this
corpus. It has no known content overlap with Perseus / First1KGreek (the
sources load_first1k below pulls from), which matters for
docs/EXPERIMENTS-2X2.md's planned ModernBERT two-stage comparison: that
comparison depends on the "new corpus" (this one) and the "ancient corpus"
stage (First1KGreek) being genuinely distinct sources, and they are.

This script rebuilds the split at the RIGHT granularity:

  * The 20% holdout is taken at the **document** level (one of the source
    parquet files / one europarl file / one Wikipedia page), so a document
    never appears on both sides of the split. Adjacent chunks of one document
    are all-train or all-val.
  * The split is **seeded** (default seed in TrainingConfig) and the chosen
    holdout documents are recorded in the saved dataset's
    ``metadata["holdout_documents"]``, so a later run's validation number is
    comparable to this run's.
  * ``wikidata`` (Modern Greek Wikipedia) is EXCLUDED by default. It is
    domain-shift dilution for an Ancient Greek model; the audit asks for an
    ablation without it. Pass --include-wikidata to reproduce the old mix.
  * europarl_el is Modern Greek as well; it is INCLUDED by default only to
    match the historical corpus (the old notebook mixed it in). It is flagged
    in the metadata and easy to drop with --exclude-europarl if you want the
    strict-Ancient-Greek corpus.
  * The old dataset also stored a ``labels`` column that was an exact copy of
    ``input_ids`` (masks/labels are generated dynamically per step by
    mlm_masking, so nothing ever read it). The new format stores only
    input_ids/attention_mask. train.py still accepts old-format datasets.

Usage (submit from the project root, same as the SLURM stages):

  python scripts/prepare_data.py \
      --out data/nlpaueb_tokenized_open_greek_dataset

  # ablation: Modern Greek Wikipedia back in the mix
  python scripts/prepare_data.py --include-wikidata \
      --out data/nlpaueb_tokenized_open_greek_dataset_with_wikidata

  # ModernBERT two-stage comparison (docs/EXPERIMENTS-2X2.md, model 5):
  # stage 1 -- modern Greek only (GreekBERT-equivalent corpus)
  python scripts/prepare_data.py --exclude-open-greek --exclude-europarl \
      --include-wikidata --include-mc4 \
      --tokenizer tokenizers/modernbert-greek-tokenizer-v2 \
      --out data/modernbert_stage1_modern_greek
  # (Europarl is excluded above only because it's folded in separately if
  # desired -- drop --exclude-europarl to include data/europarl_el/ too.)

  # stage 2 -- ancient Greek only (First1KGreek, aristoBERTo-style corpus;
  # requires slurm/fetch_external_corpora.sh to have populated
  # data/first1kgreek/ first)
  python scripts/prepare_data.py --exclude-open-greek --exclude-europarl \
      --include-first1k \
      --tokenizer tokenizers/modernbert-greek-tokenizer-v2 \
      --out data/modernbert_stage2_ancient_greek

The raw inputs must live under data/:
  data/*.parquet                       Open Greek corpus (one file per document;
                                        Harvard institutional-books, Ancient Greek
                                        subset -- see the correction note above)
  data/europarl_el/*.txt                Modern Greek Europarl
  data/wikidata_cache/                  (only with --include-wikidata)
  data/first1kgreek/                    (only with --include-first1k; populated by
                                         slurm/fetch_external_corpora.sh)
  (mC4 Greek, --include-mc4, streams from the HF Hub / its local cache --
   nothing needs to pre-exist under data/ for it specifically)
Output is a HuggingFace ``DatasetDict`` with train/test splits, saved with
``save_to_disk`` (the format train.py loads).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import TrainingConfig, tokenize_and_prepare_mlm, resolve_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                   help="Directory holding the raw corpus files.")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "nlpaueb_tokenized_open_greek_dataset"),
                   help="Where to save_to_disk the train/test DatasetDict.")
    p.add_argument("--tokenizer", default="nlpaueb/bert-base-greek-uncased-v1",
                   help="Tokenizer id/path matching the intended model. The "
                        "modernbert-greek-tokenizer is a BPE that cannot "
                        "produce the [CLS]/[SEP]/[PAD]/[MASK] special tokens this "
                        "BERT-format pipeline needs; use the WordPiece one for "
                        "this arm (the cased-BPE arm is a separate experiment).")
    p.add_argument("--test-size", type=float, default=0.20,
                   help="Fraction of DOCUMENTS held out for validation.")
    p.add_argument("--seed", type=int, default=TrainingConfig.seed,
                   help="Seed for the document split (default: TrainingConfig.seed).")
    p.add_argument("--include-wikidata", action="store_true",
                   help="Include Modern Greek Wikipedia (default: excluded, per audit item 2).")
    p.add_argument("--exclude-europarl", action="store_true",
                   help="Also exclude Modern Greek Europarl (strict Ancient Greek only).")
    p.add_argument("--exclude-open-greek", action="store_true",
                   help="Exclude data/*.parquet (Harvard institutional-books Ancient "
                        "Greek). For building a corpus for a DIFFERENT stage of the "
                        "ModernBERT two-stage comparison (docs/EXPERIMENTS-2X2.md) "
                        "that should not contain this project's own new corpus.")
    p.add_argument("--include-mc4", action="store_true",
                   help="Include Greek mC4 (legacy-datasets/mc4, config 'el') -- a "
                        "large web-crawl-based modern Greek corpus, standing in for "
                        "OSCAR (GreekBERT's own source), which is gated/suspended as "
                        "of 2026-10. Streamed and capped at --mc4-max-bytes, not "
                        "downloaded in full.")
    p.add_argument("--mc4-max-bytes", type=int, default=15 * 1024**3,
                   help="Cap on mC4 text pulled (default 15 GB). mC4 is enormous; "
                        "this is 'a large modern Greek foundation,' not a faithful "
                        "full reproduction of OSCAR's size.")
    p.add_argument("--include-first1k", action="store_true",
                   help="Include First1KGreek (data/first1kgreek/, populated by "
                        "slurm/fetch_external_corpora.sh) -- a public Ancient Greek "
                        "corpus, genuinely distinct from this project's own "
                        "data/*.parquet (Harvard institutional-books), used as the "
                        "'ancient Greek stage' corpus for the ModernBERT two-stage "
                        "comparison (an approximation of aristoBERTo's own ancient-"
                        "Greek training stage, not a byte-identical reproduction).")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Max tokens per training sequence (BERT: 512).")
    p.add_argument("--ignore-length", type=int, default=16,
                   help="Drop documents shorter than this many tokens.")
    p.add_argument("--text-shard-bytes", type=int, default=512 * 1024 * 1024,
                   help="Max total text bytes per raw-text shard. The whole "
                        "corpus is NEVER concatenated into one pyarrow string "
                        "column: HuggingFace datasets' from_dict/concat hit "
                        "pyarrow's 2 GB 'offset overflow' on a single string "
                        "array, so docs are split into shards of at most this "
                        "many bytes and tokenized shard-by-shard. (512 MiB "
                        "leaves 4x headroom under the 2 GB limit.)")
    return p.parse_args()


def load_open_greek(data_dir: Path) -> dict[str, str]:
    """Open Greek: data/*.parquet, one document per file.

    Returns {doc_id: text}. doc_id is the parquet file stem, which is stable
    across rebuilds (unlike a row index), so the held-out set is reproducible.
    """
    import pandas as pd

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {data_dir} (expected data/*.parquet)")
    docs: dict[str, str] = {}
    for f in files:
        df = pd.read_parquet(f)
        if "text" not in df.columns:
            print(f"  WARNING: {f.name} has no 'text' column ({list(df.columns)}); skipping")
            continue
        for row, text in enumerate(df["text"].astype(str)):
            if text.strip():
                # One document id per (file, row): a parquet file can hold more
                # than one document. The id is stable across rebuilds because
                # file names and row order in the source files are fixed.
                docs[f"{f.stem}#{row}"] = text
    print(f"  Open Greek: {len(docs)} documents from {len(files)} parquet files")
    return docs


def load_europarl(data_dir: Path) -> dict[str, str]:
    """europarl_el/*.txt: one document per file (Modern Greek)."""
    ep_dir = data_dir / "europarl_el"
    files = sorted(ep_dir.glob("*.txt")) if ep_dir.is_dir() else []
    if not files:
        print("  Europarl: (none found; skipped)")
        return {}
    docs = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        text = re.sub(r"<[^>]+>", "", text).strip()  # strip the residual markup
        if text:
            docs[f"europarl#{f.stem}"] = text
    print(f"  Europarl: {len(docs)} documents")
    return docs


def load_wikipedia(data_dir: Path) -> dict[str, str]:
    """Modern Greek Wikipedia (excluded by default)."""
    from datasets import load_dataset

    try:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.el",
                            cache_dir=str(data_dir / "wikidata_cache"))["train"]
    except Exception as e:  # offline cluster: the cached copy may be all we have
        raise SystemExit(f"--include-wikidata failed to load Wikipedia: {e}")
    docs = {f"wikidata#{i}": (t or "") for i, t in enumerate(wiki["text"]) if (t or "").strip()}
    print(f"  Wikidata (Modern Greek Wikipedia): {len(docs)} documents")
    return docs


def load_mc4(data_dir: Path, max_bytes: int) -> dict[str, str]:
    """Greek mC4 (legacy-datasets/mc4, config 'el'), streamed and capped.

    Standing in for OSCAR (GreekBERT's own modern-Greek web-crawl source),
    which is gated/access-suspended as of 2026-10. mC4 is itself a
    deduplicated, cleaned Common Crawl derivative, same role as OSCAR, and is
    NOT gated. Streamed (not downloaded in full -- mC4's Greek config alone is
    far larger than needed here) and capped at `max_bytes` of raw text.
    """
    from datasets import load_dataset

    try:
        mc4 = load_dataset("legacy-datasets/mc4", "el", split="train", streaming=True,
                           cache_dir=str(data_dir / "mc4_cache"), trust_remote_code=True)
    except Exception as e:  # offline cluster: pre-fetch via slurm/fetch_external_corpora.sh
        raise SystemExit(f"--include-mc4 failed to load mC4: {e}")
    docs: dict[str, str] = {}
    total_bytes = 0
    for i, row in enumerate(mc4):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        docs[f"mc4#{i}"] = text
        total_bytes += len(text.encode("utf-8"))
        if total_bytes >= max_bytes:
            break
    print(f"  mC4 (Greek web crawl, OSCAR stand-in): {len(docs)} documents, "
          f"{total_bytes / 1e9:.2f} GB (capped at {max_bytes / 1e9:.2f} GB)")
    return docs


def load_first1k(data_dir: Path) -> dict[str, str]:
    """First1KGreek (data/first1kgreek/, a shallow git clone populated by
    slurm/fetch_external_corpora.sh): every Greek work composed between Homer
    and 250 AD not already in the Perseus Digital Library. TEI/EpiDoc XML,
    one file per work; text is every surviving line/paragraph's content,
    teiHeader (metadata) and editorial <note>s excluded.
    """
    import xml.etree.ElementTree as ET

    root = data_dir / "first1kgreek"
    if not root.is_dir():
        raise SystemExit(
            f"--include-first1k requires {root} to exist first. Run:\n"
            f"  sbatch slurm/fetch_external_corpora.sh"
        )
    files = sorted(root.glob("data/**/*.xml"))
    files = [f for f in files if f.name != "__cts__.xml"]
    if not files:
        raise SystemExit(f"no work XML files found under {root}/data/ -- clone may be incomplete")

    TEI_NS = "{http://www.tei-c.org/ns/1.0}"
    docs: dict[str, str] = {}
    n_failed = 0
    for f in files:
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            n_failed += 1
            continue
        body = tree.getroot().find(f".//{TEI_NS}text")
        if body is None:
            continue
        # Drop editorial <note> (often Latin/English commentary, not Greek text).
        for note in body.findall(f".//{TEI_NS}note"):
            note.clear()
        text = " ".join(body.itertext())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            docs[f"first1k#{f.stem}"] = text
    if n_failed:
        print(f"  First1KGreek: {n_failed} files failed to parse (skipped)")
    print(f"  First1KGreek: {len(docs)} documents from {len(files)} XML files")
    return docs


def _shard_docs(doc_ids: Sequence[str], docs: dict[str, str],
                max_bytes: int) -> list[list[str]]:
    """Split doc ids into shards of at most `max_bytes` total text bytes.

    Never concatenates all texts into one pyarrow string column (that is the
    2 GB offset overflow the corpus triggers); it only groups doc IDS, and
    each shard is small enough that its own text column is safe.
    """
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for d in doc_ids:
        b = len(docs[d].encode("utf-8"))
        if cur and cur_bytes + b > max_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(d)
        cur_bytes += b
    if cur:
        shards.append(cur)
    return shards


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    from datasets import Dataset, DatasetDict, concatenate_datasets
    from transformers import AutoTokenizer

    print("Loading raw corpus...")
    docs: dict[str, str] = {}
    if not args.exclude_open_greek:
        docs.update(load_open_greek(data_dir))
    if not args.exclude_europarl:
        docs.update(load_europarl(data_dir))
    if args.include_wikidata:
        docs.update(load_wikipedia(data_dir))
    if args.include_mc4:
        docs.update(load_mc4(data_dir, args.mc4_max_bytes))
    if args.include_first1k:
        docs.update(load_first1k(data_dir))
    if not docs:
        raise SystemExit(
            "no documents loaded -- check your --include-*/--exclude-* flags "
            "(e.g. --exclude-open-greek --exclude-europarl with no --include-* "
            "flag set produces an empty corpus)"
        )

    total_bytes = sum(len(t.encode("utf-8")) for t in docs.values())
    print(f"  corpus: {len(docs)} documents, {total_bytes / 1e9:.2f} GB of text")

    doc_ids = sorted(docs)  # deterministic order before the seeded shuffle
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)
    n_holdout = max(1, int(round(args.test_size * len(doc_ids))))
    holdout, train_ids = doc_ids[:n_holdout], doc_ids[n_holdout:]

    print(f"\nSplit (seed={args.seed}, test_size={args.test_size}): "
          f"{len(train_ids)} train docs / {len(holdout)} holdout docs")
    holdout_by_source = {}
    for d in holdout:
        holdout_by_source[d.split("#", 1)[0]] = holdout_by_source.get(d.split("#", 1)[0], 0) + 1
    print(f"  holdout documents by source: {json.dumps(holdout_by_source, indent=4)}")

    # Hub id ('nlpaueb/...') or local dir ('tokenizers/modernbert-greek-tokenizer')
    # — resolve_model_path keeps both working from any cwd.
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(args.tokenizer))

    def tokenize_shard(shard_ids: list[str]) -> Dataset:
        """One shard: raw texts in, tokenized sequences out.

        `examples` is a plain list of strings, not a Dataset: that keeps the
        tokenizer call shape identical on every `datasets` version and keeps
        the text alive only for this shard.
        """
        res = tokenize_and_prepare_mlm(
            {"text": [docs[d] for d in shard_ids]}, tokenizer,
            is_main_process=True,
            chunk_size=args.chunk_size, ignore_length=args.ignore_length,
        )
        ds = Dataset.from_dict(res)
        # Drop the python lists promptly; the next shard does not need them.
        res.clear()
        return ds

    def tokenize_split(split_name: str, split_ids: list[str]) -> Dataset:
        shards = _shard_docs(split_ids, docs, args.text_shard_bytes)
        print(f"\nTokenizing {split_name}: {len(split_ids)} docs in {len(shards)} "
              f"text shards (<= {args.text_shard_bytes / 1e6:.0f} MB each)")
        parts = []
        for i, shard in enumerate(shards):
            part = tokenize_shard(shard)
            parts.append(part)
            print(f"  {split_name} shard {i + 1}/{len(shards)}: "
                  f"{len(shard)} docs -> {len(part)} sequences "
                  f"(running total {sum(len(p) for p in parts)})")
            del part
            gc.collect()
        if len(parts) == 1:
            return parts[0]
        # Concatenating equal-size shards is safe: each part's string columns
        # are already gone (only int32 token ids remain), so the combined
        # int arrays are far under any pyarrow offset limit.
        return concatenate_datasets(parts)

    train_ds = tokenize_split("train", train_ids)
    gc.collect()
    test_ds = tokenize_split("test", holdout)

    out = DatasetDict({"train": train_ds, "test": test_ds})
    out.set_format(type="torch")

    # Provenance: what the split was, so validation numbers stay comparable.
    try:
        out.metadata = {
            "split_seed": args.seed,
            "test_size": args.test_size,
            "holdout_documents": holdout,
            "holdout_by_source": holdout_by_source,
            "n_docs_train": len(train_ids),
            "n_docs_test": len(holdout),
            "chunk_size": args.chunk_size,
            "text_shard_bytes": args.text_shard_bytes,
            "include_wikidata": args.include_wikidata,
            "exclude_europarl": args.exclude_europarl,
            "exclude_open_greek": args.exclude_open_greek,
            "include_mc4": args.include_mc4,
            "include_first1k": args.include_first1k,
            "tokenizer": args.tokenizer,
            "note": ("Document-level split: a document never appears in both "
                     "train and test. No 'labels' column is stored; masks are "
                     "generated dynamically per step (mlm_masking)."),
        }
    except Exception:
        pass  # older datasets versions: metadata is a nicety, not load-bearing

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_path} ...")
    out.save_to_disk(str(out_path))
    print(f"Saved: train={len(out['train'])} sequences, test={len(out['test'])} sequences")
    print(f"\nTrain pretraining with:\n  TOKENIZED_DATASET_PATH={out_path} "
          f"RUN_NAME=<name> sbatch slurm/pretrain.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
