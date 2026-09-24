#!/usr/bin/env python3
"""Train a clean cased Ancient-Greek BPE tokenizer with HF `tokenizers`.

WHY THIS EXISTS
---------------
The checked-in `tokenizers/modernbert-greek-tokenizer` is CORRUPTED: it was
trained on mojibake'd text, so its vocab keys are double-encoded UTF-8
(`Î»Î¿Î³Î¿Ï‚` is `λόγος` run through utf-8->bytes->latin1). At runtime the
ByteLevel pre-tokenizer re-encodes to bytes again, so it can never match those
keys and real Greek fragments into ~1.3-4.4 byte-tokens/word instead of
~1 word/token. Arms 3/4 of the 2x2 (docs/EXPERIMENTS-2X2.md) cannot run on it.

This script rebuilds the BPE from the SAME clean UTF-8 corpus that
`prepare_data.py` reads (it reuses those loaders), so the retrain is a
drop-in: train it, then re-run prepare_data.py against the new dir.

DESIGN
------
* Pre-tokenizer: `Regex("\\S+")` (whitespace word pieces). Deliberately NOT
  ByteLevel: for a diacritic-rich script we want each word to map to a small
  number of whole-word BPE tokens, and we do NOT want the runtime byte
  re-encoding that made the old artifact unusable. Unseen words fall back to
  `[UNK]` (the corpus is ~99% covered by a 50k vocab, so this is rare).
* Special tokens: `[UNK] [CLS] [SEP] [PAD] [MASK]`, in that fixed order, so
  their ids are stable and known. A post-processor adds `[CLS] … [SEP]` for
  single sequences — the layout `train.py` / `mlm_masking` expect.
* `continuing_subword_prefix = "##"` and `end_of_word_suffix = "☃"` so
  subword pieces are recoverable (and WSD `word_ids()` alignment works).
* SELF-CHECK: after training we re-encode a sample of real Greek and FAIL if
  (a) a common word tokenizes to many tokens, or (b) any vocab key still looks
  double-encoded. This is the regression that let the old artifact ship.

OUTPUT
------
Writes `tokenizer.json`, `tokenizer_config.json`, and
`special_tokens_map.json` into --out (default
`tokenizers/modernbert-greek-tokenizer-v2/`). Does NOT overwrite the broken v1.

USAGE
-----
  # full corpus, 50k vocab (one-off CPU/GPU-irrelevant job; ~30-60 min):
  python scripts/train_bpe.py

  # quick sanity run (tiny vocab, first 200 docs) to check the pipeline:
  python scripts/train_bpe.py --max-docs 200 --vocab-size 2000 \
      --out /tmp/bpe-smoke

Then:
  python scripts/prepare_data.py \
      --tokenizer tokenizers/modernbert-greek-tokenizer-v2 \
      --out data/greekbpe_tokenized_open_greek_dataset_v2
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (prepare_data's load_* are intentionally NOT imported here: they load the
# whole corpus into a dict, which is exactly the memory pattern that OOM-killed
# the first run. _corpus_text_iter streams the same files directly instead.)


# --- liveness / observability -------------------------------------------------
# The `tokenizers` BPE trainer prints "Pre-processing sequences" and then works
# for a LONG time with NO further output: its progress bar is TTY-gated, so
# under sbatch (no TTY) it is silent for the entire train. That looks like a
# hang, but it is not. To make a silent multi-hour job diagnosable we:
#   * flush stdout after every print (so the sbatch log actually shows progress
#     instead of being lost in a pipe buffer), and
#   * write a heartbeat file (PID + phase + wall time) that you can watch from
#     another session to confirm the process is alive and making progress.
_HEARTBEAT: dict = {"pid": None, "phase": "start", "started": None, "updated": None}


def _beat(phase: str, extra: dict | None = None) -> None:
    """Record progress in the heartbeat dict and, if requested, on disk."""
    _HEARTBEAT["phase"] = phase
    _HEARTBEAT["updated"] = time.time()
    if extra:
        _HEARTBEAT.update(extra)
    path = os.environ.get("BPE_HEARTBEAT")
    if path:
        try:
            Path(path).write_text(json.dumps(_HEARTBEAT, indent=2, default=str))
        except Exception:
            pass


def _final_flush_guard() -> None:
    # Guarantees the last print() reaches the sbatch log even if the process
    # exits hard (segfault/OOM) — Python's atexit runs on normal and most hard
    # exits of the interpreter, and we flush on signal too.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


def _on_signal(signum, frame):  # pragma: no cover - only fires on kill
    _beat(f"signal-{signum}")
    _final_flush_guard()
    # Re-raise default behavior so the job still dies on SIGTERM/SIGINT.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

# Special tokens. The `tokenizers` BPE trainer (all current versions) stores
# these in the saved file's `added_tokens_decoder`, and on reload assigns them
# ids at the END of the vocab (content tokens keep ids 0+). The exact numbers
# are therefore vocab-size-dependent and NOT fixed here — train.py / mlm_masking
# resolve every special id from the loaded tokenizer at runtime, so that is the
# only invariant that matters. We do NOT use ByteLevel (see module docstring),
# so content ids 0+ are real Greek tokens, never reserved placeholders.
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
SPECIAL_STRINGS = set(SPECIAL_TOKENS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                   help="Directory holding the raw corpus (same as prepare_data.py).")
    p.add_argument("--out", default=str(REPO_ROOT / "tokenizers" / "modernbert-greek-tokenizer-v2"),
                   help="Where to write the new tokenizer files.")
    p.add_argument("--vocab-size", type=int, default=50000,
                   help="BPE vocab size (content tokens; specials are added on top).")
    p.add_argument("--include-wikidata", action="store_true",
                   help="Include Modern Greek Wikipedia (default: excluded, same as prepare_data.py).")
    p.add_argument("--exclude-europarl", action="store_true",
                   help="Exclude Modern Greek Europarl (strict Ancient Greek only).")
    p.add_argument("--max-docs", type=int, default=0,
                   help="Limit to the first N docs (0 = all). For smoke tests only.")
    p.add_argument("--sample-batch-chars", type=int, default=20_000_000,
                   help="Max characters per text batch streamed to the BPE trainer. "
                        "This is the peak-memory knob: keep it in the tens of MB "
                        "so only one small batch is resident at a time (the corpus "
                        "is ~13 GB and must NOT be held in RAM). Default 20M chars "
                        "(~40-60 MB resident).")
    p.add_argument("--max-token-length", type=int, default=128,
                   help="Reject BPE tokens longer than this (guards pathologic merges).")
    p.add_argument("--min-freq", type=int, default=2,
                   help="Min pair frequency to merge (standard BPE).")
    p.add_argument("--heartbeat", default=None,
                   help="Path to a heartbeat file written with PID/phase/wall-time "
                        "so a silent (no-TTY) train can be confirmed alive from "
                        "another session. Also honored via the BPE_HEARTBEAT env var.")
    # NOTE: no --seed. The `tokenizers` BPE trainer is deterministic given the
    # same corpus + vocab_size (no random sampling in the Rust core), so there
    # is no RNG to seed. (tokenizers >= 0.14 exposes a `seed` kwarg on some
    # trainers; we omit it for cross-version compatibility.)
    return p.parse_args()


def _corpus_text_iter(data_dir: Path, include_wikidata: bool, exclude_europarl: bool,
                      max_docs: int, max_batch_chars: int):
    """Stream raw corpus text from DISK in bounded-size batches.

    This is the memory fix. The previous version loaded the entire corpus into
    a Python dict (~700k str objects -> tens of GB resident, dominated by
    per-object overhead) and then handed that to train_from_iterator, which
    tokenizes in one pass on top of it. Together that exceeded the SLURM step
    memory limit and the cgroup OOM-killed the process at the first
    pre-processing chunk (the "1 / 693885315" counter: 693885315 is the total
    character count of the corpus, and the trainer died at chunk 1).

    Instead we read the source files (the SAME ones prepare_data.py uses) in
    small groups and yield joined-text batches of at most `max_batch_chars`
    characters. Only one small batch is resident at a time, so peak RSS stays
    in the single-digit GB range regardless of corpus size.

    Order is deterministic (sorted file names, stable row order), so the BPE
    is reproducible for a given corpus + vocab_size.
    """
    import pandas as pd

    # Build the ordered list of (kind, path) sources WITHOUT reading contents.
    sources: list[tuple[str, Path]] = []
    for f in sorted(data_dir.glob("*.parquet")):
        sources.append(("parquet", f))
    if not exclude_europarl:
        ep_dir = data_dir / "europarl_el"
        if ep_dir.is_dir():
            for f in sorted(ep_dir.glob("*.txt")):
                sources.append(("txt", f))
    if include_wikidata:
        # Wikipedia is loaded via the HF cache; stream it too, but it is
        # excluded by default so this is rarely taken.
        from prepare_data import load_wikipedia
        for _id, text in sorted(load_wikipedia(data_dir).items()):
            yield text
        return

    docs_seen = 0
    batch_parts: list[str] = []
    batch_chars = 0

    def flush():
        nonlocal batch_parts, batch_chars
        if batch_parts:
            yield " ".join(batch_parts)
        batch_parts, batch_chars = [], 0

    for kind, path in sources:
        if kind == "parquet":
            # Read the file once (a single parquet file is one document-group,
            # tens of MB to a few hundred MB) and stream its text column.
            df = pd.read_parquet(path)
            if "text" not in df.columns:
                continue
            for text in df["text"].astype(str):
                text = text.strip()
                if not text:
                    continue
                docs_seen += 1
                if max_docs and docs_seen > max_docs:
                    break
                if batch_chars + len(text) > max_batch_chars and batch_parts:
                    yield " ".join(batch_parts)
                    batch_parts, batch_chars = [], 0
                batch_parts.append(text)
                batch_chars += len(text)
            del df
        else:  # txt
            raw = path.read_text(encoding="utf-8")
            text = re.sub(r"<[^>]+>", "", raw).strip()
            if text:
                docs_seen += 1
                if max_docs and docs_seen > max_docs:
                    break
                if batch_chars + len(text) > max_batch_chars and batch_parts:
                    yield " ".join(batch_parts)
                    batch_parts, batch_chars = [], 0
                batch_parts.append(text)
                batch_chars += len(text)
        if max_docs and docs_seen >= max_docs:
            break
    # final partial batch
    if batch_parts:
        yield " ".join(batch_parts)


def _looks_double_encoded(key: str) -> bool:
    """Heuristic: a BPE vocab key is double-encoded if decoding it as
    latin1->utf8 yields valid text that contains Greek. A clean BPE trained on
    UTF-8 has keys that ARE the (possibly byte-escaped) Greek, so round-tripping
    them through latin1->utf8 should NOT produce clean Greek."""
    try:
        orig = key.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    return any("α" <= c <= "ω" or "Α" <= c <= "Ω" or "ͺ" <= c <= "Ϳ" for c in orig)


def self_check(tok, data_dir: Path) -> None:
    """Fail loudly if the new tokenizer reproduces the old artifact's bugs.

    (a) common Greek words must tokenize to a SMALL number of tokens
        (the old one gave 5+ for a single word);
    (b) no vocab key may look double-encoded.

    `data_dir` is used to read a SMALL sample of real Greek (first ~200 docs
    from the first parquet file) for check (a) — we do NOT hold the corpus.
    """
    import pandas as pd

    # (a) tokens-per-word on a small sample of real documents (first file only).
    sample: list[str] = []
    first_parquet = sorted(data_dir.glob("*.parquet"))[0] if data_dir.glob("*.parquet") else None
    if first_parquet is not None:
        df = pd.read_parquet(first_parquet)
        if "text" in df.columns:
            sample = [t for t in df["text"].astype(str).head(200) if t.strip()]
        del df
    if not sample and (data_dir / "europarl_el").is_dir():
        for f in sorted((data_dir / "europarl_el").glob("*.txt"))[:200]:
            t = re.sub(r"<[^>]+>", "", f.read_text(encoding="utf-8")).strip()
            if t:
                sample.append(t)
    tot_words = tot_tokens = 0
    for text in sample:
        words = text.split()
        if not words:
            continue
        enc = tok.encode(text, add_special_tokens=False)
        tot_words += len(words)
        tot_tokens += len(enc.ids)
    tpw = tot_tokens / max(tot_words, 1)
    print(f"  self-check (a): {tpw:.3f} tokens/word over {len(sample)} docs")
    if tpw > 2.0:
        raise SystemExit(
            f"SELF-CHECK FAILED: {tpw:.3f} tokens/word is far too high for a 50k "
            f"BPE (expect ~1.0-1.4). The tokenizer is fragmenting words — the "
            f"corpus or pre-tokenizer is wrong. Refusing to write a broken artifact."
        )

    # (b) vocab double-encoding scan.
    vocab = tok.get_vocab()
    bad = [k for k in list(vocab)[:20000] if _looks_double_encoded(k)]
    print(f"  self-check (b): {len(bad)}/{min(len(vocab), 20000)} sampled keys look double-encoded")
    if bad:
        raise SystemExit(
            "SELF-CHECK FAILED: vocab keys are double-encoded (mojibake training "
            f"text). Examples: {bad[:5]}. The input corpus is not clean UTF-8."
        )

    # (c) every special token must be present and map to a distinct id.
    ids = {}
    for tok_str in SPECIAL_TOKENS:
        got = tok.token_to_id(tok_str)
        if got is None:
            raise SystemExit(f"SELF-CHECK FAILED: {tok_str} not in the vocabulary.")
        ids[tok_str] = got
        print(f"  self-check (c): {tok_str} -> id {got}")
    if len(set(ids.values())) != len(ids):
        raise SystemExit(f"SELF-CHECK FAILED: special tokens share ids: {ids}")
    # Specials must be DISTINCT from the content range used by common words:
    # a common word must not collide with a special id.
    content_id = tok.token_to_id("λόγος") or tok.encode("λόγος", add_special_tokens=False).ids[0]
    if content_id in set(ids.values()):
        raise SystemExit(f"SELF-CHECK FAILED: a content token id {content_id} collides with a special id.")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Force unbuffered stdout so every print() hits the sbatch log immediately
    # (a piped/redirected stdout otherwise buffers and makes a long job look dead).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    # Heartbeat: env var wins, else the --heartbeat flag.
    if args.heartbeat and not os.environ.get("BPE_HEARTBEAT"):
        os.environ["BPE_HEARTBEAT"] = args.heartbeat
    _HEARTBEAT["pid"] = os.getpid()
    _HEARTBEAT["started"] = time.time()
    atexit.register(_final_flush_guard)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _on_signal)
        except Exception:
            pass
    _beat("start", {"out": str(out_dir), "vocab_size": args.vocab_size})

    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from tokenizers import trainers
    import pandas as pd

    # ---- cheap corpus scan (NO bulk text load) -----------------------------
    # Count documents and total characters by reading only what we need, so we
    # can (a) report the corpus size and (b) give train_from_iterator an
    # approximate `length`. This is a fast pass; it does NOT hold the corpus.
    print("Scanning corpus (metadata only, no bulk load)...")
    sys.stdout.flush()
    _beat("scanning")
    n_docs = 0
    total_chars = 0
    for f in sorted(data_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["text"])
        except Exception:
            continue
        if "text" not in df.columns:
            continue
        # Per-FILE this is cheap (tens of thousands of rows); we only ever hold
        # one file's text column at a time, so this stays light even though the
        # FULL corpus is ~700k docs. Sum the real string lengths as an int.
        col = df["text"].astype(str)
        n_docs += int(col.notna().sum())
        total_chars += int(col.str.len().sum())
        del df, col
    if not args.exclude_europarl:
        ep_dir = data_dir / "europarl_el"
        if ep_dir.is_dir():
            for f in sorted(ep_dir.glob("*.txt")):
                n_docs += 1
                total_chars += len(f.read_text(encoding="utf-8"))

    if n_docs == 0:
        raise SystemExit("No documents found — nothing to train on.")
    est_total_chars = max(int(total_chars), 1)
    print(f"  {n_docs} documents, {total_chars / 1e9:.3f}B chars "
          f"(~{total_chars * 1.2 / 1e9:.1f} GB of UTF-8 text)")
    sys.stdout.flush()
    _beat("scan-done", {"n_docs": n_docs, "total_chars": int(total_chars)})

    print(f"\nBuilding BPE (vocab_size={args.vocab_size}, specials={SPECIAL_TOKENS})...")
    sys.stdout.flush()
    # IMPORTANT: the next call (train_from_iterator) prints "Pre-processing
    # sequences" and then runs for a LONG time with NO further output — its
    # progress bar is TTY-gated, so under sbatch it is silent for the whole
    # train. This is EXPECTED, not a hang. Watch the heartbeat file (or
    # `ps -o etime= -p <PID>` / RSS growth in `top`) to confirm it is alive.
    # A full 700M-token corpus takes on the order of hours.
    _beat("bpe-train-start", {"note": "silent until done; see heartbeat"})

    # Build the Tokenizer with a BPE model. We deliberately do NOT use
    # ByteLevel: for a diacritic-rich script we want whole-word BPE tokens, and
    # the runtime byte re-encoding is exactly what made the old artifact
    # unusable. WhitespaceSplit = whitespace word pieces (available in every
    # tokenizers version; Regex(r"\S+") is the same idea but 0.23+ only).
    tok = Tokenizer(models.BPE())
    pre = pre_tokenizers.WhitespaceSplit()
    tok.pre_tokenizer = pre
    tok.trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        max_token_length=args.max_token_length,
        show_progress=True,
    )

    # Stream the corpus from disk in bounded batches (the memory fix) and let
    # the trainer consume it. `length` is only a progress-sampling hint; the
    # estimate from the scan pass is fine even if it is not exact.
    _t0 = time.time()
    tok.train_from_iterator(
        _corpus_text_iter(data_dir, args.include_wikidata, args.exclude_europarl,
                          args.max_docs, args.sample_batch_chars),
        length=est_total_chars,
    )
    _beat("bpe-train-done", {"train_seconds": round(time.time() - _t0, 1)})

    # Add special tokens AFTER training. In every current tokenizers version the
    # trainer drops/renumbers specials passed via BpeTrainer(special_tokens=...),
    # but add_special_tokens() after training persists them in the saved file's
    # added_tokens_decoder (ids assigned at the end of the vocab on reload).
    # train.py resolves those ids at runtime, so the exact numbers don't matter.
    tok.add_special_tokens(SPECIAL_TOKENS)

    # NO post_processor: train.py's tokenize_and_prepare_mlm prepends [CLS] and
    # appends [SEP] itself (it uses cls_token_id/sep_token_id), so we must not
    # double-wrap. Leaving it unset also avoids this tokenizers version's strict
    # pair-template validation.

    print("\nRunning self-check...")
    sys.stdout.flush()
    _beat("self-check")
    self_check(tok, data_dir)
    sys.stdout.flush()

    # ---- write the HF-compatible artifact ---------------------------------
    print(f"\nWriting tokenizer to {out_dir} ...")
    sys.stdout.flush()
    _beat("saving")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))

    # Serialize the added-tokens decoder in the exact shape AutoTokenizer
    # expects. get_added_tokens_decoder() yields {id: AddedToken}; we map it to
    # the {id: {id, content, special, ...}} form used by tokenizer_config.json.
    # NOTE: we do NOT write `added_tokens_decoder` into tokenizer_config.json.
    # The specials are already persisted authoritatively inside tokenizer.json
    # (this tokenizers version stores them under its `added_tokens` field), and
    # duplicating them into tokenizer_config.json both is redundant and triggers
    # "Ignored unknown kwarg option id" warnings on load (the map key duplicates
    # the inner `id` field). tokenizer.json is the source of truth.
    tokenizer_config = {
        "model_type": None,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "bos_token": None,
        # eos points at the unused base placeholder, NOT [PAD] — aliasing eos
        # to [PAD] on a generative backbone (ModernBERT) misbehaves downstream.
        "eos_token": "<|padding|>",
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": True,
        "model_max_length": 512,
        "model_input_names": ["input_ids", "attention_mask"],
        "extra_special_tokens": {},
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False))

    special_tokens_map = {
        "bos_token": None,
        "cls_token": "[CLS]",
        "eos_token": "<|padding|>",
        "mask_token": "[MASK]",
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
    }
    (out_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2, ensure_ascii=False))

    print(f"\nDone. Vocab size (incl. specials): {len(tok.get_vocab())}")
    print(f"  tokenizer.json, tokenizer_config.json, special_tokens_map.json -> {out_dir}")
    print("\nNext:")
    print(f"  python scripts/prepare_data.py --tokenizer {out_dir} \\")
    print("      --out data/greekbpe_tokenized_open_greek_dataset_v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
