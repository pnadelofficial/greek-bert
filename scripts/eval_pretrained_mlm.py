#!/usr/bin/env python3
"""Zero-shot MLM validation eval for an EXTERNAL pretrained checkpoint
(default: bowphs/GreBerta), against THIS PROJECT's own held-out validation
split -- the real, controlled counterpart to a model's self-reported numbers
(same spirit as scripts/eval_external_spacy_model.py on the spaCy side, which
did the same thing for odyCy/LatinCy).

Computes EXACTLY the same metrics train.py's own validation loop does (same
dynamic MLM masking recipe via utils.make_masking_fn/mlm_masking, same
val_loss / val_ppl / masked-token accuracy), so the result is directly
comparable to the `best_val_loss` / `final_val_loss` numbers already recorded
for this project's own 5 pretraining arms (runs/<RUN_NAME>/metrics.json,
results/table.md).

IMPORTANT: a model with its OWN tokenizer/vocab (e.g. bowphs/GreBerta, a
RoBERTa-style 52k BPE vocab -- unrelated to nlpaueb's 35k WordPiece vocab
arms 1-3 use) needs its OWN tokenized copy of the corpus. Reusing
data/nlpaueb_tokenized_open_greek_dataset_v2 directly would feed it WordPiece
ids it never saw in pretraining, producing a meaningless loss. Build the
matching dataset first with the SAME default corpus flags/seed as the
existing arms (so the document-level held-out split is IDENTICAL and this
number is actually comparable):

    python scripts/prepare_data.py --tokenizer bowphs/GreBerta \
        --out data/greberta_tokenized_open_greek_dataset_v2

(see slurm/prepare_greberta_corpus.sh, which does exactly this, and
slurm/eval_greberta_baseline.sh, which runs this script afterward.)

Usage:
    python scripts/eval_pretrained_mlm.py \
        --model bowphs/GreBerta \
        --tokenized-dataset-path data/greberta_tokenized_open_greek_dataset_v2 \
        --output results/greberta_mlm_eval.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

# scripts/ is not a package; make its siblings importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import TokenizerSpec, make_masking_fn, resolve_model_path  # noqa: E402
from runs import record_result, record_run_provenance, write_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="bowphs/GreBerta",
                   help="HF hub id or local HF-format dir of the MLM checkpoint to score.")
    p.add_argument("--tokenized-dataset-path", required=True,
                   help="A dataset built by prepare_data.py with THIS model's own "
                        "tokenizer (see slurm/prepare_greberta_corpus.sh).")
    p.add_argument("--split", default="test",
                   help="Which DatasetDict split to score (default: 'test', the "
                        "document-level held-out validation split).")
    p.add_argument("--mask-prob", type=float, default=0.30,
                   help="Must match configs/train.yaml's mask_prob for the "
                        "resulting val_loss to be comparable to this project's "
                        "own pretraining runs.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=22091997,
                   help="Matches TrainingConfig.seed -- same masking draw as "
                        "every other run's validation pass.")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--run-name", default=None,
                   help="Name under runs/ to record provenance/metrics into. "
                        "Defaults to $RUN_NAME / $RUN_DIR (see scripts/runs.py), "
                        "or runs/_adhoc if neither is set.")
    p.add_argument("--output", default=None,
                   help="Also write a flat JSON of the metrics here (same shape "
                        "as the *_eval_results.json files elsewhere in this repo), "
                        "for folding by hand into results/model-comparison.md.")
    return p.parse_args()


def make_autocast(device_type: str, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type, dtype=torch.bfloat16)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading tokenizer/model: {args.model}")
    resolved = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    config = AutoConfig.from_pretrained(resolved)

    vocab_size = len(tokenizer)
    if getattr(config, "vocab_size", None) != vocab_size:
        raise SystemExit(
            f"config.vocab_size={getattr(config, 'vocab_size', None)} != "
            f"len(tokenizer)={vocab_size} for {args.model}. This script never "
            f"resizes the LM head (unlike train.py's fresh-init arms) -- a "
            f"mismatch here means the wrong tokenizer/model pair was passed, "
            f"and every number below would be meaningless."
        )

    model = AutoModelForMaskedLM.from_pretrained(resolved, config=config)
    model.to(device)
    model.eval()

    print(f"Loading dataset: {args.tokenized_dataset_path} [split={args.split}]")
    dataset = load_from_disk(args.tokenized_dataset_path)[args.split]
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("input_ids", "attention_mask")]
    )
    dataset.set_format(type="torch")

    # extra_excluded_ids defaults to () -- fine here, this model owns its whole
    # vocab layout (no shared/reused tokenizer with unused reserved ids).
    spec = TokenizerSpec(tokenizer_path=args.model)
    mask_fn = make_masking_fn(tokenizer, spec, args.mask_prob)
    ignore_index = -100

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    use_bf16 = args.bf16 and device.type == "cuda"

    val_loss, val_batches = 0.0, 0
    correct_tokens, labelled_tokens = 0.0, 0.0
    print(f"Evaluating {len(dataset)} sequences, batch_size={args.batch_size}, "
          f"mask_prob={args.mask_prob}, device={device}...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device)

            # Same masking recipe as train.py's validation pass, so this loss
            # is comparable to every other run's val_loss.
            masked_input_ids, masked_labels = mask_fn(input_ids)
            masked_input_ids = masked_input_ids.to(device, non_blocking=True)
            masked_labels = masked_labels.to(device, non_blocking=True)

            with make_autocast(device.type, use_bf16):
                outputs = model(input_ids=masked_input_ids,
                                attention_mask=attention_mask,
                                labels=masked_labels)

            val_loss += outputs.loss.item()
            val_batches += 1

            preds = outputs.logits.argmax(dim=-1)
            hit = (preds == masked_labels) & (masked_labels != ignore_index)
            correct_tokens += hit.sum().item()
            labelled_tokens += (masked_labels != ignore_index).sum().item()

    avg_val_loss = val_loss / max(val_batches, 1)
    val_acc = correct_tokens / max(labelled_tokens, 1.0)
    val_ppl = math.exp(min(avg_val_loss, 50.0))

    print(f"\nval_loss={avg_val_loss:.4f}  val_ppl={val_ppl:.3f}  "
          f"masked-token acc={val_acc:.4f}")

    metrics = {
        "model": args.model,
        "tokenized_dataset_path": args.tokenized_dataset_path,
        "split": args.split,
        "mask_prob": args.mask_prob,
        "n_sequences": len(dataset),
        "val_loss": avg_val_loss,
        "val_ppl": val_ppl,
        "val_masked_token_acc": val_acc,
    }

    split_label = f"{Path(args.tokenized_dataset_path).name}:{args.split}"
    record_run_provenance(
        stage="external_mlm_eval",
        config_dict={"model": args.model, "mask_prob": args.mask_prob,
                     "seed": args.seed, "batch_size": args.batch_size},
        run_name=args.run_name,
    )
    write_metrics({"external_mlm_eval": metrics}, run_name=args.run_name)
    for metric_name in ("val_loss", "val_ppl", "val_masked_token_acc"):
        record_result(stage="external_mlm_eval", model=args.model, metric=metric_name,
                      value=metrics[metric_name], split=split_label,
                      seed=args.seed, run_name=args.run_name)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
