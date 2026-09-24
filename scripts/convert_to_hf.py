"""Convert a pretraining torch checkpoint into HuggingFace format.

Usage:
    python scripts/convert_to_hf.py --config-path configs/train.yaml
    python scripts/convert_to_hf.py --config-path configs/train.yaml \
        --checkpoint runs/myrun/checkpoints/best_model.pt

Everything comes from the config (plus the CHECKPOINT_DIR / HF_EXPORT_DIR env
overrides injected by slurm/env.sh). No hardcoded absolute paths.

Checkpoints are preferred in this order: explicit --checkpoint, then
best_model.pt, then final_model.pt, then the highest-numbered epoch checkpoint.
best_model.pt leads because pretraining now selects checkpoints by validation
loss (audit item 5); final_model.pt is whatever the last epoch happened to be,
which for a not-yet-converged run is not necessarily the best one.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from paths import repo_path
from runs import record_result, record_run_provenance
from utils import TrainingConfig, resolve_model_path

# Fallback tokenizer used only when training from scratch (pretrained_model: null).
SCRATCH_TOKENIZER = str(repo_path("tokenizers", "modernbert-greek-tokenizer"))

# Preference order for automatic checkpoint resolution.
CHECKPOINT_PREFERENCE = ("best_model.pt", "final_model.pt")


def resolve_checkpoint(checkpoint_dir: str, prefer: tuple[str, ...] = CHECKPOINT_PREFERENCE) -> str:
    """Locate the checkpoint to convert inside `checkpoint_dir`.

    Returns the first of `prefer` that exists, else the highest-numbered
    checkpoint_epoch_*.pt.
    """
    for name in prefer:
        candidate = Path(checkpoint_dir) / name
        if candidate.is_file():
            if name != prefer[0]:
                print(f"No {prefer[0]} found; falling back to {name}. "
                      f"best_model.pt is the val-loss-selected checkpoint — if this "
                      f"run has none, pretraining ran without validation.")
            return str(candidate)

    # Fall back to the highest-numbered epoch checkpoint.
    epoch_ckpts = sorted(
        Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if epoch_ckpts:
        print(f"No {'/'.join(prefer)} found; using latest epoch checkpoint.")
        return str(epoch_ckpts[-1])

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir} "
        f"(looked for {', '.join(prefer)} and checkpoint_epoch_*.pt)"
    )


def build_model_and_tokenizer(hp_config: TrainingConfig):
    """Rebuild the architecture the checkpoint was trained with.

    The from-scratch branch used to build a ModernBERT-base config, construct the
    model from it, and only then mutate `config.vocab_size` — so the LM head came
    out at ModernBERT's 50257 while the Greek tokenizer has 50368, and the export
    could never be loaded (audit item 9). It also routed a ModernBERT config
    through `AutoModelForMaskedLM`, which is only correct if the config's
    model_type actually resolves to ModernBERT. Both are fixed here: vocab_size is
    set before construction, the architecture comes from the config, and the
    vocab is asserted to match the tokenizer before anything is written.
    """
    # Same (architecture x tokenizer) resolution as train.py, so an export
    # always rebuilds exactly what was trained. pretrained_model set = arm 1
    # (load config+tokenizer from the checkpoint); null = fresh-init arms
    # (2: BERT 35k, 3/4: ModernBERT + 50k BPE) using model_config + tokenizer.
    if hp_config.pretrained_model:
        config = AutoConfig.from_pretrained(hp_config.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(hp_config.pretrained_model)
    else:
        tok_path = resolve_model_path(hp_config.tokenizer or SCRATCH_TOKENIZER)
        if not hp_config.model_config:
            raise SystemExit(
                "pretrained_model is null but model_config is not set. Fresh-init "
                "arms must point model_config at a BERT or ModernBERT config dir."
            )
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        config = AutoConfig.from_pretrained(resolve_model_path(hp_config.model_config))
        # vocab_size must be set BEFORE the model is built — this is the bug
        # that made the from-scratch arm unexportable (audit item 9).
        config.vocab_size = len(tokenizer)
        for attr, tok_attr in (("pad_token_id", "pad_token_id"), ("cls_token_id", "cls_token_id"),
                               ("sep_token_id", "sep_token_id"), ("bos_token_id", "bos_token_id"),
                               ("eos_token_id", "eos_token_id")):
            v = getattr(tokenizer, tok_attr, None)
            if isinstance(v, int) and v >= 0:
                setattr(config, attr, v)
        if getattr(config, "bos_token_id", None) is None:
            config.bos_token_id = 0

    if config.vocab_size != len(tokenizer):
        raise ValueError(
            f"config.vocab_size={config.vocab_size} != len(tokenizer)={len(tokenizer)}. "
            f"Refusing to write an export whose LM head cannot represent the "
            f"tokenizer's vocabulary."
        )

    # No .to(device): load_state_dict replaces every weight below, and keeping
    # the model on CPU here avoids pinning a GPU during conversion.
    model = AutoModelForMaskedLM.from_config(config)

    # Only alias eos for models with no real eos. Baking eos_token="[PAD]" into a
    # ModernBERT export silently misbehaves in generation-style consumers.
    if getattr(config, "model_type", "") == "bert":
        tokenizer.eos_token = tokenizer.pad_token

    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Convert a .pt checkpoint to HF format.")
    parser.add_argument("--config-path", dest="config_path", required=True)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=None,
        help="Explicit .pt file. Default: best_model.pt in config.checkpoint_dir.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output dir. Default: config.output_dir.",
    )
    args = parser.parse_args()

    hp_config = TrainingConfig.from_yaml(args.config_path)
    model, tokenizer, config = build_model_and_tokenizer(hp_config)

    ckpt_path = args.checkpoint or resolve_checkpoint(hp_config.checkpoint_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Architecture: {type(model).__name__} "
          f"(model_type={getattr(config, 'model_type', '?')}, "
          f"vocab_size={config.vocab_size})")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # strict=True (the default) is what we want: a shape or key mismatch must
    # hard-fail rather than silently leave randomly-initialised weights in the
    # export. The failure mode is opaque, so translate it into a diagnosis.
    try:
        load_info = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not match a freshly built "
            f"{type(model).__name__} for config "
            f"model_type={getattr(config, 'model_type', '?')}, "
            f"vocab_size={config.vocab_size}. This usually means the config's "
            f"pretrained_model does not describe the architecture that produced "
            f"this checkpoint (e.g. a ModernBERT checkpoint converted under a BERT "
            f"config, or a 50368-token tokenizer against a 50257-token LM head).\n"
            f"Underlying error:\n{exc}"
        ) from exc
    if getattr(load_info, "missing_keys", None) or getattr(load_info, "unexpected_keys", None):
        # Unreachable with strict=True, but guards a future strict=False change.
        raise RuntimeError(f"Checkpoint did not load cleanly: {load_info}")

    print(f"Loaded {len(state_dict)} tensors; "
          f"checkpoint val_loss={checkpoint.get('val_loss', 'n/a')} "
          f"epoch={checkpoint.get('epoch', 'n/a')}")

    output_dir = args.output_dir or hp_config.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Round-trip check: the thing we just wrote must reload and still agree with
    # the tokenizer. Catches a save that produced an unloadable directory.
    reloaded_config = AutoConfig.from_pretrained(output_dir)
    if reloaded_config.vocab_size != len(tokenizer):
        raise RuntimeError(
            f"Export round-trip failed: {output_dir}/config.json has "
            f"vocab_size={reloaded_config.vocab_size} but tokenizer has "
            f"{len(tokenizer)} tokens."
        )
    # Verify the exported architecture actually reloads as a masked LM. This is
    # the check that would have caught the ModernBERT-under-a-BERT-config export.
    AutoModelForMaskedLM.from_pretrained(output_dir)

    print(f"Saved HuggingFace model to: {output_dir}")

    record_run_provenance(
        stage="convert",
        config_hash=hp_config.config_hash(),
        extra={"checkpoint": ckpt_path, "output_dir": str(output_dir),
               "architecture": type(model).__name__, "vocab_size": int(config.vocab_size)},
    )
    record_result(stage="convert", model=Path(output_dir).name, metric="exported",
                  value="ok", split="", seed=hp_config.seed,
                  config_hash=hp_config.config_hash())


if __name__ == "__main__":
    main()
