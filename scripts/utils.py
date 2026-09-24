from tqdm import tqdm
import torch
import numpy as np
import torch.distributed as dist
import os
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from typing import Union

# Config keys that hold filesystem paths. Relative values are resolved against
# the directory containing the YAML file, so a run behaves identically no matter
# which directory it was launched from. Previously these resolved against the
# process cwd, which meant `torchrun train.py` only worked from scripts/.
PATH_FIELDS = (
    "checkpoint_dir",
    "tensorboard_dir",
    "tokenized_dataset_path",
    "output_dir",
)

# Environment variables that override a config value. Lets one checked-in config
# serve every run: slurm/env.sh exports CHECKPOINT_DIR / TENSORBOARD_DIR per
# RUN_NAME, so runs never overwrite each other's artifacts.
ENV_OVERRIDES = {
    "checkpoint_dir": "CHECKPOINT_DIR",
    "tensorboard_dir": "TENSORBOARD_DIR",
    "output_dir": "HF_EXPORT_DIR",
    "tokenized_dataset_path": "TOKENIZED_DATASET_PATH",
    "pretrained_model": "PRETRAINED_MODEL",
}


@dataclass
class TrainingConfig:
    mask_prob: float = 0.3
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    max_lr: float = 1e-3
    pct_start: float = 0.05
    log_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./runs"
    use_mixed_precision: bool = True
    # bf16 needs no loss scaling on H200/H100/B200 and is more stable at high LR.
    # Set False to fall back to fp16 + GradScaler (pre-Ampere GPUs).
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    # DataLoader workers PER PROCESS. The SLURM profiles allocate 16-64 CPUs and
    # the pre-tokenized dataset is pure arrow->tensor work, so 4 was the
    # bottleneck at batch 16.
    num_workers: int = 8
    prefetch_factor: int = 4
    # Seed for masks/samplers/ splits. Recorded in the run manifest so results
    # are attributable.
    seed: int = 22091997
    pretrained_model: Union[str, None] = None
    tokenized_dataset_path: str = "../data/tokenized_open_greek_dataset"
    # Where convert_to_hf.py writes the HuggingFace-format export.
    output_dir: str = "./hf_format"

    @classmethod
    def from_yaml(cls, path):
        """Load a config, apply env overrides, and resolve path fields.

        Unknown keys in the YAML are reported rather than silently dropped --
        a typo'd key used to fall back to the dataclass default with no warning.
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Env var overrides (per-run paths injected by slurm/env.sh).
        for field, env_name in ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value:
                config_dict[field] = value

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(config_dict) - known)
        if unknown:
            print(
                f"WARNING: {path} contains keys not in TrainingConfig and they "
                f"will be IGNORED: {unknown}"
            )
        config_dict = {k: v for k, v in config_dict.items() if k in known}

        config = cls(**config_dict)

        # Resolve relative paths against the config file's directory.
        base_dir = Path(path).resolve().parent
        for field in PATH_FIELDS:
            value = getattr(config, field)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                setattr(config, field, str((base_dir / value).resolve()))

        return config

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    def config_hash(self) -> str:
        """Stable short hash of the resolved config, for the results table.

        Two runs with the same hash are the same experiment (modulo seed and
        code SHA). Paths are included, so a different checkpoint_dir changes the
        hash; that is intentional for artifact provenance.
        """
        import hashlib
        import json

        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0 # if single GPU

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def tokenize_and_prepare_mlm(examples, tokenizer, is_main_process, chunk_size=1024, ignore_length=16):
    tokenized = tokenizer(examples["text"], truncation=False)

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    iterator = tqdm(tokenized["input_ids"], total=len(tokenized["input_ids"]), disable=not is_main_process)
    for input_ids in iterator:
        length = len(input_ids)
        if length < ignore_length:
            continue
        
        for i in range(0, length, chunk_size-2):
            end = min(i + chunk_size - 2, length)
            chunk_ids = input_ids[i:end]
            bert_input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
            bert_attention_mask = [1] * len(bert_input_ids)
            if len(bert_input_ids) < chunk_size:
                padding_length = chunk_size - len(bert_input_ids)
                bert_input_ids += [tokenizer.pad_token_id] * padding_length
                bert_attention_mask += [0] * padding_length
            labels = bert_input_ids.copy()
            all_input_ids.append(bert_input_ids)
            all_attention_masks.append(bert_attention_mask)
            all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels}

# Special token IDs in the BERT-family vocabularies used by this repo
# (nlpaueb/bert-base-greek-uncased-v1 and tokenizers/modernbert-greek-tokenizer
# both use the [PAD]=0 / [CLS]=101 / [SEP]=102 layout). These are only the
# fallback for positions we cannot read off the tokenizer; mlm_masking takes the
# real ids as arguments so a vocabulary with a different layout still works.
DEFAULT_SPECIAL_TOKEN_IDS = (0, 101, 102)


def mlm_masking(
    input_ids,
    mask_token_id,
    mask_prob=0.30,
    pad_token_id=0,
    ignore_index=-100,
    vocab_size=None,
    cls_token_id=None,
    sep_token_id=None,
    special_token_ids=None,
):
    """Dynamic MLM masking - generates fresh masks on every forward pass.

    This implements the RoBERTa approach (https://arxiv.org/abs/1907.11692):
    Instead of pre-computing masks once during preprocessing, we regenerate
    them for each training step. This significantly improves performance when
    training for many epochs or on large datasets because:

    1. Each training step sees a different masking pattern
    2. The model learns to predict tokens in more diverse contexts
    3. Avoids overfitting to specific mask patterns

    RoBERTa found this simple change alone improved performance by ~0.5% on GLUE.

    Masking strategy (same as BERT/RoBERTa/DeBERTa/ModernBERT):
    - mask_prob of *eligible* tokens are selected for prediction
    - 80% of selected tokens -> replaced with [MASK] token
    - 10% of selected tokens -> replaced with a random token
    - 10% of selected tokens -> left unchanged

    The 80/10/10 split is not cosmetic. The 10%-unchanged term is the only thing
    that gives the model exposure to target tokens that are still visible in the
    input, which is what corrects for the train/inference distribution shift:
    every downstream consumer in this repo (spaCy morphologizer/lemmatizer, WSD,
    the SBERT encoder) reads *unmasked* text and never sees a [MASK]. Without
    the random/unchanged terms the encoder's whole training distribution says
    "a third of what I'm looking at is [MASK]" and its residual-stream
    statistics shift at inference.

    Special tokens ([PAD]/[CLS]/[SEP], plus any in `special_token_ids`) are
    never selected. Predicting a [CLS] is free loss that looks like learning, and
    it also pollutes the position that downstream pooling relies on.

    Args:
        input_ids: Tensor of input token IDs [batch_size, seq_len]
        mask_token_id: Token ID to use for masking (usually [MASK])
        mask_prob: Probability of masking an eligible token. Default 0.30 matches
            configs/train.yaml; the previous 0.15 default here silently disagreed
            with both the YAML and TrainingConfig.
        pad_token_id: Token ID for padding (never masked, ignored in loss)
        ignore_index: Label value for ignored tokens in loss computation (-100)
        vocab_size: Upper bound for the random-token draw. Without it the 10%
            random-replacement term is skipped (and that should be treated as a
            configuration error, not a mode).
        cls_token_id: [CLS] id, excluded from masking.
        sep_token_id: [SEP] id, excluded from masking.
        special_token_ids: Extra ids excluded from masking ([UNK], <s>, ...).

    Returns:
        masked_input_ids: Input tensor with masked tokens [batch_size, seq_len]
        labels: Labels for loss computation, -100 for non-masked/padding positions

    Example:
        >>> input_ids = torch.tensor([[101, 2054, 2003, 102]])  # [CLS] this is [SEP]
        >>> masked, labels = mlm_masking(input_ids, mask_token_id=103, mask_prob=1.0)
        >>> # every eligible position is labelled; the [CLS]/[SEP] are not
    """
    excluded = {pad_token_id}
    for token_id in (cls_token_id, sep_token_id, mask_token_id):
        if token_id is not None:
            excluded.add(int(token_id))
    for token_id in (special_token_ids or ()):  # noqa: SIM110
        if token_id is not None:
            excluded.add(int(token_id))

    # Eligible = real content tokens. Special tokens are never prediction targets.
    candidates = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in excluded:
        candidates &= input_ids != int(token_id)

    # Create mask: True where we should predict
    mask = (torch.rand_like(input_ids, dtype=torch.float32) < mask_prob) & candidates

    # Create labels: -100 for padding and non-masked positions, original IDs for masked positions
    labels = torch.full_like(input_ids, ignore_index)
    labels[mask] = input_ids[mask]

    masked_input = input_ids.clone()

    # 80 / 10 / 10 split over the selected positions. One draw per token decides
    # which of the three treatments it gets, so the three sets are disjoint and
    # each is exactly its share of `mask` in expectation.
    r = torch.rand_like(input_ids, dtype=torch.float32)
    masked_input[mask & (r < 0.80)] = mask_token_id

    if vocab_size:
        random_token = torch.randint_like(input_ids, 0, int(vocab_size))
        masked_input[mask & (r >= 0.80) & (r < 0.90)] = random_token[mask & (r >= 0.80) & (r < 0.90)]
    # r >= 0.90 -> left unchanged (the exposure-bias correction term)

    return masked_input, labels