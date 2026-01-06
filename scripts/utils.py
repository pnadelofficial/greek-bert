from tqdm import tqdm
import torch
import numpy as np
import torch.distributed as dist
import os
from dataclasses import dataclass, asdict
import yaml
from typing import Union

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
    gradient_accumulation_steps: int = 1
    pretrained_model: Union[str, None] = None
    tokenized_dataset_path: str = "../data/tokenized_open_greek_dataset"
    
    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

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

def mlm_masking(input_ids, np_rng, mask_prob=0.3, mask_token=None, pad_token=None, ignore_index=-100, vocab_size=None):
    seq = input_ids.cpu().numpy()
    labels = np.where(seq == pad_token, ignore_index, seq)
    rand = np_rng.random(seq.shape)

    mask_mask = rand < mask_prob * 0.8
    random_mask = (rand >= mask_prob * 0.8) & (rand < mask_prob * 0.9)
    keep_mask = (rand >= mask_prob * 0.9) & (rand < mask_prob)

    labels = np.where(mask_mask | random_mask | keep_mask, labels, ignore_index)
    seq = np.where(mask_mask, mask_token, seq)

    max_token_id = vocab_size if vocab_size is not None else np.max(seq) + 1
    random_words = np_rng.integers(0, max_token_id, size=seq.shape)

    seq = np.where(random_mask, random_words, seq)
    return torch.tensor(seq), torch.tensor(labels)