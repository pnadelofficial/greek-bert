"""Word sense disambiguation on the glaux harmonia/kosmos data.

Compares our encoder against aristoBERTo under one shared fine-tuning recipe.

Design notes (see docs/AUDIT-2026-07.md items 7 and 8):

* Each (model, target word) pair gets its OWN `copy.deepcopy` of the encoder.
  Previously `harmonia` and `kosmos` wrapped the same `model` object, so training
  harmonia mutated the encoder that kosmos then started from — the two numbers
  were not independent, and the reported result depended on which word happened
  to be trained first.
* Sequences are padded per-batch (`padding=True` + a collate that re-derives the
  target position on the POST-padding encoding), not to `max_length=512`. glaux
  sentences are mostly short, so every batch used to be ~95% padding.
* Best-epoch weights are restored before the final evaluation, and the reported
  number is mean +/- std over several seeds, not a single run. With ~108 training
  examples and ~27 validation examples, a single-seed 0.861 vs 0.815 gap is about
  1.5 sigma and is not a result.
* The target word must survive tokenization as a real (non-`[UNK]`) position; that
  is counted and reported rather than assumed, because the corpus is ~1.26% [UNK]
  and the model is uncased.

Run:  python wsd/wsd.py                     # both models, both words, 5 seeds
      python wsd/wsd.py --models ours --words harmonia --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Repo path helpers live in scripts/ (paths.py resolves everything from the repo
# root, so nothing here needs a hardcoded absolute path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paths import model_dir as _resolve_model_dir, repo_path  # noqa: E402
from runs import record_result, record_run_provenance, write_metrics  # noqa: E402

WSD_DIR = Path(__file__).resolve().parent

# glaux data. These two JSON files are 1.9 GB and 590 MB on disk and expand to
# well over 10 GB of Python objects, so they are loaded LAZILY on first use.
# Loading them at import time made `python wsd/wsd.py --help` and any unit test
# of this module OOM-kill on a login node (the nodes cap RSS at 16 GB).
glaux_dir = repo_path("wsd", "glaux", "xml")

_glaux_cache: dict | None = None


def load_gaux(force: bool = False) -> tuple[dict, dict]:
    """Return (sentences, id2word_id), loading the glaux JSON only once."""
    global _glaux_cache
    if _glaux_cache is None or force:
        with open(glaux_dir / 'glaux_sentences.json', 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        with open(glaux_dir / 'glaux_id2word_id.json', 'r', encoding='utf-8') as f:
            id2word = json.load(f)
        _glaux_cache = (sentences, id2word)
    return _glaux_cache


class _LazyGlauxMapping:
    """Back-compat shim so `glaux_data[id]` / `id in glaux_data` keep working."""

    def __getitem__(self, key):
        return load_gaux()[0][key]

    def __contains__(self, key):
        return key in load_gaux()[0]

    def __len__(self):
        return len(load_gaux()[0])


# Module-level names kept for existing callers/tests.
class _LazyId2Word:
    def get(self, key, default=None):
        return load_gaux()[1].get(key, default)

    def __getitem__(self, key):
        return load_gaux()[1][key]


sentences = _LazyGlauxMapping()
glaux_id2word_id = _LazyId2Word()

TARGET_WORDS = {
    "harmonia": {"file": "harmonia_glaux.txt", "surface": "ἁρμονιά"},
    "kosmos": {"file": "kosmos_glaux.txt", "surface": "κόσμος"},
}

# The comparison model. Resolved locally so an offline cluster run does not fail
# on a hub lookup for "Jacobo/aristoBERTo"; override with --baseline <path>.
DEFAULT_BASELINE = str(repo_path("models", "external-aristoBERTo"))


# config
@dataclass
class WSDConfig:
    # Defaults to models/current (or $MODEL_DIR) -- see scripts/set_current_model.sh
    model_name: str = field(default_factory=_resolve_model_dir)
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 25
    weight_decay: float = 0.01
    dropout: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    # Per-batch padding instead of padding every sequence to 512.
    pad_to_max_length: bool = False
    # Restore best-by-val-accuracy weights before final eval.
    load_best_model_at_end: bool = True
    # bf16 forward/backward; no GradScaler needed.
    bf16: bool = True
    # Number of training seeds to average. n~108 means one seed proves nothing.
    num_seeds: int = 5


# dataset
class WSDDataset(Dataset):
    def __init__(self, data_path, glaux_data, tokenizer, target_word, max_length=512):
        self.tokenizer = tokenizer
        self.target_word = target_word
        self.max_length = max_length

        df = pd.read_csv(data_path, sep="\t", header=None)

        if len(df.columns) == 2:
            df.columns = ['glaux_id', 'sense']
        else:
            df.columns = ['glaux_id', 'sense', 'subsense']

        self.data = []
        missing_ids = []
        # Examples whose target could not be aligned to a real token in THIS
        # tokenizer's encoding. They are dropped rather than silently trained on,
        # because reading the hidden state at a mis-aligned position measures an
        # arbitrary neighbouring word (audit item 8).
        self.unaligned: list[str] = []

        for idx, row in df.iterrows():
            word_id = str(row["glaux_id"])
            glaux_id = glaux_id2word_id.get(word_id, None)
            sense = row["sense"]

            if glaux_id in glaux_data:
                sentense = glaux_data[glaux_id]
                text = sentense['text']
                word_ids = sentense['word_ids']
                # Index into glaux's own word list. glaux word 0 is the FIRST
                # word of the sentence, so the target word's surface form is at
                # text.split()[word_index] -- verified below.
                word_index = word_ids.index(word_id)

                token_pos = resolve_target_position(tokenizer, text, word_index)
                if token_pos is None:
                    self.unaligned.append(glaux_id)
                    continue

                self.data.append({
                    'glaux_id': glaux_id,
                    'text': text,
                    'sense': sense,
                    'word_index': word_index,
                    # Token index in the model's own encoding, not a word index.
                    'target_position': token_pos,
                })
            else:
                missing_ids.append(glaux_id)

        if missing_ids:
            print(f"Warning: {len(missing_ids)} glaux_ids not found in glaux data.")
        if self.unaligned:
            print(f"Warning: dropped {len(self.unaligned)} examples whose target word "
                  f"could not be aligned to a single token position.")
        print(f"Loaded {len(self.data)} examples for target word '{self.target_word}'.")

        self.sense_to_id = {sense: idx for idx, sense in enumerate(sorted(set(df['sense'])))}
        self.id_to_sense = {idx: sense for sense, idx in self.sense_to_id.items()}
        self.num_senses = len(self.sense_to_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['text']
        sense_label = self.sense_to_id[item['sense']]
        target_position = item['target_position']

        # Unpadded here; wsd_collate pads per batch and positions are validated
        # against the padded encoding there.
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_offsets_mapping=False,
            return_tensors='pt',
        )

        # The position was resolved against an unpadded encoding at construction
        # time. Padding is right-sided, so token indices below the sequence length
        # are unchanged by padding -- asserted rather than assumed.
        if target_position >= encoding['input_ids'].shape[-1]:
            raise IndexError(
                f"target_position {target_position} falls outside the encoded "
                f"sequence (len={encoding['input_ids'].shape[-1]}) for glaux_id "
                f"{item['glaux_id']}. Truncation cut the target word."
            )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target_position': target_position,
            'word_index': item['word_index'],
            'label': torch.tensor(sense_label, dtype=torch.long),
            'glaux_id': item['glaux_id'],
            'text': item['text'],
        }


def _offsets_of(enc) -> list | None:
    """Fetch per-token character spans across transformers versions.

    transformers 5.x stores them under `offset_mapping` (underscore); 4.x used
    `offsets_mapping`. Neither key exists unless return_offsets_mapping=True.
    """
    for key in ("offset_mapping", "offsets_mapping"):
        try:
            value = enc[key] if hasattr(enc, "__getitem__") else getattr(enc, key)
        except Exception:
            value = None
        if value:
            return list(value)
    return None


# Which token of a multi-subword word to read. WordPiece puts the highest-value
# piece first ("ἀρμον-" carries more than "-ία"), and first-piece is the
# convention used by most WSD/NER implementations, so it is the default. The
# last piece is the only one whose span covers the WHOLE word for the ~9-27% of
# targets that are single-token, and some implementations prefer it because it
# has absorbed the full contextual representation. Set TARGET_TOKEN_POSITION to
# "first" or "last" to compare the two — it is a real modelling choice, not a
# correctness one.
TARGET_TOKEN_POSITION = "first"


def resolve_target_position(tokenizer, text: str, word_index: int) -> int | None:
    """Map a glaux word-list index to a token index in `tokenizer(text)`.

    The previous code used `word_ids.index()` — an index into glaux's WORD list —
    as an index into the model's TOKEN list. Those coincide only when every word
    before the target happens to be exactly one token, which is false for a
    WordPiece model on polytonic Greek. So the hidden state being classified was
    frequently some *other* word's representation (audit item 8).

    This derives the position from the encoding the model actually sees, using
    `word_ids()` (which maps token -> source word) and the token's character span
    to confirm the alignment.

    Returns None when the target cannot be aligned to a real (non-special,
    non-empty-span) token.
    """
    enc = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    try:
        word_ids = enc.word_ids()
    except Exception:
        return None
    offsets = _offsets_of(enc)

    candidates = []
    for token_idx, wid in enumerate(word_ids):
        if wid != word_index:
            continue  # skip [CLS]/[SEP] and other words
        if offsets is not None:
            start, end = offsets[token_idx]
            if start == end:
                continue  # special/piece with no source span
        candidates.append(token_idx)

    if not candidates:
        return None
    return candidates[0] if TARGET_TOKEN_POSITION == "first" else candidates[-1]


def wsd_collate(features, tokenizer):
    """Pad to the longest sequence in the batch and re-derive target positions.

    Positions are recomputed on the padded encoding (right padding keeps earlier
    indices stable, but we verify rather than rely on it) so the hidden state we
    read is the one at the target token.
    """
    batch = tokenizer.pad(
        [{"input_ids": f["input_ids"].tolist(), "attention_mask": f["attention_mask"].tolist()}
         for f in features],
        padding=True,
        return_tensors="pt",
    )
    labels = torch.stack([f["label"] for f in features])
    positions = torch.tensor([f["target_position"] for f in features], dtype=torch.long)
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "target_position": positions,
        "label": labels,
        "glaux_id": [f["glaux_id"] for f in features],
    }


# model
class WSDClassifier(nn.Module):    
    def __init__(self, bert_model, num_senses: int, dropout: float = 0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_senses)
        
    def forward(self, input_ids, attention_mask, target_position):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        batch_size = sequence_output.size(0)
        
        # Clamp defensively: an out-of-range position used to raise an opaque CUDA
        # index error deep in the gather.
        safe_positions = target_position.clamp(0, sequence_output.size(1) - 1)
        target_embeddings = sequence_output[
            torch.arange(batch_size, device=sequence_output.device),
            safe_positions
        ]  # [batch_size, hidden_size]

        target_embeddings = self.dropout(target_embeddings)
        logits = self.classifier(target_embeddings)
        
        return logits


# trainer
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = max(len(train_loader) * config.num_epochs, 1)
        # Warmup must not exceed total steps: with ~7 batches/epoch and
        # warmup_steps=100 the schedule used to still be warming up when training
        # ended, so the LR never reached 2e-5.
        warmup = min(config.warmup_steps, int(total_steps * 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.best_state = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _amp(self):
        if not torch.cuda.is_available():
            import contextlib
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=torch.bfloat16 if self.config.bf16 else torch.float16)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_positions = batch['target_position'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            with self._amp():
                logits = self.model(input_ids, attention_mask, target_positions)
                loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        avg_loss = total_loss / max(len(self.train_loader), 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Evaluating', leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_positions = batch['target_position'].to(self.device)
                labels = batch['label'].to(self.device)

                with self._amp():
                    logits = self.model(input_ids, attention_mask, target_positions)
                    loss = self.criterion(logits, labels)

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, save_dir="./wsd_model"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

            val_loss, val_acc, val_preds, val_labels = self.evaluate()
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                # Keep the best weights in memory so they can be restored before
                # the final reported evaluation (load_best_model_at_end).
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': asdict(self.config)
                }, save_path / 'best_model.pt')
                print(f"Saved best model with Val Acc={val_acc:.4f} at epoch {epoch}.")

        if self.config.load_best_model_at_end and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"Restored best weights (epoch val acc {self.best_val_acc:.4f}) "
                  f"for final evaluation.")

        # Final evaluation, on restored-best weights when enabled.
        final_val_loss, final_val_acc, _, _ = self.evaluate()
        self.history['final_val_loss'] = final_val_loss
        self.history['final_val_acc'] = final_val_acc
        self.history['best_epoch'] = int(
            max(range(len(self.history['val_acc'])), key=lambda i: self.history['val_acc'][i]) + 1
        ) if self.history['val_acc'] else None

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'history': self.history
        }, save_path / 'final_model.pt')

        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print("Training complete. Best Val Acc: {:.4f} | Final Val Acc: {:.4f}".format(
            self.best_val_acc, final_val_acc))
        return self.history


def target_tokenization_report(dataset, tokenizer, surface: str) -> dict:
    """How often does the target word occupy exactly one token? (audit item 8)

    The encoder is uncased and its tokenizer normalizes, so an inflected target
    form can land on a subword or an [UNK]. Reading the hidden state at such a
    position silently measures something else. This reports the rate instead of
    assuming alignment.
    """
    single = 0
    unk = 0
    total = 0
    unk_id = tokenizer.unk_token_id
    for item in dataset.data:
        total += 1
        pos = item['target_position']
        # One offset-aware encoding per example gives us id AND span together.
        enc = tokenizer(item['text'], truncation=True, max_length=512,
                        return_offsets_mapping=True)
        if unk_id is not None and enc['input_ids'][pos] == unk_id:
            unk += 1
        # "exactly one token" = this token's char span covers the whole word.
        start, end = _offsets_of(enc)[pos]
        words_in_text = item['text'].split()
        word = words_in_text[item['word_index']] if item['word_index'] < len(words_in_text) else ""
        if word and item['text'][start:end] == word:
            single += 1
    return {
        "surface": surface,
        "examples": total,
        "dropped_unaligned": len(getattr(dataset, "unaligned", [])),
        "single_token_rate": (single / total) if total else 0.0,
        "unk_rate": (unk / total) if total else 0.0,
    }


def run_word(model, tokenizer, word_key: str, cfg: WSDConfig, seeds: list[int], out_root: Path,
             model_label: str) -> dict:
    """Train + evaluate one (model, word) pair across `seeds`, independently."""
    spec = TARGET_WORDS[word_key]
    full = WSDDataset(
        data_path=repo_path("wsd", "ancient-greek-wsd-data") / spec["file"],
        tokenizer=tokenizer,
        glaux_data=sentences,
        target_word=spec["surface"],
        max_length=cfg.max_length,
    )

    report = target_tokenization_report(full, tokenizer, spec["surface"])
    print(f"  target-token check [{model_label}/{word_key}]: "
          f"{report['examples']} examples, "
          f"first-piece-of-target 100% (word_ids-verified), "
          f"single-token {report['single_token_rate']:.1%}, [UNK] {report['unk_rate']:.1%}")
    if report["dropped_unaligned"]:
        print(f"  NOTE: {report['dropped_unaligned']} examples were dropped as "
              f"unaligned (target split across subwords or truncated away).")
    if report["unk_rate"] > 0:
        print(f"  WARNING: {report['unk_rate']:.1%} of targets tokenize to [UNK]; "
              f"their hidden states carry no lexical information.")

    collate = lambda feats: wsd_collate(feats, tokenizer)  # noqa: E731
    best_accs, final_accs = [], []

    for seed in seeds:
        # Fresh split per seed is what the std over seeds is measuring.
        g = torch.Generator().manual_seed(seed)
        n = len(full)
        train_size = int((1 - cfg.test_size) * n)
        train_ds, val_ds = torch.utils.data.random_split(full, [train_size, n - train_size], generator=g)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate)

        # CRITICAL: a private copy of the encoder per (word, seed). Sharing one
        # encoder across words made the second word's result depend on the first
        # word's training.
        encoder = copy.deepcopy(model)
        classifier = WSDClassifier(bert_model=encoder, num_senses=full.num_senses,
                                   dropout=cfg.dropout).to(cfg_device())

        word_cfg = dataclass_replace(cfg, random_state=seed)
        trainer = Trainer(model=classifier, train_loader=train_loader, val_loader=val_loader,
                          config=word_cfg)
        history = trainer.train(save_dir=out_root / f"seed{seed}")
        best_accs.append(trainer.best_val_acc)
        final_accs.append(history.get("final_val_acc", float("nan")))

    summary = summarize(best_accs, final_accs)
    print(f"  ==> {model_label}/{word_key}: best {summary['best_mean']:.4f} "
          f"+/- {summary['best_std']:.4f} (n_seeds={len(seeds)})")
    return {"word": word_key, "target_tokenization": report, **summary}


def cfg_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataclass_replace(cfg, **overrides):
    d = asdict(cfg)
    d.update(overrides)
    known = {f.name for f in WSDConfig.__dataclass_fields__.values()}
    return WSDConfig(**{k: v for k, v in d.items() if k in known})


def summarize(best: list[float], final: list[float]) -> dict:
    def _stats(values):
        values = [v for v in values if v == v]  # drop NaN
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    b, f = _stats(best), _stats(final)
    return {"best_mean": b["mean"], "best_std": b["std"], "best_min": b["min"], "best_max": b["max"],
            "final_mean": f["mean"], "final_std": f["std"], "best_per_seed": best, "final_per_seed": final}


def evaluate_model(model_path: str, seeds: list[int], words: list[str], num_epochs: int | None = None) -> dict:
    """Fine-tune + evaluate one encoder on all `words`, each fully independent."""
    label = Path(model_path).name if Path(model_path).name not in ("", ".") else model_path
    print(f"\n{'=' * 64}\nEncoder: {model_path}\n{'=' * 64}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(cfg_device())

    cfg = WSDConfig(model_name=model_path)
    if num_epochs:
        cfg.num_epochs = num_epochs

    out_root = WSD_DIR / "wsd_models" / label
    results = []
    for word_key in words:
        results.append(run_word(model, tokenizer, word_key, cfg, seeds, out_root, label))

    return {"model": model_path, "label": label, "words": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", default="ours,baseline",
                        help="Comma list of: ours, baseline, or any path to an HF model dir.")
    parser.add_argument("--words", default="harmonia,kosmos", help="Comma list of target words.")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per (model, word).")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="Path or hub id for the comparison model.")
    args = parser.parse_args()

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    for w in words:
        if w not in TARGET_WORDS:
            raise SystemExit(f"Unknown word '{w}'. Known: {sorted(TARGET_WORDS)}")

    model_specs = []
    for spec in [m.strip() for m in args.models.split(",") if m.strip()]:
        if spec == "ours":
            model_specs.append(_resolve_model_dir())
        elif spec == "baseline":
            model_specs.append(args.baseline)
        else:
            model_specs.append(spec)

    seeds = list(range(args.seeds))
    record_run_provenance(stage="wsd", extra={"models": model_specs, "words": words, "seeds": seeds})

    all_results = []
    for model_path in model_specs:
        all_results.append(evaluate_model(model_path, seeds, words, num_epochs=args.epochs))

    # --- report + results table --------------------------------------------
    print(f"\n{'=' * 64}\nWSD SUMMARY (mean +/- std over {len(seeds)} seeds)\n{'=' * 64}")
    metrics_block = {}
    for res in all_results:
        metrics_block[res["label"]] = {}
        for word_res in res["words"]:
            key = word_res["word"]
            metrics_block[res["label"]][key] = {
                "best_mean": word_res["best_mean"], "best_std": word_res["best_std"],
                "final_mean": word_res["final_mean"], "final_std": word_res["final_std"],
                "best_per_seed": word_res["best_per_seed"],
                "target_single_token_rate": word_res["target_tokenization"]["single_token_rate"],
            }
            print(f"  {res['label']:<28} {key:<10} "
                  f"best {word_res['best_mean']:.4f} +/- {word_res['best_std']:.4f}   "
                  f"final {word_res['final_mean']:.4f} +/- {word_res['final_std']:.4f}")
            record_result(stage="wsd", model=res["label"], metric=f"{key}_best_val_acc",
                          value=word_res["best_mean"], split="glaux-val", seed=f"{len(seeds)}-seed")
            record_result(stage="wsd", model=res["label"], metric=f"{key}_best_val_acc_std",
                          value=word_res["best_std"], split="glaux-val", seed=f"{len(seeds)}-seed")

    write_metrics({"wsd": metrics_block, "seeds": seeds}, stage="wsd")
    out = WSD_DIR / "wsd_results.json"
    out.write_text(json.dumps({"seeds": seeds, "results": all_results}, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
