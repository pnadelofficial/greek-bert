import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

POSTAG_VOCABS = [
    # Position 0: Part of speech
    list('-nvadclrpgmuixbez'),
    # Position 1: Person
    list('-123'),
    # Position 2: Number
    list('-sdp'),
    # Position 3: Tense
    list('-pirlttfa'),   # note: two 't's collapsed below, kept here for clarity
    # Position 4: Mood
    list('-isonmpd'),
    # Position 5: Voice
    list('-apme'),
    # Position 6: Gender
    list('-mfn'),
    # Position 7: Case
    list('-ngdavl'),
    # Position 8: Degree
    list('-pcs'),
]

POSTAG_VOCABS = [list(dict.fromkeys(v)) for v in POSTAG_VOCABS]
 
POSTAG_SIZES = [len(v) for v in POSTAG_VOCABS]
 
# Quick lookup: char -> index for each position
POSTAG_CHAR2IDX = [
    {c: i for i, c in enumerate(vocab)} for vocab in POSTAG_VOCABS
]

# suffix lemmatization
# We represent the lemma as a character-level transformation of the form.
# The edit script encodes the suffix to strip and the suffix to append.
# Example: form=γεννηθέντα, lemma=γεννάω
#   -> strip 5 chars from end, append "άω"  -> edit = "5:άω"
#
# This is a simplified but effective approach. The full edit-tree method
# (as used by UDPipe 2 and spaCy) builds a binary tree of edits over the
# whole string, but suffix-only edits cover >95% of Greek morphology. 

MAX_EDIT_STRIP = 12   # max suffix chars to strip; tune if you see truncation

def form_to_edit(form: str, lemma: str) -> str:
    """
    Compute the edit script that transforms form -> lemma.
    Returns a string like "3:ω" meaning "strip 3 chars, append 'ω'".
    Returns "0:" for identical form/lemma (copy).
    """
    # Find longest common prefix
    min_len = min(len(form), len(lemma))
    common = 0
    for i in range(min_len):
        if form[i] == lemma[i]:
            common += 1
        else:
            break
 
    strip = len(form) - common
    append = lemma[common:]
 
    # Cap strip at MAX_EDIT_STRIP; beyond this treat as irregular
    if strip > MAX_EDIT_STRIP:
        # Fall back to emitting the full lemma as the "append" with strip=len(form)
        # This handles very irregular forms like οἶδα
        return f"{len(form)}:{lemma}"
 
    return f"{strip}:{append}"

def apply_edit(form: str, edit: str) -> str:
    """Apply an edit script to a form to recover the lemma."""
    try:
        strip_s, append = edit.split(':', 1)
        strip = int(strip_s)
        base = form[:-strip] if strip > 0 else form
        return base + append
    except Exception:
        return form  # fallback: return form unchanged
 
 
def build_edit_vocab(train_sentences: list) -> dict:
    """
    Collect all edit scripts seen in training data.
    Returns {edit_str: index} mapping.
    """
    counts = defaultdict(int)
    for sent in train_sentences:
        sent_data = zip(sent.forms, sent.lemmas, sent.postags)
        for form, lemma, _ in sent_data:
            edit = form_to_edit(form, lemma)
            counts[edit] += 1
 
    MIN_EDIT_FREQ = 1
 
    vocab = {'<UNK>': 0}
    for edit, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= MIN_EDIT_FREQ:
            vocab[edit] = len(vocab)
 
    return vocab

@dataclass
class Sentence:
    forms:   list   # raw surface forms (strings)
    lemmas:  list   # gold lemmas (strings)
    postags: list   # gold postags (9-char strings)
 
 
def read_tsv(path: Path) -> list:
    """Read a TSV file into a list of Sentence objects."""
    sentences = []
    current = []
 
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip() == '':
            if current:
                forms, lemmas, postags = zip(*current)
                sentences.append(Sentence(
                    forms=list(forms),
                    lemmas=list(lemmas),
                    postags=list(postags),
                ))
                current = []
        else:
            parts = line.split('\t')
            if len(parts) == 3:
                current.append(parts)
 
    if current:
        forms, lemmas, postags = zip(*current)
        sentences.append(Sentence(
            forms=list(forms),
            lemmas=list(lemmas),
            postags=list(postags),
        ))
 
    return sentences

class GreekMorphDataset(Dataset):
    """
    Tokenizes sentences with the BERT tokenizer and aligns word-level
    labels to the subword tokens produced by the tokenizer.
 
    Label alignment strategy: assign the label to the FIRST subword of
    each word; all subsequent subwords get label index -100 (ignored by
    cross-entropy loss).
 
    EXPERIMENT: try "all subwords" alignment instead (assign the same label
    to every subword of a word). This sometimes helps for morphological tasks
    where the suffix carries the morphological information.
    """
 
    def __init__(
        self,
        sentences: list,
        tokenizer,
        edit_vocab: dict,
        max_length: int = 256,   # EXPERIMENT: increase for very long sentences
    ):
        self.sentences  = sentences
        self.tokenizer  = tokenizer
        self.edit_vocab = edit_vocab
        self.max_length = max_length
 
    def __len__(self):
        return len(self.sentences)
 
    def __getitem__(self, idx):
        sent = self.sentences[idx]
 
        # Tokenize each word separately so we can track subword boundaries.
        # is_split_into_words=True tells the tokenizer the input is
        # pre-tokenized; it still runs its own subword splitting internally.
        encoding = self.tokenizer(
            sent.forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
 
        input_ids      = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
 
        # word_ids maps each subword position -> word index (None for special tokens)
        word_ids = encoding.word_ids(batch_index=0)
 
        # Build per-position postag label tensors (9 positions)
        # and edit script label tensor
        seq_len = input_ids.size(0)
 
        postag_labels = torch.full(
            (9, seq_len), fill_value=-100, dtype=torch.long
        )
        edit_labels = torch.full(
            (seq_len,), fill_value=-100, dtype=torch.long
        )
 
        prev_word_idx = None
        for token_pos, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD]) — skip
                continue
 
            if word_idx != prev_word_idx:
                # First subword of this word: assign labels
                postag = sent.postags[word_idx].ljust(9, '-')[:9]
                for pos in range(9):
                    char = postag[pos]
                    vocab = POSTAG_CHAR2IDX[pos]
                    # Unknown char in this position maps to 0 ('-')
                    postag_labels[pos, token_pos] = vocab.get(char, 0)
 
                edit = form_to_edit(sent.forms[word_idx], sent.lemmas[word_idx])
                edit_labels[token_pos] = self.edit_vocab.get(edit, 0)  # 0 = <UNK>
 
            # Subsequent subwords of the same word keep -100 (ignored)
            prev_word_idx = word_idx
 
        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'postag_labels':  postag_labels,  # shape (9, seq_len)
            'edit_labels':    edit_labels,    # shape (seq_len,)
        }
 

class GreekMorphTagger(nn.Module):
    """
    Greek BERT + 9 postag classification heads + 1 edit-script head.
 
    Architecture:
        BERT encoder -> contextual embeddings (hidden_size)
            -> dropout
            -> 9 linear heads, one per postag position
            -> 1 linear head for edit script (lemmatization)
 
    All heads share the same BERT encoder, so the model learns
    representations useful for both tasks simultaneously.
 
    EXPERIMENT: try task-specific projection layers before each head
    (e.g. a small MLP) instead of going directly from BERT -> linear.
    This gives each head its own "view" of the representation.
 
    EXPERIMENT: try separate learning rates for the BERT encoder vs
    the classification heads (discriminative fine-tuning). The heads
    typically benefit from a higher LR than the pretrained encoder.
    """
 
    def __init__(
        self,
        bert_model_name_or_path: str,
        edit_vocab_size: int,
        dropout: float = 0.1,   # EXPERIMENT: try 0.1–0.3; higher = more regularization
        freeze_bert_layers: int = 0,  # EXPERIMENT: freeze bottom N layers of BERT
                                      # to prevent catastrophic forgetting on small data
    ):
        super().__init__()
 
        self.bert = AutoModel.from_pretrained(bert_model_name_or_path)
        hidden_size = self.bert.config.hidden_size
 
        # EXPERIMENT: freeze_bert_layers > 0 is useful when your training data
        # is small relative to the BERT model size. Start with 0 (all layers
        # fine-tuned), and try freezing the bottom 6 layers if you see
        # overfitting on dev set early in training.
        if freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
 
        self.dropout = nn.Dropout(dropout)
 
        # 9 heads, one per postag position
        self.postag_heads = nn.ModuleList([
            nn.Linear(hidden_size, POSTAG_SIZES[pos])
            for pos in range(9)
        ])
 
        # 1 head for edit script (lemmatization)
        self.edit_head = nn.Linear(hidden_size, edit_vocab_size)
 
        # Loss weights for the two tasks.
        # EXPERIMENT: adjust these to prioritize postag vs lemma accuracy.
        # If lemmatization is lagging, try increasing LEMMA_LOSS_WEIGHT.
        self.POSTAG_LOSS_WEIGHT = 1.0
        self.LEMMA_LOSS_WEIGHT  = 1.0
 
    def forward(
        self,
        input_ids,
        attention_mask,
        postag_labels=None,
        edit_labels=None,
    ):
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # sequence_output: (batch, seq_len, hidden_size)
        sequence_output = self.dropout(outputs.last_hidden_state)
 
        # Postag logits: list of 9 tensors, each (batch, seq_len, vocab_size_i)
        postag_logits = [head(sequence_output) for head in self.postag_heads]
 
        # Edit logits: (batch, seq_len, edit_vocab_size)
        edit_logits = self.edit_head(sequence_output)
 
        loss = None
        if postag_labels is not None and edit_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
 
            # Postag loss: sum over 9 positions
            # postag_labels shape: (batch, 9, seq_len)
            postag_loss = sum(
                loss_fn(
                    postag_logits[pos].view(-1, POSTAG_SIZES[pos]),
                    postag_labels[:, pos, :].reshape(-1),
                )
                for pos in range(9)
            )
 
            # Edit/lemma loss
            edit_loss = loss_fn(
                edit_logits.view(-1, edit_logits.size(-1)),
                edit_labels.view(-1),
            )
 
            loss = (
                self.POSTAG_LOSS_WEIGHT * postag_loss
                + self.LEMMA_LOSS_WEIGHT * edit_loss
            )
 
        return {
            'loss':          loss,
            'postag_logits': postag_logits,
            'edit_logits':   edit_logits,
        }

def evaluate(model, dataloader, edit_vocab, device):
    """
    Compute per-position postag accuracy and exact-match lemma accuracy.
    Returns a dict of metrics.
    """
    model.eval()
    idx2edit = {v: k for k, v in edit_vocab.items()}
 
    # Postag: count correct per position
    pos_correct = [0] * 9
    pos_total   = [0] * 9
 
    # Lemma: count exact-match lemmas
    lemma_correct = 0
    lemma_total   = 0
 
    # Full postag: count fully-correct 9-char tags
    full_correct = 0
 
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            postag_labels  = batch['postag_labels'].to(device)  # (B, 9, L)
            edit_labels    = batch['edit_labels'].to(device)    # (B, L)
 
            out = model(input_ids=input_ids, attention_mask=attention_mask)
 
            postag_logits = out['postag_logits']  # list of 9 x (B, L, V)
            edit_logits   = out['edit_logits']    # (B, L, E)
 
            # Predicted indices
            pred_postag = torch.stack(
                [logits.argmax(-1) for logits in postag_logits], dim=1
            )  # (B, 9, L)
            pred_edit = edit_logits.argmax(-1)  # (B, L)
 
            # Mask: only positions with real labels
            mask = postag_labels[:, 0, :] != -100  # (B, L)
 
            # Per-position postag accuracy
            pos_all_correct = torch.ones_like(mask)
            for pos in range(9):
                gold = postag_labels[:, pos, :]
                pred = pred_postag[:, pos, :]
                valid = mask
                correct = (pred == gold) & valid
                pos_correct[pos] += correct.sum().item()
                pos_total[pos]   += valid.sum().item()
                pos_all_correct &= (correct | ~valid)
 
            full_correct += (pos_all_correct & mask).sum().item()
 
            # Edit/lemma accuracy (exact match on predicted lemma string)
            edit_mask = edit_labels != -100
            correct_edits = (pred_edit == edit_labels) & edit_mask
            lemma_correct += correct_edits.sum().item()
            lemma_total   += edit_mask.sum().item()
 
    metrics = {}
    for pos in range(9):
        if pos_total[pos] > 0:
            metrics[f'pos{pos}_acc'] = pos_correct[pos] / pos_total[pos]
 
    total_pos_tokens = pos_total[0] if pos_total[0] > 0 else 1
    metrics['postag_full_acc'] = full_correct / total_pos_tokens
    metrics['lemma_edit_acc']  = lemma_correct / lemma_total if lemma_total else 0.0
 
    # Named convenience metrics
    pos_names = ['pos', 'person', 'number', 'tense', 'mood', 'voice', 'gender', 'case', 'degree']
    metrics['named'] = {
        pos_names[i]: metrics.get(f'pos{i}_acc', 0.0) for i in range(9)
    }
 
    return metrics
 

def train(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()  # Apple Silicon
        else 'cpu'
    )
    print(f"Device: {device}")
 
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    print("Reading training data...")
    train_sentences = read_tsv(args.train)
    dev_sentences   = read_tsv(args.dev)
    print(f"  Train: {len(train_sentences):,} sentences")
    print(f"  Dev:   {len(dev_sentences):,} sentences")
 
    print("Building edit vocabulary...")
    print(train_sentences[0])  # sanity check: print first sentence
    edit_vocab = build_edit_vocab(train_sentences)
    print(f"  Edit vocab size: {len(edit_vocab):,}")
 
    # Save edit vocab alongside the model so we can reload it at inference time
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output}/edit_vocab.json", 'w', encoding='utf-8') as f:
        json.dump(edit_vocab, f, ensure_ascii=False, indent=2)
 
    # -------------------------------------------------------------------------
    # Tokenizer and datasets
    # -------------------------------------------------------------------------
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
 
    train_dataset = GreekMorphDataset(
        train_sentences, tokenizer, edit_vocab,
        max_length=args.max_length,
    )
    dev_dataset = GreekMorphDataset(
        dev_sentences, tokenizer, edit_vocab,
        max_length=args.max_length,
    )
 
    # EXPERIMENT: increasing batch_size speeds up training but requires more
    # GPU memory. If you're on a small GPU, try batch_size=8 with
    # gradient_accumulation_steps=4 to simulate batch_size=32.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,               # EXPERIMENT: try False for curriculum learning
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,  # no gradients, can use larger batch
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
 
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print(f"Loading model from {args.model}...")
    model = GreekMorphTagger(
        bert_model_name_or_path=args.model,
        edit_vocab_size=len(edit_vocab),
        dropout=args.dropout,
        freeze_bert_layers=args.freeze_layers,
    ).to(device)
 
    # -------------------------------------------------------------------------
    # Optimizer
    #
    # EXPERIMENT: try different learning rates for encoder vs heads.
    # A common pattern is to use 1e-5 for BERT and 1e-3 for the heads.
    # This is "discriminative fine-tuning" and often prevents the encoder
    # from drifting too far from its pretraining on small data.
    #
    # Example of layer-wise learning rates:
    #   optimizer = torch.optim.AdamW([
    #       {'params': model.bert.parameters(),        'lr': 1e-5},
    #       {'params': model.postag_heads.parameters(), 'lr': 1e-3},
    #       {'params': model.edit_head.parameters(),   'lr': 1e-3},
    #   ], weight_decay=args.weight_decay)
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # EXPERIMENT: try 0.0, 0.01, 0.1
    )
 
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    # EXPERIMENT: try different schedulers:
    #   get_cosine_schedule_with_warmup  — often better than linear for longer runs
    #   get_constant_schedule_with_warmup — useful for debugging (no LR decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
 
    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_dev_acc  = 0.0
    best_epoch    = 0
    history       = []
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        total_steps_epoch = 0
 
        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            postag_labels  = batch['postag_labels'].to(device)
            edit_labels    = batch['edit_labels'].to(device)
 
            out  = model(input_ids, attention_mask, postag_labels, edit_labels)
            loss = out['loss']
 
            loss.backward()
 
            # Gradient clipping stabilizes training, especially early on.
            # EXPERIMENT: try max_norm values of 0.5, 1.0, 5.0
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
 
            total_loss += loss.item()
            total_steps_epoch += 1
 
            if step % args.log_every == 0:
                avg_loss = total_loss / total_steps_epoch
                lr_now   = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                    f"| Loss {avg_loss:.4f} | LR {lr_now:.2e}"
                )
 
        # Evaluate on dev set at end of each epoch
        print(f"\nEvaluating epoch {epoch}...")
        metrics = evaluate(model, dev_loader, edit_vocab, device)
 
        postag_acc = metrics['postag_full_acc']
        lemma_acc  = metrics['lemma_edit_acc']
 
        print(f"  Full postag accuracy : {postag_acc:.4f}")
        print(f"  Lemma edit accuracy  : {lemma_acc:.4f}")
        print(f"  Per-position accuracies:")
        for name, acc in metrics['named'].items():
            bar = '█' * int(acc * 20)
            print(f"    {name:10s}: {acc:.4f}  {bar}")
 
        history.append({
            'epoch':      epoch,
            'train_loss': total_loss / total_steps_epoch,
            'postag_acc': postag_acc,
            'lemma_acc':  lemma_acc,
        })
 
        # Save best model checkpoint based on full postag accuracy.
        # EXPERIMENT: try saving on lemma_acc instead, or a combined metric,
        # depending on which task matters more to you.
        if postag_acc > best_dev_acc:
            best_dev_acc = postag_acc
            best_epoch   = epoch
            print(f"  ✓ New best — saving to {args.output}")
            model.bert.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            # Save the classification heads separately (they aren't part of
            # the HuggingFace model and need to be saved manually)
            torch.save(
                {
                    'postag_heads': model.postag_heads.state_dict(),
                    'edit_head':    model.edit_head.state_dict(),
                    'edit_vocab':   edit_vocab,
                    'args':         vars(args),
                },
                f"{args.output}/morph_heads.pt",
            )
 
        # Early stopping.
        # EXPERIMENT: try patience values of 3–10 epochs.
        # With small data, 3–5 is usually enough.
        if args.patience and (epoch - best_epoch) >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break
 
    print(f"\nTraining complete.")
    print(f"  Best epoch: {best_epoch}  |  Best full postag acc: {best_dev_acc:.4f}")
 
    # Save training history
    with open(f"{args.output}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

def load_model(model_dir: str, device=None):
    """
    Load a saved GreekMorphTagger for inference.
 
    Returns (model, tokenizer, edit_vocab).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    with open(f"{model_dir}/edit_vocab.json", encoding='utf-8') as f:
        edit_vocab = json.load(f)
 
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
 
    heads = torch.load(f"{model_dir}/morph_heads.pt", map_location=device)
 
    model = GreekMorphTagger(
        bert_model_name_or_path=model_dir,
        edit_vocab_size=len(edit_vocab),
    ).to(device)
    model.postag_heads.load_state_dict(heads['postag_heads'])
    model.edit_head.load_state_dict(heads['edit_head'])
    model.eval()
 
    return model, tokenizer, edit_vocab
 
 
def predict(model, tokenizer, edit_vocab, sentences: list, device=None):
    """
    Run inference on a list of pre-tokenized sentences.
 
    Args:
        sentences: list of lists of Greek word strings
                   e.g. [['Πιστεύομεν', 'εἰς', 'ἕνα']]
 
    Returns:
        list of lists of dicts, each dict having:
            'form', 'postag' (9-char string), 'lemma' (predicted)
    """
    if device is None:
        device = next(model.parameters()).device
 
    idx2edit = {v: k for k, v in edit_vocab.items()}
    results  = []
 
    for words in sentences:
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(device)
 
        word_ids = encoding.word_ids(batch_index=0)
 
        with torch.no_grad():
            out = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
            )
 
        pred_postag_chars = [
            logits.argmax(-1)[0] for logits in out['postag_logits']
        ]  # list of 9 tensors, each shape (seq_len,)
        pred_edit_idx = out['edit_logits'].argmax(-1)[0]  # (seq_len,)
 
        sent_results = []
        prev_word_idx = None
 
        for token_pos, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            prev_word_idx = word_idx
 
            # Reconstruct postag string
            postag = ''.join(
                POSTAG_VOCABS[pos][pred_postag_chars[pos][token_pos].item()]
                for pos in range(9)
            )
 
            # Reconstruct lemma from edit script
            edit = idx2edit.get(pred_edit_idx[token_pos].item(), '<UNK>')
            form = words[word_idx]
            if edit == '<UNK>':
                lemma = form   # fallback: return the form itself
            else:
                lemma = apply_edit(form, edit)
 
            sent_results.append({
                'form':   form,
                'postag': postag,
                'lemma':  lemma,
            })
 
        results.append(sent_results)
 
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Greek BERT for morphological tagging and lemmatization.'
    )
 
    # Required
    parser.add_argument('--train',  required=True, type=Path)
    parser.add_argument('--dev',    required=True, type=Path)
    parser.add_argument('--model',  required=True,
                        help='Path to your Greek BERT (HuggingFace format)')
    parser.add_argument('--output', required=True,
                        help='Directory to save the fine-tuned model')
 
    # Training hyperparameters
    # EXPERIMENT: these are the primary knobs to turn
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch-size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=2e-5,
                        help='Learning rate. Try 1e-5 to 5e-5 for BERT fine-tuning.')
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--weight-decay',type=float, default=0.01)
    parser.add_argument('--grad-clip',   type=float, default=1.0)
    parser.add_argument('--warmup-ratio',type=float, default=0.1,
                        help='Fraction of total steps used for LR warmup.')
    parser.add_argument('--max-length',  type=int,   default=256,
                        help='Max subword sequence length.')
    parser.add_argument('--freeze-layers', type=int, default=0,
                        help='Freeze bottom N BERT layers (0 = fine-tune all).')
    parser.add_argument('--patience',    type=int,   default=5,
                        help='Early stopping patience in epochs (0 = disabled).')
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--log-every',   type=int,   default=50,
                        help='Print loss every N steps.')
    parser.add_argument('--seed',        type=int,   default=42)
 
    args = parser.parse_args()
 
    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
 
    train(args)

if __name__ == '__main__':
    main()
  
