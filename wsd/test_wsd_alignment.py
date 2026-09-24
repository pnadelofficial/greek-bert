#!/usr/bin/env python3
"""Tests for WSD target-position alignment (docs/AUDIT-2026-07.md item 8).

These encode the bug that made every pre-fix WSD number meaningless: the target
position was glaux's index into its own WORD list, used directly as an index into
the model's TOKEN list. For a WordPiece tokenizer on polytonic Greek those differ
for any sentence with a multi-subword word before the target, so the classifier
read a neighbouring word's hidden state. Measured on the real glaux data the old
logic landed on the correct word 0/538 (harmonia) and 0/1375 (kosmos) times.

Run:  python wsd/test_wsd_alignment.py      (exit 0 = pass; needs no GPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import wsd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# A sentence where early words are guaranteed multi-subword, so a word index can
# never double as a token index.
SENTENCE = "καλός κἀγαθός ἐστιν ἡ ἀρμονία"
TARGET_WORD = "ἀρμονία"


def _tokenizer():
    path = Path(wsd.DEFAULT_BASELINE)
    if not path.is_dir():
        return None
    return AutoTokenizer.from_pretrained(str(path))


def test_word_index_is_not_token_index() -> None:
    """The regression itself: word index != token index when subwords precede."""
    print("\ntest_word_index_is_not_token_index")
    tok = _tokenizer()
    if tok is None:
        print("  SKIP: baseline model dir not present")
        return

    words = SENTENCE.split()
    word_index = words.index(TARGET_WORD)
    token_pos = wsd.resolve_target_position(tok, SENTENCE, word_index)

    check("target resolves to a position", token_pos is not None, f"pos={token_pos}")
    if token_pos is None:
        return
    check("resolved position is NOT the raw word index", token_pos != word_index,
          f"word_index={word_index} token_pos={token_pos}")

    enc = tok(SENTENCE, return_offsets_mapping=True, truncation=True, max_length=512)
    # It must be the first token belonging to the target word.
    for_word = [i for i, w in enumerate(enc.word_ids()) if w == word_index]
    check("resolves to the FIRST token of the target word",
          bool(for_word) and token_pos == for_word[0],
          f"candidates={for_word} chosen={token_pos}")

    offsets = wsd._offsets_of(enc)
    start, _ = offsets[token_pos]
    span = SENTENCE[start:start + len(TARGET_WORD)]
    check("span starts at the target word", span == TARGET_WORD, f"span={span!r}")


def test_never_returns_special_tokens() -> None:
    """Position must never be [CLS]/[SEP] or an empty-span token."""
    print("\ntest_never_returns_special_tokens")
    tok = _tokenizer()
    if tok is None:
        print("  SKIP: baseline model dir not present")
        return
    for sentence in (TARGET_WORD, f"{TARGET_WORD} ἐστιν", SENTENCE, "α β γ"):
        words = sentence.split()
        if TARGET_WORD not in words:
            continue
        pos = wsd.resolve_target_position(tok, sentence, words.index(TARGET_WORD))
        if pos is None:
            continue
        enc = tok(sentence, return_offsets_mapping=True, truncation=True, max_length=512)
        start, end = wsd._offsets_of(enc)[pos]
        check(f"non-empty span ({sentence[:18]}…)", start != end, f"pos={pos} span={(start, end)}")
        check(f"not a special token ({sentence[:18]}…)", enc.word_ids()[pos] is not None)


def test_out_of_range_word_index_returns_none() -> None:
    """A word index past the end must fail loudly (None), not silently clamp."""
    print("\ntest_out_of_range_word_index_returns_none")
    tok = _tokenizer()
    if tok is None:
        print("  SKIP: baseline model dir not present")
        return
    check("word_index=999 -> None",
          wsd.resolve_target_position(tok, SENTENCE, 999) is None)


def test_offsets_helper_handles_both_key_names() -> None:
    """transformers 5.x uses offset_mapping; 4.x used offsets_mapping."""
    print("\ntest_offsets_helper_handles_both_key_names")
    tok = _tokenizer()
    if tok is None:
        print("  SKIP: baseline model dir not present")
        return
    enc = tok(SENTENCE, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = wsd._offsets_of(enc)
    check("offsets found", offsets is not None and len(offsets) == len(enc["input_ids"]))

    class _Legacy(dict):
        pass

    legacy = _Legacy(input_ids=enc["input_ids"])
    legacy["offsets_mapping"] = offsets
    check("reads the 4.x key too", wsd._offsets_of(legacy) == offsets)
    check("missing offsets -> None", wsd._offsets_of({"input_ids": [1, 2]}) is None)


def test_target_position_within_sequence() -> None:
    """Positions must survive truncation, or the dataset must raise."""
    print("\ntest_target_position_within_sequence")
    tok = _tokenizer()
    if tok is None:
        print("  SKIP: baseline model dir not present")
        return
    ds = wsd.WSDDataset(
        data_path=wsd.repo_path("wsd", "ancient-greek-wsd-data") / "harmonia_glaux.txt",
        tokenizer=tok,
        glaux_data=wsd.sentences,
        target_word="harmonia",
        max_length=512,
    )
    check("dataset loaded examples", len(ds.data) > 0, f"{len(ds.data)} examples")
    bad = 0
    for item in ds.data[:200]:
        n = len(tok(item["text"], truncation=True, max_length=512)["input_ids"])
        if item["target_position"] >= n:
            bad += 1
    check("all sampled positions inside the encoded sequence", bad == 0, f"{bad} out of range")

    # The stored position must belong to the target word. `word_ids()` is the
    # authoritative mapping (token -> source word index). Char-span comparison is
    # NOT usable here: this tokenizer normalizes polytonic Greek (breathing marks,
    # iota subscript, NFC/NFD), so spans routinely fail a string comparison even
    # when the alignment is correct — measured 116/200 vs 200/200 for word_ids().
    correct = 0
    for item in ds.data[:200]:
        enc = tok(item["text"], truncation=True, max_length=512, return_offsets_mapping=True)
        if enc.word_ids()[item["target_position"]] == item["word_index"]:
            correct += 1
    check("stored positions belong to the target word (word_ids)", correct == 200,
          f"{correct}/200")

    # And the OLD logic must be wrong here, or this test guards nothing.
    old_correct = sum(
        1 for item in ds.data[:200]
        if tok(item["text"], truncation=True, max_length=512).word_ids()[item["word_index"]]
        == item["word_index"]
    )
    check("old word-index-as-token-index logic would FAIL here", old_correct < 20,
          f"old logic accidentally correct {old_correct}/200")


def test_load_best_model_at_end_restores_best_weights() -> None:
    """`load_best_model_at_end` must restore the BEST epoch, not the last one.

    Uses a stub model and a scripted validation-accuracy sequence so the expected
    outcome is exact rather than data-dependent: 0.50 -> 0.90 (best) -> 0.30.
    """
    print("\ntest_load_best_model_at_end_restores_best_weights")
    import copy as _copy

    class Tiny(torch.nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 8})()
            self.fc = torch.nn.Linear(8, n)

        def forward(self, input_ids, attention_mask, target_position):
            return self.fc(torch.zeros(input_ids.size(0), 8))

    batch = {
        "input_ids": torch.zeros(2, 3).long(),
        "attention_mask": torch.ones(2, 3).long(),
        "target_position": torch.tensor([1, 1]),
        "label": torch.tensor([0, 1]),
    }
    cfg = wsd.WSDConfig(model_name="stub", num_epochs=3, batch_size=2,
                        load_best_model_at_end=True)
    trainer = wsd.Trainer(model=Tiny(), train_loader=[batch], val_loader=[batch], config=cfg)

    scripted = [0.50, 0.90, 0.30]
    state = {"i": 0}
    original_eval = wsd.Trainer.evaluate

    def fake_eval(self):
        acc = scripted[min(state["i"], len(scripted) - 1)]
        state["i"] += 1
        return (1.0 - acc, acc, [], [])

    saved_best: dict = {}
    original_save = torch.save

    def spy_save(obj, path, *a, **k):
        if str(path).endswith("best_model.pt"):
            saved_best.update({kk: vv.clone() for kk, vv in obj["model_state_dict"].items()})
        return original_save(obj, path, *a, **k)

    wsd.Trainer.evaluate = fake_eval
    torch.save = spy_save
    try:
        history = trainer.train(save_dir="/tmp/wsd_restore_test")
    finally:
        torch.save = original_save
        wsd.Trainer.evaluate = original_eval

    check("best val acc is the max of the scripted sequence",
          abs(trainer.best_val_acc - 0.90) < 1e-9, f"{trainer.best_val_acc}")
    check("best_epoch recorded as 2", history.get("best_epoch") == 2, f"{history.get('best_epoch')}")
    live = trainer.model.state_dict()
    check("live weights equal the BEST epoch snapshot (not the last epoch)",
          all(torch.equal(live[k], v) for k, v in saved_best.items()))


def main() -> int:
    print("=" * 64)
    print("WSD target-position alignment tests")
    print("=" * 64)
    for test in (
        test_word_index_is_not_token_index,
        test_never_returns_special_tokens,
        test_out_of_range_word_index_returns_none,
        test_offsets_helper_handles_both_key_names,
        test_target_position_within_sequence,
        test_load_best_model_at_end_restores_best_weights,
    ):
        test()
    print("\n" + "=" * 64)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


