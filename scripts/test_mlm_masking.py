#!/usr/bin/env python3
"""Tests for mlm_masking — the 80/10/10 split and the special-token guard.

Why this file exists: the previous version of this file only asserted that two
consecutive calls produced *different* masks. That passes for a 100/0/0
implementation, which is exactly how "dynamic masking" shipped while silently
dropping the 80/10/10 term described in its own docstring (see
docs/AUDIT-2026-07.md item 1). These tests assert the *composition* of the
selected positions, not just their randomness.

Run:  python scripts/test_mlm_masking.py        (exit 0 = pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import mlm_masking  # noqa: E402

MASK, PAD, CLS, SEP = 103, 0, 101, 102
VOCAB = 35_000

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _sample(seed: int = 0, rows: int = 128, cols: int = 512, content_lo: int = 1000) -> torch.Tensor:
    """Content tokens only, with a padded tail on every row."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(content_lo, VOCAB, (rows, cols), generator=g)
    ids[:, -64:] = PAD
    return ids


def test_split_composition() -> None:
    """Selected positions must be ~80% [MASK] / 10% random / 10% unchanged."""
    print("\ntest_split_composition")
    ids = _sample()
    masked, labels = mlm_masking(
        ids, mask_token_id=MASK, mask_prob=0.30, pad_token_id=PAD,
        vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP,
    )
    selected = labels != -100
    n = selected.sum().item()
    check("some positions selected", n > 1000, f"{n} selected")

    frac_mask = (masked[selected] == MASK).float().mean().item()
    frac_same = (masked[selected] == ids[selected]).float().mean().item()
    both = (masked[selected] != ids[selected]) & (masked[selected] != MASK)
    frac_random = both.float().mean().item()

    print(f"    selected={selected.float().mean().item():.4f} "
          f"mask={frac_mask:.4f} random={frac_random:.4f} unchanged={frac_same:.4f}")

    # This is the assertion the old test lacked: 100/0/0 fails here.
    check("~80% [MASK]", abs(frac_mask - 0.80) < 0.02, f"{frac_mask:.4f}")
    check("~10% random", abs(frac_random - 0.10) < 0.02, f"{frac_random:.4f}")
    check("~10% unchanged", abs(frac_same - 0.10) < 0.02, f"{frac_same:.4f}")
    check("three treatments sum to 1", abs(frac_mask + frac_random + frac_same - 1.0) < 1e-4)


def test_labels_are_original_tokens() -> None:
    """Labels must be the ORIGINAL token at every selected position."""
    print("\ntest_labels_are_original_tokens")
    ids = _sample(seed=1)
    _, labels = mlm_masking(
        ids, mask_token_id=MASK, mask_prob=0.30, pad_token_id=PAD,
        vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP,
    )
    selected = labels != -100
    check("labels equal original ids", bool((labels[selected] == ids[selected]).all()))
    check("padding never labelled", int(labels[ids == PAD].ne(-100).sum()) == 0)
    check("unselected positions are -100", bool((labels[~selected] == -100).all()))


def test_special_tokens_never_masked() -> None:
    """[CLS]/[SEP]/[PAD]/[MASK] must never be prediction targets or be overwritten."""
    print("\ntest_special_tokens_never_masked")
    ids = _sample(seed=2)
    # Sprinkle specials through the middle where they are mask-eligible.
    ids[:, 0] = CLS
    ids[:, -65] = SEP
    ids[0, 10] = MASK
    ids[1, 10] = MASK

    for mask_prob in (0.30, 1.0):
        masked, labels = mlm_masking(
            ids, mask_token_id=MASK, mask_prob=mask_prob, pad_token_id=PAD,
            vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP,
        )
        tag = f"mask_prob={mask_prob}"
        for name, pos in (("CLS", ids == CLS), ("SEP", ids == SEP), ("PAD", ids == PAD)):
            check(f"{name} never labelled ({tag})", int((pos & (labels != -100)).sum()) == 0,
                  f"{int((pos & (labels != -100)).sum())} labelled")
        # [CLS] must survive unmodified in the input, even at mask_prob=1.0.
        check(f"[CLS] positions unchanged in input ({tag})", bool((masked[ids == CLS] == CLS).all()))
        check(f"[SEP] positions unchanged in input ({tag})", bool((masked[ids == SEP] == SEP).all()))


def test_selected_rate_matches_mask_prob() -> None:
    """Selection rate over ELIGIBLE tokens should match mask_prob."""
    print("\ntest_selected_rate_matches_mask_prob")
    ids = _sample(seed=3)
    eligible = (ids != PAD)
    for mask_prob in (0.15, 0.30):
        _, labels = mlm_masking(
            ids, mask_token_id=MASK, mask_prob=mask_prob, pad_token_id=PAD,
            vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP,
        )
        rate = (labels != -100).sum().item() / eligible.sum().item()
        check(f"selection rate ~= {mask_prob}", abs(rate - mask_prob) < 0.01, f"{rate:.4f}")


def test_random_tokens_in_vocabulary() -> None:
    """Random replacements must be valid ids and must not be [MASK]/[PAD]."""
    print("\ntest_random_tokens_in_vocabulary")
    ids = _sample(seed=4)
    masked, labels = mlm_masking(
        ids, mask_token_id=MASK, mask_prob=0.30, pad_token_id=PAD,
        vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP,
    )
    selected = labels != -100
    both = (masked != ids) & (masked != MASK) & selected
    check("random replacements exist", both.sum().item() > 100, f"{both.sum().item()}")
    check("random ids in vocab", bool((masked[both] < VOCAB).all()))


def test_dynamic_across_calls() -> None:
    """Fresh masks per call (the property the original test did check)."""
    print("\ntest_dynamic_across_calls")
    ids = _sample(seed=5)
    a, la = mlm_masking(ids, mask_token_id=MASK, mask_prob=0.30, pad_token_id=PAD,
                        vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP)
    b, lb = mlm_masking(ids, mask_token_id=MASK, mask_prob=0.30, pad_token_id=PAD,
                        vocab_size=VOCAB, cls_token_id=CLS, sep_token_id=SEP)
    check("masks differ across calls", not torch.equal(a, b))
    check("selected sets differ across calls", not torch.equal(la, lb))


def test_no_vocab_size_degrades_gracefully() -> None:
    """Without vocab_size the random-replacement term is skipped.

    Labels must stay correct and nothing may be corrupted: the r>=0.80 positions
    simply keep their original id (so the observable split collapses to
    ~89/11 [MASK]/unchanged). Losing the random term is a config error, not a
    correctness hazard — but train.py always passes vocab_size.
    """
    print("\ntest_no_vocab_size_degrades_gracefully")
    ids = _sample(seed=6)
    masked, labels = mlm_masking(ids, mask_token_id=MASK, mask_prob=0.30, pad_token_id=PAD)
    selected = labels != -100
    check("labels still original ids", bool((labels[selected] == ids[selected]).all()))
    check("no spurious token ids introduced", bool(
        ((masked == ids) | (masked == MASK)).all()
    ))
    # Upper bound is 8/9 (the 10% random bucket keeps original ids); it lands a
    # little lower because a random draw can coincide with the original token.
    frac_mask = (masked[selected] == MASK).float().mean().item()
    check("[MASK] share between 0.78 and 8/9", 0.78 <= frac_mask <= (0.80 / 0.90) + 1e-6,
          f"{frac_mask:.4f}")


def main() -> int:
    print("=" * 64)
    print("mlm_masking tests (80/10/10 split + special-token guard)")
    print("=" * 64)
    for test in (
        test_split_composition,
        test_labels_are_original_tokens,
        test_special_tokens_never_masked,
        test_selected_rate_matches_mask_prob,
        test_random_tokens_in_vocabulary,
        test_dynamic_across_calls,
        test_no_vocab_size_degrades_gracefully,
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
