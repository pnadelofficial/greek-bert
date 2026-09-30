"""Build the training-set lemma lexicon for the odyCy-style frequency
lemmatizer (custom_components/lemmatizer.py::FrequencyLemmatizer).

Reads GOLD annotations straight out of the spaCy training corpus (NOT model
predictions -- this must run on train only, never dev/test, or the lexicon
leaks the evaluation signal the whole "lexicon vs neural" comparison depends
on) and produces two JSON files consumed by the @misc registry functions
added at the bottom of custom_components/lemmatizer.py, which
[initialize.components.frequency_lemmatizer] in
configs/gpu_default_freq_lemma.cfg points at:

  assets/lemmas/table.json   form -> [{lemma, upos, <morph features>, frequency}, ...]
  assets/lemmas/lookup.json  form -> single most frequent lemma (POS/morph-agnostic)

table.json backs FrequencyLemmatizer's primary lexicon-match layer; lookup.json
backs its stage-2 fallback (spaCy's plain lookup table, before the neural
edit-tree lemmatizer gets a turn). Both are gitignored (assets/ and corpus/
are generated artifacts) -- rerun this whenever corpus/train.spacy changes.

Run (from spacy_code/):
    uv run python build_lemma_lexicon.py
    uv run python build_lemma_lexicon.py --train corpus/train.spacy --out-dir assets/lemmas
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from spacy.training.corpus import Corpus


def build_lexicon(train_path: str, lang: str) -> tuple[dict, dict, int]:
    # A blank pipeline is enough: we only read GOLD Example.reference docs,
    # never run the model, so we just need a Vocab/tokenizer for the "grc"
    # lang to deserialize the DocBin (spaCy ships spacy.lang.grc natively).
    nlp = spacy.blank(lang)
    corpus = Corpus(train_path)

    entry_counts: dict[str, Counter] = defaultdict(Counter)
    lemma_counts: dict[str, Counter] = defaultdict(Counter)

    n_tokens = 0
    for example in corpus(nlp):
        for token in example.reference:
            if token.is_space or not token.lemma_:
                continue
            n_tokens += 1
            form = token.orth_.lower()
            lemma = token.lemma_
            upos = token.pos_
            # Only SET morph features land in the key (to_dict() omits unset
            # ones) -- this matches match_lemma()'s `.get(prop, "")` compare,
            # where a feature missing from BOTH sides counts as equal.
            morph_items = tuple(sorted(token.morph.to_dict().items()))
            entry_counts[form][(lemma, upos, morph_items)] += 1
            lemma_counts[form][lemma] += 1

    table = {}
    for form, counter in entry_counts.items():
        entries = []
        for (lemma, upos, morph_items), freq in counter.items():
            entry = {"form": form, "lemma": lemma, "upos": upos, "frequency": freq}
            entry.update(dict(morph_items))
            entries.append(entry)
        table[form] = entries

    lookup = {
        form: counter.most_common(1)[0][0] for form, counter in lemma_counts.items()
    }

    return table, lookup, n_tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--train",
        default="corpus/train.spacy",
        help="GOLD training corpus (.spacy DocBin). Train split only.",
    )
    p.add_argument(
        "--out-dir",
        default="assets/lemmas",
        help="Where to write table.json / lookup.json.",
    )
    p.add_argument("--lang", default="grc")
    args = p.parse_args()

    if not Path(args.train).is_file():
        raise SystemExit(
            f"error: no corpus at {args.train} -- build it first "
            f"(see spacy_code/scripts/combine_treebanks.sh / spacy convert)."
        )

    table, lookup, n_tokens = build_lexicon(args.train, args.lang)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table.json").write_text(json.dumps(table))
    (out_dir / "lookup.json").write_text(json.dumps(lookup))

    n_entries = sum(len(v) for v in table.values())
    print(f"Read {n_tokens} lemma-bearing tokens from {args.train}")
    print(f"table.json : {len(table)} distinct surface forms, {n_entries} (form, lemma, upos, morph) entries")
    print(f"lookup.json: {len(lookup)} surface forms -> single most-frequent lemma")
    print(f"Wrote {out_dir}/table.json and {out_dir}/lookup.json")


if __name__ == "__main__":
    main()
