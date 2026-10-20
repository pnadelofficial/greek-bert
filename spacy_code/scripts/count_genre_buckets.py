#!/usr/bin/env python3
"""
Count sentences/tokens per genre/period bucket in a combined CoNLL-U file
(assets/train.conllu, assets/dev.conllu, or assets/test.conllu -- the files
produced by combine_treebanks.sh).

Provenance is read straight from the CoNLL-U comment lines already present
in the source treebanks, not re-derived:
  - Perseus sentences carry "# newdoc id = tlg0008.tlg001...." ONCE per
    document (not per sentence) -- state is carried forward across all
    following sentences until the next "# newdoc id" line.
  - PROIEL sentences each carry their own "# source = <Work>, Book X, ..."
    comment line directly.

Bucket assignments below come from the Perseus/Open Greek and Latin CTS
catalog (github.com/PerseusDL/canonical-greekLit), verified work-by-work,
not guessed from memory -- see results/model-comparison.md discussion.
Herodotus's *Histories* appears in BOTH treebanks (Perseus tlg0016.tlg001
and PROIEL's "Histories") and is deliberately pooled into one bucket here.

Usage:
    python3 spacy_code/scripts/count_genre_buckets.py assets/test.conllu
    python3 spacy_code/scripts/count_genre_buckets.py assets/train.conllu assets/dev.conllu assets/test.conllu
"""
import re
import sys
from collections import Counter

PERSEUS_CODE_RE = re.compile(r"tlg\d{4}\.tlg\d{3}")

PERSEUS_BUCKETS = {
    "tlg0012.tlg001": "archaic_epic",
    "tlg0013.tlg002": "archaic_epic",
    "tlg0020.tlg001": "archaic_epic",
    "tlg0020.tlg002": "archaic_epic",
    "tlg0020.tlg003": "archaic_epic",
    "tlg0085.tlg001": "classical_tragedy",
    "tlg0011.tlg001": "classical_tragedy",
    "tlg0011.tlg002": "classical_tragedy",
    "tlg0011.tlg003": "classical_tragedy",
    "tlg0011.tlg004": "classical_tragedy",
    "tlg0011.tlg005": "classical_tragedy",
    "tlg0003.tlg001": "classical_historiography",
    "tlg0016.tlg001": "classical_historiography",
    "tlg0060.tlg001": "hellenistic_imperial_prose",
    "tlg0007.tlg004": "hellenistic_imperial_prose",
    "tlg0007.tlg015": "hellenistic_imperial_prose",
    "tlg0008.tlg001": "hellenistic_imperial_prose",
}

PROIEL_SOURCE_BUCKETS = {
    "Histories": "classical_historiography",  # == Perseus tlg0016.tlg001 (Herodotus)
    "The Greek New Testament": "koine_nt",
}


def classify_file(path):
    sent_counts = Counter()
    token_counts = Counter()
    raw_code_sent_counts = Counter()
    unrecognized = Counter()
    total_sents_seen = 0
    unmapped_sents = 0

    current_perseus_bucket = None
    current_sent_bucket = None
    current_sent_tokens = 0
    current_sent_has_tokens = False

    def flush_sentence():
        nonlocal current_sent_tokens, total_sents_seen, unmapped_sents, current_sent_has_tokens
        if current_sent_has_tokens:
            total_sents_seen += 1
            if current_sent_bucket is not None:
                sent_counts[current_sent_bucket] += 1
                token_counts[current_sent_bucket] += current_sent_tokens
            else:
                unmapped_sents += 1
        current_sent_tokens = 0
        current_sent_has_tokens = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# newdoc id"):
                flush_sentence()
                m = PERSEUS_CODE_RE.search(line)
                if m:
                    code = m.group(0)
                    current_perseus_bucket = PERSEUS_BUCKETS.get(code)
                    if current_perseus_bucket is None:
                        unrecognized[f"perseus:{code}"] += 1
                continue

            if line.startswith("# source"):
                # New sentence-level source comment -> flush whatever
                # sentence we were accumulating under the PREVIOUS bucket
                # before switching.
                flush_sentence()
                value = line.split("=", 1)[1].strip()
                value = value.split(",", 1)[0].strip()
                current_sent_bucket = PROIEL_SOURCE_BUCKETS.get(value)
                if current_sent_bucket is None:
                    unrecognized[f"proiel:{value}"] += 1
                raw_code_sent_counts[f"proiel:{value}"] += 1
                continue

            if line.startswith("# sent_id"):
                flush_sentence()
                # Perseus sentences don't repeat "# source"/"# newdoc id"
                # per sentence -- inherit the last-seen document bucket.
                if current_perseus_bucket is not None:
                    current_sent_bucket = current_perseus_bucket
                continue

            if line.startswith("#"):
                continue

            if line.strip() == "":
                continue

            # A real CoNLL-U token line. Skip multiword-token / empty-node
            # rows (IDs like "3-4" or "3.1") -- only count real tokens.
            first_field = line.split("\t", 1)[0]
            if "-" in first_field or "." in first_field:
                continue
            current_sent_tokens += 1
            current_sent_has_tokens = True

    flush_sentence()

    # Perseus raw-code sentence counts: re-derive by a second light pass
    # restricted to newdoc-id-triggered spans would double the complexity
    # for little benefit here -- the per-bucket totals above are what
    # matters for deciding bucket viability. Raw PROIEL codes are already
    # tracked above since they're per-sentence.
    return sent_counts, token_counts, raw_code_sent_counts, unrecognized, total_sents_seen, unmapped_sents


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    grand_sent = Counter()
    grand_tok = Counter()
    grand_unrecognized = Counter()
    grand_total = 0
    grand_unmapped = 0

    for path in sys.argv[1:]:
        sent_counts, token_counts, raw_proiel, unrecognized, total_sents, unmapped = classify_file(path)
        print(f"=== {path} ===")
        for bucket in sorted(sent_counts, key=lambda b: -sent_counts[b]):
            print(f"  {bucket:30s} sents={sent_counts[bucket]:6d}  tokens={token_counts[bucket]:8d}")
        if unrecognized:
            print("  -- unrecognized provenance (not bucketed, check mapping) --")
            for k, v in unrecognized.most_common():
                print(f"  {k:30s} sents={v}")
        if unmapped:
            print(f"  WARNING: {unmapped} sentence(s) had no provenance comment at all "
                  f"(e.g. appeared before any '# newdoc id'/'# source' line) and were not "
                  f"counted in any bucket.")
        print(f"  (total sentences seen: {total_sents}, bucketed: {total_sents - unmapped})")
        print()
        grand_sent.update(sent_counts)
        grand_tok.update(token_counts)
        grand_unrecognized.update(unrecognized)
        grand_total += total_sents
        grand_unmapped += unmapped

    if len(sys.argv) > 2:
        print("=== TOTAL across all files given ===")
        for bucket in sorted(grand_sent, key=lambda b: -grand_sent[b]):
            print(f"  {bucket:30s} sents={grand_sent[bucket]:6d}  tokens={grand_tok[bucket]:8d}")
        if grand_unrecognized:
            print("  -- unrecognized provenance (not bucketed, check mapping) --")
            for k, v in grand_unrecognized.most_common():
                print(f"  {k:30s} sents={v}")
        if grand_unmapped:
            print(f"  WARNING: {grand_unmapped} total sentence(s) with no provenance comment at all.")
        print(f"  (total sentences seen: {grand_total}, bucketed: {grand_total - grand_unmapped})")


if __name__ == "__main__":
    main()
