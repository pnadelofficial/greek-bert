import argparse
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

EMPTY_POSTAG = '---------'

def extract_sentences(path: Path) -> tuple[list, list]:
    """
    Parse one XML file and return (sentences, warnings).
 
    Each sentence is a list of (form, lemma, postag) triples.
    Tokens are skipped (not the whole sentence) if:
      - marked artificial/elliptic
      - missing form, lemma, or postag
      - postag is all dashes (uninformative)
      - lemma is the punctuation placeholder 'punc1'
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return [], [f"{path.name}: XML parse error — {e}"]
 
    sentences = []
    warnings = []
    skipped_tokens = 0
 
    for sent in tree.iter('sentence'):
        tokens = []
        for word in sent.iter('word'):
            # Skip artificial/elliptic nodes
            if word.get('artificial'):
                continue
 
            form   = word.get('form',   '').strip()
            lemma  = word.get('lemma',  '').strip()
            postag = word.get('postag', '').strip()
 
            # Skip if any core field is missing
            if not form or not lemma or not postag:
                skipped_tokens += 1
                continue
 
            # Skip uninformative postags (all dashes, or z = Pedalion unknown)
            if postag == EMPTY_POSTAG or all(c == '-' for c in postag) or postag[0] == 'z':
                skipped_tokens += 1
                continue
 
            # Skip bare underscore lemmas (Pedalion unresolved tokens)
            if lemma == '_':
                skipped_tokens += 1
                continue
 
            # Normalize punctuation placeholder lemma to the form itself
            if lemma == 'punc1':
                lemma = form
 
            tokens.append((form, lemma, postag))
 
        if tokens:
            sentences.append(tokens)
 
    if skipped_tokens:
        warnings.append(f"{path.name}: skipped {skipped_tokens} tokens (missing fields or uninformative postag)")
 
    return sentences, warnings

def compute_stats(all_sentences: list) -> dict:
    total_tokens = sum(len(s) for s in all_sentences)
    postag_counts = defaultdict(int)
    pos_counts    = defaultdict(int)
    lemma_counts  = defaultdict(int)
 
    for sent in all_sentences:
        for form, lemma, postag in sent:
            postag_counts[postag] += 1
            pos_counts[postag[0]] += 1
            lemma_counts[lemma]   += 1
 
    return {
        'sentences':    len(all_sentences),
        'tokens':       total_tokens,
        'unique_postags': len(postag_counts),
        'unique_lemmas':  len(lemma_counts),
        'pos_counts':     dict(pos_counts),
        'top_postags':    sorted(postag_counts.items(), key=lambda x: -x[1])[:20],
    }
 
POS_NAMES = {
    'n': 'noun', 'v': 'verb', 'a': 'adjective', 'p': 'pronoun',
    'd': 'adverb', 'c': 'conjunction', 'r': 'preposition', 'i': 'interjection',
    'm': 'numeral', 'u': 'punctuation', 'g': 'particle', 'l': 'article',
    'x': 'irregular', '-': 'unknown',
}
 
def print_stats(stats: dict):
    print(f"\n{'='*55}")
    print(f"  Corpus statistics")
    print(f"{'='*55}")
    print(f"  Sentences     : {stats['sentences']:>10,}")
    print(f"  Tokens        : {stats['tokens']:>10,}")
    print(f"  Unique postags: {stats['unique_postags']:>10,}")
    print(f"  Unique lemmas : {stats['unique_lemmas']:>10,}")
    print(f"\n  POS distribution:")
    for char, count in sorted(stats['pos_counts'].items(), key=lambda x: -x[1]):
        name = POS_NAMES.get(char, '?')
        pct  = 100 * count / stats['tokens']
        bar  = '█' * int(pct / 2)
        print(f"    {char} {name:14s} {count:>8,}  {pct:5.1f}%  {bar}")
    print(f"\n  Top 20 postags:")
    for tag, count in stats['top_postags']:
        print(f"    {tag}  {count:>8,}")


def write_tsv(sentences: list, path: Path):
    lines = []
    for sent in sentences:
        for form, lemma, postag in sent:
            lines.append(f"{form}\t{lemma}\t{postag}")
        lines.append('')   # blank line between sentences
    path.write_text('\n'.join(lines), encoding='utf-8')


def split_sentences(sentences: list, ratios: list[float], seed: int = 42
                    ) -> tuple[list, list, list]:
    """
    Shuffle and split sentences into train/dev/test.
    ratios must sum to 1.0, e.g. [0.8, 0.1, 0.1]
    """
    assert len(ratios) == 3, "Need exactly three ratios: train dev test"
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
 
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
 
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_dev   = int(n * ratios[1])
 
    train = shuffled[:n_train]
    dev   = shuffled[n_train:n_train + n_dev]
    test  = shuffled[n_train + n_dev:]
    return train, dev, test


def main():
    parser = argparse.ArgumentParser(
        description='Convert AGDT XML treebanks to TSV (form, lemma, postag).'
    )
    parser.add_argument(
        'inputs', nargs='+', type=Path,
        help='Input XML files. Use shell glob: treebanks/*/*.xml'
    )
    parser.add_argument(
        '-o', '--output', type=Path, required=True,
        help='Output TSV file (or base name if --split is used)'
    )
    parser.add_argument(
        '--split', nargs=3, type=float, metavar=('TRAIN', 'DEV', 'TEST'),
        help='Split into train/dev/test with given ratios, e.g. --split 0.8 0.1 0.1'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for shuffling when splitting (default: 42)'
    )
    parser.add_argument(
        '--stats', action='store_true',
        help='Print corpus statistics after conversion'
    )
    parser.add_argument(
        '--skip-punct', action='store_true',
        help='Exclude punctuation tokens (postag starting with u) from output'
    )
    args = parser.parse_args()
 
    # Collect and parse all files
    all_sentences = []
    all_warnings  = []
    file_count    = 0
 
    for path in sorted(args.inputs):
        if not path.exists():
            print(f"[WARN] Not found: {path}")
            continue
        sentences, warnings = extract_sentences(path)
        all_warnings.extend(warnings)
        all_sentences.extend(sentences)
        file_count += 1
 
    print(f"Parsed {file_count} files → {len(all_sentences):,} sentences")
 
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  {w}")
 
    if not all_sentences:
        print("[ERROR] No sentences extracted. Check your input paths.")
        return
 
    # Optional: drop punctuation
    if args.skip_punct:
        all_sentences = [
            [(f, l, p) for f, l, p in sent if p[0] != 'u']
            for sent in all_sentences
        ]
        all_sentences = [s for s in all_sentences if s]
        print(f"After removing punctuation: {len(all_sentences):,} sentences")
 
    # Stats
    if args.stats:
        print_stats(compute_stats(all_sentences))
 
    # Write output
    if args.split:
        train, dev, test = split_sentences(all_sentences, args.split, args.seed)
        stem   = args.output.with_suffix('')
        suffix = args.output.suffix or '.tsv'
 
        train_path = Path(f"{stem}.train{suffix}")
        dev_path   = Path(f"{stem}.dev{suffix}")
        test_path  = Path(f"{stem}.test{suffix}")
 
        write_tsv(train, train_path)
        write_tsv(dev,   dev_path)
        write_tsv(test,  test_path)
 
        print(f"\nWrote split files:")
        print(f"  train: {len(train):>6,} sentences → {train_path}")
        print(f"  dev  : {len(dev):>6,} sentences → {dev_path}")
        print(f"  test : {len(test):>6,} sentences → {test_path}")
    else:
        write_tsv(all_sentences, args.output)
        print(f"\nWrote {len(all_sentences):,} sentences → {args.output}")
 
 
if __name__ == '__main__':
    main()
