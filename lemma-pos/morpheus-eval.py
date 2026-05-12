#!/usr/bin/env python3
"""
evaluate_morpheus.py
--------------------
Compare the fine-tuned Greek BERT morphological tagger against Perseus Morpheus
on a sample drawn from the treebank gold standard.

Rather than trying to recover provenance from the TSV (which has none), this
script goes back to the source XML files, samples ~N tokens with full metadata
(document_id, subdoc, word position), queries Morpheus for each, and compares
both lemma and POS against the treebank gold standard.

The Morpheus `doc_ref` parameter is constructed from the CTS URN and subdoc
that are stored in every treebank XML sentence, so each query is made in the
correct textual context — which matters because Morpheus uses prior-word and
document context to rank competing analyses.

On the CTS URN → Morpheus doc_ref mapping
------------------------------------------
Morpheus expects the old-style Perseus text identifier:
    Perseus:text:1999.01.0125:book=1:chapter=1:section=1
The CTS URNs in the treebank (urn:cts:greekLit:tlg0016.tlg001.perseus-grc1)
encode the same work via TLG number. We maintain a hand-built mapping for the
works present in the corpus. For unmapped works the script still queries
Morpheus but without a doc_ref (works fine, slightly less context for ranking).

Pedalion sentences use internal document IDs (e.g. "0019-001") with no
Perseus parallel, so they are excluded from the Morpheus comparison sample
but can still be evaluated against BERT alone.

Usage:
    python evaluate_morpheus.py \
        --xml-dir  treebanks/agdt treebanks/gorman \
        --model    ./greek-morph-model \
        --output   morpheus_eval.json \
        --n-tokens 500

    # Dry run: show sampled tokens and example URLs without hitting Morpheus
    python evaluate_morpheus.py --xml-dir treebanks/agdt --dry-run

    # Re-use cached Morpheus responses on a second run
    python evaluate_morpheus.py --xml-dir treebanks/agdt \
        --cache morpheus_cache.json --model ./greek-morph-model

Requirements:
    pip install requests beautifulsoup4 torch transformers
"""

import argparse
import json
import random
import re
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


# =============================================================================
# CTS URN → Perseus Morpheus doc_ref mapping
#
# Values are bare Perseus text IDs; the subdoc from the XML sentence is
# appended as a structured passage reference.
#
# To find the Perseus ID for any work: search https://www.perseus.tufts.edu,
# navigate to a passage, and read the `doc` parameter from the URL.
#
# Add entries here as you encounter unmapped works in the evaluation output —
# the script will report them so you know what to add.
# =============================================================================

CTS_TO_PERSEUS: dict[str, str] = {
    # Thucydides, History of the Peloponnesian War
    "tlg0003.tlg001.perseus-grc1": "Perseus:text:1999.01.0199",
    "tlg0003.tlg001.perseus-grc2": "Perseus:text:1999.01.0199",
    # Plato
    "tlg0059.tlg001.perseus-grc1": "Perseus:text:1999.01.0170",  # Apology
    "tlg0059.tlg005.perseus-grc1": "Perseus:text:1999.01.0172",  # Phaedo
    "tlg0059.tlg011.perseus-grc1": "Perseus:text:1999.01.0174",  # Republic
    # Sophocles
    "tlg0011.tlg001.perseus-grc2": "Perseus:text:1999.01.0185",  # Ajax
    "tlg0011.tlg002.perseus-grc2": "Perseus:text:1999.01.0187",  # Electra
    "tlg0011.tlg003.perseus-grc1": "Perseus:text:1999.01.0189",  # Oedipus Tyrannus
    "tlg0011.tlg004.perseus-grc1": "Perseus:text:1999.01.0191",  # Oedipus at Colonus
    "tlg0011.tlg005.perseus-grc2": "Perseus:text:1999.01.0193",  # Antigone
    # Aeschylus
    "tlg0085.tlg001.perseus-grc2": "Perseus:text:1999.01.0015",  # Agamemnon
    "tlg0085.tlg002.perseus-grc2": "Perseus:text:1999.01.0017",  # Libation Bearers
    "tlg0085.tlg003.perseus-grc2": "Perseus:text:1999.01.0019",  # Eumenides
    "tlg0085.tlg006.perseus-grc2": "Perseus:text:1999.01.0023",  # Persians
    # Herodotus, Histories
    "tlg0016.tlg001.perseus-grc1": "Perseus:text:1999.01.0125",
    # Hesiod
    "tlg0020.tlg001.perseus-grc1": "Perseus:text:1999.01.0129",  # Theogony
    "tlg0020.tlg003.perseus-grc1": "Perseus:text:1999.01.0131",  # Works and Days
    # Homer
    "tlg0012.tlg001.perseus-grc1": "Perseus:text:1999.01.0133",  # Iliad
    "tlg0012.tlg002.perseus-grc1": "Perseus:text:1999.01.0135",  # Odyssey
    # Demosthenes
    "tlg0014.tlg004.perseus-grc1": "Perseus:text:1999.01.0073",
    "tlg0014.tlg018.perseus-grc2": "Perseus:text:1999.01.0083",  # On the Crown
    "tlg0014.tlg059.perseus-grc2": "Perseus:text:1999.01.0107",  # Against Neaira
    # Aeschines
    "tlg0026.tlg001.perseus-grc1": "Perseus:text:1999.01.0001",
    # Lysias
    "tlg0540.tlg001.perseus-grc1": "Perseus:text:1999.01.0153",
    # Xenophon
    "tlg0032.tlg001.perseus-grc1": "Perseus:text:1999.01.0209",  # Hellenica
    "tlg0032.tlg002.perseus-grc1": "Perseus:text:1999.01.0211",  # Anabasis
    # Andocides
    "tlg0027.tlg001.perseus-grc1": "Perseus:text:1999.01.0025",
    # Antiphon
    "tlg0028.tlg001.perseus-grc1": "Perseus:text:1999.01.0027",
}


# =============================================================================
# Work-level labels for genre evaluation
#
# Keys are the short TLG work identifiers (tlg{author}.tlg{work}).
# Add any work you want to evaluate by name here — the label is used
# only for display in the report and the output JSON.
#
# To run a genre comparison, call the script three times with --work:
#   python evaluate_morpheus.py --work tlg0003.tlg001 ...  # Thucydides (prose)
#   python evaluate_morpheus.py --work tlg0011.tlg003 ...  # Sophocles OT (drama)
#   python evaluate_morpheus.py --work tlg0012.tlg001 ...  # Homer Iliad (epic)
# =============================================================================

WORK_LABELS: dict[str, str] = {
    # Prose
    "tlg0003.tlg001": "Thucydides, History",
    "tlg0016.tlg001": "Herodotus, Histories",
    "tlg0059.tlg001": "Plato, Apology",
    "tlg0059.tlg011": "Plato, Republic",
    "tlg0014.tlg018": "Demosthenes, On the Crown",
    "tlg0032.tlg001": "Xenophon, Hellenica",
    "tlg0032.tlg002": "Xenophon, Anabasis",
    "tlg0026.tlg001": "Aeschines, Against Timarchus",
    "tlg0540.tlg001": "Lysias, Orations",
    # Drama (tragedy)
    "tlg0011.tlg001": "Sophocles, Ajax",
    "tlg0011.tlg003": "Sophocles, Oedipus Tyrannus",
    "tlg0011.tlg004": "Sophocles, Oedipus at Colonus",
    "tlg0011.tlg005": "Sophocles, Antigone",
    "tlg0085.tlg001": "Aeschylus, Agamemnon",
    "tlg0085.tlg002": "Aeschylus, Libation Bearers",
    "tlg0085.tlg003": "Aeschylus, Eumenides",
    "tlg0085.tlg006": "Aeschylus, Persians",
    # Epic / poetry
    "tlg0012.tlg001": "Homer, Iliad",
    "tlg0012.tlg002": "Homer, Odyssey",
    "tlg0020.tlg001": "Hesiod, Theogony",
    "tlg0020.tlg003": "Hesiod, Works and Days",
}


def work_label(cts_urn: str) -> str:
    """Return a human-readable label for a CTS URN, or the raw TLG key."""
    m = re.search(r'(tlg\d+\.tlg\d+)', cts_urn)
    if not m:
        return cts_urn
    return WORK_LABELS.get(m.group(1), m.group(1))


def cts_matches_filter(cts_urn: str, work_filter) -> bool:
    """
    Return True if cts_urn contains work_filter as a substring, or if
    work_filter is None (no filtering). The filter is matched against
    the TLG portion only, so 'tlg0003.tlg001' matches both
    'tlg0003.tlg001.perseus-grc1' and 'tlg0003.tlg001.perseus-grc2'.
    """
    if work_filter is None:
        return True
    m = re.search(r'tlg\d+\.tlg\d+', cts_urn)
    return bool(m and work_filter in m.group(0))


def cts_to_doc_ref(cts_urn: str, subdoc: str) -> Optional[str]:
    """
    Convert a CTS URN + subdoc to a Morpheus doc_ref parameter string.

    Returns None if the work is not in CTS_TO_PERSEUS — the caller should
    still query Morpheus but without the d= parameter.
    """
    match = re.search(r'(tlg\d+\.tlg\d+\.[a-z\-]+\d*)', cts_urn)
    if not match:
        return None
    work_key = match.group(1)
    perseus_id = CTS_TO_PERSEUS.get(work_key)
    if not perseus_id:
        return None

    # subdoc "1.22.4" → ":book=1:chapter=22:section=4"
    if subdoc:
        parts = subdoc.split('.')
        keys  = ['book', 'chapter', 'section', 'line']
        passage = ':'.join(
            f'{keys[i]}={p}' for i, p in enumerate(parts) if i < len(keys)
        )
        return f"{perseus_id}:{passage}"

    return perseus_id


# =============================================================================
# Unicode Greek → Beta Code
# Ported directly from the notebook.
# =============================================================================

_COMBINING_TO_BETA: dict[str, str] = {
    "\u0313": ")",   # smooth breathing
    "\u0314": "(",   # rough breathing
    "\u0301": "/",   # acute
    "\u0300": "\\",  # grave
    "\u0342": "=",   # circumflex (perispomeni)
    "\u0308": "+",   # diaeresis
    "\u0345": "|",   # iota subscript
    "\u0304": "&",   # macron
    "\u0306": "'",   # breve
}

_GREEK_BASE_TO_BETA: dict[str, str] = {
    "\u03b1": "a", "\u03b2": "b", "\u03b3": "g", "\u03b4": "d",
    "\u03b5": "e", "\u03b6": "z", "\u03b7": "h", "\u03b8": "q",
    "\u03b9": "i", "\u03ba": "k", "\u03bb": "l", "\u03bc": "m",
    "\u03bd": "n", "\u03be": "c", "\u03bf": "o", "\u03c0": "p",
    "\u03c1": "r", "\u03c2": "s", "\u03c3": "s", "\u03c4": "t",
    "\u03c5": "u", "\u03c6": "f", "\u03c7": "x", "\u03c8": "y",
    "\u03c9": "w",
}


def greek_to_beta(word: str) -> str:
    """Convert Unicode Ancient Greek to Perseus Beta Code."""
    nfd = unicodedata.normalize('NFD', word.lower())
    parts = []
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        base_beta = _GREEK_BASE_TO_BETA.get(ch)
        if base_beta is None:
            if not unicodedata.category(ch).startswith('M') and ch.isascii():
                parts.append(ch)
            i += 1
            continue
        diacritics = []
        j = i + 1
        while j < len(nfd) and unicodedata.category(nfd[j]).startswith('M'):
            d = _COMBINING_TO_BETA.get(nfd[j])
            if d:
                diacritics.append(d)
            j += 1
        parts.append(base_beta + ''.join(diacritics))
        i = j
    return ''.join(parts)


# =============================================================================
# Morpheus HTTP client
# =============================================================================

MORPHEUS_BASE = "https://www.perseus.tufts.edu/hopper/morph"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.perseus.tufts.edu/",
}


class MorpheusClient:
    def __init__(
        self,
        rate_limit_secs: float = 1.0,
        timeout: float = 15.0,
        session: Optional[requests.Session] = None,
    ):
        self.rate_limit_secs = rate_limit_secs
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update(_BROWSER_HEADERS)
        self._last_request_time: float = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_secs:
            time.sleep(self.rate_limit_secs - elapsed)
        self._last_request_time = time.time()

    def query(
        self,
        word: str,
        prior_word: str = '',
        doc_ref: str = '',
        index: int = 1,
    ) -> Optional[str]:
        """Query Morpheus for one word. Returns raw HTML or None on failure."""
        self._throttle()
        beta_word = greek_to_beta(word)
        params = {
            'l':   beta_word,
            'la':  'greek',
            'can': f'{beta_word}{index}',
            'i':   str(index),
        }
        if prior_word:
            params['prior'] = greek_to_beta(prior_word)
        if doc_ref:
            params['d'] = doc_ref
        url = MORPHEUS_BASE + '?' + urllib.parse.urlencode(params)
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"    [Morpheus error] {word!r}: {e}")
            return None


# =============================================================================
# Morpheus HTML parser
# =============================================================================

# =============================================================================
# Morpheus form string → AGDT postag parser
#
# Morpheus winner rows contain a free-text morphological description in
# cell[1], e.g. "verb 3rd sg aor ind act" or "noun pl masc acc".
# We parse these into a full 9-character AGDT postag string so the
# comparison against the treebank gold standard is apples-to-apples.
#
# The first token of the form string determines POS; the remaining tokens
# are matched against feature vocabularies. Unknown/dialect tokens
# (attic, epic, ionic, doric, enclitic, proclitic, nu_movable, etc.)
# are silently ignored — they carry no AGDT postag information.
# =============================================================================

_MORPH_FIRST_TOKEN_TO_POS: dict[str, str] = {
    "noun":    "n",
    "verb":    "v",
    "part":    "v",   # participle is a verb form in AGDT (mood=p)
    "adj":     "a",
    "pron":    "p",
    "adv":     "d",
    "partic":  "g",   # particle (τε, δέ, γε …) — distinct from "part" (participle)
    "conj":    "c",
    "prep":    "r",
    "article": "l",
    "numeral": "m",
    "irreg":   "x",   # indeclinable irregular forms
}

_MORPH_PERSON:  dict[str, str] = {"1st": "1", "2nd": "2", "3rd": "3"}
_MORPH_NUMBER:  dict[str, str] = {"sg": "s", "pl": "p", "dual": "d"}
_MORPH_TENSE:   dict[str, str] = {
    "pres": "p", "imperf": "i", "aor": "a", "fut": "f",
    "perf": "r", "plup":   "l", "futperf": "t",
}
_MORPH_MOOD:    dict[str, str] = {
    "ind": "i", "subj": "s", "opt": "o", "imp": "m", "inf": "n",
    # "part" as a mood token is handled via the first-token rule below
}
_MORPH_VOICE:   dict[str, str] = {"act": "a", "mid": "m", "pass": "p", "mp": "e"}
_MORPH_GENDER:  dict[str, str] = {"masc": "m", "fem": "f", "neut": "n"}
_MORPH_CASE:    dict[str, str] = {"nom": "n", "gen": "g", "dat": "d", "acc": "a", "voc": "v"}
_MORPH_DEGREE:  dict[str, str] = {
    "comp": "c", "superl": "s",
    "irreg_comp": "c", "irreg_superl": "s", "comp_only": "c",
}


def _parse_form_str(form_str: str) -> tuple[str, str]:
    """
    Parse a Morpheus free-text morphological description into
    (pos_char, postag_9char) where postag_9char is a full AGDT-style
    9-character string with '-' for unfilled positions.

    Examples:
        "verb 3rd sg aor ind act"          -> ("v", "v3saia---")
        "noun pl masc acc"                 -> ("n", "n-p---ma-")
        "part sg pres act masc nom"        -> ("v", "v-spamn-")
        "partic enclitic indeclform"       -> ("g", "g--------")
        "article sg masc nom indeclform"   -> ("l", "l-s---mn-")
    """
    if not form_str:
        return "-", "---------"

    tokens = form_str.lower().split()
    token_set = set(tokens)
    first = tokens[0] if tokens else ""

    pos = _MORPH_FIRST_TOKEN_TO_POS.get(first, "x")

    # Mood: participles set mood="p" via the first token;
    # other verb forms read mood from the token set.
    if first == "part":
        mood = "p"
    else:
        mood = next((v for t, v in _MORPH_MOOD.items() if t in token_set), "-")

    person = next((v for t, v in _MORPH_PERSON.items()  if t in token_set), "-")
    number = next((v for t, v in _MORPH_NUMBER.items()  if t in token_set), "-")
    tense  = next((v for t, v in _MORPH_TENSE.items()   if t in token_set), "-")
    voice  = next((v for t, v in _MORPH_VOICE.items()   if t in token_set), "-")
    gender = next((v for t, v in _MORPH_GENDER.items()  if t in token_set), "-")
    case   = next((v for t, v in _MORPH_CASE.items()    if t in token_set), "-")
    degree = next((v for t, v in _MORPH_DEGREE.items()  if t in token_set), "-")

    postag = pos + person + number + tense + mood + voice + gender + case + degree
    return pos, postag


@dataclass
class MorpheusResult:
    lemma:    Optional[str] = None
    form_str: Optional[str] = None   # raw Morpheus morphological description
    postag:   Optional[str] = None   # parsed 9-char AGDT-style postag
    pos_char: Optional[str] = None   # postag[0], convenience field
    error:    Optional[str] = None

    @property
    def found(self) -> bool:
        return self.lemma is not None


def parse_morpheus_html(html: str) -> MorpheusResult:
    """
    Extract the winner lemma and morphological features from a Morpheus
    HTML response page.

    Lemma source
    ------------
    The winner <tr class="winner"> is always inside a <div class="lemma">
    whose <h4> holds the dictionary headword. We climb from the winner row
    to its parent lemma div and read the h4 — this is more reliable than
    reading cell[0] of the winner row, which contains the inflected form
    (with a † dagger) rather than the lemma.

    Postag
    ------
    Cell[1] of the winner row contains the free-text morphological
    description (e.g. "verb 3rd sg aor ind act"). We parse this into a
    full 9-character AGDT postag via _parse_form_str().
    """
    if not html:
        return MorpheusResult(error="no html")

    soup = BeautifulSoup(html, "html.parser")
    winner = soup.find("tr", class_="winner")
    if winner is None:
        return MorpheusResult(error="no winner")

    cells = winner.find_all("td")
    if len(cells) < 2:
        return MorpheusResult(error="malformed row")

    # ── Lemma: climb to the enclosing lemma div and read its h4 ──────────
    lemma_div = winner.find_parent("div", class_="lemma")
    if lemma_div:
        h4 = lemma_div.find("h4")
        lemma = h4.get_text(strip=True) if h4 else None
    else:
        # Fallback: strip the dagger from cell[0] (inflected form, not ideal)
        lemma = re.sub(r"[†*\d]", "", cells[0].get_text(strip=True)).strip() or None

    # ── Postag: parse the free-text morphological description ────────────
    form_str = cells[1].get_text(strip=True)
    pos_char, postag = _parse_form_str(form_str)

    return MorpheusResult(
        lemma=lemma,
        form_str=form_str,
        postag=postag,
        pos_char=pos_char,
    )


# =============================================================================
# Gold token sampling from XML
# =============================================================================

@dataclass
class GoldToken:
    form:        str
    lemma:       str
    postag:      str
    pos_char:    str
    prior_form:  str
    doc_ref:     str        # Morpheus d= value (may be empty)
    source_file: str
    sent_id:     str
    word_id:     str


def extract_gold_sample(
    xml_dirs: list[Path],
    n_tokens: int = 500,
    seed: int = 42,
    exclude_pos: str = 'u',
    work_filter: Optional[str] = None,
) -> list[GoldToken]:
    """
    Sample n_tokens gold tokens from XML files, preserving sentence context
    (so we have prior_form for each token) and full provenance for doc_ref.

    Only draws from sentences where a CTS→Perseus mapping exists.
    Unmapped works are reported so you can extend CTS_TO_PERSEUS.

    work_filter : optional TLG work identifier string, e.g. 'tlg0003.tlg001'.
        When set, only sentences from that work are included. Both version
        suffixes (perseus-grc1, perseus-grc2) are matched automatically.
        Use --list-works to see available identifiers.
    """
    rng = random.Random(seed)
    candidates = []
    unmapped_works: set[str] = set()

    for xml_dir in xml_dirs:
        for xml_file in sorted(Path(xml_dir).glob('*.xml')):
            try:
                tree = ET.parse(xml_file)
            except ET.ParseError:
                continue

            for sent in tree.iter('sentence'):
                doc_id = sent.get('document_id', '')
                subdoc = sent.get('subdoc', '')
                doc_ref = cts_to_doc_ref(doc_id, subdoc)

                # Track unmapped works for reporting
                if doc_ref is None and 'tlg' in doc_id:
                    m = re.search(r'tlg\d+\.tlg\d+\.[a-z\-]+\d*', doc_id)
                    if m:
                        unmapped_works.add(m.group(0))

                # Work-level filter: skip if this sentence is not from the
                # requested work. Checked before the token loop for efficiency.
                if not cts_matches_filter(doc_id, work_filter):
                    continue

                # We require a mapping — skip Pedalion etc.
                if not doc_ref:
                    continue

                words = []
                for w in sent.iter('word'):
                    if w.get('artificial'):
                        continue
                    postag = w.get('postag', '')
                    form   = w.get('form', '').strip()
                    lemma  = w.get('lemma', '').strip()
                    if not form or not lemma or not postag:
                        continue
                    if all(c == '-' for c in postag) or postag[0] == 'z':
                        continue
                    if postag[0] in exclude_pos:
                        continue
                    if lemma in ('_', 'punc1'):
                        continue
                    words.append((w.get('id', ''), form, lemma, postag))

                if len(words) >= 2:
                    candidates.append((
                        words, doc_ref, xml_file.name, sent.get('id', '')
                    ))

    if unmapped_works:
        print(f"\nUnmapped works (add to CTS_TO_PERSEUS to include them):")
        for w in sorted(unmapped_works):
            print(f"  {w}")

    if work_filter:
        label = WORK_LABELS.get(work_filter, work_filter)
        print(f"\nWork filter: {work_filter}  ({label})")
    print(f"Found {len(candidates):,} candidate sentences with Morpheus mapping")
    rng.shuffle(candidates)

    tokens: list[GoldToken] = []
    for words, doc_ref, source_file, sent_id in candidates:
        if len(tokens) >= n_tokens:
            break
        for i, (word_id, form, lemma, postag) in enumerate(words):
            prior_form = words[i - 1][1] if i > 0 else ''
            tokens.append(GoldToken(
                form=form, lemma=lemma, postag=postag, pos_char=postag[0],
                prior_form=prior_form, doc_ref=doc_ref,
                source_file=source_file, sent_id=sent_id, word_id=word_id,
            ))
            if len(tokens) >= n_tokens:
                break

    sources = len(set(t.source_file for t in tokens))
    work_desc = f"  (work: {work_filter})" if work_filter else ""
    print(f"Sampled {len(tokens):,} tokens from {sources} files{work_desc}")
    return tokens


# =============================================================================
# BERT inference (deferred import — works without torch in --dry-run mode)
# =============================================================================

def load_bert_predict_fn(model_dir: str):
    """
    Load the fine-tuned GreekMorphTagger and return a predict function.
    Mirrors the inference logic in train_morph_tagger.py.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    POSTAG_VOCABS = [
        list('-nvadclrpgmuixbez'),
        list('-123'),
        list('-sdp'),
        list(dict.fromkeys('-pirltfa')),
        list('-isonmpd'),
        list('-apme'),
        list('-mfn'),
        list('-ngdavl'),
        list('-pcs'),
    ]
    POSTAG_SIZES = [len(v) for v in POSTAG_VOCABS]

    def apply_edit(form, edit):
        try:
            strip_s, append = edit.split(':', 1)
            strip = int(strip_s)
            return (form[:-strip] if strip > 0 else form) + append
        except Exception:
            return form

    class _Tagger(nn.Module):
        def __init__(self, bert_path, edit_vocab_size):
            super().__init__()
            self.bert = AutoModel.from_pretrained(bert_path)
            h = self.bert.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.postag_heads = nn.ModuleList([
                nn.Linear(h, POSTAG_SIZES[i]) for i in range(9)
            ])
            self.edit_head = nn.Linear(h, edit_vocab_size)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq = self.dropout(out.last_hidden_state)
            return (
                [head(seq) for head in self.postag_heads],
                self.edit_head(seq),
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f'{model_dir}/edit_vocab.json', encoding='utf-8') as f:
        edit_vocab = json.load(f)
    idx2edit = {v: k for k, v in edit_vocab.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    heads = torch.load(f'{model_dir}/morph_heads.pt', map_location=device, weights_only=False)

    model = _Tagger(model_dir, len(edit_vocab)).to(device)
    model.postag_heads.load_state_dict(heads['postag_heads'])
    model.edit_head.load_state_dict(heads['edit_head'])
    model.eval()

    def predict(sentences: list[list[str]]) -> list[list[dict]]:
        results = []
        for words in sentences:
            enc = tokenizer(
                words, is_split_into_words=True, return_tensors='pt',
                truncation=True, max_length=256,
            ).to(device)
            word_ids = enc.word_ids(batch_index=0)
            with torch.no_grad():
                postag_logits, edit_logits = model(
                    enc['input_ids'], enc['attention_mask']
                )
            pred_postag = [lg.argmax(-1)[0] for lg in postag_logits]
            pred_edit   = edit_logits.argmax(-1)[0]
            sent_out, prev = [], None
            for pos, wid in enumerate(word_ids):
                if wid is None or wid == prev:
                    continue
                prev = wid
                postag = ''.join(
                    POSTAG_VOCABS[i][pred_postag[i][pos].item()] for i in range(9)
                )
                edit  = idx2edit.get(pred_edit[pos].item(), '<UNK>')
                lemma = apply_edit(words[wid], edit) if edit != '<UNK>' else words[wid]
                sent_out.append({'form': words[wid], 'postag': postag, 'lemma': lemma})
            results.append(sent_out)
        return results

    return predict


# =============================================================================
# Evaluation metrics
# =============================================================================

POS_NAMES = {
    'n': 'noun', 'v': 'verb', 'a': 'adjective', 'p': 'pronoun',
    'd': 'adverb', 'c': 'conjunction', 'r': 'preposition',
    'i': 'interjection', 'm': 'numeral', 'g': 'particle', 'l': 'article',
    'x': 'irregular', 'b': 'particle(b)',
}


def normalize_lemma(lemma: str) -> str:
    """Lowercase, strip editorial marks, NFC normalize."""
    lemma = unicodedata.normalize('NFC', lemma)
    return re.sub(r'[†*\d\s]', '', lemma).lower().strip()


def compute_metrics(results: list[dict], system: str) -> dict:
    """
    Compute accuracy metrics for one system (morpheus or bert).
    system: 'morpheus' or 'bert' — selects which keys to read from results.
    """
    total         = len(results)
    found         = sum(1 for r in results if r[f'{system}_found'])
    lemma_correct = sum(1 for r in results if r[f'{system}_lemma_match'])
    pos_correct   = sum(1 for r in results if r[f'{system}_pos_match'])
    both_correct  = sum(
        1 for r in results
        if r[f'{system}_lemma_match'] and r[f'{system}_pos_match']
    )
    postag_correct = sum(1 for r in results if r.get(f'{system}_postag_match'))

    by_pos = defaultdict(lambda: {'total': 0, 'lemma': 0, 'pos': 0})
    for r in results:
        if not r[f'{system}_found']:
            continue
        p = r['gold_pos']
        by_pos[p]['total'] += 1
        if r[f'{system}_lemma_match']:
            by_pos[p]['lemma'] += 1
        if r[f'{system}_pos_match']:
            by_pos[p]['pos'] += 1

    return {
        'n_tokens':    total,
        'n_found':     found,
        'coverage':    found / total if total else 0.0,
        'lemma_acc':   lemma_correct  / total if total else 0.0,
        'pos_acc':     pos_correct    / total if total else 0.0,
        'postag_acc':  postag_correct / total if total else 0.0,
        'both_acc':    both_correct   / total if total else 0.0,
        'by_pos':      dict(by_pos),
    }


def print_report(bert: dict, morpheus: dict, work_filter: Optional[str] = None):
    n = bert['n_tokens']
    print()
    print('=' * 65)
    print('  EVALUATION REPORT: Greek BERT vs Perseus Morpheus')
    print('=' * 65)
    if work_filter:
        label = WORK_LABELS.get(work_filter, work_filter)
        print(f'  Work              : {label}  [{work_filter}]')
    print(f'  Gold tokens       : {n}')
    print(f'  Morpheus coverage : {morpheus["coverage"]:.1%}  '
          f'({n - morpheus["n_found"]} no-analysis)')
    if bert['n_found'] > 0:
        print(f'  BERT coverage     : {bert["coverage"]:.1%}')
    print()
    print(f'  {"Metric":<28}  {"BERT":>8}  {"Morpheus":>8}  {"Δ":>8}')
    print(f'  {"-"*56}')
    for label, key in [
        ('Lemma accuracy',            'lemma_acc'),
        ('POS accuracy (coarse)',     'pos_acc'),
        ('Full postag accuracy',      'postag_acc'),
        ('Lemma + POS accuracy',      'both_acc'),
    ]:
        bv = bert.get(key, 0.0)
        mv = morpheus.get(key, 0.0)
        d  = bv - mv
        print(f'  {label:<28}  {bv:>7.1%}  {mv:>7.1%}  '
              f'{"+" if d >= 0 else ""}{d:>6.1%}')

    print()
    print('  Per-POS breakdown — lemma accuracy (Morpheus | BERT):')
    print(f'  {"POS":<14}  {"N":>5}  {"Morpheus":>8}  {"BERT":>8}')
    print(f'  {"-"*42}')
    all_pos = sorted(set(bert['by_pos']) | set(morpheus['by_pos']))
    for pos in all_pos:
        bd = bert['by_pos'].get(pos, {})
        md = morpheus['by_pos'].get(pos, {})
        n_shown = max(bd.get('total', 0), md.get('total', 0))
        if n_shown == 0:
            continue
        bl = bd['lemma'] / bd['total'] if bd.get('total') else 0.0
        ml = md['lemma'] / md['total'] if md.get('total') else 0.0
        name = POS_NAMES.get(pos, pos)
        print(f'  {name:<14}  {n_shown:>5}  {ml:>7.1%}  {bl:>7.1%}')
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Greek BERT tagger vs Perseus Morpheus on gold XML data.'
    )
    parser.add_argument('--xml-dir', nargs='+', type=Path, required=True,
                        help='Directories of downloaded treebank XML files.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to fine-tuned model dir (optional).')
    parser.add_argument('--output', type=Path, default=Path('morpheus_eval.json'))
    parser.add_argument('--n-tokens', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rate-limit', type=float, default=1.0,
                        help='Seconds between Morpheus requests.')
    parser.add_argument('--cache', type=Path, default=None,
                        help='JSON file to cache/restore Morpheus HTML responses.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Sample tokens and show URLs; do not query Morpheus.')
    parser.add_argument(
        '--work', type=str, default=None, metavar='TLG_ID',
        help=(
            'Restrict evaluation to a single work, identified by its TLG '
            'work key, e.g. tlg0003.tlg001 (Thucydides). Both version '
            'suffixes (perseus-grc1/grc2) are matched automatically. '
            'Use --list-works to see available identifiers.'
        ),
    )
    parser.add_argument(
        '--list-works', action='store_true',
        help='Print available work identifiers and exit.',
    )
    args = parser.parse_args()

    if args.list_works:
        print("\nAvailable work identifiers (pass to --work):\n")
        # Group by genre based on position in WORK_LABELS
        prose  = [k for k in WORK_LABELS if k.startswith(('tlg0003','tlg0016','tlg0059','tlg0014','tlg0032','tlg0026','tlg0540','tlg0027','tlg0028','tlg0010','tlg0017','tlg0008'))]
        drama  = [k for k in WORK_LABELS if k.startswith(('tlg0011','tlg0085'))]
        poetry = [k for k in WORK_LABELS if k.startswith(('tlg0012','tlg0020'))]
        for heading, group in [('Prose', prose), ('Drama', drama), ('Epic / Poetry', poetry)]:
            print(f"  {heading}:")
            for k in group:
                print(f"    {k:30s}  {WORK_LABELS[k]}")
            print()
        return

    # -------------------------------------------------------------------------
    # Sample gold tokens
    # -------------------------------------------------------------------------
    tokens = extract_gold_sample(args.xml_dir, args.n_tokens, args.seed, work_filter=args.work)
    if not tokens:
        print("[ERROR] No tokens sampled. Check --xml-dir and CTS_TO_PERSEUS mapping.")
        return

    if args.dry_run:
        print(f"\nFirst 10 sampled tokens:")
        for tok in tokens[:10]:
            print(f"  {tok.form:20s}  lemma={tok.lemma:20s}  "
                  f"postag={tok.postag}  ref={tok.doc_ref[:55]}")
        t = tokens[0]
        beta = greek_to_beta(t.form)
        example_url = (f"{MORPHEUS_BASE}?l={beta}&la=greek"
                       f"&d={urllib.parse.quote(t.doc_ref)}&i=1")
        print(f"\nExample Morpheus URL:\n  {example_url}")
        return

    # -------------------------------------------------------------------------
    # Load Morpheus cache
    # -------------------------------------------------------------------------
    morph_cache: dict[str, str] = {}
    if args.cache and args.cache.exists():
        morph_cache = json.loads(args.cache.read_text(encoding='utf-8'))
        print(f"Loaded {len(morph_cache)} cached Morpheus responses.")

    # -------------------------------------------------------------------------
    # BERT inference
    # -------------------------------------------------------------------------
    bert_preds: dict[tuple, dict] = {}
    if args.model:
        print(f"\nRunning BERT inference...")
        predict_fn = load_bert_predict_fn(args.model)

        # Group tokens by sentence to give BERT full sentential context
        sent_groups: dict[tuple, list] = OrderedDict()
        for tok in tokens:
            sent_groups.setdefault((tok.source_file, tok.sent_id), []).append(tok)

        for (src, sid), sent_toks in sent_groups.items():
            words = [t.form for t in sent_toks]
            preds = predict_fn([words])[0]
            for tok, pred in zip(sent_toks, preds):
                bert_preds[(tok.source_file, tok.sent_id, tok.word_id)] = pred
        print(f"BERT inference complete ({len(bert_preds)} predictions).")
    else:
        print("\nNo --model provided; skipping BERT inference.")

    # -------------------------------------------------------------------------
    # Query Morpheus
    # -------------------------------------------------------------------------
    client = MorpheusClient(rate_limit_secs=args.rate_limit)
    eta_min = len(tokens) * args.rate_limit / 60
    print(f"\nQuerying Morpheus for {len(tokens)} tokens (~{eta_min:.1f} min)...")

    results = []
    for i, tok in enumerate(tokens):
        cache_key = f"{tok.form}|{tok.doc_ref}"
        if cache_key in morph_cache:
            raw_html = morph_cache[cache_key]
        else:
            raw_html = client.query(
                word=tok.form,
                prior_word=tok.prior_form,
                doc_ref=tok.doc_ref,
                index=int(tok.word_id) if tok.word_id.isdigit() else i + 1,
            )
            morph_cache[cache_key] = raw_html or ''

        morph = parse_morpheus_html(raw_html) if raw_html else MorpheusResult(error='no response')

        gold_lemma_n = normalize_lemma(tok.lemma)
        morph_lemma_n = normalize_lemma(morph.lemma) if morph.lemma else ''

        bert_key  = (tok.source_file, tok.sent_id, tok.word_id)
        bert_pred = bert_preds.get(bert_key)
        bert_lemma_n = normalize_lemma(bert_pred['lemma']) if bert_pred else ''

        # Full postag comparison (all 9 positions) — only meaningful where
        # Morpheus/BERT coverage exists. We also report pos-only (position 0)
        # as a coarser metric that is less sensitive to feature parsing gaps.
        morph_postag_match = (
            morph.found and morph.postag is not None
            and morph.postag == tok.postag
        )
        bert_postag = bert_pred['postag'] if bert_pred else None
        bert_postag_match = bool(bert_pred) and bert_postag == tok.postag

        results.append({
            # Gold
            'form':        tok.form,
            'gold_lemma':  tok.lemma,
            'gold_postag': tok.postag,
            'gold_pos':    tok.pos_char,
            'source_file': tok.source_file,
            'doc_ref':     tok.doc_ref,
            # Morpheus
            'morpheus_found':        morph.found,
            'morpheus_lemma':        morph.lemma,
            'morpheus_form_str':     morph.form_str,
            'morpheus_postag':       morph.postag,
            'morpheus_pos':          morph.pos_char,
            'morpheus_lemma_match':  morph.found and morph_lemma_n == gold_lemma_n,
            'morpheus_pos_match':    morph.found and morph.pos_char == tok.pos_char,
            'morpheus_postag_match': morph_postag_match,
            # BERT
            'bert_found':        bert_pred is not None,
            'bert_lemma':        bert_pred['lemma'] if bert_pred else None,
            'bert_postag':       bert_postag,
            'bert_lemma_match':  bool(bert_pred) and bert_lemma_n == gold_lemma_n,
            'bert_pos_match':    bool(bert_pred) and bert_postag is not None and bert_postag[0] == tok.pos_char,
            'bert_postag_match': bert_postag_match,
        })

        if (i + 1) % 50 == 0:
            n_found = sum(1 for r in results if r['morpheus_found'])
            print(f"  {i+1}/{len(tokens)}  Morpheus found: {n_found}/{i+1}")

    # Save cache after all queries
    if args.cache:
        args.cache.write_text(
            json.dumps(morph_cache, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f"Morpheus cache saved to {args.cache}")

    # -------------------------------------------------------------------------
    # Metrics + report
    # -------------------------------------------------------------------------
    morpheus_metrics = compute_metrics(results, 'morpheus')
    bert_metrics     = compute_metrics(results, 'bert')

    print_report(bert_metrics, morpheus_metrics, work_filter=args.work)

    # -------------------------------------------------------------------------
    # Save full results
    # -------------------------------------------------------------------------
    args.output.write_text(
        json.dumps({
            'metadata': {
                'n_tokens': len(tokens), 'seed': args.seed,
                'model': args.model,
                'xml_dirs': [str(d) for d in args.xml_dir],
                'work': args.work,
                'work_label': WORK_LABELS.get(args.work, args.work) if args.work else None,
            },
            'morpheus_metrics': morpheus_metrics,
            'bert_metrics':     bert_metrics,
            'tokens':           results,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"Detailed results → {args.output}")


if __name__ == '__main__':
    main()