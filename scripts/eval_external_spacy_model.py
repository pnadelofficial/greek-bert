#!/usr/bin/env python3
"""Evaluate an installed spaCy pipeline against one of THIS PROJECT's own
corpus/*.spacy test sets, for a genuinely controlled comparison against our
own models' numbers (same held-out documents, same Scorer, same metrics).

Deliberately dependency-light: only needs `spacy`, nothing from this
project's own packages. Meant to run inside whatever separate, isolated venv
has the EXTERNAL pipeline installed (e.g. odyCy, pip-installed from its own
wheel) -- not this project's root or spacy_code venvs.

Builds predicted Docs from the GOLD tokenization (not raw text run through
the external pipeline's own tokenizer), same reasoning as the Pindar eval:
avoids alignment mismatches if the external pipeline would segment text
differently, and matches how corpus/test.spacy itself was scored for every
one of this project's own models.

Usage:
    python eval_external_spacy_model.py \
        --pipeline grc_odycy_joint_trf \
        --test-spacy /path/to/corpus/test.spacy \
        --output results/odycy_eval_results.json
"""

from __future__ import annotations

import argparse
import json

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example

METRICS = (
    "tag_acc", "pos_acc", "morph_acc", "lemma_acc",
    "dep_uas", "dep_las", "sents_p", "sents_r", "sents_f",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pipeline", required=True,
                   help="Installed spaCy pipeline package name, e.g. grc_odycy_joint_trf")
    p.add_argument("--test-spacy", required=True,
                   help="Path to the gold corpus/test.spacy DocBin to score against")
    p.add_argument("--output", required=True,
                   help="Where to write the eval_results.json-shaped output")
    args = p.parse_args()

    print(f"Loading pipeline: {args.pipeline}")
    nlp = spacy.load(args.pipeline)
    print("Pipeline components:", nlp.pipe_names)

    print(f"Loading gold corpus: {args.test_spacy}")
    doc_bin = DocBin().from_disk(args.test_spacy)
    reference_docs = list(doc_bin.get_docs(nlp.vocab))
    print(f"  {len(reference_docs)} reference docs")

    examples = []
    for ref_doc in reference_docs:
        words = [t.text for t in ref_doc]
        pred_doc = Doc(nlp.vocab, words=words)  # bare tokenization only -- nlp.evaluate() runs the pipeline on this
        examples.append(Example(pred_doc, ref_doc))

    print(f"Scoring {len(examples)} documents...")
    scores = nlp.evaluate(examples)

    result = {k: scores[k] for k in METRICS if k in scores}
    print(json.dumps({k: round(v * 100, 2) for k, v in result.items()}, indent=2))

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
