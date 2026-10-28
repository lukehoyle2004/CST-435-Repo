"""Write tests/fixtures/reference.json: the exported models' outputs on fixed reviews.

tests/test_regression.py checks that the served models still produce exactly these
numbers, so a change to the tokenizer, vocab, masking or weights cannot slip through.
Re-run only when the models are deliberately retrained:

    python -m scripts.make_reference
"""
from __future__ import annotations

import json

from seqsight import MODELS_DIR, PROJECT_ROOT
from seqsight.inference import classify_text, load_vocab
from seqsight.model import MODEL_KINDS, load_exported

REVIEWS = [
    "An absolutely wonderful film. The acting was superb and I loved every minute!",
    "Dull, predictable and far too long. I wanted my two hours back.",
    "I expected a dull remake, but the performances are wonderful and the last twenty minutes "
    "had me on the edge of my seat. Not perfect, the middle drags, but well worth it.",
    "Not the best, not the worst.",
    "This is the worst movie I have ever seen. Don't bother.",
]
OUT = PROJECT_ROOT / "tests" / "fixtures" / "reference.json"


def main() -> None:
    vocab = load_vocab()
    models = {k: load_exported(MODELS_DIR / k) for k in MODEL_KINDS}
    cases = []
    for text in REVIEWS:
        case = {"text": text}
        for kind, (model, config) in models.items():
            r = classify_text(model, vocab, text)
            case[kind] = {"run_id": config["run_id"], "class_name": r["class_name"],
                          "probabilities": r["probabilities"], "tokens": r["tokens"],
                          "attention": r["attention"]}
        cases.append(case)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"vocab_sha256": vocab.sha256(), "cases": cases}, indent=1))
    for c in cases:
        print(f"{c['attention']['class_name']:>8} {c['attention']['probabilities']['positive']:.3f} | "
              f"{c['lstm']['class_name']:>8} {c['lstm']['probabilities']['positive']:.3f} | {c['text'][:50]}")


if __name__ == "__main__":
    main()
