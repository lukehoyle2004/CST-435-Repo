"""Regenerate tests/fixtures/reference_outputs.json (the regression fixture).

    python -m scripts.make_reference

Runs every model endpoint on fixed inputs with deterministic settings and saves
the answers. tests/test_regression.py then fails if a dependency upgrade, a
checkpoint change or a code edit silently changes what the service returns.
Only regenerate it on purpose, and say why in the commit message.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api import db
from api.main import app

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "reference_outputs.json"

CASES = [
    ("/encoder_attention", {"text": "dog bites man bites dog"}),
    ("/encoder_attention", {"text": "the cat sat on the mat"}),
    ("/bert_fillmask", {"text": "I drank a cup of [MASK] this morning.", "top_k": 5}),
    ("/bert_fillmask", {"text": "The capital of France is [MASK].", "top_k": 5}),
    ("/gpt2_continue", {"text": "The best way to learn a new language is", "max_new_tokens": 20}),
    ("/t5_summarize", {"text": (
        "The city council voted on Tuesday to expand the bike lane network after a year of public "
        "meetings. Supporters said the new lanes would make downtown streets safer for commuters, "
        "while several shop owners worried about losing parking in front of their stores."),
        "max_new_tokens": 40}),
]
KEEP = ("prediction", "probabilities", "candidates", "continuation", "summary", "checkpoint", "output_tokens")


def main() -> None:
    db.log_inference = lambda row: 0          # never write to a real database from here
    out = []
    with TestClient(app) as c:
        for path, payload in CASES:
            body = c.post(path, json=payload).json()
            out.append({"path": path, "request": payload, "expected": {k: body[k] for k in KEEP if k in body}})
    FIXTURE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} cases to {FIXTURE}")


if __name__ == "__main__":
    main()
