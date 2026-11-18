"""Text cleaning and hashing, shared by the data loader and the API.

Two different normalisations, on purpose:

* ``clean`` is what the model sees. It only trims and collapses whitespace, so
  casing and punctuation survive (DistilBERT-uncased lower-cases on its own).
* ``dedup_key`` is used ONLY to find duplicates across splits. It is aggressive
  (lower-case, punctuation stripped) so "Where is my card?" and "where is my card"
  count as the same message and cannot sit on both sides of the train/test line.
"""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def sha256(text: str) -> str:
    """SHA-256 of the cleaned text: the only trace of a customer message that is stored."""
    return hashlib.sha256(clean(text).encode("utf-8")).hexdigest()
