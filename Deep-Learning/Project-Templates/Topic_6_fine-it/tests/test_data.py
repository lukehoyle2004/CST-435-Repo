"""The corpus and the routing contract: complete mapping, clean splits, no leakage."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from shared.queues import INTENT_TO_QUEUE, PRIORITY_QUEUE, QUEUE_NAMES, QUEUES
from shared.text import clean, dedup_key, sha256

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.csv"


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS)


def test_every_intent_has_exactly_one_known_queue():
    assert len(INTENT_TO_QUEUE) == 77
    assert set(INTENT_TO_QUEUE.values()) == set(QUEUE_NAMES)
    assert PRIORITY_QUEUE in QUEUES
    assert len(QUEUE_NAMES) == 8


def test_splits_and_sizes(corpus):
    counts = Counter(corpus["split"])
    assert set(counts) == {"train", "validation", "test"}
    assert counts["train"] > 8000 and counts["validation"] > 900 and counts["test"] > 3000


def test_no_message_crosses_a_split(corpus):
    owners = corpus.assign(key=corpus["text"].map(dedup_key)).groupby("key")["split"].nunique()
    assert (owners > 1).sum() == 0


def test_every_intent_appears_in_every_split(corpus):
    assert (corpus.groupby("split")["intent"].nunique() == 77).all()


def test_labels_follow_the_mapping_and_hashes_match(corpus):
    assert (corpus["intent"].map(INTENT_TO_QUEUE) == corpus["queue"]).all()
    sample = corpus.sample(200, random_state=0)
    assert (sample["text"].map(sha256) == sample["text_sha256"]).all()
    assert corpus["text_sha256"].is_unique


def test_text_helpers():
    assert clean("  where   is\nmy card? ") == "where is my card?"
    assert dedup_key("Where is my CARD?!") == dedup_key("where is my card")
    assert sha256("where  is my card?") == sha256("where is my card?")
    assert len(sha256("x")) == 64
