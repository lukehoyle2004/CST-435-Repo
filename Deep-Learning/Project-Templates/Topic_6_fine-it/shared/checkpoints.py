"""Pinned upstream revisions, so every number in the repo can be reproduced.

A Hugging Face repo name alone is a moving target; a commit sha is not.
"""
from __future__ import annotations

DATASET_REPO = "PolyAI/banking77"
DATASET_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"  # CC-BY-4.0

BASE_MODEL = "distilbert-base-uncased"
BASE_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"  # Apache-2.0

ZERO_SHOT_MODEL = "google/flan-t5-small"
ZERO_SHOT_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"  # Apache-2.0

# Every message is tokenised to at most this many WordPiece tokens. The longest
# Banking77 message is well under it (see docs/data/dataset_stats.json), so in
# practice nothing is truncated; the cap only bounds a hostile 10 kB input.
MAX_LENGTH = 128

SPLIT_SEED = 42
