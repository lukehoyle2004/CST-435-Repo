"""Fingerprint of a checkpoint's weights, shared by the sweep, the fetcher and the API.

Kept free of training dependencies so the Render build can import it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def weights_sha256(directory: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(directory).glob("*.safetensors")):
        h.update(f.read_bytes())
    return h.hexdigest()
