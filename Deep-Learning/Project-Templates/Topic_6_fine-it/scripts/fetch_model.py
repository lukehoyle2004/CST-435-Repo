"""Fetch the deployed checkpoint (Render build step and CI). Weights are too big for git.

    python -m scripts.fetch_model

Sources, first match wins:
    FTF_MODEL_URL    a .tar.gz of models/deployed (e.g. a GitHub release asset)
    FTF_MODEL_REPO   a Hugging Face model repo id (optionally FTF_MODEL_REVISION)

Either way the archive's sha256 of the weights is checked against
docs/experiments/results.json, so the service can only ever serve the exact
checkpoint the leaderboard says is deployed. Also pre-downloads flan-t5-small
so the zero-shot endpoint never waits on the Hub at request time.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

from shared.checkpoints import ZERO_SHOT_MODEL, ZERO_SHOT_REVISION
from shared.artifacts import weights_sha256

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(os.environ.get("FTF_MODEL_DIR", ROOT / "models" / "deployed"))


def expected() -> dict:
    rows = json.loads((ROOT / "docs" / "experiments" / "results.json").read_text())
    return next(r for r in rows if r["is_deployed"])


def from_url(url: str) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(DEST, filter="data")


def from_hub(repo: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=os.environ.get("FTF_MODEL_REVISION"), local_dir=DEST)


def main() -> None:
    want = expected()
    if (DEST / "config.json").exists() and weights_sha256(DEST) == want["artifact_sha256"]:
        print(f"{DEST} already holds {want['run_name']}")
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        if os.environ.get("FTF_MODEL_URL"):
            from_url(os.environ["FTF_MODEL_URL"])
        elif os.environ.get("FTF_MODEL_REPO"):
            from_hub(os.environ["FTF_MODEL_REPO"])
        else:
            sys.exit("set FTF_MODEL_URL or FTF_MODEL_REPO")
        got = weights_sha256(DEST)
        if got != want["artifact_sha256"]:
            sys.exit(f"weights sha256 {got[:12]} != deployed {want['artifact_sha256'][:12]} ({want['run_name']})")
        (DEST / "ftf_run.json").write_text(json.dumps({"run_name": want["run_name"],
                                                       "artifact_sha256": want["artifact_sha256"]}, indent=2))
        print(f"fetched {want['run_name']} ({got[:12]})")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION)
    AutoModelForSeq2SeqLM.from_pretrained(ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION)
    print("zero-shot model cached")
    if os.environ.get("FTF_ZERO_SHOT_HALF_WEIGHTS", "0") == "1":
        from src.zero_shot import save_half_copy
        print(f"float16 copy saved to {save_half_copy()}")


if __name__ == "__main__":
    main()
