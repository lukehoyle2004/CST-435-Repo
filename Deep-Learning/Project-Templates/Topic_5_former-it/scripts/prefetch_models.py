"""Download the three checkpoints into the Hugging Face cache at build time.

Render runs this in the build step (see render.yaml) with HF_HOME pointing inside
the service directory, so the files ship with the deploy and the first request
does not wait on a download. Each model is loaded once to fetch exactly the
files from_pretrained needs, then freed.

    python -m scripts.prefetch_models
"""
from __future__ import annotations

import os

from src import pipelines


def main() -> None:
    print("HF_HOME =", os.environ.get("HF_HOME", "(default cache)"))
    for name, spec in pipelines.SPECS.items():
        bundle = pipelines.load_bundle(spec)
        print(f"{name:5s} {bundle.checkpoint_id}  {bundle.params / 1e6:.1f}M params")
        del bundle
        pipelines.release_memory()


if __name__ == "__main__":
    main()
