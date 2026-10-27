"""Download the IMDB train/test splits into ``data/imdb/`` (gitignored).

Source: the Hugging Face mirror of the Maas et al. (2011) dataset,
https://huggingface.co/datasets/stanfordnlp/imdb (``plain_text`` config), as two
parquet files (~21 MB each). The original tarball at
https://ai.stanford.edu/~amaas/data/sentiment/ holds the same 25k + 25k labeled
reviews but was unreachable when this project was built.

    python -m scripts.download_imdb
"""
from __future__ import annotations

import urllib.request

from shared.imdb import DEFAULT_ROOT, SPLITS

BASE_URL = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"


def main() -> None:
    DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = DEFAULT_ROOT / f"{split}.parquet"
        if target.exists():
            print(f"Already present: {target}")
            continue
        url = f"{BASE_URL}/{split}-00000-of-00001.parquet"
        print(f"Downloading {url} ...")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(target)
        print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
