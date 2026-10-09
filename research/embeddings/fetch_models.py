"""Download the pinned model files and verify their digests.

    uv run --project research/embeddings --frozen python research/embeddings/fetch_models.py \
        --root <empty or existing directory> [--model minilm ...]

Files land in <root>/<model>/. Nothing is loaded from a floating tag: the revision in
models.toml is the Hub commit sha, and every file's sha256 is checked after download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from embed import load_models
from emb_common import sha256_file


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", action="append", help="model key; default all")
    args = parser.parse_args(argv)
    from huggingface_hub import hf_hub_download

    models = load_models()
    failures = []
    for key in args.model or list(models):
        spec = models[key]
        target = args.root / key
        for name, expected in spec["files"].items():
            path = Path(
                hf_hub_download(spec["id"], name, revision=spec["revision"], local_dir=str(target))
            )
            if sha256_file(path) != expected:
                failures.append(f"{key}/{name}")
            else:
                print(f"ok {key}/{name}")
    if failures:
        print("DIGEST MISMATCH:", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
