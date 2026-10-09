"""Compute pinned CPU sentence embeddings for a text pool.

    uv run --project research/embeddings --frozen python research/embeddings/embed.py \
        --model minilm --pool q003 --model-dir <dir with the pinned files> \
        --out research/embeddings/vectors-q003.f16 --record <run.json>

Pipeline (per models.toml): tokenizer, truncate to max_length, forward pass in fp32 on
CPU, pool (mean over the attention mask, or the first token), L2-normalise, store as
little-endian IEEE half, row-major, in pool order.

The model files are digest-checked against models.toml before they are loaded, so a
wrong or tampered snapshot stops the run. torch CPU output is not bit-reproducible
across machines or thread counts; this is why the result is a committed, digest-pinned
file and harnesses never recompute it per seed. Same machine, same thread count, same
batch size gives identical bytes (checked by `--twice`).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tomllib
from pathlib import Path

from emb_common import HERE, sha256_bytes, sha256_file
from pool_gen import load_pool


def load_models(path: Path = HERE / "models.toml") -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["models"]


def verify_model_dir(spec: dict, model_dir: Path) -> None:
    problems = []
    for name, expected in spec["files"].items():
        path = model_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(path) != expected:
            problems.append(f"digest mismatch {name}")
    if problems:
        raise SystemExit(f"model directory {model_dir} does not match the pinned files: {problems}")


def embed_texts(texts: list[str], spec: dict, model_dir: Path, threads: int, batch_size: int):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model.eval()
    prefix = spec.get("document_prefix", "")
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=spec["max_length"],
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            if spec["pooling"] == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            elif spec["pooling"] == "cls":
                pooled = hidden[:, 0]
            else:
                raise SystemExit(f"unknown pooling {spec['pooling']!r}")
            if spec["normalize"]:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.to(torch.float32).numpy())
    matrix = np.concatenate(out, axis=0)
    if matrix.shape[1] != spec["dimension"]:
        raise SystemExit(f"dimension {matrix.shape[1]} != pinned {spec['dimension']}")
    return matrix.astype("<f2")


def tool_versions() -> dict:
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--record", type=Path, help="write a JSON run record here")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="embed only the first N rows (timing runs)")
    parser.add_argument("--twice", action="store_true", help="embed twice and compare digests")
    args = parser.parse_args(argv)

    spec = load_models()[args.model]
    verify_model_dir(spec, args.model_dir)
    rows = load_pool(args.pool)
    if args.limit:
        rows = rows[: args.limit]
    texts = [row["text"] for row in rows]

    digests, seconds = [], []
    matrix = None
    for _ in range(2 if args.twice else 1):
        started = time.monotonic()
        matrix = embed_texts(texts, spec, args.model_dir, args.threads, args.batch_size)
        seconds.append(round(time.monotonic() - started, 1))
        digests.append(sha256_bytes(matrix.tobytes()))
    args.out.write_bytes(matrix.tobytes())
    record = {
        "model": args.model,
        "pool": args.pool,
        "file": args.out.name,
        "rows": len(rows),
        "dimension": spec["dimension"],
        "threads": args.threads,
        "batch_size": args.batch_size,
        "seconds": seconds,
        "run_digests": digests,
        "deterministic_on_this_machine": len(set(digests)) == 1,
        "vectors_sha256": sha256_file(args.out),
        "tools": tool_versions(),
    }
    print(json.dumps(record, indent=2))
    if args.record:
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if record["deterministic_on_this_machine"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
