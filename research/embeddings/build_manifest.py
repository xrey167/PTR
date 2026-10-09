"""Write MANIFEST.toml from the files on disk and embed.py run records.

    python3 build_manifest.py --record run-minilm-q003.json --record run-minilm-m008.json ...

Pool and script digests are computed from the files; the vectors entries come from the
run records (tool versions, timings, the digests of repeated runs) plus the file digest.
Entries already in MANIFEST.toml for other (model, pool) pairs are kept. Verification
is `verify_manifest.py`; nothing here is trusted by it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

from emb_common import HERE, TOKEN_PATTERN, sha256_file
from pool_gen import SPECS, load_pool, stats

SCRIPT_FILES = [
    "emb_common.py",
    "pool_gen.py",
    "embed.py",
    "fetch_models.py",
    "build_manifest.py",
    "verify_manifest.py",
    "models.toml",
    "pyproject.toml",
    "uv.lock",
]
BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(scalar(item) for item in value) + "]"
    raise TypeError(f"cannot emit {value!r}")


def key(name: str) -> str:
    return name if BARE.match(name) else json.dumps(name)


def emit(table: dict, path: tuple[str, ...] = ()) -> list[str]:
    lines = []
    plain = {k: v for k, v in table.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in table.items() if isinstance(v, dict)}
    if path:
        lines.append("")
        lines.append("[" + ".".join(key(p) for p in path) + "]")
    for k, v in plain.items():
        lines.append(f"{key(k)} = {scalar(v)}")
    for k, v in nested.items():
        lines.extend(emit(v, (*path, k)))
    return lines


def build(root: Path, records: list[dict], repeats: list[dict] | None = None) -> dict:
    previous = {}
    existing = root / "MANIFEST.toml"
    if existing.is_file():
        previous = tomllib.loads(existing.read_text(encoding="utf-8"))
    models = tomllib.loads((root / "models.toml").read_text(encoding="utf-8"))["models"]
    manifest: dict = {"version": 1}
    manifest["pipeline"] = {
        "tokenizer": "the model's own tokenizer.json, truncation at max_length",
        "forward": "transformers AutoModel, float32, CPU, torch.inference_mode",
        "pooling": "per model: minilm mean over the attention mask; bge-m3 first token",
        "normalise": "L2, in float32",
        "storage": "little-endian IEEE half (float16), row-major, one row per pool row, pool order",
        "token_count_pattern": TOKEN_PATTERN,
    }
    tools = previous.get("tools", {})
    for record in records:
        for name, value in record["tools"].items():
            if tools.get(name, value) != value:
                raise SystemExit(f"run records disagree on tool {name}: {tools[name]} vs {value}")
            tools[name] = value
    manifest["tools"] = tools
    manifest["scripts"] = {name: sha256_file(root / name) for name in SCRIPT_FILES}
    manifest["models"] = {
        key_: {"id": spec["id"], "revision": spec["revision"], "license": spec["license"],
               "files": dict(spec["files"])}
        for key_, spec in models.items()
    }
    manifest["pools"] = {}
    for name, spec in SPECS.items():
        rows = load_pool(name, root)
        summary = stats(rows)
        manifest["pools"][name] = {
            "file": f"pool-{name}.jsonl",
            "sha256": sha256_file(root / f"pool-{name}.jsonl"),
            "rows": len(rows),
            "facts": summary["facts"]["count"],
            "queries": summary["queries"]["count"],
            "max_fact_tokens": summary["facts"]["max"],
            "max_query_tokens": summary["queries"]["max"],
        }
    vectors = dict(previous.get("vectors", {}))
    for record in records:
        name = f"{record['model']}-{record['pool']}"
        file_name = vectors.get(name, {}).get("file") or record.get("file")
        vectors[name] = {
            "file": file_name,
            "model": record["model"],
            "pool": record["pool"],
            "rows": record["rows"],
            "dimension": record["dimension"],
            "dtype": "float16-le",
            "bytes": record["rows"] * record["dimension"] * 2,
            "sha256": record["vectors_sha256"],
            "threads": record["threads"],
            "batch_size": record["batch_size"],
            "seconds": record["seconds"],
            "repeat_runs": len(record["run_digests"]),
            "run_digests": record["run_digests"],
            "determinism_checked": len(record["run_digests"]) > 1,
            "deterministic_on_this_machine": record["deterministic_on_this_machine"],
        }
    for repeat in repeats or []:
        name = f"{repeat['model']}-{repeat['pool']}"
        entry = vectors[name]
        entry["determinism_checked"] = True
        entry["fresh_process_repeat_sha256"] = repeat["vectors_sha256"]
        entry["fresh_process_repeat_seconds"] = repeat["seconds"][0]
    manifest["vectors"] = vectors
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", default=[], type=Path)
    parser.add_argument("--repeat", action="append", default=[], type=Path,
                        help="run record of a second, fresh-process computation to compare")
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args(argv)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.record]
    repeats = [json.loads(path.read_text(encoding="utf-8")) for path in args.repeat]
    manifest = build(args.root, records, repeats)
    header = ["# Generated by build_manifest.py; verified by verify_manifest.py."]
    (args.root / "MANIFEST.toml").write_text("\n".join(header + emit(manifest)) + "\n", encoding="utf-8")
    print("wrote MANIFEST.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
