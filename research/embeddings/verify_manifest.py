"""Re-hash everything MANIFEST.toml names and fail on any mismatch.

    python3 research/embeddings/verify_manifest.py [--models-root DIR] [--deep] [--regenerate]

Always checked (standard library only, no model needed):
  * every script / lockfile digest, models.toml agrees with the manifest's model table;
  * every pool file: digest, row count, per-row token count recomputed with the pinned
    pattern, ids unique, fact/query counts, max token counts;
  * every vectors file: digest, size == rows * dimension * 2, rows == the pool's rows.
Optional:
  --models-root DIR   re-hash the model files in DIR/<model>/ against the manifest
  --deep              decode every vector: finite, L2 norm within 2e-3 of 1
  --regenerate        regenerate the pools from pool_gen.py and compare bytes

Exit status 0 only if nothing mismatches.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tomllib
from pathlib import Path

from emb_common import HERE, TOKEN_PATTERN, count_tokens, sha256_file


def verify(root: Path, models_root: Path | None = None, deep: bool = False,
           regenerate: bool = False) -> list[str]:
    problems: list[str] = []

    def bad(message: str) -> None:
        problems.append(message)

    try:
        manifest = tomllib.loads((root / "MANIFEST.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return [f"MANIFEST.toml unreadable: {error}"]
    if manifest.get("version") != 1:
        bad("MANIFEST.toml: unsupported version")
    if manifest.get("pipeline", {}).get("token_count_pattern") != TOKEN_PATTERN:
        bad("MANIFEST.toml: token_count_pattern differs from emb_common.TOKEN_PATTERN")

    for name, expected in manifest.get("scripts", {}).items():
        path = root / name
        if not path.is_file():
            bad(f"scripts.{name}: file missing")
        elif sha256_file(path) != expected:
            bad(f"scripts.{name}: digest mismatch")

    try:
        pinned = tomllib.loads((root / "models.toml").read_text(encoding="utf-8"))["models"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        pinned = {}
        bad(f"models.toml unreadable: {error}")
    for key, model in manifest.get("models", {}).items():
        spec = pinned.get(key)
        if spec is None:
            bad(f"models.{key}: not in models.toml")
            continue
        for field in ("id", "revision"):
            if spec.get(field) != model.get(field):
                bad(f"models.{key}.{field}: differs from models.toml")
        if spec.get("files") != model.get("files"):
            bad(f"models.{key}.files: differ from models.toml")
        if models_root is not None:
            for name, expected in model.get("files", {}).items():
                path = models_root / key / name
                if not path.is_file():
                    bad(f"model file {key}/{name}: missing under {models_root}")
                elif sha256_file(path) != expected:
                    bad(f"model file {key}/{name}: digest mismatch")

    pool_rows: dict[str, int] = {}
    for name, pool in manifest.get("pools", {}).items():
        path = root / pool.get("file", "")
        if not path.is_file():
            bad(f"pools.{name}: file {pool.get('file')!r} missing")
            continue
        if sha256_file(path) != pool.get("sha256"):
            bad(f"pools.{name}: digest mismatch")
        ids, facts, queries, max_fact, max_query = set(), 0, 0, 0, 0
        count = 0
        for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad(f"pools.{name}:{number}: not JSON")
                continue
            count += 1
            if row["id"] in ids:
                bad(f"pools.{name}:{number}: duplicate id {row['id']}")
            ids.add(row["id"])
            if count_tokens(row["text"]) != row["tokens"]:
                bad(f"pools.{name}:{number}: token count {row['tokens']} != recount")
            if row["kind"] == "fact":
                facts += 1
                max_fact = max(max_fact, row["tokens"])
            else:
                queries += 1
                max_query = max(max_query, row["tokens"])
        pool_rows[name] = count
        for field, value in (("rows", count), ("facts", facts), ("queries", queries),
                             ("max_fact_tokens", max_fact), ("max_query_tokens", max_query)):
            if pool.get(field) != value:
                bad(f"pools.{name}.{field}: manifest {pool.get(field)} != recomputed {value}")

    for name, vec in manifest.get("vectors", {}).items():
        path = root / vec.get("file", "")
        if not path.is_file():
            bad(f"vectors.{name}: file {vec.get('file')!r} missing")
            continue
        size = path.stat().st_size
        expected_size = vec["rows"] * vec["dimension"] * 2
        if size != expected_size or vec.get("bytes") != expected_size:
            bad(f"vectors.{name}: size {size} != rows*dimension*2 = {expected_size}")
        if vec.get("pool") not in pool_rows:
            bad(f"vectors.{name}: pool {vec.get('pool')!r} is not in the manifest")
        elif pool_rows[vec["pool"]] != vec["rows"]:
            bad(f"vectors.{name}: {vec['rows']} rows but the pool has {pool_rows[vec['pool']]}")
        if sha256_file(path) != vec.get("sha256"):
            bad(f"vectors.{name}: digest mismatch")
        if vec.get("model") not in manifest.get("models", {}):
            bad(f"vectors.{name}: model {vec.get('model')!r} is not in the manifest")
        if vec.get("determinism_checked") and not vec.get("deterministic_on_this_machine", False):
            bad(f"vectors.{name}: the recorded repeat runs were not identical")
        again = vec.get("fresh_process_repeat_sha256")
        if again is not None and again != vec.get("sha256"):
            bad(f"vectors.{name}: the fresh-process repeat digest differs from the vectors digest")
        if len(set(vec.get("run_digests", []))) > 1:
            bad(f"vectors.{name}: recorded run digests differ")
        if deep and size == expected_size:
            dim = vec["dimension"]
            data = path.read_bytes()
            for row in range(vec["rows"]):
                values = struct.unpack_from(f"<{dim}e", data, row * dim * 2)
                norm = math.sqrt(sum(v * v for v in values))
                if not math.isfinite(norm) or abs(norm - 1.0) > 2e-3:
                    bad(f"vectors.{name}: row {row} norm {norm}")
                    break

    if regenerate:
        from pool_gen import SPECS, build_pool, render

        for name, spec in SPECS.items():
            path = root / f"pool-{name}.jsonl"
            if not path.is_file() or path.read_bytes() != render(build_pool(spec)):
                bad(f"pools.{name}: differs from the generator output")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args(argv)
    problems = verify(args.root, args.models_root, args.deep, args.regenerate)
    for problem in problems:
        print("MISMATCH:", problem)
    if problems:
        return 1
    print("MANIFEST.toml verified")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
