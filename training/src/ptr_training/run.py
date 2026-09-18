from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))

def file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"

def build_manifest(config_path: Path) -> dict:
    cfg = load(config_path)
    dataset_registry = load(ROOT / "datasets" / "registry.toml")
    dataset_name = cfg["data"]["dataset"]
    datasets = {d["name"]: d for d in dataset_registry.get("dataset", [])}
    if dataset_name not in datasets:
        raise ValueError(f"unknown dataset: {dataset_name}")
    dataset = datasets[dataset_name]
    return {
        "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config": cfg,
        "dataset": dataset,
        "cargo_lock_sha256": file_sha(ROOT / "Cargo.lock"),
        "uv_lock_sha256": file_sha(ROOT / "training" / "uv.lock"),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "training" / "configs" / "run-default.toml",
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    manifest = build_manifest(config_path)
    out = args.out
    if out is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "training" / "runs" / f"run-{stamp}.json"
    elif not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))

    if args.execute:
        backend = manifest["config"]["training"]["backend"]
        if backend in {"dry-run", "unconfigured"}:
            raise SystemExit(
                "training backend is not selected; run manifest prepared but no weights were trained"
            )
        raise SystemExit(f"training backend {backend!r} is not implemented yet")

if __name__ == "__main__":
    main()
