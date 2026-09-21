from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

from ptr_training import codebook as codebook_artifact

ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def file_fingerprint(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha(path),
    }


def verify_declared_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict:
    fingerprint = file_fingerprint(path)
    if expected_sha256 and fingerprint["sha256"] != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {fingerprint['path']}: "
            f"expected {expected_sha256}, got {fingerprint['sha256']}"
        )
    if expected_bytes is not None and fingerprint["bytes"] != expected_bytes:
        raise ValueError(
            f"size mismatch for {fingerprint['path']}: "
            f"expected {expected_bytes}, got {fingerprint['bytes']}"
        )
    return fingerprint


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_worktree_state() -> dict:
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
        )
        return {
            "dirty": bool(porcelain.strip()),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {
            "dirty": None,
            "tracked_diff_sha256": None,
        }


def input_fingerprint(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(config_path: Path) -> dict:
    config_path = config_path.resolve()
    cfg = load(config_path)

    dataset_registry_path = ROOT / "datasets" / "registry.toml"
    dataset_registry = load(dataset_registry_path)
    dataset_name = cfg["data"]["dataset"]
    datasets = {item["name"]: item for item in dataset_registry.get("dataset", [])}
    if dataset_name not in datasets:
        raise ValueError(f"unknown dataset: {dataset_name}")

    dataset = dict(datasets[dataset_name])
    dataset_path = ROOT / "datasets" / dataset["path"]
    dataset_artifact = verify_declared_file(
        dataset_path,
        expected_sha256=dataset.get("sha256"),
        expected_bytes=dataset.get("bytes"),
    )

    dataset_card = None
    if dataset.get("card"):
        dataset_card = file_fingerprint(ROOT / "datasets" / dataset["card"])

    model_config_path = ROOT / cfg["model"]["config"]
    model_config = file_fingerprint(model_config_path)

    hardware_profile_path = ROOT / cfg["run"]["hardware_profile"]
    hardware_profile = file_fingerprint(hardware_profile_path)

    # A run is only interpretable against the assignment its codes belong to, so
    # the config must name it and the artifact must agree. Both fields are
    # required: a version with no fingerprint cannot catch a table edited in
    # place, and neither can be defaulted without adopting an assignment nobody
    # chose.
    book = codebook_artifact.load()
    codebook_artifact.require_identity(
        cfg.get("codebook", {}),
        book,
        version_field="version",
        fingerprint_field="fingerprint",
        where=f"{config_path.name} [codebook]",
    )
    codebook = {
        "version": book["version"],
        "fingerprint_sha256": book["fingerprint_sha256"],
        "artifact": codebook_artifact.artifact_fingerprint(),
    }

    config_file = file_fingerprint(config_path)
    cargo_lock = file_fingerprint(ROOT / "Cargo.lock")
    uv_lock = file_fingerprint(ROOT / "training" / "uv.lock")
    git = {"sha": git_sha(), **git_worktree_state()}

    provenance = {
        "git": git,
        "config": config_file,
        "dataset_registry": file_fingerprint(dataset_registry_path),
        "dataset_artifact": dataset_artifact,
        "dataset_card": dataset_card,
        "model_config": model_config,
        "hardware_profile": hardware_profile,
        "cargo_lock": cargo_lock,
        "a0_manifest": file_fingerprint(ROOT / "model/burn-a0/Cargo.toml"),
        "a0_cargo_lock": file_fingerprint(ROOT / "model/burn-a0/Cargo.lock"),
        "uv_lock": uv_lock,
        "codebook": codebook,
    }

    return {
        "schema_version": 4,
        "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": git["sha"],
        "git_dirty": git["dirty"],
        "git_tracked_diff_sha256": git["tracked_diff_sha256"],
        "platform": platform.platform(),
        "python": sys.version,
        "config_path": str(config_path.relative_to(ROOT)),
        "config": cfg,
        "config_sha256": config_file["sha256"],
        "dataset": dataset,
        "dataset_artifact": dataset_artifact,
        "dataset_card": dataset_card,
        "model_config": model_config,
        "hardware_profile": hardware_profile,
        "codebook": codebook,
        "cargo_lock_sha256": cargo_lock["sha256"],
        "uv_lock_sha256": uv_lock["sha256"],
        "input_fingerprint_sha256": input_fingerprint(provenance),
        "provenance": provenance,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "training" / "configs" / "run-default.toml",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    manifest = build_manifest(config_path)
    out = args.out
    if out is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out = ROOT / "training" / "runs" / f"run-{stamp}.json"
    elif not out.is_absolute():
        out = ROOT / out

    write_manifest(out, manifest)
    print(out.relative_to(ROOT))

    if args.execute:
        if manifest["git_dirty"]:
            raise SystemExit(
                "refusing training execution from a dirty tracked worktree; "
                "commit/stash changes first"
            )
        backend = manifest["config"]["training"]["backend"]
        if backend in {"dry-run", "unconfigured"}:
            raise SystemExit(
                "training backend is not selected; run manifest prepared but no weights were trained"
            )
        raise SystemExit(f"training backend {backend!r} is not implemented yet")


if __name__ == "__main__":
    main()
