#!/usr/bin/env python3
"""Run the fixed, non-claimable M002-v5 architecture pilot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path("benchmarks/operator-routing-v2/splits.lock.json")
DATA_PATH = Path("datasets/generated/operator_routing_v2")
FOLDS = ("evidence-interventional", "claim-temporal-development")
SEEDS = (7, 13)
RANKS = (8, 16)
BIAS_LIMITS = (1, 2)
METADATA_DROPOUTS = (0.0, 0.1)
TOOLCHAIN = "1.95.0-x86_64-pc-windows-gnu"
HARDWARE_PROFILE = Path("hardware/a0-cpu-12core-windows.toml")
DEFAULT_OUTPUT = ROOT.parent / f"{ROOT.name}-m002-v5-pilot-output"


class PilotError(RuntimeError):
    """A refusal caused by invalid inputs, provenance, or existing evidence."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fold_digests(root: Path = ROOT) -> dict[str, str]:
    """Read and validate the two pilot fold identities from the dataset lock."""
    path = root / LOCK_PATH
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
        folds = lock["folds"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PilotError(f"cannot read fold digests from {path}: {error}") from None

    expected = {
        "evidence-interventional": ("diagnostic", "evidence", "interventional"),
        "claim-temporal-development": ("development", "claim", "temporal"),
    }
    result: dict[str, str] = {}
    for name in FOLDS:
        try:
            fold = folds[name]
            identity = (fold["kind"], fold["target_role"], fold["target_regime"])
            digest = fold["data_fnv1a64"]
        except (KeyError, TypeError) as error:
            raise PilotError(f"{path}: invalid {name} entry: {error}") from None
        if identity != expected[name]:
            raise PilotError(
                f"{path}: {name} identity is {identity!r}, expected {expected[name]!r}"
            )
        if not isinstance(digest, str) or len(digest) != 16:
            raise PilotError(f"{path}: {name} data_fnv1a64 must be 16 hexadecimal digits")
        try:
            int(digest, 16)
        except ValueError:
            raise PilotError(f"{path}: {name} data_fnv1a64 is not hexadecimal") from None
        result[name] = digest.lower()
    return result


def candidates() -> Iterable[dict[str, int | float]]:
    for rank in RANKS:
        for bias_limit in BIAS_LIMITS:
            for metadata_dropout in METADATA_DROPOUTS:
                yield {
                    "rank": rank,
                    "bias_limit": bias_limit,
                    "metadata_dropout": metadata_dropout,
                }


def cell_name(candidate: dict[str, int | float], seed: int) -> str:
    dropout = str(candidate["metadata_dropout"]).replace(".", "p")
    return (
        f"rank-{candidate['rank']}-bias-{candidate['bias_limit']}"
        f"-dropout-{dropout}-seed-{seed}"
    )


def command_for(
    candidate: dict[str, int | float], seed: int, digests: dict[str, str]
) -> list[str]:
    folds = ",".join(f"{name}={digests[name]}" for name in FOLDS)
    return [
        "cargo",
        f"+{TOOLCHAIN}",
        "run",
        "--release",
        "--locked",
        "--quiet",
        "--jobs",
        "1",
        "--manifest-path",
        "model/burn-a0/Cargo.toml",
        "--example",
        "a0_ablation",
        "--",
        "--phase",
        "paired-v5",
        "--experiment",
        "M002-v5",
        "--arms",
        "factorized-v2,factorized-v2-off",
        "--seed",
        str(seed),
        "--steps",
        "1500",
        "--lr",
        "0.005",
        "--data",
        DATA_PATH.as_posix(),
        "--folds",
        folds,
        "--d-model",
        "48",
        "--rank",
        str(candidate["rank"]),
        "--bias-limit",
        str(candidate["bias_limit"]),
        "--metadata-dropout",
        str(candidate["metadata_dropout"]),
    ]


def source_sha(root: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or len(result.stdout.strip()) != 40:
        raise PilotError(f"cannot resolve committed HEAD: {result.stderr.strip()}")
    return result.stdout.strip()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def require_clean_detached_worktree(output: Path, root: Path = ROOT) -> str:
    """Refuse mutable source inputs, while permitting verified own output.

    A completed cell is immutable evidence and intentionally lives below
    ``output``.  It is the only untracked content a resumed invocation may
    observe; every other modified, staged, or untracked path is a refusal.
    The actual subprocesses run from a second detached worktree created from
    this SHA, so the archive cannot change their inputs.
    """
    sha = source_sha(root)
    branch = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if branch.returncode == 0:
        raise PilotError("refusing real pilot run: source worktree must be detached at a commit")
    if branch.returncode not in (0, 1):
        raise PilotError(f"cannot inspect source HEAD attachment: {branch.stderr.strip()}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode:
        raise PilotError(f"cannot inspect working tree: {status.stderr.strip()}")
    root_resolved = root.resolve()
    output_resolved = output.resolve()
    for line in status.stdout.splitlines():
        if not line.startswith("?? "):
            raise PilotError("refusing real pilot run: working tree is not clean and committed")
        candidate = root_resolved / line[3:]
        if not _inside(candidate, output_resolved):
            raise PilotError("refusing real pilot run: working tree is not clean and committed")
    return sha


def provenance_environment() -> dict[str, str]:
    """Digest only build-relevant environment values; never archive secrets."""
    names = (
        "PATH",
        "CARGO_BUILD_TARGET",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_INCREMENTAL",
        "CC",
        "CFLAGS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTDOCFLAGS",
        "RUSTFLAGS",
    )
    return {name: os.environ.get(name, "") for name in names}


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise PilotError(f"cannot identify {' '.join(command)}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def execution_provenance(root: Path, source: str) -> dict[str, object]:
    """Bound toolchain, environment, host and source-tree identity for every cell."""
    cargo = shutil.which("cargo")
    if cargo is None:
        raise PilotError("cannot resolve cargo executable")
    cargo_path = Path(cargo).resolve()
    try:
        cargo_digest = sha256(cargo_path.read_bytes())
        profile_bytes = (root / HARDWARE_PROFILE).read_bytes()
    except OSError as error:
        raise PilotError(f"cannot read execution provenance input: {error}") from None
    tree = command_output(["git", "rev-parse", f"{source}^{{tree}}"], cwd=root)
    if len(tree) != 40:
        raise PilotError("cannot resolve source tree identity")
    environment = provenance_environment()
    host = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "release": platform.release(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
    }
    return {
        "source_tree_sha": tree,
        "cargo": {
            "path": str(cargo_path),
            "sha256": cargo_digest,
            "version": command_output(["cargo", f"+{TOOLCHAIN}", "--version"]),
        },
        "rustc_version": command_output(["rustc", f"+{TOOLCHAIN}", "-Vv"]),
        "environment_sha256": sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "host": host,
        "hardware_profile": {
            "path": HARDWARE_PROFILE.as_posix(),
            "sha256": sha256(profile_bytes),
        },
    }


def expected_identity(
    source: str,
    command: list[str],
    candidate: dict[str, int | float],
    seed: int,
    digests: dict[str, str],
    execution: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "source_sha": source,
        "command": command,
        "candidate": candidate,
        "seed": seed,
        "fold_digests": digests,
        "execution_provenance": execution,
    }


def artifact_paths(output: Path, name: str) -> tuple[Path, Path, Path]:
    return (
        output / f"{name}.stdout",
        output / f"{name}.stderr",
        output / f"{name}.json",
    )


def resumable(output: Path, name: str, expected: dict[str, object]) -> bool:
    stdout_path, stderr_path, metadata_path = artifact_paths(output, name)
    paths = (stdout_path, stderr_path, metadata_path)
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths):
        raise PilotError(f"{name}: incomplete existing artifacts; refusing to overwrite")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotError(f"{name}: invalid existing metadata: {error}") from None
    if not isinstance(metadata, dict):
        raise PilotError(f"{name}: existing metadata is not a JSON object")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise PilotError(f"{name}: existing artifacts do not match this exact cell")
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    duration = metadata.get("duration_seconds")
    complete = (
        metadata.get("exit_code") == 0
        and metadata.get("stdout_sha256") == sha256(stdout)
        and metadata.get("stderr_sha256") == sha256(stderr)
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration >= 0
        and metadata.get("stdout_file") == stdout_path.name
        and metadata.get("stderr_file") == stderr_path.name
    )
    if not complete:
        raise PilotError(f"{name}: existing artifacts are not a complete successful cell")
    return True


def write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise PilotError(f"refusing to overwrite existing artifact {path}") from None


def validate_existing_archive(
    output: Path,
    cells: list[tuple[str, dict[str, int | float], int, list[str]]],
    source: str,
    digests: dict[str, str],
    execution: dict[str, object],
) -> None:
    """Allow resumable, complete evidence only; reject stray or partial files."""
    if not output.exists():
        return
    if not output.is_dir():
        raise PilotError(f"pilot output is not a directory: {output}")
    expected_paths = set()
    for name, candidate, seed, command in cells:
        expected = expected_identity(source, command, candidate, seed, digests, execution)
        expected_paths.update(artifact_paths(output, name))
        resumable(output, name, expected)
    for path in output.rglob("*"):
        if path.is_file() and path not in expected_paths:
            raise PilotError(f"pilot archive has an unexpected artifact: {path.name}")
        if path.is_dir() and path != output:
            raise PilotError(f"pilot archive has an unexpected directory: {path.name}")


def snapshot_status(root: Path, source: str) -> None:
    head = source_sha(root)
    if head != source:
        raise PilotError("immutable execution snapshot moved away from its source commit")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode or status.stdout:
        raise PilotError("immutable execution snapshot was modified")


@contextlib.contextmanager
def immutable_snapshot(root: Path, source: str) -> Iterable[Path]:
    """Materialize a private detached worktree from exactly ``source``.

    The source checkout may retain its versioned archive while a run resumes.
    Cargo receives this private detached checkout instead, so it cannot consume
    an edit made in the archival checkout after provenance was captured.
    """
    with tempfile.TemporaryDirectory(prefix="m002-v5-source-", dir=root.parent) as parent:
        snapshot = Path(parent) / "source"
        created = subprocess.run(
            ["git", "worktree", "add", "--detach", str(snapshot), source],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode:
            raise PilotError(f"cannot create immutable execution snapshot: {created.stderr.strip()}")
        try:
            snapshot_status(snapshot, source)
            yield snapshot
        finally:
            removed = subprocess.run(
                ["git", "worktree", "remove", "--force", str(snapshot)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if removed.returncode:
                raise PilotError(f"cannot remove immutable execution snapshot: {removed.stderr.strip()}")


def run_cell(
    root: Path,
    output: Path,
    name: str,
    command: list[str],
    expected: dict[str, object],
    environment: dict[str, str],
) -> int:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    duration = time.monotonic() - started
    stdout = completed.stdout
    stderr = completed.stderr
    stdout_path, stderr_path, metadata_path = artifact_paths(output, name)
    metadata = {
        **expected,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "stdout_sha256": sha256(stdout),
        "stderr_sha256": sha256(stderr),
    }
    write_exclusive(stdout_path, stdout)
    write_exclusive(stderr_path, stderr)
    write_exclusive(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return completed.returncode


def plan(digests: dict[str, str]) -> Iterable[tuple[str, dict[str, int | float], int, list[str]]]:
    for candidate in candidates():
        for seed in SEEDS:
            yield cell_name(candidate, seed), candidate, seed, command_for(candidate, seed, digests)


def execute(output: Path, dry_run: bool, root: Path = ROOT) -> int:
    if dry_run:
        digests = fold_digests(root)
        cells = list(plan(digests))
        for name, _, _, command in cells:
            print(json.dumps({"cell": name, "command": command}, sort_keys=True))
        return 0

    output = output if output.is_absolute() else root / output
    source = require_clean_detached_worktree(output, root)
    digests = fold_digests(root)
    cells = list(plan(digests))
    execution = execution_provenance(root, source)
    validate_existing_archive(output, cells, source, digests, execution)
    output.mkdir(parents=True, exist_ok=True)
    with immutable_snapshot(root, source) as snapshot:
        with tempfile.TemporaryDirectory(prefix="m002-v5-pilot-", dir=root.parent) as isolated:
            isolated_path = Path(isolated)
            environment = os.environ.copy()
            environment.update(
                {
                    "CARGO_HOME": str(isolated_path / "cargo-home"),
                    "CARGO_TARGET_DIR": str(isolated_path / "cargo-target"),
                    "TEMP": str(isolated_path / "temp"),
                    "TMP": str(isolated_path / "temp"),
                }
            )
            for path in (environment["CARGO_HOME"], environment["CARGO_TARGET_DIR"], environment["TEMP"]):
                Path(path).mkdir(parents=True, exist_ok=True)
            for name, candidate, seed, command in cells:
                expected = expected_identity(source, command, candidate, seed, digests, execution)
                if resumable(output, name, expected):
                    print(f"resume {name}")
                    continue
                print(f"run {name}", flush=True)
                exit_code = run_cell(snapshot, output, name, command, expected, environment)
                snapshot_status(snapshot, source)
                if exit_code:
                    raise PilotError(f"{name}: cargo exited with {exit_code}; immutable evidence retained")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return execute(args.output_dir, args.dry_run)
    except PilotError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
