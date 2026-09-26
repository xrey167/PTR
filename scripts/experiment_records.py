"""Bind an experiment's aggregated result to the code its run records exercised.

`scripts/run_experiment.py run` stamps every run record with the experiment
id, the commit it ran at (`git_sha`), the manifest, the Cargo.lock hash, the
parameters and the entrypoint. An aggregator that combines one record per seed
must not publish a verdict for code those records never ran, so before it
combines them it asks `source_revision`, which refuses records that

- belong to another experiment, or do not name the commit they ran at;
- disagree about the Cargo.lock hash, the parameters or the entrypoint, or ran
  under a manifest that differs from the current `experiment.toml` in anything
  but its `status` (which changes when the experiment completes);
- ran at commits whose Rust sources, Cargo manifests, lock file, toolchain,
  Cargo configuration or SQL files (`CODE_PATHS`) differ from each other, or
  from the checkout the aggregate is written in.

Records archived one commit at a time sit at different commits; they still
agree when nothing under `CODE_PATHS` changed between those commits.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

# The files that decide what an experiment binary does, as git pathspecs. A
# pathspec without magic matches `*` across `/`, so `*.rs` is every Rust file.
CODE_PATHS = (
    "*.rs",
    "*.sql",
    "Cargo.toml",
    "*/Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
)
COMMIT = re.compile(r"[0-9a-f]{7,64}")
AGREED = (
    ("cargo_lock_sha256", "Cargo.lock"),
    ("parameters", "parameters"),
    ("entrypoint", "entrypoint"),
)


class ProvenanceError(Exception):
    """The run records cannot be published as one result of the current code."""


def without_status(manifest: dict) -> dict:
    """A manifest apart from its `status`, which completing an experiment changes."""
    return {key: value for key, value in manifest.items() if key != "status"}


def agreement_errors(experiment_id: str, manifest: dict, records: dict[str, dict]) -> list[str]:
    """Why `records` (file name -> run record) cannot be seeds of one run of
    `experiment_id`, whose manifest is now `manifest`; empty when they can.
    This compares the records alone; `source_revision` also compares code."""
    errors = []
    for name, record in records.items():
        if record.get("experiment_id") != experiment_id:
            errors.append(f"{name} is a record of {record.get('experiment_id')!r}, not {experiment_id!r}")
        if not COMMIT.fullmatch(str(record.get("git_sha", ""))):
            errors.append(f"{name} names no commit it ran at: {record.get('git_sha')!r}")
    for key, label in AGREED:
        groups: dict[str, list[str]] = {}
        for name, record in records.items():
            groups.setdefault(json.dumps(record.get(key), sort_keys=True), []).append(name)
        if len(groups) > 1:
            errors.append(
                f"the records disagree on {label}: "
                + "; ".join(f"{value} in {', '.join(names)}" for value, names in groups.items())
            )
    current = without_status(manifest)
    for name, record in records.items():
        if without_status(record.get("manifest") or {}) != current:
            errors.append(f"{name} ran under an experiment.toml that differs from the current one")
    return errors


def code_changes(base: str, head: str | None, root: Path) -> list[str]:
    """The files under `CODE_PATHS` that differ between commit `base` and
    commit `head`, or the working tree of `root` when `head` is None. Raises
    `ProvenanceError` when git cannot compare them (say a commit is missing)."""
    revisions = [base] if head is None else [base, head]
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", *revisions, "--", *CODE_PATHS],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ProvenanceError(f"cannot run git: {error}") from error
    if diff.returncode != 0:
        against = "the checkout" if head is None else head
        raise ProvenanceError(f"cannot compare {base} with {against}: {diff.stderr.strip()}")
    return diff.stdout.splitlines()


def listed(paths: list[str], limit: int = 5) -> str:
    more = f" and {len(paths) - limit} more" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def source_revision(experiment_id: str, manifest: dict, records: dict[str, dict], root: Path) -> str:
    """The commit the results in `records` were produced at: the earliest
    record's `git_sha`, once every record agrees (`agreement_errors`) and
    every record's commit and the checkout in `root` have the same files
    under `CODE_PATHS`. Raises `ProvenanceError` naming what disagrees."""
    if not records:
        raise ProvenanceError("no run records")
    errors = agreement_errors(experiment_id, manifest, records)
    if errors:
        raise ProvenanceError("; ".join(errors))
    ordered = sorted(records.items(), key=lambda item: (str(item[1].get("started_at", "")), item[0]))
    revision = ordered[0][1]["git_sha"]
    for name, record in ordered[1:]:
        if record["git_sha"] != revision:
            changed = code_changes(revision, record["git_sha"], root)
            if changed:
                errors.append(f"{name} ran at {record['git_sha']}, whose code differs in {listed(changed)}")
    if errors:
        raise ProvenanceError(f"the records ran different code from {revision}: " + "; ".join(errors))
    changed = code_changes(revision, None, root)
    if changed:
        raise ProvenanceError(
            f"the records ran at {revision}, and the checkout's code has changed since, "
            f"in {listed(changed)}; rerun the seeds or aggregate at that commit"
        )
    return revision
