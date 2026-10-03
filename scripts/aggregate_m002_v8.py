"""Strict M002-v8 decision adapter over the immutable matched-pair decider.

M002-v8 keeps M002-v5's statistic and gates but has runner-owned v8 arm
labels. This adapter accepts only v8 records, maps those labels in memory for
the immutable statistic, then restores the original v8 provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "M002-v8"
CORE_PATH = ROOT / "scripts/aggregate_m002_v5.py"
ARM_RENAMES = (
    ("factorized-v2-off-v8", "factorized-v2-off"),
    ("factorized-v2-v8", "factorized-v2"),
)


class EvidenceError(ValueError):
    """The candidate records cannot support an M002-v8 decision."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_core():
    spec = importlib.util.spec_from_file_location("_m002_v8_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise EvidenceError("cannot load the immutable M002-v5 decision core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v5_arm_labels(stdout: object, name: str) -> str:
    """Map only exact versioned v8 labels for the immutable v5 core."""
    if not isinstance(stdout, str):
        raise EvidenceError(f"{name}: stdout is not text")
    seen = re.findall(
        r"(?<![A-Za-z0-9-])(factorized-v2(?:-off)?-v8[A-Za-z0-9-]*)(?![A-Za-z0-9-])",
        stdout,
    )
    allowed = {source for source, _ in ARM_RENAMES}
    if any(label not in allowed for label in seen):
        raise EvidenceError(f"{name}: unrecognized v8 arm label")
    normalized = stdout
    for source, target in ARM_RENAMES:
        normalized = normalized.replace(source, target)
    if "-v8" in normalized:
        raise EvidenceError(f"{name}: unrecognized v8 arm label")
    return normalized


def normalized_records(records: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Adapt v8 identity and labels only for the version-specific core checker."""
    normalized = []
    for name, record in records:
        if not isinstance(record, dict) or record.get("experiment_id") != EXPERIMENT_ID:
            raise EvidenceError(f"{name}: experiment_id is not {EXPERIMENT_ID}")
        manifest = record.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("id") != EXPERIMENT_ID:
            raise EvidenceError(f"{name}: recorded manifest identity is not {EXPERIMENT_ID}")
        normalized.append((
            name,
            {
                **record,
                "experiment_id": "M002-v5",
                "manifest": {**manifest, "id": "M002-v5"},
                "stdout": v5_arm_labels(record.get("stdout"), name),
            },
        ))
    return normalized


def decide(records: list[tuple[str, dict]], root: Path = ROOT, protocol: dict | None = None) -> dict:
    """Apply the unchanged conjunction of confirmatory evidence gates."""
    core = load_core()
    try:
        result = core.decide(normalized_records(records), root, protocol)
    except core.EvidenceError as error:
        raise EvidenceError(str(error)) from None
    result["experiment_id"] = EXPERIMENT_ID
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError("decision core returned malformed provenance")
    provenance["decision_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance["source_records"] = [
        {"path": name, "canonical_sha256": hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()}
        for name, record in sorted(records)
    ]
    return result


def load_records(paths: list[Path]) -> list[tuple[str, dict]]:
    records = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceError(f"{path}: unreadable run record: {error}") from None
        try:
            name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            name = path.name
        records.append((name, value))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = decide(load_records(args.records))
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
