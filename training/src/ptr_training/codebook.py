"""The kernel's codebook, as the training stack reads it.

One loader for every consumer here. The member sets, the version and the
assignment fingerprint all come from `datasets/generated/codebook.json`, which
`scripts/generate_codebook.py` emits from the Rust tables themselves. A second
copy of any of it — a retyped member list, a pasted fingerprint — is the drift the
codebook exists to prevent, and a copy cannot notice when it stops being true.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = ROOT / "datasets/generated/codebook.json"


def load(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Read the generated artifact, with member sets by family.

    A missing artifact is an error rather than a fallback: nothing here can be
    checked without it, and checking nothing silently is the failure mode.
    """
    if not path.is_file():
        raise ValueError(f"{path} is missing; run scripts/generate_codebook.py")
    document = json.loads(path.read_text(encoding="utf-8"))
    for field in ("version", "fingerprint_sha256", "canonical_bytes_hex", "families"):
        if field not in document:
            raise ValueError(f"{path} has no {field}")
    recomputed = hashlib.sha256(
        bytes.fromhex(document["canonical_bytes_hex"])
    ).hexdigest()
    if recomputed != document["fingerprint_sha256"]:
        # The artifact disagrees with itself, so neither field can be trusted as
        # the identity of anything.
        raise ValueError(
            f"{path}: fingerprint_sha256 is not the digest of canonical_bytes_hex"
        )
    document["members"] = {
        family["family"]: {member["name"] for member in family["members"]}
        for family in document["families"]
    }
    return document


def require_identity(
    obj: dict[str, Any],
    document: dict[str, Any],
    *,
    version_field: str = "type_codebook_version",
    fingerprint_field: str = "type_codebook_fingerprint",
    where: str = "record",
) -> None:
    """Require both identity fields and check them against `document`.

    Both are mandatory and there is no defaulting path. Something that names no
    codebook cannot be checked against one, and treating "absent" as "current" is
    how a stale integer assignment gets adopted without anyone deciding to.
    """
    version = obj.get(version_field)
    if version is None:
        raise ValueError(f"{where}: {version_field} is required")
    if version != document["version"]:
        raise ValueError(
            f"{where}: {version_field} {version!r} is not this build's "
            f"{document['version']!r}"
        )

    fingerprint = obj.get(fingerprint_field)
    if fingerprint is None:
        raise ValueError(f"{where}: {fingerprint_field} is required")
    if fingerprint != document["fingerprint_sha256"]:
        # A distinct reason from an unknown version: the version can be right
        # while the table behind it has moved, which is the silent remapping the
        # fingerprint exists to catch.
        raise ValueError(
            f"{where}: {fingerprint_field} does not match this build's assignment"
        )


def artifact_fingerprint(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """The artifact file's own identity, for a run's provenance."""
    if not path.is_file():
        raise ValueError(f"{path} is missing; run scripts/generate_codebook.py")
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
