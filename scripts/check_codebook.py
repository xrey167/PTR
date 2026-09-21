"""Check the generated codebook artifact for self-consistency, offline.

`crates/ptr-types/tests/codebook_artifact.rs` is what catches drift between the
kernel and this file; it needs cargo. This checker needs nothing but the file and
answers the question that remains: is the fingerprint the digest of the bytes
recorded beside it, and does the document describe what it claims to?

A hand-edited fingerprint is the failure worth catching here. Every dataset is
checked against it, so a wrong one either rejects valid data or admits data
produced under a table that has moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "datasets/generated/codebook.json"


def tracked(path: Path) -> bool:
    """Whether git has this file, rather than merely this machine.

    The artifact existing locally is not the same as it being in the repository,
    and the difference is invisible to every other check here. `datasets/generated`
    is ignored wholesale, so the first version of this artifact was generated,
    validated, committed around and never actually committed — four CI jobs then
    failed on a file that only ever existed on one machine. Nothing else notices
    that, because everything else reads the working tree.
    """
    try:
        return (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", str(path)],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
    except OSError:
        # No git available: say nothing rather than claim the file is untracked.
        return True


def check(path: Path, require_tracked: bool = False) -> list[str]:
    if not path.is_file():
        return [f"{path} is missing; run scripts/generate_codebook.py"]

    if require_tracked and not tracked(path):
        return [
            f"{path} is not tracked by git, so nothing outside this machine can "
            "read or check it; add it (it is deliberately excepted from the "
            "datasets/generated ignore rule)"
        ]

    document = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if document.get("schema") != "1":
        errors.append("unsupported schema")
        return errors

    hex_bytes = document.get("canonical_bytes_hex")
    if not isinstance(hex_bytes, str) or not hex_bytes:
        errors.append("canonical_bytes_hex is absent")
        return errors
    try:
        canonical = bytes.fromhex(hex_bytes)
    except ValueError:
        errors.append("canonical_bytes_hex is not hex")
        return errors

    expected = hashlib.sha256(canonical).hexdigest()
    if document.get("fingerprint_sha256") != expected:
        errors.append(
            "fingerprint_sha256 is not the digest of canonical_bytes_hex; "
            "run scripts/generate_codebook.py"
        )

    families = document.get("families")
    if not isinstance(families, list) or not families:
        errors.append("families is absent or empty")
        return errors

    for family in families:
        name = family.get("family")
        members = family.get("members")
        if not isinstance(name, str) or not name:
            errors.append("a family has no name")
            continue
        if not isinstance(members, list):
            errors.append(f"{name}: members is not a list")
            continue
        if family.get("cardinality") != len(members):
            errors.append(
                f"{name}: cardinality {family.get('cardinality')} does not match "
                f"{len(members)} members"
            )
        # Codes index an embedding table, so they must be exactly 0..n with no
        # gap and no repeat; a gap would leave a row nothing can reach.
        codes = [member.get("code") for member in members]
        if codes != list(range(len(members))):
            errors.append(f"{name}: codes are not dense 0..{len(members) - 1}: {codes}")
        names = [member.get("name") for member in members]
        if len(set(names)) != len(names):
            errors.append(f"{name}: duplicate member names")
        # The canonical bytes commit to every member name, so a name present here
        # and absent there means the document was edited rather than generated.
        for member in names:
            if isinstance(member, str) and member.encode("utf-8") not in canonical:
                errors.append(f"{name}: {member!r} is not in the canonical bytes")

    errors.extend(check_exceptions(document, canonical, {f.get("family") for f in families}))
    return errors


def check_exceptions(document: dict, canonical: bytes, family_names: set) -> list[str]:
    """Widths that model tables are sized by and the taxonomy deliberately omits.

    A recorded exception is a decision; an unrecorded one is indistinguishable
    from an oversight, which is the whole reason this section exists. What is
    checked here is that the record is well formed and says why — that the width
    matches the model built from it is Rust's to assert, and
    `crates/ptr-types/tests/codebook_artifact.rs` ties this file to the kernel.
    """
    errors: list[str] = []
    exceptions = document.get("exceptions")
    if not isinstance(exceptions, list):
        return ["exceptions is absent; an empty list is how a kernel says it has none"]

    seen = set()
    for exception in exceptions:
        name = exception.get("name") if isinstance(exception, dict) else None
        if not isinstance(name, str) or not name:
            errors.append("an exception has no name")
            continue
        if name in seen:
            errors.append(f"{name}: recorded twice")
        seen.add(name)
        width = exception.get("width")
        # Zero rows is a table nothing can index, which is not an exception but a
        # mistake; a negative or non-integer width is not a table at all.
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            errors.append(f"{name}: width {width!r} is not a positive integer")
        if not (exception.get("reason") or "").strip():
            errors.append(f"{name}: no reason recorded, so it reads as an oversight")
        # An exception is by definition not a family. A name in both places would
        # mean the document says it is inside and outside the taxonomy at once.
        if name in family_names:
            errors.append(f"{name}: recorded as an exception and as a family")
        if name.encode("utf-8") in canonical:
            errors.append(
                f"{name}: appears in the canonical bytes, which commit to the code "
                "assignment; an exception assigns no codes and must stay outside it"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ARTIFACT)
    args = parser.parse_args()

    # The CLI, which is what CI runs, insists the artifact is in the repository.
    errors = check(args.path, require_tracked=True)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    document = json.loads(args.path.read_text(encoding="utf-8"))
    members = sum(len(family["members"]) for family in document["families"])
    print(
        f"OK: codebook version {document['version']}, "
        f"{len(document['families'])} families, {members} members, "
        f"{len(document.get('exceptions', []))} recorded exception(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
