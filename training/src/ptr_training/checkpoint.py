"""A run that produced a checkpoint has to record the seal produced with it.

A checkpoint carries the assignment it was trained under. It does not carry what
it was trained *from* — the committed position, the values that were read, the
generations that were live. `ptrctl seal` attaches those and writes two files: the
sealed artifact and, separately, its anchor.

Separately matters. A digest read back out of the artifact it describes proves
nothing, so the anchor is evidence only while something else retains it. The run
manifest is that something else, which is why this module refuses a run that names
a checkpoint and stops there.

There is no defaulting path. Each refusal below names one missing or disagreeing
thing, because "the seal is probably next to it" is how an unbound artifact gets
recorded as a bound one.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path


class SealError(ValueError):
    """A declared checkpoint whose seal is missing, unreadable or not its own."""


def _fingerprint(root: Path, declared: str, *, field: str, where: str) -> dict:
    path = root / declared
    if not path.is_file():
        raise SealError(f"{where}: {field} names a file that is not there: {declared}")

    data = path.read_bytes()
    return {
        "path": declared,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def require_seal(section: dict, root: Path, *, where: str) -> dict | None:
    """Resolve a run config's ``[checkpoint]`` section, or refuse it.

    Returns ``None`` when the run declares no checkpoint at all — a run that
    trains nothing it keeps is not thereby wrong. Returns the three fingerprints
    when it declares one, so they reach the run's input fingerprint rather than
    merely sitting beside it.
    """
    if not section:
        return None

    artifact = section.get("artifact")
    sealed = section.get("sealed")
    anchor = section.get("anchor")

    # Stated one at a time. A single "incomplete [checkpoint]" would not say which
    # half of the seal the run is missing.
    if not artifact:
        raise SealError(f"{where}: a checkpoint section must name its artifact")
    if not sealed:
        raise SealError(
            f"{where}: {artifact} is declared with no sealed artifact; "
            "a checkpoint that was never bound records nothing about what it was trained from"
        )
    if not anchor:
        raise SealError(
            f"{where}: {sealed} is declared with no anchor; "
            "a sealed artifact whose anchor is not retained elsewhere cannot be reopened"
        )

    artifact_fp = _fingerprint(root, artifact, field="artifact", where=where)
    sealed_fp = _fingerprint(root, sealed, field="sealed", where=where)
    anchor_fp = _fingerprint(root, anchor, field="anchor", where=where)

    try:
        retained = tomllib.loads((root / anchor).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise SealError(f"{where}: {anchor} is not readable as an anchor: {error}") from error

    for field in ("sealed", "codebook", "journal_index", "journal_digest", "digest"):
        if field not in retained:
            raise SealError(f"{where}: {anchor} records no {field}")

    # The anchor names the artifact it describes. Without this, a run could retain
    # the anchor of some *other* sealed state and every field would still parse.
    named = Path(retained["sealed"]).name
    if named != Path(sealed).name:
        raise SealError(
            f"{where}: {anchor} describes {named}, not the declared {Path(sealed).name}"
        )

    return {
        "artifact": artifact_fp,
        "sealed": sealed_fp,
        "anchor": anchor_fp,
        "codebook": retained["codebook"],
        "journal_index": retained["journal_index"],
        "journal_digest": retained["journal_digest"],
        "digest": retained["digest"],
    }
