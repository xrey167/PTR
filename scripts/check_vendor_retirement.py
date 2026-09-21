"""Check that every retained downstream patch records why it exists and what would retire it.

`check_vendor_integrity.py` answers "did these files change since the upstream
archive". It says nothing about *why* a package is patched, so a new patch added
for an entirely different reason passes it while the prose describing the set
silently stops being true.

This checker closes that: `vendor/RETIREMENT.json` carries a per-package record,
and every field in it is re-derived from the tree rather than trusted:

* the package set must equal `vendor/ORIGINS.json` exactly, in both directions;
* the recorded workspace must be the one whose `[patch.crates-io]` actually
  patches that directory;
* a class with a `touches` allow-list must cover every file `CURRENT.json`
  records as changed, so a Cargo.toml-only class cannot quietly grow a source
  patch;
* a `paste-alias` record must have the alias in its vendored manifest.

Retirement candidates are reported from `upstream_observed`. That field is a
recorded observation, not a live lookup: the packages here are patched out of
the registry, so cargo never caches their index entries, and this checker does
no network. `--require-observations` fails on entries nobody has refreshed,
which is a maintainer's gate rather than CI's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Manifest path relative to the tree root, and the prefix its patch paths carry.
WORKSPACE_MANIFESTS = {
    "root": ("Cargo.toml", ""),
    "burn-a0": ("model/burn-a0/Cargo.toml", "../../"),
}
PASTE_ALIAS_MARKER = 'package = "pastey"'


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def patched_directories(manifest: Path, prefix: str) -> set[str]:
    """Directories the manifest's [patch.crates-io] section points at."""
    text = manifest.read_text(encoding="utf-8")
    if "[patch.crates-io]" not in text:
        return set()
    section = re.split(r"\n\[(?!patch\.crates-io])", text.split("[patch.crates-io]", 1)[1], 1)[0]
    found = set()
    for match in re.finditer(r'path\s*=\s*"([^"]+)"', section):
        path = match.group(1)
        found.add(path[len(prefix):] if prefix and path.startswith(prefix) else path)
    return found


def release_key(version: str) -> tuple:
    """Order versions so a pre-release sorts below the release it precedes."""
    core, _, pre = version.partition("-")
    numbers = tuple(int(part) if part.isdigit() else 0 for part in core.split("."))
    # An absent pre-release outranks any present one, as SemVer requires.
    return (numbers, 1, ()) if not pre else (numbers, 0, tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"[.]", pre)
    ))


def check(root: Path, require_observations: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    origins = load(root / "vendor/ORIGINS.json")
    current = load(root / "vendor/CURRENT.json")
    document = load(root / "vendor/RETIREMENT.json")

    if document.get("schema") != "1":
        errors.append("vendor/RETIREMENT.json: unsupported schema")
        return errors, notes

    classes = document.get("classes") or {}
    for name, entry in classes.items():
        for field in ("change", "retire_when"):
            if not (entry.get(field) or "").strip():
                errors.append(f"class {name}: empty {field}")

    changed_files = {
        package["directory"]: set(package.get("changes") or {})
        for package in current.get("packages", [])
    }
    # Every declared workspace stays in the vocabulary even when its manifest is
    # gone, so a record naming it reports the missing manifest rather than
    # degrading into a misleading "unknown workspace".
    patched = {}
    for workspace, (manifest, prefix) in WORKSPACE_MANIFESTS.items():
        path = root / manifest
        if not path.exists():
            errors.append(f"workspace {workspace}: {manifest} is missing")
            patched[workspace] = set()
            continue
        patched[workspace] = patched_directories(path, prefix)

    recorded = {}
    for package in document.get("packages", []):
        key = (package["name"], package["version"])
        if key in recorded:
            errors.append(f"{key[0]} {key[1]}: recorded twice")
        recorded[key] = package

    expected = {
        (origin["name"], origin["version"]): origin.get(
            "directory", "vendor/" + origin["name"]
        )
        for origin in origins
    }

    for key in sorted(expected.keys() - recorded.keys()):
        errors.append(f"{key[0]} {key[1]}: vendored with no retirement record")
    for key in sorted(recorded.keys() - expected.keys()):
        errors.append(f"{key[0]} {key[1]}: retirement record for a package that is not vendored")

    for key in sorted(expected.keys() & recorded.keys()):
        package = recorded[key]
        name, version = key
        directory = package.get("directory")
        if directory != expected[key]:
            errors.append(f"{name} {version}: directory {directory!r} is not {expected[key]!r}")
            continue

        workspace = package.get("workspace")
        if workspace not in patched:
            errors.append(f"{name} {version}: unknown workspace {workspace!r}")
        elif directory not in patched[workspace]:
            owners = sorted(w for w, dirs in patched.items() if directory in dirs)
            errors.append(
                f"{name} {version}: recorded under {workspace!r} but patched by "
                + (", ".join(owners) if owners else "no workspace")
            )

        entry = classes.get(package.get("class"))
        if entry is None:
            errors.append(f"{name} {version}: unknown class {package.get('class')!r}")
            continue

        allowed = entry.get("touches")
        if allowed is not None:
            beyond = sorted(changed_files.get(directory, set()) - set(allowed))
            if beyond:
                errors.append(
                    f"{name} {version}: class {package['class']} allows only "
                    f"{sorted(allowed)} but {beyond} changed"
                )

        if package["class"] == "paste-alias":
            manifest = root / directory / "Cargo.toml"
            if PASTE_ALIAS_MARKER not in manifest.read_text(encoding="utf-8"):
                errors.append(
                    f"{name} {version}: recorded as paste-alias but its manifest "
                    "does not alias pastey"
                )

        observed = package.get("upstream_observed")
        if observed is None:
            if require_observations:
                errors.append(f"{name} {version}: no upstream version has been observed")
            else:
                notes.append(f"{name} {version}: upstream unobserved")
        elif release_key(observed) > release_key(version):
            notes.append(
                f"{name} {version}: RETIREMENT CANDIDATE, upstream {observed} observed "
                f"on {package.get('observed_at')} — {entry['retire_when']}"
            )

    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-observations",
        action="store_true",
        help="fail when a package has no recorded upstream observation",
    )
    args = parser.parse_args()

    errors, notes = check(ROOT, args.require_observations)
    for note in notes:
        print(f"note: {note}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: {len(load(ROOT / 'vendor/ORIGINS.json'))} retained patches carry a retirement record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
