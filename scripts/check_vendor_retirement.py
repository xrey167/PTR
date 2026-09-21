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

**A newer version is not a retirement.** The first form of this checker reported a
candidate whenever `upstream_observed` outranked the pinned version, which asks
about a version number and not about the condition actually recorded in
`retire_when`. It produced two false actionables:

* `macerator 0.4.0` was reported as a candidate although it still depends on
  `paste ^1`, so the alias it would retire has exactly as much left to redirect
  as before. It never satisfied `retire_when` at all.
* `netlink-packet-core 0.9.0` *does* satisfy it — the release is paste-free — but
  four dependents (`netdev`, `netlink-packet-route`, `netlink-proto`, `netwatch`)
  declare `^0.8.x`, which admits nothing in `0.9`. The move is real and its
  blocker is elsewhere.

So a package with a newer upstream carries two further fields, and this checker
reports three states rather than one: `upstream_retire_when` says whether the
observed release satisfies the recorded condition, and `blocked_by` names the
dependents whose requirements exclude it.

**What is re-derived and what is observed.** A dependent's *existence* is a fact
this repository holds: `Cargo.lock` records it, so every lockfile-visible
dependent must appear in `blocked_by` at the version the lock pins, in both
directions. A dependent's *requirement* is not — a lockfile records resolved
versions, never requirements, and these dependents are not vendored — so the
requirement string is an observation like `upstream_observed`, refreshed by
`refresh_vendor_upstream.py`. A dependent that is optional and currently
inactive (`burn-flex` requires `macerator ^0.3.4`) does not appear in the
lockfile at all and is recorded with `in_lockfile = false`; it constrains a bump
regardless of whether its dependency is selected today.

**Raising a vendored version is not retiring a patch.** Both patches use the
renamed-key form, which patches one specific version, so bumping the vendored
copy past what a dependent requires leaves that dependent on the *unpatched*
upstream while cargo merely notes that the patch went unused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Manifest path relative to the tree root, and the prefix its patch paths carry.
WORKSPACE_MANIFESTS = {
    "root": ("Cargo.toml", ""),
    "burn-a0": ("model/burn-a0/Cargo.toml", "../../"),
}
PASTE_ALIAS_MARKER = 'package = "pastey"'
# The lockfile that records what each workspace actually resolved.
WORKSPACE_LOCKFILES = {
    "root": "Cargo.lock",
    "burn-a0": "model/burn-a0/Cargo.lock",
}
RETIRE_WHEN_STATES = ("satisfied", "unsatisfied", "unknown")


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


def lockfile_dependents(lock: Path, name: str, version: str) -> set[tuple[str, str]]:
    """Packages the lockfile shows depending on this exact name and version.

    A lock entry names a dependency as `"name"` when only one version of it is in
    the graph and as `"name version"` when several are, so both forms are read.
    An *optional* dependency that is not selected does not appear here at all,
    which is why `blocked_by` may legitimately name more than this returns.
    """
    if not lock.exists():
        return set()
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    unique = len({p["version"] for p in packages if p["name"] == name}) == 1
    found = set()
    for package in packages:
        for entry in package.get("dependencies", []):
            parts = entry.split()
            if parts[0] != name:
                continue
            if (len(parts) == 2 and parts[1] == version) or (len(parts) == 1 and unique):
                found.add((package["name"], package["version"]))
    return found


def caret_bounds(version: tuple[int, ...]) -> tuple[int, ...]:
    """The exclusive upper bound of a caret requirement, as cargo defines it.

    Leading zeros narrow the range: `^1.2.3` reaches into all of `1.x`, `^0.8`
    stops before `0.9`, and `^0.0.3` stops before `0.0.4`.
    """
    major, minor, patch = version
    if major > 0:
        return (major + 1, 0, 0)
    if minor > 0:
        return (0, minor + 1, 0)
    return (0, 0, patch + 1)


def requirement_admits(requirement: str, version: str):
    """Whether a caret requirement admits a version. `None` when unrecognised.

    Deliberately narrow: every requirement in play is a caret, and a requirement
    form this does not understand is reported as unknown rather than guessed at,
    because guessing here produces exactly the false actionable this checker was
    changed to stop producing.
    """
    text = requirement.strip()
    if not text.startswith("^"):
        return None
    if "-" in version:  # a caret without a pre-release never matches one
        return None
    try:
        wanted = tuple(int(part) for part in text[1:].split("."))
        actual = tuple(int(part) for part in version.split("."))
    except ValueError:
        return None
    wanted = wanted + (0,) * (3 - len(wanted))
    actual = actual + (0,) * (3 - len(actual))
    if len(wanted) != 3 or len(actual) != 3:
        return None
    return wanted <= actual < caret_bounds(wanted)


def check_blockers(root: Path, package: dict, errors: list[str]) -> list[str]:
    """Validate `blocked_by` against the lockfile and return the blocking lines.

    Existence is re-derived, requirements are observations. Both directions are
    checked, so a dependent that appears in the lock and not in the record fails
    just as loudly as one recorded that the lock does not show.
    """
    name, version = package["name"], package["version"]
    observed = package["upstream_observed"]
    lock = root / WORKSPACE_LOCKFILES.get(package.get("workspace"), "Cargo.lock")
    seen = lockfile_dependents(lock, name, version)

    recorded = package.get("blocked_by")
    if not isinstance(recorded, list):
        errors.append(f"{name} {version}: newer upstream but no blocked_by list")
        return []

    named = set()
    blocking = []
    for entry in recorded:
        if not isinstance(entry, dict) or not {"name", "version", "requirement"} <= set(entry):
            errors.append(f"{name} {version}: blocked_by entry needs name, version, requirement")
            continue
        key = (entry["name"], entry["version"])
        named.add(key)
        in_lock = key in seen
        if entry.get("in_lockfile", True) != in_lock:
            errors.append(
                f"{name} {version}: blocked_by {entry['name']} {entry['version']} records "
                f"in_lockfile={entry.get('in_lockfile', True)} but {lock.name} "
                f"{'shows' if in_lock else 'does not show'} it"
            )
        admits = requirement_admits(entry["requirement"], observed)
        if admits is None:
            blocking.append(f"{entry['name']} {entry['version']} at {entry['requirement']} (unread)")
        elif not admits:
            blocking.append(f"{entry['name']} {entry['version']} at {entry['requirement']}")
    for missing in sorted(seen - named):
        errors.append(
            f"{name} {version}: {lock.name} shows {missing[0]} {missing[1]} depending on it, "
            "and blocked_by does not name it"
        )
    return blocking


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
        # A class says whether its retirement condition has a mechanical signal.
        # `paste-alias` does - the upstream release either still depends on a paste
        # macro crate or does not. `raft-protobuf` does not: "carries that
        # migration" is something a person reads, and a class that pretended
        # otherwise would be back to deciding retirement from a version number.
        if "retire_when_signal" not in entry:
            errors.append(f"class {name}: no retire_when_signal (use null when there is none)")

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
            when = package.get("upstream_retire_when")
            if when not in RETIRE_WHEN_STATES:
                errors.append(
                    f"{name} {version}: newer upstream {observed} but "
                    f"upstream_retire_when is {when!r}, not one of {list(RETIRE_WHEN_STATES)}"
                )
                continue
            blocking = check_blockers(root, package, errors)
            seen_on = package.get("observed_at")
            if when == "unsatisfied":
                # The case that produced a false actionable: a higher version
                # number that retires nothing, because the condition recorded in
                # `retire_when` is about the release and not about its number.
                notes.append(
                    f"{name} {version}: NOT A CANDIDATE. Upstream {observed} (seen {seen_on}) "
                    f"does not satisfy retire_when - {entry['retire_when']}"
                )
            elif when == "unknown":
                notes.append(
                    f"{name} {version}: upstream {observed} (seen {seen_on}) needs reading - "
                    f"this class has no mechanical signal for {entry['retire_when']}"
                )
            elif blocking:
                notes.append(
                    f"{name} {version}: BLOCKED. Upstream {observed} (seen {seen_on}) satisfies "
                    f"retire_when, but {len(blocking)} dependent(s) exclude it: "
                    + "; ".join(blocking)
                    + ". Retiring this patch means moving them, not bumping the vendored copy."
                )
            else:
                notes.append(
                    f"{name} {version}: RETIREMENT CANDIDATE. Upstream {observed} (seen "
                    f"{seen_on}) satisfies retire_when and no dependent excludes it - "
                    f"{entry['retire_when']}"
                )

    dates = sorted(
        package["observed_at"]
        for package in document.get("packages", [])
        if package.get("observed_at")
    )
    if dates:
        # Staleness is reported, never failed on: a check that goes red with the
        # passage of time fails commits that changed nothing.
        notes.append(f"{len(dates)} observations recorded, oldest {dates[0]}")

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
