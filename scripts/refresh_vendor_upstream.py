"""Record, from the crates.io sparse index, the newest published version of each retained patch.

This is the online half of the retirement workflow and is deliberately not a
checker: it writes observations, it does not pass or fail. `check_vendor_retirement.py`
stays offline and reports candidates from what this recorded, so CI never depends
on the network and a report is reproducible from the repository alone.

Yanked versions are excluded — a yanked release is not something a patch can move
to — and "newest" is the highest version by precedence, not the last line of the
index, which is publication order and can end on a backport.

For a package whose newest release outranks the pinned one it records two further
observations, because a version number does not decide a retirement:

* `upstream_retire_when` — whether that release satisfies the class's condition.
  For `paste-alias` the signal is mechanical: the release either still declares a
  dependency on the crate named `paste` or it does not. `macerator 0.4.0` does,
  which is why it was never the candidate the old report called it. A class whose
  `retire_when_signal` is null is recorded as `unknown`, because "carries that
  migration" is something a person reads.
* `blocked_by` — for each dependent, the requirement it declares on this package.
  The dependents themselves come from the lockfile, which is offline; their
  requirements come from the index, which is why this lives here. A dependent that
  is optional and not currently selected is absent from the lockfile and is kept
  with `in_lockfile = false` rather than dropped: `burn-flex` requires
  `macerator ^0.3.4` and constrains a bump whether or not it is selected today.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://index.crates.io"

sys.path.insert(0, str(ROOT / "scripts"))
from check_vendor_retirement import (  # noqa: E402
    WORKSPACE_LOCKFILES,
    lockfile_dependents,
    release_key,
)

# The crate the paste-alias class exists to redirect away from. `pastey`, the
# maintained drop-in, is the destination and is not this.
PASTE_CRATE = "paste"


def index_path(name: str) -> str:
    """The sparse index lays crates out by name length, lowercased."""
    lowered = name.lower()
    if len(lowered) <= 2:
        return f"{len(lowered)}/{lowered}"
    if len(lowered) == 3:
        return f"3/{lowered[0]}/{lowered}"
    return f"{lowered[:2]}/{lowered[2:4]}/{lowered}"


def unyanked_versions(body: str) -> list[str]:
    """Every version the index offers, minus the yanked ones."""
    versions = []
    for line in body.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("yanked"):
            continue
        versions.append(entry["vers"])
    return versions


def newest(body: str) -> str | None:
    """The highest version by precedence, not the last line.

    The index is in publication order, which can end on a backport to an older
    series, so taking the last line would record a version older than one
    already published.
    """
    versions = unyanked_versions(body)
    return max(versions, key=release_key) if versions else None


def fetch(name: str, timeout: float) -> str:
    url = f"{INDEX}/{index_path(name)}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def entry_for(body: str, version: str) -> dict | None:
    """The index line describing one exact version."""
    for line in body.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if parsed["vers"] == version:
            return parsed
    return None


def depends_on_paste(entry: dict) -> bool:
    return any(dep["name"] == PASTE_CRATE for dep in entry.get("deps", []))


def requirement_on(entry: dict, target: str) -> str | None:
    for dep in entry.get("deps", []):
        if dep["name"] == target:
            return dep["req"]
    return None


def refresh_blockers(package: dict, signal, timeout: float, failures: list[str]) -> None:
    """Record what the observed release owes, and who refuses it.

    The dependent set is re-derived from the lockfile every time, so a dependent
    that has disappeared stops being recorded and a new one is picked up. A
    recorded dependent the lockfile does not show is kept rather than dropped:
    an optional dependency that is not selected is invisible there and still
    constrains a bump.
    """
    name, pinned = package["name"], package["version"]
    observed = package["upstream_observed"]

    if signal == "paste-free":
        try:
            entry = entry_for(fetch(name, timeout), observed)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{name}: reading {observed}: {error}")
            return
        if entry is None:
            failures.append(f"{name}: the index does not list {observed}")
            return
        package["upstream_retire_when"] = (
            "unsatisfied" if depends_on_paste(entry) else "satisfied"
        )
    else:
        package["upstream_retire_when"] = "unknown"

    lock = ROOT / WORKSPACE_LOCKFILES.get(package.get("workspace"), "Cargo.lock")
    seen = lockfile_dependents(lock, name, pinned)
    existing = {(b["name"], b["version"]): b for b in package.get("blocked_by", [])}
    # Recorded-but-invisible entries survive; recorded-and-visible ones are
    # re-derived; anything else the lockfile no longer shows is dropped.
    keys = sorted(seen | {k for k, b in existing.items() if not b.get("in_lockfile", True)})

    blocked = []
    for dependent, dependent_version in keys:
        requirement = (existing.get((dependent, dependent_version)) or {}).get("requirement")
        try:
            entry = entry_for(fetch(dependent, timeout), dependent_version)
            if entry is None:
                failures.append(f"{dependent}: the index does not list {dependent_version}")
            else:
                found = requirement_on(entry, name)
                if found is None:
                    failures.append(
                        f"{dependent} {dependent_version}: declares no dependency on {name}"
                    )
                else:
                    requirement = found
        except Exception as error:  # noqa: BLE001
            failures.append(f"{dependent}: {error}")
        if requirement is None:
            continue
        blocked.append(
            {
                "name": dependent,
                "version": dependent_version,
                "requirement": requirement,
                "in_lockfile": (dependent, dependent_version) in seen,
            }
        )
    package["blocked_by"] = blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be recorded"
    )
    args = parser.parse_args()

    path = ROOT / "vendor/RETIREMENT.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    classes = document.get("classes") or {}
    today = dt.date.today().isoformat()
    failures = []

    for package in document["packages"]:
        name, pinned = package["name"], package["version"]
        try:
            observed = newest(fetch(name, args.timeout))
        except Exception as error:  # noqa: BLE001 - reported, never silently skipped
            failures.append(f"{name}: {error}")
            continue
        if observed is None:
            failures.append(f"{name}: the index lists no unyanked version")
            continue
        newer = release_key(observed) > release_key(pinned)
        print(f"{name} {pinned}: newest published {observed}{' NEWER' if newer else ''}")
        if args.dry_run:
            continue
        package["upstream_observed"] = observed
        package["observed_at"] = today
        if newer:
            signal = classes.get(package.get("class"), {}).get("retire_when_signal")
            refresh_blockers(package, signal, args.timeout, failures)
            print(
                f"  retire_when {package['upstream_retire_when']}, "
                f"{len(package.get('blocked_by', []))} dependent(s) recorded"
            )
        else:
            # No newer release, so there is nothing to ask either question about.
            package.pop("upstream_retire_when", None)
            package.pop("blocked_by", None)

    if not args.dry_run:
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"recorded {len(document['packages']) - len(failures)} observations on {today}")

    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    # An observation that could not be made is left as it was rather than
    # replaced with a guess; the checker keeps reporting it as unobserved.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
