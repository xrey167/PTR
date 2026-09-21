"""Record, from the crates.io sparse index, the newest published version of each retained patch.

This is the online half of the retirement workflow and is deliberately not a
checker: it writes observations, it does not pass or fail. `check_vendor_retirement.py`
stays offline and reports candidates from what this recorded, so CI never depends
on the network and a report is reproducible from the repository alone.

Yanked versions are excluded — a yanked release is not something a patch can move
to — and "newest" is the highest version by precedence, not the last line of the
index, which is publication order and can end on a backport.
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
from check_vendor_retirement import release_key  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be recorded"
    )
    args = parser.parse_args()

    path = ROOT / "vendor/RETIREMENT.json"
    document = json.loads(path.read_text(encoding="utf-8"))
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
        marker = " CANDIDATE" if release_key(observed) > release_key(pinned) else ""
        print(f"{name} {pinned}: newest published {observed}{marker}")
        if not args.dry_run:
            package["upstream_observed"] = observed
            package["observed_at"] = today

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
