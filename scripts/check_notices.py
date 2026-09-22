"""Check that THIRD-PARTY-NOTICES.md still describes the dependency graph.

The notices file is only worth committing if it cannot go stale unnoticed, which
is the same failure the generated component docs already guard against. This
checker recomputes the third-party package set from the owned workspaces'
lockfiles and compares it against what the document records, naming every
difference in both directions.

It reads no crate archives and touches no network, so it runs in the ordinary
invariants job. Regenerating is a separate, heavier step that needs the pinned
archives.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_notices import collect_packages, package_set_digest  # noqa: E402

import json  # noqa: E402

HEADING = re.compile(r"^\| ([^|\s]+) \| ([^|\s]+) \|")
DIGEST = re.compile(r"^`Package-set-digest: sha256:([0-9a-f]{64})`$", re.MULTILINE)


def expected_packages(root: Path) -> set[tuple[str, str]]:
    origins = json.loads((root / "vendor/ORIGINS.json").read_text(encoding="utf-8"))
    vendored = {(origin["name"], origin["version"]) for origin in origins}
    return {
        key
        for key, entry in collect_packages(root).items()
        if entry["source"] is not None or key in vendored
    }


def recorded_packages(document: str) -> set[tuple[str, str]]:
    found = set()
    inside = False
    for line in document.splitlines():
        if line.startswith("| Package | Version |"):
            inside = True
            continue
        if inside:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            match = HEADING.match(line)
            if match:
                found.add((match.group(1), match.group(2)))
    return found


def check(root: Path, path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path.name} is missing; run scripts/generate_notices.py"]

    document = path.read_text(encoding="utf-8")
    expected = expected_packages(root)
    recorded = recorded_packages(document)

    errors = []
    for name, version in sorted(expected - recorded):
        errors.append(f"{name} {version}: in a lockfile but not in the notices")
    for name, version in sorted(recorded - expected):
        errors.append(f"{name} {version}: in the notices but in no lockfile")

    match = DIGEST.search(document)
    if not match:
        errors.append("no Package-set-digest line; the document was not generated")
    elif not errors and match.group(1) != package_set_digest(expected):
        # Only meaningful once the sets agree; otherwise it restates the above.
        errors.append("the Package-set-digest does not match the recorded packages")

    if errors:
        errors.append("run scripts/generate_notices.py after `cargo fetch --locked`")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "THIRD-PARTY-NOTICES.md")
    args = parser.parse_args()

    errors = check(ROOT, args.path)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: {len(expected_packages(ROOT))} third-party packages carry notices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
