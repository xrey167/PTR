from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require component metadata changes when implementation files change."
    )
    parser.add_argument("--base", required=True, help="Base commit/ref to compare with HEAD.")
    args = parser.parse_args()

    try:
        changed = git("diff", "--name-only", f"{args.base}...HEAD").splitlines()
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"ERROR: unable to diff against {args.base}; component freshness is unverified: {error}")
        return 1

    changed_set = set(changed)
    implementation_changed: dict[str, list[str]] = {}

    for path in changed:
        parts = Path(path).parts
        if len(parts) < 3 or parts[0] != "crates":
            continue
        crate = parts[1]
        is_source = len(parts) >= 4 and parts[2] == "src"
        is_manifest = len(parts) == 3 and parts[2] == "Cargo.toml"
        if is_source or is_manifest:
            implementation_changed.setdefault(crate, []).append(path)

    stale = []
    for crate, files in sorted(implementation_changed.items()):
        meta = f"crates/{crate}/component.toml"
        if meta not in changed_set:
            stale.append((crate, files, meta))

    if stale:
        print("Component implementation changed without implementation-status metadata update:")
        for crate, files, meta in stale:
            print(f"\n{crate}: update {meta}")
            for path in files:
                print(f"  - {path}")
        print(
            "\nAfter updating component.toml, run: "
            "python3 scripts/update_component_docs.py --write"
        )
        return 1

    print(f"OK: metadata freshness verified for {len(implementation_changed)} changed components")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
