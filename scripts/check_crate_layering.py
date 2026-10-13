#!/usr/bin/env python3
"""Enforce the crate layering declared in research/catalogs/crate-layers.toml.

Rules, checked on the direct normal and build dependencies of every workspace member
(optional and target-specific ones included, `dev-dependencies` exempt):

1. A crate depends only on crates of the same or a lower layer rank.
2. A backend crate is a direct dependency only of its declared owners.
3. Every `crates/*` and `bins/*` workspace member is placed in exactly one layer.

Known violations live in `[[allowed_exception]]`; an exception that no longer matches a
violation is an error, so the list can only shrink.

The source is `cargo metadata --locked --no-deps --format-version 1`: the declared
dependencies are all the rules need, so nothing is resolved or downloaded.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research" / "catalogs" / "crate-layers.toml"
PLACED_DIRS = ("crates", "bins")
CHECKED_KINDS = (None, "build")


def load_catalog(path: Path = CATALOG) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def read_metadata(root: Path = ROOT) -> dict:
    command = ["cargo", "+stable", "metadata", "--locked", "--no-deps", "--format-version", "1"]
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def workspace_members(metadata: dict) -> dict[str, dict]:
    """Workspace packages by name, each with its direct checked dependencies."""
    root = Path(metadata["workspace_root"])
    member_ids = set(metadata["workspace_members"])
    members: dict[str, dict] = {}
    for package in metadata["packages"]:
        if package["id"] not in member_ids:
            continue
        manifest = Path(package["manifest_path"])
        try:
            top = manifest.relative_to(root).parts[0]
        except ValueError:
            top = ""
        members[package["name"]] = {
            "placed": top in PLACED_DIRS,
            "deps": sorted(
                {dep["name"] for dep in package["dependencies"] if dep.get("kind") in CHECKED_KINDS}
            ),
        }
    return members


def backend_matches(pattern: str, name: str) -> bool:
    return fnmatch.fnmatchcase(name, pattern)


def check(metadata: dict, catalog: dict) -> list[str]:
    errors: list[str] = []
    members = workspace_members(metadata)

    rank_of: dict[str, int] = {}
    for layer in catalog.get("layer", []):
        for crate in layer["crates"]:
            if crate in rank_of:
                errors.append(f"{crate} is placed in more than one layer")
            rank_of[crate] = layer["rank"]
    layer_name = {layer["rank"]: layer["name"] for layer in catalog.get("layer", [])}

    for crate in sorted(rank_of):
        if crate not in members:
            errors.append(f"{crate} is placed in crate-layers.toml but is not a workspace member")
    for name, member in sorted(members.items()):
        if member["placed"] and name not in rank_of:
            errors.append(f"{name} is not placed in any layer of crate-layers.toml")

    backends = catalog.get("backend_owner", [])
    for entry in backends:
        for owner in entry["owners"]:
            if owner not in members:
                errors.append(f"backend {entry['crate']}: owner {owner} is not a workspace member")

    violations: dict[tuple[str, str], str] = {}
    for name, member in sorted(members.items()):
        if name not in rank_of:
            continue
        for dep in member["deps"]:
            if dep in members:
                if dep in rank_of and rank_of[dep] > rank_of[name]:
                    violations[(name, dep)] = (
                        f"{name} ({layer_name[rank_of[name]]}) depends on {dep} "
                        f"({layer_name[rank_of[dep]]}), a higher layer"
                    )
                continue
            for entry in backends:
                if backend_matches(entry["crate"], dep) and name not in entry["owners"]:
                    owners = ", ".join(entry["owners"]) or "no crate"
                    violations[(name, dep)] = (
                        f"{name} depends on backend crate {dep}; allowed owners: {owners}"
                    )

    allowed: set[tuple[str, str]] = set()
    for exception in catalog.get("allowed_exception", []):
        key = (exception["from"], exception["to"])
        if not exception.get("reason", "").strip():
            errors.append(f"allowed_exception {key[0]} -> {key[1]} has no reason")
        if key in allowed:
            errors.append(f"allowed_exception {key[0]} -> {key[1]} is listed twice")
        allowed.add(key)
        if key not in violations:
            errors.append(
                f"allowed_exception {key[0]} -> {key[1]} no longer matches a violation; "
                "remove it from crate-layers.toml"
            )
    for key, message in violations.items():
        if key not in allowed:
            errors.append(message)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="read cargo metadata JSON from this file")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args(argv)
    try:
        if args.metadata:
            metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        else:
            metadata = read_metadata()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: cannot read cargo metadata: {error}")
        return 1
    errors = check(metadata, load_catalog(args.catalog))
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print("OK: crate layering, backend ownership and layer placement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
