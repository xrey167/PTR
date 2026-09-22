"""Repository-shape invariants: required files, parseable data, per-crate documentation.

Structured as `check(root)` rather than as a script body so that the workspace
membership rule below can be driven against a fixture tree. Every other checker in
this directory already has a test; this one had none, and the rule it gained is
precisely the kind that is only ever wrong about a directory nobody thought of.
"""

from __future__ import annotations

import fnmatch
import json
import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "target", "vendor", ".venv", "node_modules"}
# Directories whose sub-directories are Cargo packages, so a package appearing
# here must be accounted for by the workspace.
PACKAGE_AREAS = ("crates", "bins")
REQUIRED = [
    "Cargo.toml",
    "README.md",
    "docs/DEFINITION_OF_DONE.md",
    "experiments/registry.toml",
    "datasets/registry.toml",
    "crates/ptr-types/src/lib.rs",
    "crates/ptr-semdb/src/lib.rs",
    "crates/ptr-core/src/lib.rs",
    "evaluations/README.md",
    "docs/components/STATUS.md",
    "scripts/update_component_docs.py",
    "docs/RUST_API_STYLE.md",
    "research/catalogs/rust-api-layout.toml",
    "scripts/check_architecture_catalog.py",
    "scripts/check_rust_conventions.py",
    "scripts/report_rust_api.py",
    "docs/TESTING.md",
    "templates/rust-crate/Cargo.toml",
]
COMPONENT_KEYS = [
    "maturity",
    "last_reviewed",
    "implemented",
    "missing",
    "next",
    "experiments",
    "evaluations",
    "decisions",
    "checks",
]


def owned(root, paths):
    return (p for p in paths if not SKIP.intersection(p.relative_to(root).parts))


def matches(relative: str, pattern: str) -> bool:
    """Cargo's member/exclude globs: anchored at the workspace root, segment by segment.

    `PurePosixPath.match` was the wrong tool and produced a false pass, which is
    the one direction this check must not fail in. It matches a pattern with no
    separator against the *end* of the path, so `members = ["ptrctl"]` read as
    covering `bins/ptrctl`. Cargo resolves a member path from the workspace root:
    put to `cargo metadata`, that manifest fails with "failed to read
    <root>/ptrctl/Cargo.toml" - it never looks in `bins/`. A package registered
    nowhere would have read as registered.

    `*` matches within one segment and `**` spans any number of them; both were
    confirmed against `cargo metadata`, which accepted `bins/*` and `**/ptrctl`
    for a package at `bins/ptrctl`.
    """
    return segments_match(PurePosixPath(relative).parts, PurePosixPath(pattern).parts)


def segments_match(parts: tuple[str, ...], globs: tuple[str, ...]) -> bool:
    if not globs:
        return not parts
    if globs[0] == "**":
        return any(segments_match(parts[index:], globs[1:]) for index in range(len(parts) + 1))
    if not parts:
        return False
    return fnmatch.fnmatchcase(parts[0], globs[0]) and segments_match(parts[1:], globs[1:])


def check_workspace_membership(root: Path) -> list[str]:
    """Every Cargo package under a package area is a member or an exclusion.

    `members = ["bins/ptr-*"]` needs the hyphen, so `bins/ptrctl` and `bins/ptrd`
    matched neither that glob nor `exclude` and were built, tested, linted and
    formatted by nothing at all. A glob that silently omits a package is not a
    decision anybody made, and the omission is invisible: `cargo` reports no error
    for a directory it was never told about.
    """
    errors: list[str] = []
    manifest = root / "Cargo.toml"
    if not manifest.exists():
        return ["missing Cargo.toml"]
    try:
        workspace = tomllib.loads(manifest.read_text(encoding="utf-8")).get("workspace")
    except Exception as error:  # reported in full by the TOML pass
        return [f"Cargo.toml: {error}"]
    if workspace is None:
        return []
    members = workspace.get("members", [])
    exclude = workspace.get("exclude", [])
    for area in PACKAGE_AREAS:
        base = root / area
        if not base.is_dir():
            continue
        for package in sorted(p for p in base.iterdir() if (p / "Cargo.toml").exists()):
            relative = f"{area}/{package.name}"
            if any(matches(relative, pattern) for pattern in members):
                continue
            if any(matches(relative, pattern) for pattern in exclude):
                continue
            errors.append(
                f"{relative}: Cargo package is in no workspace - "
                "it matches no members glob and no exclude entry, so nothing "
                "builds, tests, lints or formats it"
            )
    return errors


def check(root: Path) -> tuple[list[str], str]:
    errors: list[str] = []

    for rel in REQUIRED:
        if not (root / rel).exists():
            errors.append(f"missing {rel}")

    errors.extend(check_workspace_membership(root))

    for p in owned(root, root.rglob("*.toml")):
        try:
            tomllib.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"TOML {p.relative_to(root)}: {e}")
    for p in owned(root, root.rglob("*.json")):
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"JSON {p.relative_to(root)}: {e}")
    for p in owned(root, root.rglob("*.jsonl")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as e:
                errors.append(f"JSONL {p.relative_to(root)}:{n}: {e}")

    # Reported, not raised. `check(root)` is driven by fixtures as well as by the
    # real tree, and a fixture missing a registry should come back as an error in
    # the list like every other finding rather than as a traceback that hides the
    # ones already accumulated.
    def registry(relative: str, key: str) -> set[str]:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            return set()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return {x["id"] for x in raw.get(key, [])}

    exp_ids = registry("experiments/registry.toml", "experiment")
    eval_ids = registry("evaluations/registry.toml", "component")

    if not (root / "crates").is_dir():
        errors.append("missing crates")
        return errors, summary
    crate_dirs = sorted(
        p for p in (root / "crates").iterdir() if p.is_dir() and (p / "Cargo.toml").exists()
    )
    for crate in crate_dirs:
        name = crate.name
        for rel in ["README.md", "component.toml", "src/lib.rs"]:
            if not (crate / rel).exists():
                errors.append(f"{name}: missing {rel}")
        if not (root / "docs/diagrams/components" / f"{name}.mmd").exists():
            errors.append(f"{name}: missing component diagram")
        meta_path = crate / "component.toml"
        if meta_path.exists():
            try:
                meta = tomllib.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("id") != name:
                    errors.append(f"{name}: component.toml id mismatch")
                for key in COMPONENT_KEYS:
                    if key not in meta:
                        errors.append(f"{name}: component.toml missing {key}")
                for eid in meta.get("experiments", []):
                    if eid not in exp_ids:
                        errors.append(f"{name}: unknown experiment {eid}")
                for eid in meta.get("evaluations", []):
                    if eid not in eval_ids:
                        errors.append(f"{name}: unknown evaluation {eid}")
                for adr in meta.get("decisions", []):
                    if not (root / "research/decisions" / adr).exists():
                        errors.append(f"{name}: missing ADR {adr}")
            except Exception as e:
                errors.append(f"{name}: invalid component.toml: {e}")
        readme = crate / "README.md"
        if readme.exists():
            txt = readme.read_text(encoding="utf-8")
            if "<!-- PTR:STATUS:BEGIN -->" not in txt or "<!-- PTR:STATUS:END -->" not in txt:
                errors.append(f"{name}: README generated status block missing")

    # Every declared workspace-area config must have a sibling tests directory.
    for cfg in owned(root, root.rglob("config.toml")):
        if ".git" in cfg.parts:
            continue
        if not (cfg.parent / "tests").exists():
            errors.append(f"{cfg.parent.relative_to(root)}: config.toml requires tests/")

    # Every Rust workspace crate must carry local config + integration-test directory.
    for crate in sorted((root / "crates").glob("ptr-*")):
        if not (crate / "Cargo.toml").exists():
            continue
        if not (crate / "config.toml").exists():
            errors.append(f"{crate.name}: missing config.toml")
        if not (crate / "tests").exists():
            errors.append(f"{crate.name}: missing tests/")

    mods = list((root / "model/modifications").glob("MOD-*.md"))
    if len(mods) < 10:
        errors.append("expected >=10 model modification specs")
    comps = list((root / "evaluations/components").glob("*/candidates.toml"))
    if len(comps) < 20:
        errors.append("expected >=20 component evaluations")
    exps = list((root / "experiments").glob("**/experiment.toml"))
    if len(exps) < 15:
        errors.append("expected >=15 experiments")

    summary = (
        f"OK: {len(crate_dirs)} documented crates, {len(comps)} component evaluations, "
        f"{len(exps)} experiments, {len(mods)} model modifications"
    )
    return errors, summary


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else ROOT
    errors, summary = check(root)
    if errors:
        print("\n".join("ERROR: " + x for x in errors))
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
