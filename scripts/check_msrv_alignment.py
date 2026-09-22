"""Every package is linted at the MSRV it declares.

Clippy reads its MSRV from the nearest `clippy.toml` walking upward from a crate,
and that file wins over `rust-version` in `Cargo.toml`. So a workspace nested
under a root that carries a `clippy.toml` is linted at the *root's* MSRV, however
much newer its own is, and every lint whose suggestion needs a newer compiler is
suppressed.

`model/burn-a0` requires 1.95 and was linted at the root's 1.85.0 for as long as
it existed. Clippy is not silent about it — it prints "the MSRV in `clippy.toml`
and `Cargo.toml` differ" on every run — but the message is not a named lint, so
`-D warnings` cannot promote it and the job exits 0. A printed warning nothing
enforces is why this checker exists.

Two details decide whether such a check works at all, and both are tested:

* it compares **semantically**, because the root declares `rust-version = "1.85"`
  against `msrv = "1.85.0"` and clippy treats those as the same version. A string
  comparison fails on this repository's own root workspace on day one;
* it resolves `rust-version.workspace = true`, which every crate and both binaries
  use, so the comparison is against the version that actually governs the package.

A package that declares no `rust-version` is not reported: it makes no claim, so
there is nothing for a `clippy.toml` to contradict.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "target", "vendor", ".venv", "node_modules"}
CLIPPY_NAMES = ("clippy.toml", ".clippy.toml")


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def parse_version(text: str) -> tuple[int, ...]:
    """`1.85` and `1.85.0` are the same version, which is why this is not `==`."""
    parts = [int(piece) for piece in str(text).split(".")]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def enclosing_workspace(manifest: Path, root: Path) -> Path | None:
    """The nearest ancestor manifest declaring `[workspace]`, as cargo resolves it."""
    for parent in manifest.parent.parents:
        if not parent.is_relative_to(root):
            break
        candidate = parent / "Cargo.toml"
        if candidate.exists():
            try:
                if "workspace" in load(candidate):
                    return candidate
            except Exception:
                continue
    return None


def declared_rust_version(manifest: Path, root: Path):
    """The `rust-version` governing this package, following workspace inheritance.

    Returns `(version, source)` or `(None, None)` when the package declares none.
    """
    try:
        document = load(manifest)
    except Exception:
        return None, None
    package = document.get("package", {})
    value = package.get("rust-version")
    if isinstance(value, dict) and value.get("workspace"):
        # Inherited: from this manifest's own `[workspace.package]` if it is a
        # workspace root, otherwise from the enclosing one.
        own = document.get("workspace", {}).get("package", {}).get("rust-version")
        if own is not None:
            return own, f"{manifest.name} [workspace.package]"
        enclosing = enclosing_workspace(manifest, root)
        if enclosing is None:
            return None, None
        inherited = load(enclosing).get("workspace", {}).get("package", {}).get("rust-version")
        if inherited is None:
            return None, None
        return inherited, f"{enclosing.relative_to(root)} [workspace.package]"
    if value is not None:
        return value, f"{manifest.relative_to(root)} [package]"
    # A virtual workspace root carries its MSRV in `[workspace.package]`.
    own = document.get("workspace", {}).get("package", {}).get("rust-version")
    if own is not None:
        return own, f"{manifest.relative_to(root)} [workspace.package]"
    return None, None


def governing_clippy(directory: Path, root: Path) -> Path | None:
    """The `clippy.toml` clippy would use for a crate in this directory."""
    for candidate_dir in [directory, *directory.parents]:
        if not candidate_dir.is_relative_to(root):
            break
        for name in CLIPPY_NAMES:
            candidate = candidate_dir / name
            if candidate.exists():
                return candidate
    return None


def manifests(root: Path):
    for manifest in sorted(root.rglob("Cargo.toml")):
        if SKIP.intersection(manifest.relative_to(root).parts):
            continue
        yield manifest


def check(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    checked: list[str] = []
    for manifest in manifests(root):
        version, source = declared_rust_version(manifest, root)
        if version is None:
            continue
        clippy = governing_clippy(manifest.parent, root)
        if clippy is None:
            continue
        try:
            msrv = load(clippy).get("msrv")
        except Exception as error:
            errors.append(f"{clippy.relative_to(root)}: unreadable: {error}")
            continue
        if msrv is None:
            continue
        try:
            same = parse_version(msrv) == parse_version(version)
        except ValueError as error:
            errors.append(f"{clippy.relative_to(root)}: unparseable version: {error}")
            continue
        where = manifest.relative_to(root)
        if not same:
            errors.append(
                f"{where}: linted at msrv {msrv} from {clippy.relative_to(root)} "
                f"but rust-version {version} from {source} - every lint whose "
                "suggestion needs a newer compiler is suppressed, and clippy's own "
                "notice about it is not a lint, so -D warnings cannot catch it"
            )
        else:
            checked.append(f"{where}: {version} == {msrv} ({clippy.relative_to(root)})")
    return errors, checked


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else ROOT
    errors, checked = check(root)
    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        return 1
    print(f"OK: {len(checked)} packages linted at the MSRV they declare")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
