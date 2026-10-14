"""Audit every owned Cargo workspace; never mutate a lockfile to make it pass."""
from __future__ import annotations
import argparse
import fnmatch
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_OR_GENERATED = {".git", "target", "vendor", ".venv", "node_modules"}
EXPECTED = {"Cargo.toml", "model/burn-a0/Cargo.toml", "fuzz/Cargo.toml", "templates/rust-crate/Cargo.toml"}
TEXT_OUTPUT = {"encoding": "utf-8", "errors": "replace"}

# Advisories cargo-audit may ignore, per workspace. Only for a lockfile that
# frozen experiment records pin and that therefore may not be updated, and
# only for a package no target and no feature of that workspace builds. Each
# entry names the exact locked package: once the lockfile no longer pins it,
# the entry is stale and the scan fails until it is removed. cargo-deny is
# never given an exception (deny.toml keeps advisories.ignore empty).
AUDIT_EXCEPTIONS = {
    "model/burn-a0/Cargo.toml": {
        "package": ("rustls", "0.23.45"),
        "advisories": ("RUSTSEC-2026-0346", "RUSTSEC-2026-0347",
                       "RUSTSEC-2026-0348", "RUSTSEC-2026-0349"),
        "reason": (
            "model/burn-a0 (Cargo.lock included) is pinned by "
            "implementation_tree_git_digest in frozen M001/M002/M003 records; "
            "rustls is in its lockfile but `cargo tree --target all "
            "--all-features -i rustls` finds no path to it, so it is never built"
        ),
    },
}


def workspaces(root: Path) -> list[Path]:
    found = []
    root_manifest = root / "Cargo.toml"
    root_data = (tomllib.loads(root_manifest.read_text(encoding="utf-8"))
                 if root_manifest.is_file() else {})
    excluded = root_data.get("workspace", {}).get("exclude", [])
    for manifest in root.rglob("Cargo.toml"):
        relative = manifest.relative_to(root)
        if EXTERNAL_OR_GENERATED.intersection(relative.parts):
            continue
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        # Cargo permits excluded standalone packages without a [workspace]
        # table (the real fuzz package uses this form). Their lockfiles must
        # not disappear from security coverage merely because of that syntax.
        standalone = any(fnmatch.fnmatchcase(relative.parent.as_posix(), pattern.rstrip("/"))
                         for pattern in excluded)
        if ("workspace" in data or relative.as_posix() in EXPECTED
                or standalone or manifest.with_name("Cargo.lock").is_file()):
            found.append(relative)
    paths = {path.as_posix() for path in found}
    missing = EXPECTED - paths
    if missing:
        raise ValueError(f"required workspaces disappeared: {sorted(missing)}")
    # New owned workspaces are included, not silently skipped.
    return sorted(found)


def scanner_command(kind: str, manifest: Path, root: Path) -> list[str]:
    if kind == "audit":
        command = ["cargo", "+stable", "audit", "--file",
                   str(manifest.with_name("Cargo.lock")), "--json"]
        exception = AUDIT_EXCEPTIONS.get(manifest.relative_to(root).as_posix())
        for advisory in exception["advisories"] if exception else ():
            command += ["--ignore", advisory]
        return command
    if kind == "deny":
        # cargo-deny 0.20 (the version .github/workflows/security.yml installs):
        # --config is a root option, before the subcommand. After `check` it is
        # rejected ("unexpected argument '--config' found"), which failed the
        # deny job for every workspace.
        return ["cargo", "+stable", "deny", "--format", "json",
                "--manifest-path", str(manifest), "--config",
                str(root / "deny.toml"), "check"]
    raise ValueError(f"unsupported security scanner: {kind}")


def stale_audit_exceptions(root: Path) -> list[str]:
    """Exceptions whose workspace or exact locked package no longer exists."""
    stale = []
    for manifest, exception in sorted(AUDIT_EXCEPTIONS.items()):
        lock = root / manifest
        lock = lock.with_name("Cargo.lock")
        name, version = exception["package"]
        packages = (tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
                    if lock.is_file() else [])
        if not any(p.get("name") == name and p.get("version") == version for p in packages):
            stale.append(f"{manifest}: {name} {version} is no longer locked")
    return stale


def reachable_audit_exceptions(root: Path, run=subprocess.run) -> list[str]:
    """Exceptions whose package the workspace can build for some target or feature.

    `cargo tree -i <name>@<version>` over every target, every feature and every
    edge kind prints nothing when no path leads to the package; any output (or
    a failed command) means the exception no longer holds.
    """
    reachable = []
    for manifest, exception in sorted(AUDIT_EXCEPTIONS.items()):
        name, version = exception["package"]
        command = ["cargo", "+stable", "tree", "--manifest-path", str(root / manifest),
                   "--locked", "--target", "all", "--all-features",
                   "-e", "normal,build,dev", "-i", f"{name}@{version}"]
        result = run(command, cwd=root, capture_output=True, **TEXT_OUTPUT)
        if result.returncode != 0 or result.stdout.strip():
            reachable.append(f"{manifest}: {name} {version} is reachable or the graph "
                             f"check failed (exit {result.returncode})")
    return reachable


def workspace_name(relative: Path) -> str:
    """Stable flat evidence prefix for a workspace manifest."""
    return (
        "workspace"
        if relative.parent == Path(".")
        else relative.parent.as_posix().replace("/", "-")
    )


def scan(kind: str, output: Path, root: Path = ROOT) -> int:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    failed = False
    if kind == "audit":
        stale = stale_audit_exceptions(root)
        if stale:
            raise ValueError(f"stale audit exceptions, remove them: {stale}")
        reachable = reachable_audit_exceptions(root)
        if reachable:
            raise ValueError(f"audit exceptions no longer hold: {reachable}")
    for relative in workspaces(root):
        name = workspace_name(relative)
        manifest = root / relative
        lock = manifest.with_name("Cargo.lock")
        if not lock.is_file():
            raise ValueError(f"missing committed lockfile: {lock.relative_to(root)}")
        subprocess.run(["git", "ls-files", "--error-unmatch", str(lock.relative_to(root))], cwd=root,
                       check=True, capture_output=True, **TEXT_OUTPUT)
        before = hashlib.sha256(lock.read_bytes()).hexdigest()
        commands = {
            "metadata": ["cargo", "+stable", "metadata", "--manifest-path", str(manifest), "--all-features", "--locked", "--format-version", "1"],
            kind: scanner_command(kind, manifest, root),
        }
        statuses = {}
        for stage, command in commands.items():
            result = subprocess.run(command, cwd=root, capture_output=True, **TEXT_OUTPUT)
            (output / f"{name}-{stage}.stdout").write_text(result.stdout, encoding="utf-8")
            (output / f"{name}-{stage}.stderr").write_text(result.stderr, encoding="utf-8")
            statuses[stage] = result.returncode
            failed |= result.returncode != 0
        after = hashlib.sha256(lock.read_bytes()).hexdigest()
        failed |= before != after
        record = {"manifest": relative.as_posix(), "lock_sha256": before,
                  "lock_unchanged": before == after, "exit_codes": statuses}
        if kind == "audit" and relative.as_posix() in AUDIT_EXCEPTIONS:
            record["ignored_advisories"] = AUDIT_EXCEPTIONS[relative.as_posix()]
        records.append(record)
    def version(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=root, **TEXT_OUTPUT).strip()
    report = {"source_sha": version(["git", "rev-parse", "HEAD"]), "rust": version(["rustc", "+stable", "-Vv"]),
              "scanner": version(["cargo", "+stable", kind, "--version"]), "workspaces": records}
    database = Path.home() / ".cargo/advisory-db"
    if database.exists():
        report["advisory_db_sha"] = version(["git", "-C", str(database), "rev-parse", "HEAD"])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["audit", "deny", "list"])
    parser.add_argument("--output", type=Path, default=Path("security-results"))
    args = parser.parse_args()
    try:
        if args.kind == "list":
            print("\n".join(map(str, workspaces(ROOT))))
            return 0
        return scan(args.kind, args.output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # Preserve diagnostic evidence even if discovery/metadata fails before
        # a scanner can run. This is explicitly incomplete, never a clean scan.
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.json").write_text(
            json.dumps({"complete": False, "kind": args.kind,
                        "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        print(f"ERROR: security verification incomplete: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
