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


def scan(kind: str, output: Path, root: Path = ROOT) -> int:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    failed = False
    for relative in workspaces(root):
        name = str(relative.parent).replace("/", "-") if relative.parent != Path(".") else "workspace"
        manifest = root / relative
        lock = manifest.with_name("Cargo.lock")
        if not lock.is_file():
            raise ValueError(f"missing committed lockfile: {lock.relative_to(root)}")
        subprocess.run(["git", "ls-files", "--error-unmatch", str(lock.relative_to(root))], cwd=root,
                       check=True, capture_output=True, text=True)
        before = hashlib.sha256(lock.read_bytes()).hexdigest()
        commands = {
            "metadata": ["cargo", "+stable", "metadata", "--manifest-path", str(manifest), "--all-features", "--locked", "--format-version", "1"],
            kind: (["cargo", "+stable", "audit", "--file", str(lock), "--json"] if kind == "audit" else
                   ["cargo", "+stable", "deny", "--format", "json", "--manifest-path", str(manifest), "check", "--config", str(root / "deny.toml")]),
        }
        statuses = {}
        for stage, command in commands.items():
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            (output / f"{name}-{stage}.stdout").write_text(result.stdout, encoding="utf-8")
            (output / f"{name}-{stage}.stderr").write_text(result.stderr, encoding="utf-8")
            statuses[stage] = result.returncode
            failed |= result.returncode != 0
        after = hashlib.sha256(lock.read_bytes()).hexdigest()
        failed |= before != after
        records.append({"manifest": str(relative), "lock_sha256": before,
                        "lock_unchanged": before == after, "exit_codes": statuses})
    def version(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=root, text=True).strip()
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
