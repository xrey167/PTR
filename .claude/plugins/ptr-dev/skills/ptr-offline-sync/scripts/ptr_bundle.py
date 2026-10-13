#!/usr/bin/env python3
"""Move a git repository between machines without a remote, using git bundles.

Subcommands (Python 3.11+, standard library and the `git` executable only):

  create --out DIR [--since REF] [--split-mib 25]
      Bundle every ref (or only the commits in REF..--all), split the bundle
      into parts of at most --split-mib MiB and write a SHA256SUMS file.
  join DIR
      Concatenate the parts in DIR into one .bundle, verify the sha256 of every
      part and of the result, then run `git bundle verify`.
  apply BUNDLE [--branch-prefix import/]
      Fetch the bundle's branches into refs/heads/<prefix>*. The current branch
      and the working tree are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

BUNDLE_NAME = "repo.bundle"
SUMS_NAME = "SHA256SUMS"
CHUNK = 1024 * 1024


class BundleError(Exception):
    """A user-facing failure; main() prints it and exits with status 1."""


def git(*args: str, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        raise BundleError("git executable not found on PATH") from None
    if proc.returncode != 0:
        raise BundleError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def part_name(index: int) -> str:
    return f"{BUNDLE_NAME}.part{index:03d}"


def split_file(path: Path, out_dir: Path, part_bytes: int) -> list[Path]:
    """Split `path` into out_dir/repo.bundle.partNNN files of at most part_bytes."""
    parts: list[Path] = []
    with path.open("rb") as src:
        while True:
            block = src.read(min(CHUNK, part_bytes))
            if not block:
                break
            target = out_dir / part_name(len(parts))
            written = 0
            with target.open("wb") as dst:
                while block:
                    dst.write(block)
                    written += len(block)
                    block = src.read(min(CHUNK, part_bytes - written)) if written < part_bytes else b""
            parts.append(target)
    return parts


def write_sums(out_dir: Path, files: list[Path]) -> Path:
    sums = out_dir / SUMS_NAME
    lines = [f"{sha256_file(f)}  {f.name}\n" for f in files]
    sums.write_text("".join(lines), encoding="utf-8", newline="\n")
    return sums


def read_sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise BundleError(f"{path} is missing")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if not name:
            raise BundleError(f"malformed line in {path}: {line!r}")
        result[name.strip()] = digest.strip()
    return result


def cmd_create(args: argparse.Namespace) -> int:
    if args.split_mib <= 0:
        raise BundleError("--split-mib must be positive")
    git("rev-parse", "--git-dir")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise BundleError(f"{out_dir} is not empty; use a fresh directory")
    bundle = out_dir / BUNDLE_NAME
    spec = ["--all"]
    if args.since:
        git("rev-parse", "--verify", f"{args.since}^{{commit}}")
        spec = ["--all", f"^{args.since}"]
    git("bundle", "create", str(bundle), *spec)
    part_bytes = args.split_mib * 1024 * 1024
    if bundle.stat().st_size <= part_bytes:
        files = [bundle]
        print(f"bundle: {bundle} ({bundle.stat().st_size} bytes, single file)")
    else:
        files = split_file(bundle, out_dir, part_bytes)
        bundle.unlink()
        print(f"bundle split into {len(files)} parts of at most {args.split_mib} MiB")
    sums = write_sums(out_dir, files)
    print(f"checksums: {sums}")
    for f in files:
        print(f"  {f.name}")
    return 0


def expected_name(name: str) -> bool:
    """True only for the names `create` writes: the bundle itself or one of its parts."""
    if name == BUNDLE_NAME:
        return True
    prefix = f"{BUNDLE_NAME}.part"
    return name.startswith(prefix) and name[len(prefix):].isdigit() and name.isascii()


def cmd_join(args: argparse.Namespace) -> int:
    directory = Path(args.dir).resolve()
    sums = read_sums(directory / SUMS_NAME)
    names = sorted(sums)
    # The manifest comes from another machine: check every name before any path is opened, so
    # an absolute path or a `../` entry can never make this read (and hash) a file elsewhere.
    unexpected = [name for name in names if not expected_name(name)]
    if unexpected:
        raise BundleError(f"unexpected entries in {SUMS_NAME}: {unexpected}")
    if BUNDLE_NAME in names and len(names) > 1:
        raise BundleError(f"{SUMS_NAME} lists both {BUNDLE_NAME} and parts: {names}")
    for name in names:
        part = directory / name
        if not part.is_file():
            raise BundleError(f"missing file listed in {SUMS_NAME}: {name}")
        actual = sha256_file(part)
        if actual != sums[name]:
            raise BundleError(f"sha256 mismatch for {name}: expected {sums[name]}, got {actual}")
        print(f"ok  {name}")
    target = directory / BUNDLE_NAME
    if names == [BUNDLE_NAME]:
        print(f"single-file bundle, nothing to join: {target}")
    else:
        with target.open("wb") as dst:
            for name in names:
                with (directory / name).open("rb") as src:
                    while block := src.read(CHUNK):
                        dst.write(block)
        print(f"joined {len(names)} parts into {target}")
    # `git bundle verify` needs a repository; prerequisites must exist in it.
    out = git("bundle", "verify", str(target))
    print(out.strip())
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).resolve()
    if not bundle.is_file():
        raise BundleError(f"{bundle} does not exist")
    prefix = args.branch_prefix
    if not prefix or prefix.startswith("/") or ".." in prefix:
        raise BundleError(f"unsafe --branch-prefix {prefix!r}")
    git("rev-parse", "--git-dir")
    git("bundle", "verify", str(bundle))
    # Force-update only inside the import namespace; heads and tags of the
    # current work stay untouched. A leading '+' is limited to refs/heads/<prefix>*.
    git("fetch", str(bundle), f"+refs/heads/*:refs/heads/{prefix}*")
    heads = git("for-each-ref", "--format=%(refname:short) %(objectname:short)", f"refs/heads/{prefix}")
    print(heads.strip() or "(bundle contained no branches)")
    print(f"Inspect with: git log --oneline {prefix}<branch>; current branch was not changed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptr_bundle.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="bundle the repository into split parts plus SHA256SUMS")
    create.add_argument("--out", required=True, help="empty or new output directory")
    create.add_argument("--since", default=None, help="only commits not reachable from REF")
    create.add_argument("--split-mib", type=int, default=25, help="part size in MiB (default 25)")
    create.set_defaults(func=cmd_create)
    join = sub.add_parser("join", help="verify parts, concatenate, git bundle verify")
    join.add_argument("dir")
    join.set_defaults(func=cmd_join)
    apply = sub.add_parser("apply", help="fetch bundle branches into refs/heads/<prefix>*")
    apply.add_argument("bundle")
    apply.add_argument("--branch-prefix", default="import/")
    apply.set_defaults(func=cmd_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BundleError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
