#!/usr/bin/env python3
"""Merge check: every experiment's preregistration rule digest against both parents.

    python3 scripts/freeze/digest_parents.py [PARENT1 PARENT2] [--root <tree>]

For each experiment in the working tree's experiments/preregistration.toml the canonical rule digest
(scripts/experiment_records.py canonical_digest of its entry) must equal the digest of the same id on
at least one parent. Without parent arguments: HEAD and MERGE_HEAD while a merge is in progress,
else HEAD^1 and HEAD^2 (a committed merge; the working tree equals HEAD then).

  MISMATCH                 digest equals neither parent's (a merge that split or rewrote rule lines,
                           or an id new in the merge)                                    -> error
  DROPPED                  an id of a parent that the merged list no longer has           -> error
  DIFFERS-BETWEEN-PARENTS  the parents disagree and the merge took one side              -> info

Exit 1 when there is any error, else 0; 2 for unusable input.
"""

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

LIST = "experiments/preregistration.toml"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)


def default_parents(root):
    if git(root, "rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0:
        return "HEAD", "MERGE_HEAD"
    return "HEAD^1", "HEAD^2"


def table_at(root, rev):
    if rev is None:
        text = (Path(root) / LIST).read_text(encoding="utf-8")
    else:
        done = git(root, "show", f"{rev}:{LIST}")
        if done.returncode != 0:
            raise SystemExit(f"digest_parents: cannot read {LIST} at {rev}: {done.stderr.strip()}")
        text = done.stdout
    return tomllib.loads(text).get("experiment", {})


def check(root, parent_a, parent_b, canonical_digest, out=print):
    current, a, b = table_at(root, None), table_at(root, parent_a), table_at(root, parent_b)
    problems = 0
    for ident, entry in current.items():
        digest = canonical_digest(entry)
        da = canonical_digest(a[ident]) if ident in a else None
        db = canonical_digest(b[ident]) if ident in b else None
        if digest not in (da, db):
            problems += 1
            out(f"MISMATCH {ident}")
        elif da is not None and db is not None and da != db:
            out(f"DIFFERS-BETWEEN-PARENTS {ident} merged takes {'parent1' if digest == da else 'parent2'}")
    for ident in sorted(set(a) | set(b)):
        if ident not in current:
            problems += 1
            out(f"DROPPED {ident}")
    out(f"digest check: {len(current)} experiments, {problems} problems")
    return problems


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parents", nargs="*")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    if len(args.parents) not in (0, 2):
        parser.error("give both parents or none")
    root = Path(args.root).resolve()
    scripts = root / "scripts"
    if not (scripts / "experiment_records.py").is_file():
        scripts = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scripts))
    import experiment_records as records  # noqa: E402

    parent_a, parent_b = args.parents or default_parents(root)
    return 1 if check(root, parent_a, parent_b, records.canonical_digest) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
