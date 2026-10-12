#!/usr/bin/env python3
"""File edits after an experiment's aggregate. Commit-free: the caller commits with explicit paths.

    # 1. status prepared -> completed|failed (per results/run.json recommended_status) in
    #    experiment.toml and experiments/registry.toml, and one STATUS.md line after an anchor line
    python3 scripts/freeze/finish_experiment.py status --id E007 --dir experiments/system/E007-split-pod-kv \
        --status-line-file /path/line.md --anchor '- **Q002 evidence-promotion completed'

    # 2. results/STALE.toml, once the commit of step 1 exists
    python3 scripts/freeze/finish_experiment.py stale --id E007 --dir experiments/system/E007-split-pod-kv \
        --results-sha <aggregate-parent-sha> --stale-since <docs-commit-sha> --tag e007-prereg

`status` refuses (exit 4, nothing written) when the verdict recommends no status (NOT_SHOWN, INVALID:
recommended_status null), when the experiment is not `prepared`, or when the anchor line is missing or
the line is already present. `--anchor` must match the start of exactly one STATUS.md line (the new line goes right after that line).
`stale` needs an explicit --tag (the preregistration tag; its commit is read for the reason text).
All edits are computed first and written together. The repository is the one holding this script
unless --root is given.
"""

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomic_write import atomic_write_all  # noqa: E402

FINAL = ("completed", "failed")


class FinishError(Exception):
    pass


def recommended_status(results: Path):
    for name, pick in (("run.json", lambda d: d.get("recommended_status")),
                       ("metrics.json", lambda d: (d.get("decision") or {}).get("recommended_status"))):
        path = results / name
        if path.is_file():
            try:
                value = pick(json.loads(path.read_text(encoding="utf-8")))
            except ValueError as error:
                raise FinishError(f"{path} is not valid JSON: {error}")
            return value, name
    raise FinishError(f"no run.json or metrics.json in {results}: run the aggregate first")


def do_status(args):
    root = Path(args.root).resolve()
    directory = root / args.dir
    value, source = recommended_status(directory / "results")
    if value not in FINAL:
        raise FinishError(f"{source} recommends status {value!r}: the experiment stays prepared, nothing to finish")
    manifest_path = directory / "experiment.toml"
    manifest = manifest_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(manifest)
    if parsed.get("id") != args.id or parsed.get("status") != "prepared":
        raise FinishError(f"experiment.toml of {args.id} must have status 'prepared' (is {parsed.get('status')!r})")
    manifest = re.sub(r'^status = "prepared"', f'status = "{value}"', manifest, count=1, flags=re.MULTILINE)

    registry_path = root / "experiments/registry.toml"
    registry = registry_path.read_text(encoding="utf-8")
    registry, count = re.subn(r'(id = "%s"\npath = "[^"]+"\nstatus = )"prepared"' % re.escape(args.id),
                              rf'\1"{value}"', registry)
    if count != 1:
        raise FinishError(f"registry entry of {args.id} with status prepared found {count} times")

    status_path = root / args.status_file
    text = status_path.read_text(encoding="utf-8")
    line = Path(args.status_line_file).read_text(encoding="utf-8").strip()
    if not line:
        raise FinishError("status line file is empty")
    if line in text:
        raise FinishError(f"{args.status_file} already contains this line")
    lines = text.splitlines(keepends=True)
    hits = [index for index, candidate in enumerate(lines) if candidate.startswith(args.anchor)]
    if not hits:
        raise FinishError(f"anchor {args.anchor!r} matches no line start of {args.status_file}")
    if len(hits) > 1:
        raise FinishError(f"anchor {args.anchor!r} matches {len(hits)} lines of {args.status_file}; make it unique")
    at = hits[0]
    if not lines[at].endswith("\n"):
        lines[at] += "\n"
    lines.insert(at + 1, line + "\n")
    text = "".join(lines)

    atomic_write_all({manifest_path: manifest, registry_path: registry, status_path: text})
    print(f"{args.id}: status {value} (from {source}) in experiment.toml and registry; line added to {args.status_file}")
    return 0


def rev(root, name):
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", f"{name}^{{commit}}"],
                          text=True, capture_output=True)
    if done.returncode != 0:
        raise FinishError(f"{name!r} is not a commit of {root}")
    return done.stdout.strip()


def do_stale(args):
    root = Path(args.root).resolve()
    directory = root / args.dir
    manifest = tomllib.loads((directory / "experiment.toml").read_text(encoding="utf-8"))
    if manifest.get("id") != args.id:
        raise FinishError(f"experiment.toml id is {manifest.get('id')!r}, not {args.id!r}")
    results_sha, since, tag = rev(root, args.results_sha), rev(root, args.stale_since), rev(root, args.tag)
    if not (directory / "results").is_dir():
        raise FinishError(f"{args.dir}/results does not exist")
    status = manifest.get("status")
    reason = args.reason or (
        f"No code under test changed after {args.id}'s seed records and mutation check. {since[:8]} records the "
        f"{status} status and the result in STATUS.md and regenerates the component docs, which the "
        f"repository-wide provenance counts as a change. {args.id} was preregistered at {args.tag} "
        f"({tag[:8]}). No rerun under the same preregistration follows (each seed runs once)."
    )
    body = (f'results_git_sha = "{results_sha}"\nstale_since = "{since}"\n'
            f'reason = {json.dumps(reason)}\n')
    tomllib.loads(body)
    (directory / "results/STALE.toml").write_text(body, encoding="utf-8", newline="")
    print(f"{args.id}: {args.dir}/results/STALE.toml written (results {results_sha[:8]}, stale since {since[:8]}, tag {args.tag})")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("status")
    st.add_argument("--id", required=True)
    st.add_argument("--dir", required=True)
    st.add_argument("--status-line-file", required=True)
    st.add_argument("--anchor", required=True, help="text the anchor line of STATUS.md starts with")
    st.add_argument("--status-file", default="STATUS.md")
    sl = sub.add_parser("stale")
    sl.add_argument("--id", required=True)
    sl.add_argument("--dir", required=True)
    sl.add_argument("--results-sha", required=True)
    sl.add_argument("--stale-since", required=True)
    sl.add_argument("--tag", required=True, help="explicit preregistration tag, e.g. q003v3-prereg")
    sl.add_argument("--reason")
    args = parser.parse_args(argv)
    return {"status": do_status, "stale": do_stale}[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except FinishError as error:
        print(f"FINISH-EXPERIMENT ERROR: {error}", file=sys.stderr)
        sys.exit(4)
