"""Mutation checks for experiment harnesses: can a broken implementation pass?

An experiment that has only ever passed says little until it has been shown to
fail. Each experiment that opts in lists defects in `tests/mutations.toml`: a
source file, an exact text that must occur there once, and its replacement, plus
optional further edits (`[[mutation.also]]`, each with its own `find`,
`replace` and optionally `file`) when one defect spans several places.
Each mutation also names the hard counters that show its defect (`expect`).
This script plants one defect at a time, rebuilds the harness, runs it, and
restores the file whatever happens: each edit replaces the file in one step, so
a write that fails never leaves it truncated, and restoring renames the
original, kept under a second name beside it before the defect is planted,
back into place, which needs no free space; should even that fail, the script
names each file that still holds the defect and where its original is. The
rename brings back the original's modification time, which is older than the
mutated build, and cargo rebuilds a crate only when a source is newer than the
stamp it took when it last built that crate. So each restored file is then
stamped, and read back, as modified after the harness binary that build
linked, which cargo writes after every crate's stamp: otherwise the next
mutation of another crate, and the unmutated rebuild, would still link the
defect. A
mutation is *killed* when the harness exits with status 1 and one of its
expected counters fired. A run that fails only on other counters is recorded as
`failed-elsewhere`: the defect may have broken something unrelated first (a
query that no longer binds, say), which shows nothing about whether the
harness sees the defect itself. Exiting any other way (a panic, a build
failure, a timeout) or passing is recorded as it is. The record goes to the
experiment's `results/mutations.json`, written whole or not at all and stamped
with the commit HEAD was at when the run started, before the plan was read, so
a run that writes it refuses to start from a working tree with uncommitted or
untracked provenance files, and writes no record, exiting 2, when HEAD moved or
a provenance file, the plan included, was written, created or removed while it
ran, apart from the checker's own edits of the files it plants defects in
(`ProvenanceWatch` in `scripts/experiment_records.py`); the aggregators accept
it only while its commit has the checkout's code, checker and mutation plan.
Every name given to `--only` must be one the plan lists; a run with `--only`
writes no record and so checks no tree. Afterwards the unmutated harness is
rebuilt, and the run fails, writing no record, if that rebuild does: nothing
downstream would see that the run was invalid.

    python scripts/mutation_check.py L004
    python scripts/mutation_check.py L003 --only append-without-row-lock
    python scripts/mutation_check.py L004 --check   # anchors only, no build

The harness reads its server from the environment the experiment names (for
the PostgreSQL experiments `PTR_PG_EXPERIMENT_DSN`); nothing here stores it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import secrets
import stat
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_records  # noqa: E402


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def experiment_root(exp_id: str) -> Path:
    for item in load_toml(REGISTRY).get("experiment", []):
        if item["id"] == exp_id:
            return ROOT / "experiments" / item["path"]
    raise SystemExit(f"unknown experiment: {exp_id}")


def load_plan(exp_root: Path) -> dict:
    path = exp_root / "tests/mutations.toml"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} does not exist")
    plan = load_toml(path)
    for key in ("package", "features", "subcommand", "cases", "seed"):
        if key not in plan:
            raise SystemExit(f"{path.relative_to(ROOT)}: missing {key!r}")
    names = [mutation["name"] for mutation in plan.get("mutation", [])]
    if len(names) != len(set(names)):
        raise SystemExit(f"{path.relative_to(ROOT)}: duplicate mutation names")
    errors = expectation_errors(plan)
    if errors:
        raise SystemExit(f"{path.relative_to(ROOT)}: " + "; ".join(errors))
    return plan


def expectation_errors(plan: dict) -> list[str]:
    """Every mutation must name at least one expected counter, and each must
    be one of the plan's hard counters."""
    hard = set(plan.get("hard_counters", []))
    errors = []
    for mutation in plan.get("mutation", []):
        expected = mutation.get("expect", [])
        if not expected:
            errors.append(f"{mutation['name']}: no expected counter")
        for counter in expected:
            if counter not in hard:
                errors.append(f"{mutation['name']}: {counter!r} is not a hard counter")
    return errors


def edits(mutation: dict) -> list[tuple[str, str, str]]:
    """The (file, find, replace) edits of one mutation, in order."""
    first = [(mutation["file"], mutation["find"], mutation["replace"])]
    return first + [
        (extra.get("file", mutation["file"]), extra["find"], extra["replace"])
        for extra in mutation.get("also", [])
    ]


def anchor_errors(plan: dict, root: Path = ROOT) -> list[str]:
    """Every edit's text must occur exactly once in its file, and its
    replacement must differ; otherwise the list has drifted from the code."""
    errors = []
    for mutation in plan.get("mutation", []):
        name = mutation["name"]
        for file, find, replace in edits(mutation):
            path = root / file
            if not path.exists():
                errors.append(f"{name}: {file} does not exist")
                continue
            count = path.read_text(encoding="utf-8").count(find)
            if count != 1:
                errors.append(f"{name}: the anchor occurs {count} times in {file}")
            if find == replace:
                errors.append(f"{name}: the replacement equals the anchor")
    return errors


def binary_command(plan: dict, cases: int, seed: int) -> tuple[list[str], list[str]]:
    toolchain = [f"+{plan['toolchain']}"] if plan.get("toolchain") else []
    build = [
        "cargo",
        *toolchain,
        "build",
        "--release",
        "--locked",
        "-p",
        plan["package"],
        "--features",
        plan["features"],
    ]
    run = [
        str(ROOT / "target/release" / plan["package"]),
        plan["subcommand"],
        str(cases),
        str(seed),
    ]
    return build, run


def last_json_line(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {}


def classify(returncode: int, metrics: dict, plan: dict, mutation: dict) -> tuple[str, dict]:
    """The result of one run of a mutated harness and the hard counters that
    fired."""
    fired = {
        key: value
        for key, value in metrics.items()
        if key in plan.get("hard_counters", []) and isinstance(value, int) and value > 0
    }
    hard = int(metrics.get("hard_failures", 0))
    if returncode == 0:
        return "survived", fired
    if returncode == 1 and hard > 0:
        if any(counter in fired for counter in mutation.get("expect", [])):
            return "killed", fired
        return "failed-elsewhere", fired
    return "crashed", fired


def own_writes(watch: experiment_records.ProvenanceWatch | None, files):
    """The checker's own rewrite of `files`, which `watch` (when a record is
    to be written) accepts as no change of the tree."""
    return watch.rewriting(files) if watch is not None else contextlib.nullcontext()


def keep_original(path: Path) -> Path:
    """A second name beside `path` for its current content, from which
    `restore_originals` puts it back by a rename, which needs no free space:
    a hard link to the file, or, where the filesystem has none, a copy made
    before anything is planted."""
    kept = path.with_name(f".{path.name}.{secrets.token_hex(6)}.orig")
    try:
        os.link(path, kept)
    except OSError:
        return experiment_records.write_temporary(path, path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
    return kept


# How far past the newest stamp a restored file must be to be stored as later
# than it, tried in turn: a filesystem that keeps whole seconds, or two as FAT
# does, rounds a nanosecond or a millisecond away.
STAMP_STEPS_NS = (1, 1_000_000, 1_000_000_000, 2_000_000_000)


def stamp_after(target: Path, planted: int, built: Path | None) -> None:
    """Stamp `target` as modified after both the planted file it replaced,
    whose modification time was `planted`, and the harness binary `built`
    from it, and read the stored time back to make sure it is later.

    Cargo rebuilds a crate only when one of its sources is stored as newer
    than the stamp cargo took when it last built that crate. The binary is
    linked after every crate of its build, so a source stored as newer than
    the binary is newer than each of those stamps, whatever either
    filesystem rounds to and wherever the clock stood. Raises `OSError` when
    no step of `STAMP_STEPS_NS` leaves a later stored time."""
    newest = planted
    if built is not None:
        try:
            newest = max(newest, built.stat().st_mtime_ns)
        except FileNotFoundError:
            pass
    base = max(time.time_ns(), newest)
    for step in STAMP_STEPS_NS:
        fresh = base + step
        os.utime(target, ns=(fresh, fresh))
        if target.stat().st_mtime_ns > newest:
            return
    raise OSError(f"{target} is still stored as no newer than {newest} ns after every stamp step")


def restore_originals(name: str, kept: dict[str, Path], built: Path | None = None) -> None:
    """Put each file of `kept` (file -> the second name `keep_original` gave
    its original) back by renaming the original over it, then stamp it as
    modified after the planted file it replaces and the harness binary
    `built` from it (`stamp_after`). A restore that fails names every file
    that still holds the mutation `name` and where its original is, and
    raises; so does a restored file whose stamp cannot be moved past both.

    The rename brings back the original's modification time, which is older
    than the build of the mutation, and cargo rebuilds a crate only when one
    of its sources is newer than that crate's last build. Left so, the
    restored crate stays built with the defect: the next mutation, when it is
    in another crate, runs with both defects, and the unmutated rebuild
    rebuilds nothing, leaving the harness mutated for whatever runs next."""
    unrestored = []
    unstamped = []
    for file, original in kept.items():
        target = ROOT / file
        try:
            planted = target.stat().st_mtime_ns
        except OSError:
            planted = 0
        try:
            os.replace(original, target)
            # A rename between two names of one file, as when the plant never
            # replaced it, leaves both names.
            original.unlink(missing_ok=True)
        except OSError as error:
            unrestored.append((file, original, error))
        else:
            try:
                stamp_after(target, planted, built)
            except OSError as error:
                unstamped.append((file, error))
        experiment_records.sync_directory(target.parent)
    if unrestored:
        print(
            "ERROR: restoring the planted files failed; "
            + "; ".join(
                f"{file} still holds the mutation {name}, and its original is "
                f"{original.relative_to(ROOT)} ({error})"
                for file, original, error in unrestored
            ),
            flush=True,
        )
        raise unrestored[0][2]
    if unstamped:
        print(
            "ERROR: restored the planted files, but could not stamp them as newer than the "
            f"build of the mutation {name} and the harness it linked, so cargo may keep building it; "
            + "; ".join(f"{file} ({error})" for file, error in unstamped),
            flush=True,
        )
        raise unstamped[0][1]


def run_mutation(
    plan: dict, mutation: dict, timeout: int, watch: experiment_records.ProvenanceWatch | None = None
) -> dict:
    """Plant `mutation`, build and run the harness, and restore the edited
    files whatever happens: before planting, each original gets a second
    name beside it (`keep_original`), and restoring renames it back, which a
    full disk cannot stop. With a `watch`, the checker's own writes of the
    edited files are accepted and any other write to them is left for
    `watch.changes` to report."""
    cases = int(mutation.get("cases", plan["cases"]))
    seed = int(mutation.get("seed", plan["seed"]))
    build, run = binary_command(plan, cases, seed)
    planned = edits(mutation)
    originals = {file: (ROOT / file).read_text(encoding="utf-8") for file, _, _ in planned}
    outcome: dict = {"name": mutation["name"], "file": mutation["file"], "cases": cases, "seed": seed}
    kept: dict[str, Path] = {}
    try:
        mutated = dict(originals)
        for file, find, replace in planned:
            mutated[file] = mutated[file].replace(find, replace, 1)
        with own_writes(watch, mutated):
            for file in mutated:
                kept[file] = keep_original(ROOT / file)
            for file, text in mutated.items():
                experiment_records.write_atomically(ROOT / file, text)
        built = subprocess.run(build, cwd=ROOT, capture_output=True, text=True, check=False)
        if built.returncode != 0:
            outcome.update(result="build-failed", detail=built.stderr[-2000:])
            return outcome
        try:
            ran = subprocess.run(
                run, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            outcome.update(result="timeout")
            return outcome
    finally:
        with own_writes(watch, originals):
            restore_originals(mutation["name"], kept, Path(run[0]))
    metrics = last_json_line(ran.stdout)
    result, fired = classify(ran.returncode, metrics, plan, mutation)
    outcome.update(
        result=result,
        exit_code=ran.returncode,
        hard_failures=int(metrics.get("hard_failures", 0)),
        expected=list(mutation.get("expect", [])),
        counters=fired,
    )
    if result != "killed":
        outcome["detail"] = ran.stderr[-2000:]
    return outcome


def select(plan: dict, only: list[str]) -> tuple[list[dict], list[str]]:
    """The mutations to run: every one the plan lists or, given `only`, those
    it names, and the errors in that request. A name the plan does not list is
    an error, and so is an empty selection: a typo must not pass by testing
    nothing."""
    mutations = plan.get("mutation", [])
    known = {mutation["name"] for mutation in mutations}
    errors = [f"no mutation named {name!r}" for name in only if name not in known]
    selected = [mutation for mutation in mutations if not only or mutation["name"] in only]
    if not selected:
        errors.append("no mutation selected")
    return selected, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--check", action="store_true", help="verify anchors only")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    exp_root = experiment_root(args.id)
    watch = None
    if not args.check and not args.only:
        # The record names HEAD as the code it mutated, so HEAD must hold
        # every file that decides it, the plan included, from before the plan
        # is read until the record is written: refuse before reading it.
        try:
            watch = experiment_records.ProvenanceWatch(
                ROOT,
                experiment_records.tree_pathspecs(
                    exp_root,
                    exp_root / "results",
                    ROOT,
                    experiment_records.mutation_record_paths(exp_root, ROOT),
                ),
            )
        except experiment_records.ProvenanceError as error:
            print(f"ERROR: {error}")
            return 2
        if watch.uncommitted:
            print(
                "ERROR: refusing to record mutations from a working tree whose sources HEAD does not "
                f"hold; commit or remove {experiment_records.listed(watch.uncommitted)}"
            )
            return 2
    plan = load_plan(exp_root)
    errors = anchor_errors(plan)
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    if args.check:
        print(f"OK: {len(plan.get('mutation', []))} mutation anchors for {args.id}")
        return 0

    selected, errors = select(plan, args.only)
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 2
    outcomes = []
    for mutation in selected:
        outcome = run_mutation(plan, mutation, args.timeout, watch)
        outcomes.append(outcome)
        print(f"{outcome['name']}: {outcome['result']} {outcome.get('counters', {})}", flush=True)

    # Rebuild the unmutated harness so no mutated binary is left behind. Until
    # that succeeds the binary may still hold the last mutation, so a failed
    # rebuild fails the run whatever the mutations did, and a run that fails
    # writes no record: mutations.json says nothing of the rebuild, so the
    # aggregators would take it as valid evidence.
    build, run = binary_command(plan, 1, int(plan["seed"]))
    rebuilt = subprocess.run(build, cwd=ROOT, capture_output=True, text=True, check=False)
    restored = rebuilt.returncode == 0
    if not restored:
        print(
            f"ERROR: rebuilding the unmutated harness failed; {run[0]} may still hold a mutation, "
            "and no record is written\n" + rebuilt.stderr[-2000:]
        )

    killed = all(outcome["result"] == "killed" for outcome in outcomes)
    if watch is not None:
        # Every mutated build read the tree while it ran: the record may name
        # HEAD only if the tree stayed so, apart from the checker's own edits.
        try:
            changes = watch.changes()
        except experiment_records.ProvenanceError as error:
            changes = [str(error)]
        if changes:
            print(
                "ERROR: not recording the mutations: their sources changed while they ran, so "
                f"{watch.head} may not be the code they mutated; {'; '.join(changes)}; "
                "rerun from a working tree that stays at HEAD"
            )
            return 2
    if watch is not None and restored:
        record = {
            "experiment_id": args.id,
            "git_sha": watch.head,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "subcommand": plan["subcommand"],
            "killed": sum(outcome["result"] == "killed" for outcome in outcomes),
            "total": len(outcomes),
            "mutations": outcomes,
        }
        out = exp_root / "results/mutations.json"
        experiment_records.write_atomically(out, experiment_records.json_text(record))
        print(out.relative_to(ROOT))
    return 0 if restored and killed else 1


if __name__ == "__main__":
    sys.exit(main())
