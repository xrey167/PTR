from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_research_gates  # noqa: E402
import experiment_records  # noqa: E402

PLACEHOLDER = experiment_records.PLACEHOLDER


def load(path: Path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def registry():
    return {e["id"]: e for e in load(REGISTRY).get("experiment", [])}


def resolve(exp_id: str):
    item = registry().get(exp_id)
    if not item:
        raise SystemExit(f"unknown experiment: {exp_id}")
    root = ROOT / "experiments" / item["path"]
    return item, root, load(root / "experiment.toml")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json_exclusive(path: Path, record: dict) -> None:
    """Write `record` to the new file `path`, whole or not at all: a write
    that fails leaves no partial record for an aggregator to select. The
    runner refuses a manifest holding a TOML date or time before anything
    runs (`dated_manifest`); were one to reach this point, it would be
    written as its ISO 8601 text (`experiment_records.toml_time`) rather
    than lose the record of a run that ran. Raises `FileExistsError` when
    `path` exists; a record never replaces another."""
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment_records.write_exclusively(path, json.dumps(record, indent=2, default=experiment_records.toml_time) + "\n")


def write_json_replacing(path: Path, record: dict) -> None:
    """Write `record` to `path` in one step, replacing the reservation the
    run made there (`reserve_run`): the record appears whole or not at all,
    and a failed write leaves the reservation."""
    temporary = experiment_records.write_temporary(
        path, (json.dumps(record, indent=2, default=experiment_records.toml_time) + "\n").encode("utf-8")
    )
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    experiment_records.sync_directory(path.parent)


def run_lock(exp_id: str) -> Path | None:
    """Take the lock on runs of `exp_id`, a file in git's own directory
    created only if absent, and return it, or return None, printing why,
    when another run holds it: two runs of a listed experiment at once could
    both find a seed not yet run, both run it and keep the better record.
    A lock left by a run that died is removed by hand, once no run of it is
    in progress."""
    common = experiment_records.git(ROOT, "rev-parse", "--git-common-dir")
    if common.returncode != 0:
        print(f"ERROR: cannot find git's directory: {common.stderr.strip()}", file=sys.stderr)
        return None
    lock = (ROOT / common.stdout.strip()).resolve() / f"ptr-run-{exp_id}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(
            f"ERROR: another run of {exp_id} holds {lock}; a listed experiment runs one seed at a time, and the lock "
            "of a run that died is removed by hand once no run of it is in progress",
            file=sys.stderr,
        )
        return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    return lock


def validate():
    schema = load(ROOT / "experiments/schema.toml")
    required = set(schema["required"])
    allowed = set(schema["allowed_status"])
    errors = []
    for exp_id, item in registry().items():
        root = ROOT / "experiments" / item["path"]
        path = root / "experiment.toml"
        if not path.exists():
            errors.append(f"{exp_id}: missing experiment.toml")
            continue
        data = load(path)
        missing = required - set(data)
        if missing:
            errors.append(f"{exp_id}: missing {sorted(missing)}")
        if data.get("id") != exp_id:
            errors.append(f"{exp_id}: id mismatch")
        if data.get("status") not in allowed:
            errors.append(f"{exp_id}: invalid status")
        if item.get("status") != data.get("status"):
            errors.append(f"{exp_id}: registry/manifest status mismatch")
        if data.get("status") in {"running", "completed", "failed"} and not str(
            data.get("entrypoint", "")
        ).strip():
            errors.append(f"{exp_id}: active experiment needs executable entrypoint")
        for key in experiment_records.temporal_keys(data):
            errors.append(
                f"{exp_id}: experiment.toml holds a TOML date or time at {key}, which a run record, holding the "
                "manifest as JSON, cannot tell from a string; write it as a string"
            )
        if not (root / "config.toml").exists():
            errors.append(f"{exp_id}: missing config.toml")
        if not (root / "tests").exists():
            errors.append(f"{exp_id}: missing tests/")
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print(f"OK: validated {len(registry())} experiments")
    return 0


def base_record(exp_id: str, data: dict, root: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": exp_id,
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "manifest": data,
        "manifest_sha256": sha(root / "experiment.toml"),
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "uv_lock_sha256": sha(ROOT / "training/uv.lock"),
        "hardware_profile": data.get("hardware_profile"),
    }


def dated_manifest(exp_id: str, data: dict) -> bool:
    """Print why the manifest `data` of `exp_id` cannot be recorded and
    return True, or return False when it can: a run record holds the
    manifest as JSON, which has no date or time, so a TOML date and the
    string of its text would be one manifest to aggregation
    (`experiment_records.temporal_keys`)."""
    keys = experiment_records.temporal_keys(data)
    for key in keys:
        print(
            f"ERROR: {exp_id}: experiment.toml holds a TOML date or time at {key}, which a run record, holding the "
            "manifest as JSON, cannot tell from a string; write it as a string",
            file=sys.stderr,
        )
    return bool(keys)


def is_listed(exp_id: str) -> bool:
    """Whether `experiments/preregistration.toml` names `exp_id`, so its
    launches are bound to its frozen preregistration."""
    entries = load(ROOT / check_research_gates.PREREGISTRATION).get("experiment", {})
    return isinstance(entries, dict) and exp_id in entries


def seed_runs(results: Path, seed: int) -> list[str]:
    """The run records in `results` of a run of `seed` whose command ran,
    as repository paths: every record naming the seed but one whose command
    failed to launch, which saw no outcome. Raises `ValueError` for a record
    it cannot read, since it could be one."""
    found = []
    for path in sorted(results.glob("run-*.json")):
        name = path.relative_to(ROOT).as_posix()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} cannot be read, so whether seed {seed} ran is unknown: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{name} is not a JSON object, so whether seed {seed} ran is unknown")
        if "seed" in record and record["seed"] == seed and record.get("status") != "failed-to-launch":
            found.append(name)
    return found


def launch_refused(exp_id: str) -> bool:
    """Print why `exp_id` may not be run or prepared now and return True, or
    return False when it may: an experiment `experiments/preregistration.toml`
    lists runs only once its preregistration is frozen
    (`check_research_gates.launch_errors`)."""
    problems = check_research_gates.launch_errors(ROOT, exp_id)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return bool(problems)


def results_directory(root: Path, data: dict) -> Path | None:
    """The results directory of the experiment at `root` whose manifest is
    `data`, or None, printing why, when it cannot hold its records: it must
    lie below the experiment's directory, outside git's own directory
    (`check_research_gates.is_git_administration`), where no record could be
    committed, and be reached through no symlink and no file. The runner
    holds the rest of that directory to HEAD, so a results directory that is
    the experiment's own, or a link that carries records elsewhere, would
    take sources out of that watch; and a file on the way would leave no
    directory to write the record into once the run had run."""
    named = data.get("results_dir", "results")
    relative = PurePosixPath(named) if isinstance(named, str) and named else None
    if relative is None or relative.is_absolute() or not relative.parts or ".." in relative.parts:
        print(f"ERROR: results_dir {named!r} is not a directory below {root.relative_to(ROOT)}", file=sys.stderr)
        return None
    if any(check_research_gates.is_git_administration(part) for part in relative.parts):
        print(
            f"ERROR: results_dir {named!r} passes through git's own directory, where no record can be committed",
            file=sys.stderr,
        )
        return None
    step = root
    for part in relative.parts:
        step = step / part
        if step.is_symlink():
            print(f"ERROR: results directory {step.relative_to(ROOT)} is a symlink", file=sys.stderr)
            return None
        if step.exists() and not step.is_dir():
            print(f"ERROR: results directory {step.relative_to(ROOT)} is not a directory", file=sys.stderr)
            return None
    return root / relative


def launch_watch(
    exp_id: str, root: Path, results: Path, data: dict, outputs: tuple[str, ...] = experiment_records.RESULT_OUTPUTS
) -> experiment_records.ProvenanceWatch | None:
    """The watch on every file that decides a launch of `exp_id` (the
    provenance files, the experiment's directory but the `outputs` the tools
    write into `results`, and `check_research_gates.launch_inputs`), or
    None, printing why, when git cannot tell, HEAD does not hold one of
    them, or `results` holds a file git ignores, which the watch would not
    see: a record names HEAD as what it ran, so it may be written only from
    a tree that holds HEAD. A listed experiment's watch leaves out only the
    record the run itself writes, so every earlier record and output, which
    a command could read, is one HEAD holds. A
    listed experiment also needs HEAD to hold it frozen as the tree launches
    it (`check_research_gates.launch_commit_errors`), each input a regular
    file HEAD holds, so the gate can find the freeze from HEAD alone. And
    `data`, the manifest the command and the record are built from, read
    before the watch looked, must be the one the watched tree holds: a
    manifest changed or committed in between would run what HEAD does not
    hold."""
    try:
        watch = experiment_records.ProvenanceWatch(
            ROOT,
            [
                *experiment_records.tree_pathspecs(
                    root, results, ROOT, experiment_records.seed_record_paths(root, ROOT), outputs
                ),
                # What decided that this experiment may launch: an edit to it
                # would be undone after the outcome is seen.
                *check_research_gates.launch_inputs(ROOT, exp_id),
            ],
        )
    except experiment_records.ProvenanceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return None
    if watch.uncommitted:
        print(
            "ERROR: refusing to run from a working tree whose sources HEAD does not hold; "
            f"commit or remove {experiment_records.listed(watch.uncommitted)}",
            file=sys.stderr,
        )
        return None
    # The watch leaves out what git ignores, and the repository ignores
    # most of a results directory: code or input kept there could change
    # between runs while every record named one commit.
    try:
        ignored = experiment_records.listed_names(
            ROOT, "--literal-pathspecs", "ls-files", "-z", "--others", "--ignored",
            experiment_records.PER_DIRECTORY, "--", results.relative_to(ROOT).as_posix(),
        )
    except experiment_records.ProvenanceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return None
    if ignored:
        print(
            f"ERROR: refusing to run while the results directory holds {experiment_records.listed(ignored)}, which git "
            "ignores; it holds the records the tools write and files HEAD holds, nothing a command could run or read "
            "unrecorded",
            file=sys.stderr,
        )
        return None
    problems = check_research_gates.launch_commit_errors(ROOT, exp_id, watch.head)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        return None
    try:
        held = load(root / "experiment.toml")
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        held = None
    if held is None or not check_research_gates.same_value(held, data):
        print(
            "ERROR: experiment.toml changed while the launch was checked; rerun from a tree that holds HEAD",
            file=sys.stderr,
        )
        return None
    return watch


def prepare(exp_id: str):
    """Write a `prepared` record of `exp_id` at HEAD, once it may launch
    (`launch_refused`) and HEAD holds every file that decides that
    (`launch_watch`), and its manifest holds no TOML date or time
    (`dated_manifest`)."""
    _, root, data = resolve(exp_id)
    if dated_manifest(exp_id, data) or launch_refused(exp_id):
        return 2
    results = results_directory(root, data)
    if results is None:
        return 2
    timestamp = utc_stamp()
    out = results / f"run-{timestamp}.json"
    outputs = (out.name,) if is_listed(exp_id) else experiment_records.RESULT_OUTPUTS
    watch = launch_watch(exp_id, root, results, data, outputs)
    if watch is None:
        return 2
    record = base_record(exp_id, data, root)
    record.update({"git_sha": watch.head, "status": "prepared", "prepared_at": timestamp})
    write_json_exclusive(out, record)
    print(out.relative_to(ROOT))
    return 0


def parse_params(items: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"parameter must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if not PLACEHOLDER.fullmatch(f"<{key}>"):
            raise ValueError(f"invalid parameter name: {key}")
        if key in params:
            raise ValueError(f"duplicate parameter: {key}")
        params[key] = value
    return params


def build_command(
    data: dict,
    *,
    entrypoint: str,
    seed: int,
    params: dict[str, str] | None = None,
) -> list[str]:
    seeds = data.get("seeds", [])
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not declared in experiment seeds {seeds}")
    if "seed" in (params or {}):
        # The record names the seed --seed gave; a parameter would run another.
        raise ValueError("the seed is given by --seed, not by a parameter")

    template = str(data.get(entrypoint, "")).strip()
    if not template:
        raise ValueError(f"experiment has no executable {entrypoint!r}")

    values = {"seed": str(seed)}
    values.update(params or {})
    tokens = shlex.split(template)
    taken = {name for token in tokens for name in PLACEHOLDER.findall(token)}
    missing = sorted(taken - values.keys())
    if missing:
        raise ValueError(f"missing value for <{missing[0]}>")
    # A value no placeholder takes would be recorded as a parameter of a
    # run it had no part in.
    unused = sorted(values.keys() - taken - {"seed"})
    if unused:
        raise ValueError(f"no placeholder of {entrypoint!r} takes --set {', '.join(unused)}")
    # One pass, so a value is taken as itself: text in it that reads as a
    # placeholder is not substituted again.
    return [PLACEHOLDER.sub(lambda match: values[match.group(1)], token) for token in tokens]


def token_text(name: str, value) -> str:
    """The preregistered `value` of the placeholder `<name>` as a command
    token takes it: an integer in decimal, a boolean as TOML writes it, a
    string as itself. Raises `ValueError` for any other value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return str(value)
    raise ValueError(f"<{name}> is preregistered as a {type(value).__name__}, which no command token takes")


def command_parameters(exp_id: str, root: Path, data: dict, entrypoint: str, params: dict[str, str]) -> dict[str, str]:
    """The values the placeholders other than `<seed>` take in the command
    the manifest `data` of `exp_id`, whose directory is `root`, names at
    `entrypoint`. An experiment the list names runs only through
    `entrypoint`, the manifest's own command, and takes each value from its
    frozen `[preregistration]` table (`token_text`), a `--set` value only
    repeating it: a command or a value chosen at launch could be chosen
    after an outcome was seen, the runs made with the others discarded. A
    variant of its command is a placeholder the table fixes. Any other
    experiment runs through any `entrypoint` with the `--set` values
    `params`, each of which a placeholder of that command must take
    (`build_command`). Raises `ValueError`, for a listed experiment, for
    another entrypoint, a placeholder its table does not hold or holds as a
    list, a `--set` naming no placeholder the table fills, and a `--set`
    value that is not the preregistered one."""
    if not is_listed(exp_id):
        return params
    if entrypoint != "entrypoint":
        raise ValueError(
            f"--entrypoint {entrypoint}: a listed experiment runs only through its manifest's entrypoint; a variant "
            "of its command is a placeholder its frozen [preregistration] table fixes"
        )
    table = load(root / "config.toml").get("preregistration", {})
    template = str(data.get(entrypoint, ""))
    names = sorted({name for token in shlex.split(template) for name in PLACEHOLDER.findall(token)} - {"seed"})
    values = {}
    for name in names:
        if name not in table:
            raise ValueError(
                f"<{name}> is no key of the frozen [preregistration] table; a listed experiment's command "
                "takes only preregistered values"
            )
        values[name] = token_text(name, table[name])
    for key, value in params.items():
        if key not in values:
            raise ValueError(f"--set {key} names no placeholder the command takes from the frozen [preregistration] table")
        if value != values[key]:
            raise ValueError(f"--set {key}={value} is not the preregistered value {values[key]}")
    return values


def execute_command(command: list[str]) -> dict:
    """Run `command` from the repository's root and return its exit status,
    output, launch error and duration. Python in it reads and writes its
    bytecode cache in a fresh directory (`PYTHONPYCACHEPREFIX`), so no
    `__pycache__` entry the tree holds, which git ignores and HEAD does not
    hold, runs in place of a tracked source."""
    started = time.perf_counter_ns()
    cache = tempfile.TemporaryDirectory()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPYCACHEPREFIX": cache.name},
        )
        result = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "launch_error": None,
        }
    except OSError as error:
        result = {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "launch_error": f"{type(error).__name__}: {error}",
        }
    finally:
        cache.cleanup()
    result["duration_ns"] = time.perf_counter_ns() - started
    return result


def run_experiment(
    exp_id: str,
    *,
    entrypoint: str,
    seed: int,
    params: dict[str, str] | None = None,
) -> int:
    """Run one seed of `exp_id` through `entrypoint` and write its record to
    the experiment's results, stamped with the commit HEAD was at when the
    run started. Refuses with status 2, before anything runs or is written,
    an undeclared seed, an unresolved command, and a working tree with
    uncommitted or untracked provenance files or experiment files
    (`experiment_records.uncommitted_files`, which also covers the files that
    decide whether it may launch, `check_research_gates.launch_inputs`), and
    a listed experiment whose preregistration is not frozen
    (`launch_refused`), and a manifest holding a TOML date or time
    (`dated_manifest`). A listed experiment runs only through its
    `entrypoint`, with its preregistered values (`command_parameters`), and
    each seed once (`seed_runs`): a seed run again after its outcome was
    seen could keep whichever run came out best. Its run holds a lock on
    the experiment's runs (`run_lock`), reads no uncommitted file in its
    results, and reserves its record before the command starts, the whole
    record replacing the reservation once it ends (`write_json_replacing`).
    After the command ends it looks at the same tree again
    (`experiment_records.ProvenanceWatch`) and writes no record, returning
    2, when HEAD moved or a provenance or experiment file was written,
    created or removed while the command ran, even if its content was put
    back, so the record's `git_sha` is the code that ran as far as that
    watch can see (its `changes` names what it cannot); a listed
    experiment's reservation then stays, as the record that its seed ran.
    The record is written whole or not at all (`write_json_exclusive`)."""
    _, root, data = resolve(exp_id)
    if dated_manifest(exp_id, data) or launch_refused(exp_id):
        return 2

    # The record names HEAD as the code it ran, so HEAD must hold every file
    # that decides the run: refuse before anything runs or is written.
    results = results_directory(root, data)
    if results is None:
        return 2
    listed = is_listed(exp_id)
    timestamp = utc_stamp()
    out = results / f"run-{timestamp}-seed-{seed}.json"
    lock = run_lock(exp_id) if listed else None
    if listed and lock is None:
        return 2
    try:
        return launch_and_record(
            exp_id, root, data, results, out, timestamp, entrypoint=entrypoint, seed=seed, params=params, listed=listed
        )
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


def launch_and_record(
    exp_id: str,
    root: Path,
    data: dict,
    results: Path,
    out: Path,
    timestamp: str,
    *,
    entrypoint: str,
    seed: int,
    params: dict[str, str] | None,
    listed: bool,
) -> int:
    """The part of `run_experiment` from the watch on, for the record `out`
    stamped `timestamp`; a listed experiment's run holds its lock
    throughout."""
    watch = launch_watch(exp_id, root, results, data, (out.name,) if listed else experiment_records.RESULT_OUTPUTS)
    if watch is None:
        return 2
    # Built once the watch holds the tree to HEAD, so a listed experiment's
    # preregistered values are the ones HEAD holds.
    try:
        params = command_parameters(exp_id, root, data, entrypoint, params or {})
        command = build_command(data, entrypoint=entrypoint, seed=seed, params=params)
        # A listed experiment runs each seed once: every run of it is
        # evidence, so none can be chosen by its outcome. The results
        # directory holds every record committed, which the gate keeps.
        ran = seed_runs(results, seed) if listed else []
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if ran:
        print(
            f"ERROR: seed {seed} of {exp_id} already ran ({', '.join(ran)}); a listed experiment runs each seed once, "
            "so no run of it is chosen by its outcome",
            file=sys.stderr,
        )
        return 2

    record = base_record(exp_id, data, root)
    record.update(
        {
            "git_sha": watch.head,
            "status": "started",
            "started_at": timestamp,
            "entrypoint": entrypoint,
            "seed": seed,
            "parameters": params,
            "command": command,
        }
    )
    if listed:
        # The reservation: the record, as far as it is known before the
        # command starts, which the whole record replaces once it ends. A run
        # that dies or goes unrecorded leaves it, and its seed has run.
        try:
            write_json_exclusive(out, record)
        except OSError as error:
            print(f"ERROR: cannot reserve {out.relative_to(ROOT)}: {error}", file=sys.stderr)
            return 2
    kept = f"; {out.relative_to(ROOT)} stays as the record that seed {seed} ran" if listed else ""

    execution = execute_command(command)
    exit_code = execution["exit_code"]
    # The command read the tree while it ran (a `cargo run` entrypoint
    # compiles it first): the record may name HEAD only if the tree stayed so.
    try:
        changes = watch.changes()
    except experiment_records.ProvenanceError as error:
        changes = [str(error)]
    if changes:
        print(
            f"ERROR: not recording the run (exit status {exit_code}): its sources changed while it ran, "
            f"so {watch.head} may not be the code it ran; {'; '.join(changes)}; "
            f"rerun from a working tree that stays at HEAD{kept}",
            file=sys.stderr,
        )
        return 2
    # The command could have put a file or a link where the results
    # directory is, which the watch leaves out.
    if results_directory(root, data) is None:
        print(
            f"ERROR: not recording the run (exit status {exit_code}): its results directory changed while it ran{kept}",
            file=sys.stderr,
        )
        return 2
    record.update(
        {
            "status": (
                "completed"
                if exit_code == 0
                else "failed" if exit_code is not None else "failed-to-launch"
            ),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **execution,
        }
    )

    if listed:
        write_json_replacing(out, record)
    else:
        write_json_exclusive(out, record)
    print(out.relative_to(ROOT))
    return exit_code if exit_code is not None else 127


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("validate")

    show = sub.add_parser("show")
    show.add_argument("id")

    prep = sub.add_parser("prepare")
    prep.add_argument("id")

    run = sub.add_parser("run")
    run.add_argument("id")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--entrypoint", default="entrypoint")
    run.add_argument("--set", dest="params", action="append", default=[])

    args = parser.parse_args()
    if args.cmd == "list":
        for key, value in registry().items():
            print(key, value["status"], value["path"])
        return
    if args.cmd == "validate":
        raise SystemExit(validate())
    if args.cmd == "show":
        _, _, data = resolve(args.id)
        print(json.dumps(data, indent=2))
        return
    if args.cmd == "prepare":
        raise SystemExit(prepare(args.id))
    if args.cmd == "run":
        try:
            params = parse_params(args.params)
        except ValueError as error:
            parser.error(str(error))
        raise SystemExit(
            run_experiment(
                args.id,
                entrypoint=args.entrypoint,
                seed=args.seed,
                params=params,
            )
        )


if __name__ == "__main__":
    main()
