from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.parse
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"

# The runner writes no bytecode cache of the modules it imports into the
# tree: a `__pycache__` directory is one git ignores, and a listed experiment
# runs only from a checkout that holds none (`ignored_files`).
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_research_gates  # noqa: E402
import experiment_records  # noqa: E402

PLACEHOLDER = experiment_records.PLACEHOLDER
# The environment a listed experiment's command runs in, taken from the
# runner's: where its programs and toolchains are found, the home they
# default to, the temporary directory and the locale. Nothing else reaches
# the command, so no `PYTHONPATH`, `LD_PRELOAD`, `RUSTC_WRAPPER`,
# `RUSTUP_TOOLCHAIN` or `RUSTFLAGS` loads code the commit does not hold,
# and no credential or network setting reaches it.
COMMAND_ENVIRONMENT = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "CARGO_HOME", "RUSTUP_HOME")
# What the runner sets in that environment whatever the runner's holds:
# Python reads no packages from the user's own site directory under `HOME`,
# which the commit does not hold, and hashes strings the same way in every
# run, so an order that follows hashing (a set's) is the same for every
# seed rather than one each process draws at random.
FIXED_ENVIRONMENT = {"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"}
# The names Cargo reads its configuration from, in a `.cargo` directory and
# in its home.
CARGO_CONFIGURATIONS = ("config", "config.toml")
# rustup's proxies, which take a first argument `+<toolchain>` naming the
# toolchain to run (`cargo +stable run`).
RUSTUP_PROXIES = frozenset(
    ("cargo", "rustc", "rustdoc", "rustfmt", "cargo-fmt", "cargo-clippy", "clippy-driver", "cargo-miri", "rust-gdb",
     "rust-gdbgui", "rust-lldb", "rust-analyzer", "rls")
)


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
    runs (`unrecordable_manifest`); were one to reach this point, it would
    be written as its ISO 8601 text (`experiment_records.toml_time`) rather
    than lose the record of a run that ran. Raises `FileExistsError` when
    `path` exists; a record never replaces another."""
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment_records.write_exclusively(path, json.dumps(record, indent=2, default=experiment_records.toml_time) + "\n")


def write_json_replacing(path: Path, record: dict) -> None:
    """Write `record` to `path` in one step, replacing the reservation the
    run made there (`launch_and_record`): the record appears whole or not at
    all, and a failed write leaves the reservation."""
    replace_file(path, (json.dumps(record, indent=2, default=experiment_records.toml_time) + "\n").encode("utf-8"))


def replace_file(path: Path, data: bytes) -> None:
    """Write `data` to `path` in one step, creating its directory when it
    has none and replacing what is there: the file appears whole or not at
    all, and a failed write leaves what was there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = experiment_records.write_temporary(path, data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    experiment_records.sync_directory(path.parent)


def git_directory() -> Path | None:
    """git's own directory for the repository, which its worktrees share,
    or None, printing why, when git cannot name it."""
    common = experiment_records.git(ROOT, "rev-parse", "--git-common-dir")
    if common.returncode != 0:
        print(f"ERROR: cannot find git's directory: {common.stderr.strip()}", file=sys.stderr)
        return None
    return (ROOT / common.stdout.strip()).resolve()


def attempts_directory(directory: Path, results: Path) -> Path:
    """Where, in git's own `directory`, the runner keeps a copy of the
    record of every run of a listed experiment whose results directory is
    `results`: under the results directory's repository path, out of the
    reach of a command that clears its results. The copy is written before
    the command starts and replaced by the whole record once it ends, so it
    outlasts a reservation the command removed or rewrote while its runner
    was stopped; `seed_runs` reads it, and `restore_records` puts back a
    record it holds that the results directory lost."""
    return directory / "ptr-runs" / results.relative_to(ROOT)


def restore_records(results: Path, attempts: Path) -> list[str]:
    """Put back into `results`, from the copies in `attempts`
    (`attempts_directory`), each record of a run at HEAD or a commit on its
    history that HEAD does not hold and that the results directory holds no
    longer, or holds otherwise than the copy, and return their repository
    paths. A record HEAD holds is the gate's, which keeps it as it was
    committed. A run at a commit on another line of history, another branch
    or another worktree's, belongs to that history, whose record the gate
    here would refuse: its copy is not put back, and still counts its seed
    as run (`seed_runs`). Raises `OSError` when a copy cannot be read or a
    record written, and `check_research_gates.HistoryUnreadable` when git
    cannot read HEAD."""
    restored = []
    for kept in sorted(attempts.glob("run-*.json")):
        path = results / kept.name
        relative = path.relative_to(ROOT).as_posix()
        if check_research_gates.tree_entry(ROOT, "HEAD", relative) is not None:
            continue
        data = kept.read_bytes()
        try:
            commit = json.loads(data.decode("utf-8")).get("git_sha")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            # `seed_runs` refuses a copy it cannot read.
            continue
        if not (
            isinstance(commit, str)
            and experiment_records.COMMIT.fullmatch(commit)
            and experiment_records.is_ancestor(commit, "HEAD", ROOT)
        ):
            continue
        if not path.is_symlink() and path.is_file() and path.read_bytes() == data:
            continue
        replace_file(path, data)
        restored.append(relative)
    return restored


# The longest part of a file name the runner makes from an experiment's id,
# well within the 255 bytes most file systems allow a name.
ID_NAME_LIMIT = 120


def id_file_name(exp_id: str) -> str:
    """`exp_id` as part of a file name: every character a file name could
    not hold as itself percent-encoded (`team/trial` as `team%2Ftrial`),
    and where that runs longer than `ID_NAME_LIMIT`, as a long id or one
    of characters that encode to several bytes does, its beginning with the
    SHA-256 of the id, so every id names one file a file system can hold."""
    quoted = urllib.parse.quote(exp_id, safe="")
    if len(quoted) <= ID_NAME_LIMIT:
        return quoted
    digest = hashlib.sha256(exp_id.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{quoted[: ID_NAME_LIMIT - len(digest) - 1]}-{digest}"


def run_lock(exp_id: str, directory: Path) -> Path | None:
    """Take the lock on runs of `exp_id`, a file in git's own `directory`
    created only if absent, and return it, or return None, printing why,
    when another run holds it: two runs of a listed experiment at once could
    both find a seed not yet run, both run it and keep the better record.
    A lock left by a run that died is removed by hand, once no run of it is
    in progress; the copy of its record the run kept (`attempts_directory`)
    stays, so its seed has run all the same. The lock is named by the
    experiment's id as a file name can hold it (`id_file_name`); a lock
    that cannot be created is refused as well."""
    lock = directory / f"ptr-run-{id_file_name(exp_id)}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(
            f"ERROR: another run of {exp_id} holds {lock}; a listed experiment runs one seed at a time, and the lock "
            "of a run that died is removed by hand once no run of it is in progress",
            file=sys.stderr,
        )
        return None
    except OSError as error:
        print(f"ERROR: cannot take the lock on runs of {exp_id} in {lock}: {error}", file=sys.stderr)
        return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    return lock


def validate():
    schema = load(ROOT / "experiments/schema.toml")
    required = set(schema["required"])
    allowed = set(schema["allowed_status"])
    errors = []
    # `registry` keeps one entry of an id: a second, in another directory,
    # would go unchecked here and leave the runner no one place to launch.
    ids = [entry.get("id") for entry in load(REGISTRY).get("experiment", []) if isinstance(entry, dict)]
    for exp_id in sorted({exp_id for exp_id in ids if ids.count(exp_id) > 1}, key=str):
        errors.append(f"{exp_id}: registered {ids.count(exp_id)} times; an experiment is registered once")
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
        errors.extend(f"{exp_id}: {problem}" for problem in experiment_records.manifest_problems(data))
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


def unrecordable_manifest(exp_id: str, data: dict) -> bool:
    """Print why the manifest `data` of `exp_id` cannot be recorded and
    return True, or return False when it can: a run record holds the
    manifest as JSON, which has no date or time and writes a NaN of either
    sign as `NaN`, so a TOML date and the string of its text, or `nan` and
    `-nan`, would be one manifest to aggregation
    (`experiment_records.manifest_problems`)."""
    problems = experiment_records.manifest_problems(data)
    for problem in problems:
        print(f"ERROR: {exp_id}: {problem}", file=sys.stderr)
    return bool(problems)


def is_listed(exp_id: str) -> bool:
    """Whether `experiments/preregistration.toml` names `exp_id`, so its
    launches are bound to its frozen preregistration."""
    entries = load(ROOT / check_research_gates.PREREGISTRATION).get("experiment", {})
    return isinstance(entries, dict) and exp_id in entries


def seed_runs(results: Path, seed: int, attempts: Path) -> list[str]:
    """The records of a run of `seed` whose command ran, as the repository
    paths of the records in `results`: every record naming the seed but one
    whose command failed to launch, which saw no outcome, in `results` or
    among the copies the runner keeps in `attempts` (`attempts_directory`),
    which a command cannot clear with its results. Raises `ValueError` for a
    record or copy it cannot read, since it could be one."""
    found = set()
    for directory in (results, attempts):
        for path in sorted(directory.glob("run-*.json")):
            name = path.relative_to(ROOT).as_posix() if directory == results else str(path)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{name} cannot be read, so whether seed {seed} ran is unknown: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{name} is not a JSON object, so whether seed {seed} ran is unknown")
            if "seed" in record and record["seed"] == seed and record.get("status") != "failed-to-launch":
                found.add((results / path.name).relative_to(ROOT).as_posix())
    return sorted(found)


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


def ignored_files() -> list[str]:
    """What git ignores in the repository, by any rule (a build's output
    under `target/`, a virtual environment, a `__pycache__` directory): a
    directory all of whose files it ignores is named as the directory. No
    commit holds such a file and the watch does not see one, and a listed
    experiment's command could run or read it. Raises
    `experiment_records.ProvenanceError` when git cannot list them."""
    return experiment_records.listed_names(
        ROOT, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"
    )


def unlisted_entries() -> list[str]:
    """What the repository holds that git neither tracks nor lists, sorted,
    so no watch or record sees it: an empty directory, whose presence a
    command can test as a file's, and an entry named as git's own directory
    below the root, however a platform names it
    (`check_research_gates.is_git_administration`: `.GIT`, and on Windows
    `.git.` and `git~1`), which git keeps for itself and never looks into
    (`scripts/.git/helper.py`), or at the root under any name but `.git`.
    A walk of the checkout, leaving out git's
    own directory at the root and what git ignores (`ignored_files`), which
    a listed run refuses by itself. Raises
    `experiment_records.ProvenanceError` when git cannot list what it
    ignores."""
    ignored = {entry.rstrip("/") for entry in ignored_files()}
    found = []
    for directory, subdirectories, files in os.walk(ROOT):
        relative = Path(directory).relative_to(ROOT).as_posix()
        here = "" if relative == "." else relative
        if here and not subdirectories and not files:
            found.append(here)
        kept = []
        for name in subdirectories:
            path = f"{here}/{name}" if here else name
            if check_research_gates.is_git_administration(name):
                if here or name != ".git":
                    found.append(path)
            elif path not in ignored:
                kept.append(name)
        subdirectories[:] = kept
        found.extend(
            f"{here}/{name}" if here else name
            for name in files
            if check_research_gates.is_git_administration(name) and (here or name != ".git")
        )
    return sorted(found)


def launch_watch(
    exp_id: str,
    root: Path,
    results: Path,
    data: dict,
    outputs: tuple[str, ...] = experiment_records.RESULT_OUTPUTS,
    listed: bool = False,
) -> experiment_records.ProvenanceWatch | None:
    """The watch on every file that decides a launch of `exp_id` (the
    provenance files, the experiment's directory but the `outputs` the tools
    write into `results`, and `check_research_gates.launch_inputs`), or
    None, printing why, when git cannot tell, HEAD does not hold one of
    them, or `results` holds a file git ignores, which the watch would not
    see: a record names HEAD as what it ran, so it may be written only from
    a tree that holds HEAD. A listed experiment's watch is on the whole
    repository, since its command may run or read any file there (a script
    under `scripts/`, say), and leaves out only the record the run itself
    writes, so every earlier record and output, which a command could read,
    is one HEAD holds. What git ignores (build output, caches) is no file
    of the commit. A
    listed experiment also needs HEAD to hold it frozen as the tree launches
    it (`check_research_gates.launch_commit_errors`), each input a regular
    file HEAD holds, so the gate can find the freeze from HEAD alone; and
    HEAD decides whether the run is listed, `listed` having been read from
    the tree before the watch looked
    (`check_research_gates.launch_mode_errors`): the list as HEAD holds it
    names `exp_id` exactly when `listed`, and an unlisted run's experiment
    is one the list has never named on HEAD's history. And
    `data`, the manifest the command and the record are built from, read
    before the watch looked, must be the one the watched tree holds: a
    manifest changed or committed in between would run what HEAD does not
    hold."""
    try:
        watch = experiment_records.ProvenanceWatch(
            ROOT,
            [
                *experiment_records.tree_pathspecs(
                    root, results, ROOT, (".",) if listed else experiment_records.seed_record_paths(root, ROOT), outputs
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
    # Whether the run is listed was read from the tree before the watch
    # looked: the commit the record would name decides, and an experiment
    # the list has named runs only as listed.
    problems = check_research_gates.launch_mode_errors(
        ROOT, exp_id, watch.head, listed
    ) or check_research_gates.launch_commit_errors(ROOT, exp_id, watch.head)
    # What decided the launch was read before the watch looked, and a commit
    # made in between could have changed it: decide again from the tree the
    # watch holds to HEAD, and from the directory its registry places the
    # experiment in, where its command's values are read.
    if not problems:
        problems = check_research_gates.launch_errors(ROOT, exp_id)
    if not problems:
        placed = registry().get(exp_id)
        if not placed or (ROOT / "experiments" / str(placed.get("path"))).resolve() != root.resolve():
            problems = [
                f"{check_research_gates.REGISTRY} placed {exp_id} elsewhere while its launch was checked; rerun from a "
                "tree that holds HEAD"
            ]
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        return None
    # A listed experiment's command may run or read any file of the
    # repository, one git ignores (a build's output, a virtual environment)
    # as well, which the watch does not see: it runs from a checkout that
    # holds none, and builds into a directory of its own outside it
    # (`execute_command`).
    if listed:
        try:
            ignored = ignored_files()
        except experiment_records.ProvenanceError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return None
        if ignored:
            print(
                f"ERROR: refusing to run {exp_id} while the repository holds {experiment_records.listed(ignored)}, "
                "which git ignores and its command could run or read unrecorded; a listed experiment runs from a "
                "checkout that holds no such file (`git clean -ndX` lists them; a fresh worktree holds none)",
                file=sys.stderr,
            )
            return None
        try:
            unlisted = unlisted_entries()
        except experiment_records.ProvenanceError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return None
        if unlisted:
            print(
                f"ERROR: refusing to run {exp_id} while the repository holds {experiment_records.listed(unlisted)}, "
                "which git does not list (an empty directory, or an entry named .git below the root) and its command "
                "could read unrecorded; a listed experiment runs from a checkout that holds none",
                file=sys.stderr,
            )
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
    (`launch_watch`), and its manifest holds no TOML date or time and no
    NaN (`unrecordable_manifest`). For a listed experiment it holds the
    lock on its runs meanwhile (`run_lock`): a seed run watches every file
    of its results but its own record, so a record written while its
    command ran would leave that run unrecorded and its seed spent."""
    _, root, data = resolve(exp_id)
    if unrecordable_manifest(exp_id, data) or launch_refused(exp_id):
        return 2
    results = results_directory(root, data)
    if results is None:
        return 2
    if not is_listed(exp_id):
        return write_prepared(exp_id, root, data, results, listed=False)
    directory = git_directory()
    if directory is None:
        return 2
    lock = run_lock(exp_id, directory)
    if lock is None:
        return 2
    try:
        return write_prepared(exp_id, root, data, results, listed=True)
    finally:
        lock.unlink(missing_ok=True)


def write_prepared(exp_id: str, root: Path, data: dict, results: Path, *, listed: bool) -> int:
    """Write `prepare`'s record of `exp_id` into `results`, once HEAD holds
    every file that decides a launch (`launch_watch`)."""
    timestamp = utc_stamp()
    out = results / f"run-{timestamp}.json"
    outputs = (out.name,) if listed else experiment_records.RESULT_OUTPUTS
    watch = launch_watch(exp_id, root, results, data, outputs, listed)
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


def command_environment() -> dict[str, str]:
    """The environment a listed experiment's command runs in: the variables
    `COMMAND_ENVIRONMENT` names that the runner's environment sets, and
    `FIXED_ENVIRONMENT`, and no other. A listed run is refused while one of
    them names a directory by a relative path (`relative_directories`)."""
    environment = {name: os.environ[name] for name in COMMAND_ENVIRONMENT if name in os.environ}
    return {**environment, **FIXED_ENVIRONMENT}


# The variables of a listed experiment's environment besides `PATH` that
# name a directory.
DIRECTORY_VARIABLES = ("HOME", "TMPDIR", "CARGO_HOME", "RUSTUP_HOME")


def relative_directories(environment: dict[str, str]) -> list[str]:
    """The variables of `environment` that name a directory by a relative
    path: `PATH` when an entry of it is relative (an empty one is the
    current directory), and each of `DIRECTORY_VARIABLES` set to one. Each
    program resolves such a path from wherever it starts (the runner from
    its own directory, the command from the root, a program the command
    starts from its own), so no record could say which directory the run
    used, nor would seeds run from two checkouts name one environment; a
    listed run is refused while one does."""
    found = []
    if "PATH" in environment and any(not os.path.isabs(entry) for entry in environment["PATH"].split(os.pathsep)):
        found.append("PATH")
    found.extend(name for name in DIRECTORY_VARIABLES if name in environment and not os.path.isabs(environment[name]))
    return found


def earlier_run_problems(exp_id: str, results: Path, head: str, record: dict) -> list[str]:
    """Why a listed run of `exp_id` at `head`, whose record would be
    `record`, could not be one more seed of the runs its results directory
    already holds (committed, as the launch reads none that is not), empty
    when it could: the gate refuses seeds that ran another program,
    toolchain or environment, or another repository but for their seed
    records (`check_research_gates.archived_errors`), and a seed found so
    only once it ran would be spent. A record whose command failed to launch
    ran nothing and counts for neither."""
    earlier = []
    for path in sorted(results.glob("run-*.json")):
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return [f"{path.relative_to(ROOT)} cannot be read, so what the earlier runs ran is unknown: {error}"]
        if isinstance(held, dict) and "seed" in held and held.get("status") != "failed-to-launch":
            earlier.append((str(held.get("started_at", "")), path.name, held))
    if not earlier:
        return []
    problems = []
    ran = json.dumps([record.get(key) for key in ("executable", "toolchain", "environment")], sort_keys=True)
    others = [
        name for _, name, held in earlier
        if json.dumps([held.get(key) for key in ("executable", "toolchain", "environment")], sort_keys=True) != ran
    ]
    if others:
        problems.append(
            f"{exp_id}: {experiment_records.listed(others)} ran another program, toolchain or environment than this run "
            "would; the seeds of a listed experiment run one of each"
        )
    _, first, held = min(earlier)
    paths = experiment_records.listed_record_paths(results.relative_to(ROOT).as_posix())
    try:
        changed = experiment_records.code_changes(str(held.get("git_sha")), head, ROOT, paths)
    except experiment_records.ProvenanceError as error:
        return [*problems, f"{exp_id}: cannot compare the repository with {first}'s: {error}"]
    if changed:
        problems.append(
            f"{exp_id}: the repository at {head[:12]} differs from {str(held.get('git_sha'))[:12]}'s, where {first} ran, "
            f"in {experiment_records.listed(changed)}; the seeds of a listed experiment run one tree, only their seed "
            "records committed between them"
        )
    return problems


def outside_cargo_configurations(environment: dict[str, str]) -> list[str]:
    """The Cargo configuration files outside the repository that Cargo, run
    from the repository's root in `environment`, would read: a
    `.cargo/config` or `.cargo/config.toml` in a directory above the root,
    and `config` or `config.toml` in Cargo's home (`CARGO_HOME`, or `.cargo`
    under `HOME`). The commit holds none of them, and one could set a rustc
    wrapper, flags or sources for the build; a listed experiment's run is
    refused while one exists. The repository's own `.cargo/config.toml` is
    a provenance file."""
    directories = [directory / ".cargo" for directory in ROOT.parents]
    if environment.get("CARGO_HOME"):
        directories.append(Path(environment["CARGO_HOME"]))
    elif environment.get("HOME"):
        directories.append(Path(environment["HOME"]) / ".cargo")
    found = []
    for directory in directories:
        for name in CARGO_CONFIGURATIONS:
            if os.path.lexists(directory / name) and str(directory / name) not in found:
                found.append(str(directory / name))
    return found


def named_toolchain(command: list[str]) -> str | None:
    """The toolchain `command` names itself, which rustup runs in place of
    the one it resolves from the root, or None: the first argument
    `+<toolchain>` of one of rustup's proxies (`cargo +stable run`), or the
    toolchain of `rustup run <toolchain> <program>`, which rustup documents
    as the same (`rustup run stable cargo run`), past rustup's own options
    and `+<toolchain>` before `run` and the options of `run` (`--install`)
    before the toolchain. Each program is named as a platform runs it
    (`experiment_records.program_name`: `rustup.exe` is rustup)."""
    program = experiment_records.program_name(command[0]) if command else ""
    if program in RUSTUP_PROXIES:
        return command[1][1:] if len(command) > 1 and command[1].startswith("+") else None
    if program != "rustup":
        return None
    rest = command[1:]
    while rest and rest[0].startswith(("-", "+")):
        rest = rest[1:]
    if not rest or rest[0] != "run":
        return None
    rest = rest[1:]
    while rest and rest[0].startswith("-"):
        rest = rest[1:]
    return rest[0] if rest else None


def is_rustup_proxy(path: str, rustup: str) -> bool:
    """Whether the program at `path` is rustup itself under a proxy's name,
    as rustup installs its proxies: the same file (a hard or symbolic link)
    or a copy of it."""
    try:
        if os.path.samefile(path, rustup):
            return True
        return os.path.getsize(path) == os.path.getsize(rustup) and Path(path).read_bytes() == Path(rustup).read_bytes()
    except OSError:
        return False


def toolchain(
    environment: dict[str, str],
    command: list[str],
    stamps: dict[str, experiment_records.Stamp | None] | None = None,
    selected: dict[str, str] | None = None,
) -> dict:
    """The Rust toolchain `command`, run from the repository's root in
    `environment`, would build with, each of `rustc` and `cargo` named as
    `resolved_executable` names a program (and stamped into `stamps` as it
    does), by nothing when it cannot be resolved: as rustup resolves it
    there, after its overrides and `rust-toolchain.toml`, or for the
    toolchain the command names itself (`named_toolchain`: `cargo +stable
    run`, `rustup run stable cargo run`), where the `PATH`'s tool of that
    name is rustup's proxy (`is_rustup_proxy`) or the command is `rustup
    run`; and otherwise as the `PATH` holds it, such as a standalone Cargo
    before rustup's proxies, which runs whatever rustup would resolve.
    rustup keeps its toolchains outside the repository, where the commit
    holds none. Given `selected`, the name of the toolchain rustup resolved
    each tool from goes in under the tool (`toolchain_name`)."""
    search = os.pathsep.join(os.get_exec_path(environment))
    rustup = shutil.which("rustup", path=search)
    name = named_toolchain(command)
    named = [] if name is None else ["--toolchain", name]
    run_by_rustup = bool(command) and experiment_records.program_name(command[0]) == "rustup"
    found = {}
    for tool in ("rustc", "cargo"):
        on_path = shutil.which(tool, path=search)
        if rustup is None or not (run_by_rustup or (on_path is not None and is_rustup_proxy(on_path, rustup))):
            found[tool] = resolved_executable([tool], environment, stamps)
            continue
        try:
            which = subprocess.run(
                [rustup, "which", *named, tool], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120
            )
        except (OSError, subprocess.SubprocessError):
            which = None
        resolved = which.stdout.strip() if which is not None and which.returncode == 0 else ""
        found[tool] = resolved_executable([resolved], environment, stamps) if os.path.isabs(resolved) else {
            "path": None, "sha256": None,
        }
        name = toolchain_name(resolved) if os.path.isabs(resolved) else None
        if selected is not None and name is not None:
            selected[tool] = name
    return found


def toolchain_name(path: str) -> str | None:
    """The name of the toolchain whose tool rustup names at `path`
    (`<RUSTUP_HOME>/toolchains/<name>/bin/<tool>`, as `rustup which` prints
    it), or None for a path laid out otherwise."""
    tool = PurePosixPath(path.replace("\\", "/"))
    if tool.parent.name == "bin" and tool.parent.parent.parent.name == "toolchains":
        return tool.parent.parent.name
    return None


def resolved_executable(
    command: list[str], environment: dict[str, str], stamps: dict[str, experiment_records.Stamp | None] | None = None
) -> dict:
    """The program `command` starts, as the command runs it from the
    repository's root in `environment` (`found_program`), named as
    `named_program` names it."""
    return named_program(found_program(command, environment), stamps)


def has_slash(program: str) -> bool:
    """Whether the command's first token `program` names a path, which the
    process that starts it reads as it stands rather than looking it up."""
    return os.sep in program or bool(os.altsep and os.altsep in program)


def found_program(command: list[str], environment: dict[str, str]) -> str | None:
    """Where the process that starts `command` from the repository's root in
    `environment` finds its program, or None when it finds none: a name
    without a slash on `environment`'s `PATH`, as that process looks it up,
    and one with a slash from the root, where it must be an executable
    file."""
    program = command[0]
    if has_slash(program):
        found = str(ROOT / program)
        return found if os.path.isfile(found) and os.access(found, os.X_OK) else None
    return shutil.which(program, path=os.pathsep.join(os.get_exec_path(environment)))


def named_program(found: str | None, stamps: dict[str, experiment_records.Stamp | None] | None = None) -> dict:
    """The program at `found`: its path with every link resolved, and the
    SHA-256 of its content; both None when there is none, and the digest
    None when the file cannot be read or changed while it was read. Given
    `stamps`, the stamps (`experiment_records.file_stamp`) of `found`, of
    every link it is reached through and of the file, from before the file
    was read, go in under their paths, unless one is there already: while
    they stay, the name the command starts leads to the file the digest
    names, holding that content."""
    if found is None:
        return {"path": None, "sha256": None}
    before: dict[str, experiment_records.Stamp | None] = {}
    hop = found
    for _ in range(40):
        before.setdefault(hop, experiment_records.file_stamp(Path(hop)))
        try:
            hop = os.path.join(os.path.dirname(hop), os.readlink(hop))
        except OSError:
            break
    target = Path(os.path.realpath(found))
    stamp = experiment_records.file_stamp(target)
    before.setdefault(str(target), stamp)
    try:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        digest = None
    if stamp is None or experiment_records.file_stamp(target) != stamp:
        digest = None
    if stamps is not None:
        for path, taken in before.items():
            stamps.setdefault(path, taken)
    return {"path": str(target), "sha256": digest}


def scratch_directory(exp_id: str, environment: dict[str, str]) -> Path:
    """The directory a listed run of `exp_id` builds and caches in, fresh for
    each run (`execute_command`): one of the experiment's name in the
    command's temporary directory (`TMPDIR` in `environment`, or the
    runner's), its id named as its lock's is (`id_file_name`). Its path is
    the same for every seed run there, so the variables naming it
    (`scratch_variables`), which the command can read, are part of the
    environment its record names and its seeds share."""
    temporary = environment.get("TMPDIR") or tempfile.gettempdir()
    return Path(temporary) / f"ptr-run-{id_file_name(exp_id)}"


def scratch_variables(directory: str, cargo: bool) -> dict[str, str]:
    """The variables naming a command's fresh directories under `directory`:
    Python's bytecode cache (`PYTHONPYCACHEPREFIX`) and, for a listed
    experiment's build (`cargo`), Cargo's target directory
    (`CARGO_TARGET_DIR`)."""
    variables = {"PYTHONPYCACHEPREFIX": directory}
    if cargo:
        variables["CARGO_TARGET_DIR"] = os.path.join(directory, "cargo-target")
    return variables


@contextlib.contextmanager
def removed_after(directory: Path):
    """Yield `directory`, made by the caller, as text, and remove it with
    what it holds once the block ends; what cannot be removed is left."""
    try:
        yield str(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def execute_command(
    command: list[str],
    environment: dict[str, str] | None = None,
    program: str | None = None,
    scratch: Path | None = None,
) -> dict:
    """Run `command` from the repository's root in `environment`, or in the
    runner's own environment when None, and return its exit status,
    output, launch error and duration. Given `program`, the command starts
    that file under the name `command[0]`, as the process would once it
    looked the name up: a program put earlier on the `PATH` meanwhile does
    not run in its place. Python in it reads and writes its bytecode cache
    in a fresh directory (`PYTHONPYCACHEPREFIX`), so no `__pycache__` entry
    the tree holds, which git ignores and HEAD does not hold, runs in place
    of a tracked source. Given an environment (a listed
    experiment's), Cargo in it builds into a fresh directory too
    (`CARGO_TARGET_DIR`), from the commit's sources alone and outside the
    repository, which then holds no build output a later run could take
    in place of what the commit builds. That directory is `scratch` when
    named (a listed run's, `scratch_directory`, made here only if absent,
    so nothing an earlier run left is read), and a temporary one of a fresh
    name otherwise; it is removed once the command has run. A command that
    could not start, for want of that directory or of its program, or given
    an argument no process can take (a NUL character), has no exit status
    and its launch error: it saw no outcome. Once it started, an exception raised in the
    runner while it waits (an interrupt, say) kills the command and is
    raised, as `subprocess.run` does, and a cache that cannot be removed
    afterwards is left: the command ran. A runner killed by a signal leaves
    the command running; the copy of its reservation in git's own directory
    (`attempts_directory`) keeps that its seed ran."""
    started = time.perf_counter_ns()

    def not_started(error: Exception) -> dict:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "launch_error": f"{type(error).__name__}: {error}",
            "duration_ns": time.perf_counter_ns() - started,
        }

    try:
        if scratch is None:
            cache = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            os.mkdir(scratch)
            cache = removed_after(scratch)
    except OSError as error:
        return not_started(error)
    with cache as directory:
        try:
            process = subprocess.Popen(
                command,
                executable=program,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **(os.environ if environment is None else environment),
                    **scratch_variables(directory, environment is not None),
                },
            )
        except (OSError, ValueError) as error:
            return not_started(error)
        with process:
            try:
                stdout, stderr = process.communicate()
            except BaseException:
                process.kill()
                raise
    return {
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "launch_error": None,
        "duration_ns": time.perf_counter_ns() - started,
    }


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
    (`launch_refused`), and a manifest holding a TOML date or time or a NaN
    (`unrecordable_manifest`). A listed experiment runs only through its
    `entrypoint`, with its preregistered values (`command_parameters`), and
    each seed once (`seed_runs`): a seed run again after its outcome was
    seen could keep whichever run came out best. Its run holds a lock on
    the experiment's runs (`run_lock`), puts back first a record of an
    earlier run its results lost (`restore_records`), reads no uncommitted
    file in its results, and reserves its record before the command starts,
    in its results and in a copy in git's own directory
    (`attempts_directory`), the whole record replacing both once it ends
    (`write_json_replacing`). After the command ends it looks at the same
    tree again (`experiment_records.ProvenanceWatch`) and writes no record,
    returning 2, when HEAD moved or a provenance or experiment file was
    written, created or removed while the command ran, even if its content
    was put back, so the record's `git_sha` is the code that ran as far as
    that watch can see (its `changes` names what it cannot), and for a
    listed experiment when the repository holds a file git ignores, which
    the command could have run or read (`ignored_files`), or Cargo would
    read configuration from outside the repository
    (`outside_cargo_configurations`); a listed experiment's reservation
    then stays, as the record that its seed ran. A command that could not
    start ran no code, so its record, that it failed to launch, is written
    whatever the watch saw. A listed experiment's command runs in the
    environment the runner allows (`command_environment`), which its record
    names with the program it started (`resolved_executable`) and the
    toolchain (`toolchain`), each by its content: one that cannot be read,
    or changes while it is read, refuses the run, the command starts its
    program by the name the launch found it by, and fails to launch when
    the launch found none, and a run during which one of them, or a link on
    the way to it, changed (its stamp moved, `experiment_records.file_stamp`)
    is not recorded. The record is written whole or not at all
    (`write_json_exclusive`)."""
    _, root, data = resolve(exp_id)
    if unrecordable_manifest(exp_id, data) or launch_refused(exp_id):
        return 2

    # The record names HEAD as the code it ran, so HEAD must hold every file
    # that decides the run: refuse before anything runs or is written.
    results = results_directory(root, data)
    if results is None:
        return 2
    timestamp = utc_stamp()
    out = results / f"run-{timestamp}-seed-{seed}.json"
    if not is_listed(exp_id):
        return launch_and_record(
            exp_id, root, data, results, out, timestamp, entrypoint=entrypoint, seed=seed, params=params, attempts=None
        )
    directory = git_directory()
    if directory is None:
        return 2
    lock = run_lock(exp_id, directory)
    if lock is None:
        return 2
    try:
        return launch_and_record(
            exp_id,
            root,
            data,
            results,
            out,
            timestamp,
            entrypoint=entrypoint,
            seed=seed,
            params=params,
            attempts=attempts_directory(directory, results),
        )
    finally:
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
    attempts: Path | None,
) -> int:
    """The part of `run_experiment` from the watch on, for the record `out`
    stamped `timestamp`. `attempts` is where a listed experiment's run keeps
    the copy of its record (`attempts_directory`), and None for an unlisted
    experiment; a listed experiment's run holds its lock throughout."""
    listed = attempts is not None
    if listed:
        # A record the results lost (the command of a run whose runner was
        # stopped could clear them) comes back from its copy, and is
        # committed before any other run: the seed it names has run.
        try:
            restored = restore_records(results, attempts)
        except (OSError, check_research_gates.HistoryUnreadable) as error:
            print(f"ERROR: cannot put back the records of {exp_id}'s runs kept in {attempts}: {error}", file=sys.stderr)
            return 2
        if restored:
            print(
                f"ERROR: put back {', '.join(restored)} from the copies kept in {attempts}: records of runs at commits "
                "on HEAD's history, which the results directory held otherwise or not at all; commit them before the "
                "next run",
                file=sys.stderr,
            )
            return 2
    watch = launch_watch(
        exp_id, root, results, data, (out.name,) if listed else experiment_records.RESULT_OUTPUTS, listed
    )
    if watch is None:
        return 2
    # Built once the watch holds the tree to HEAD, so a listed experiment's
    # preregistered values are the ones HEAD holds.
    try:
        params = command_parameters(exp_id, root, data, entrypoint, params or {})
        command = build_command(data, entrypoint=entrypoint, seed=seed, params=params)
        # A listed experiment runs each seed once: every run of it is
        # evidence, so none can be chosen by its outcome. The results
        # directory holds every record committed, which the gate keeps, and
        # git's own directory a copy of every record this clone wrote.
        ran = seed_runs(results, seed, attempts) if listed else []
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
    # A listed experiment's command runs in the environment the runner
    # allows (`command_environment`), with no Cargo configuration from
    # outside the repository, and its record names that environment, the
    # program it starts and the Rust toolchain, which lie outside what the
    # commit holds.
    environment = command_environment() if listed else None
    if listed:
        relative = relative_directories(environment)
        if relative:
            print(
                f"ERROR: refusing to run {exp_id}: {', '.join(relative)} names a directory by a relative path, which "
                "each program resolves from wherever it starts, so the record could not say which one the run used; "
                "set absolute directories for the run",
                file=sys.stderr,
            )
            return 2
        outside = outside_cargo_configurations(environment)
        if outside:
            print(
                f"ERROR: refusing to run {exp_id}: Cargo would read {', '.join(outside)}, configuration outside the "
                "repository that could set a rustc wrapper, flags or sources the commit does not hold; move it aside "
                "for the run, or commit what it sets in the repository's .cargo/config.toml",
                file=sys.stderr,
            )
            return 2
        # The command builds and caches in a directory of the experiment's
        # name, the same for every seed, which the environment names.
        scratch = scratch_directory(exp_id, environment)
        if os.path.lexists(scratch):
            print(
                f"ERROR: refusing to run {exp_id}: {scratch} exists, left by a run that died or held by one in "
                "progress; a listed run builds in a fresh directory there, so remove it once no run of "
                f"{exp_id} is in progress",
                file=sys.stderr,
            )
            return 2
        environment = {**environment, **scratch_variables(str(scratch), True)}
        # Each program's stamp from before it was read: while it stays, the
        # file holds what its digest names, through the run.
        stamps: dict[str, experiment_records.Stamp | None] = {}
        found = found_program(command, environment)
        executable = named_program(found, stamps)
        selected: dict[str, str] = {}
        tools = toolchain(environment, command, stamps, selected)
        # rustup's proxies resolve the toolchain anew each time they run, from
        # overrides outside the repository: the one resolved here is the one
        # every proxy the command starts runs (`RUSTUP_TOOLCHAIN`), and the
        # record names it.
        if len(set(selected.values())) == 1:
            environment = {**environment, "RUSTUP_TOOLCHAIN": next(iter(selected.values()))}
        # A program named by its path alone could be replaced between seeds
        # while every record named the same: one found that cannot be read,
        # such as a binary only executable, or that changed while it was
        # read, is refused.
        unread = list(dict.fromkeys(
            program["path"] for program in (executable, *tools.values())
            if program["path"] is not None and program["sha256"] is None
        ))
        if unread:
            print(
                f"ERROR: refusing to run {exp_id}: {', '.join(unread)} cannot be read, or changed while it was read, "
                "so its record could not name by its content a program the run starts or builds with; make it "
                "readable, and leave it unchanged, for the run",
                file=sys.stderr,
            )
            return 2
        record.update({"environment": environment, "executable": executable, "toolchain": tools})
        # A record git would ignore could not be committed as written, and
        # would read as a file the command could have run.
        try:
            ignored = experiment_records.git(
                ROOT, "check-ignore", "-q", "--no-index", "--", out.relative_to(ROOT).as_posix()
            ).returncode
        except experiment_records.ProvenanceError as error:
            ignored = str(error)
        if ignored != 1:
            print(
                f"ERROR: refusing to run {exp_id}: git would ignore its record {out.relative_to(ROOT)}"
                + (f" ({ignored})" if isinstance(ignored, str) else "")
                + ", which could then not be committed as written; no rule of the repository or the clone may ignore it",
                file=sys.stderr,
            )
            return 2
        # A seed that could not join the earlier ones would be spent: the
        # gate refuses it once its record exists.
        problems = earlier_run_problems(exp_id, results, watch.head, record)
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        if problems:
            return 2
    kept = attempts / out.name if listed else None
    if listed:
        # The reservation: the record, as far as it is known before the
        # command starts, which the whole record replaces once it ends. A run
        # that dies or goes unrecorded leaves it, and its seed has run. Its
        # copy in git's own directory is written first, and outlasts what
        # the command does to its results.
        try:
            write_json_exclusive(kept, record)
        except OSError as error:
            print(f"ERROR: cannot reserve {out.relative_to(ROOT)} in {kept}: {error}", file=sys.stderr)
            return 2
        try:
            write_json_exclusive(out, record)
        except OSError as error:
            # Nothing ran: the copy goes, so the seed may run.
            problem = str(error)
            try:
                kept.unlink()
            except OSError as removal:
                problem += f"; and {kept}, which says seed {seed} ran, cannot be removed: {removal}"
            print(f"ERROR: cannot reserve {out.relative_to(ROOT)}: {problem}", file=sys.stderr)
            return 2
    stays = f"; {out.relative_to(ROOT)} stays as the record that seed {seed} ran" if listed else ""

    if not listed:
        execution = execute_command(command)
    elif found is None:
        # No program was found for the record to name: the process would
        # look the name up again, and could find one put there meanwhile.
        execution = {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "launch_error": f"FileNotFoundError: the launch found no program {command[0]!r} to start",
            "duration_ns": 0,
        }
    else:
        # The name as the process would find it, so a script reads its own
        # name as it would (`$0`); a name with a slash it reads as given.
        execution = execute_command(command, environment, None if has_slash(command[0]) else found, scratch)
    exit_code = execution["exit_code"]
    launched = exit_code is not None
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
    # A command that could not start ran no code and saw no outcome: its
    # record says so whatever the tree did meanwhile, so its seed may run.
    if launched:
        # The command read the tree while it ran (a `cargo run` entrypoint
        # compiles it first): the record may name HEAD only if the tree
        # stayed so. A listed experiment's command could also have run or
        # read a file git ignores, which the watch does not see, put there
        # after the launch looked for one: the tree still holds none.
        try:
            changes = watch.changes()
            ignored = ignored_files() if listed else []
            if ignored:
                changes.append(
                    f"the repository holds {experiment_records.listed(ignored)}, which git ignores and the command "
                    "could have run or read unrecorded"
                )
            unlisted = unlisted_entries() if listed else []
            if unlisted:
                changes.append(
                    f"the repository holds {experiment_records.listed(unlisted)}, which git does not list and the "
                    "command could have read unrecorded"
                )
        except experiment_records.ProvenanceError as error:
            changes = [str(error)]
        # So could Cargo configuration outside the repository, put there
        # after the launch looked for one.
        outside = outside_cargo_configurations(environment) if listed else []
        if outside:
            changes.append(
                f"Cargo would read {', '.join(outside)}, configuration outside the repository its build could have "
                "read"
            )
        # A program the record names by its digest could have been replaced
        # while the command ran, and put back: every write or replacement
        # moves its stamp.
        moved = [
            path for path, stamp in stamps.items() if experiment_records.file_stamp(Path(path)) != stamp
        ] if listed else []
        if moved:
            changes.append(
                f"{experiment_records.listed(moved)} changed since the launch read it, so the program that ran may not "
                "be the one its digest in the record names"
            )
        if changes:
            print(
                f"ERROR: not recording the run (exit status {exit_code}): its sources changed while it ran, "
                f"so {watch.head} may not be the code it ran; {'; '.join(changes)}; "
                f"rerun from a working tree that stays at HEAD{stays}",
                file=sys.stderr,
            )
            return 2
    # The command could have put a file or a link where the results
    # directory is, which the watch leaves out.
    results_changed = results_directory(root, data) is None
    if results_changed:
        print(
            f"ERROR: not recording the run (exit status {exit_code}): its results directory changed while it ran"
            f"{stays if launched else ''}",
            file=sys.stderr,
        )
    if not listed:
        if results_changed:
            return 2
        write_json_exclusive(out, record)
        print(out.relative_to(ROOT))
        return exit_code if exit_code is not None else 127
    if not launched:
        # Nothing ran: the record that the command failed to launch replaces
        # the reservation and its copy, the copy first; where it cannot be
        # written, the reservation and its copy go, so the seed may run. The
        # next run puts the record back once the results directory is one
        # again.
        try:
            write_json_replacing(kept, record)
            if not results_changed:
                write_json_replacing(out, record)
        except OSError as error:
            problem = f"cannot write the record of seed {seed}'s launch, which failed: {error}"
            for reservation in (kept,) if results_changed else (kept, out):
                try:
                    reservation.unlink(missing_ok=True)
                except OSError as removal:
                    problem += f"; and {reservation}, which says seed {seed} ran, cannot be removed: {removal}"
            print(f"ERROR: {problem}", file=sys.stderr)
            return 2
        if results_changed:
            return 2
        print(out.relative_to(ROOT))
        return 127
    if results_changed:
        return 2
    # The copy first: a runner stopped between the two writes leaves the
    # whole record in the copy, which the next run puts back. A copy git's
    # own directory cannot hold (full, say) goes, so that no reservation
    # there is put back over the record, which the results hold alone until
    # it is committed.
    try:
        write_json_replacing(kept, record)
    except OSError as error:
        problem = f"cannot write the record's copy {kept}: {error}"
        try:
            kept.unlink(missing_ok=True)
        except OSError as removal:
            problem += f"; nor remove it, so the next run would put its reservation back: {removal}"
        print(f"ERROR: {problem}; commit {out.relative_to(ROOT)} before the next run", file=sys.stderr)
    write_json_replacing(out, record)
    print(out.relative_to(ROOT))
    return exit_code


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
