"""Bind an experiment's aggregated result to the code its run records exercised.

`scripts/run_experiment.py run` stamps every run record with the experiment
id, the commit it ran at (`git_sha`), the manifest, the Cargo.lock hash, the
parameters, the entrypoint and the command. An aggregator that combines one
record per seed must not publish a verdict for code those records never ran,
so before it combines them it asks `source_revision`, which refuses records
that

- belong to another experiment, or do not name the commit they ran at;
- disagree about the Cargo.lock hash, the parameters or the entrypoint, ran an
  entrypoint the manifest does not declare, or ran under a manifest that
  differs from the current `experiment.toml` in anything but its `status`
  (which changes when the experiment completes);
- ran at commits whose provenance files (`seed_record_paths`: the Rust
  sources, Cargo manifests, lock file, toolchain, Cargo configuration, SQL
  and protobuf files of `CODE_PATHS`, the scripts that record and judge seed
  runs, and the experiment's own `aggregate.py`) differ from each other, or
  from the checkout the aggregate is written in (`code_changes`, which also
  counts a symlink or submodule within reach of the build whose entry
  differs, since git holds a link's target path and no pathspec matches it);
- are aggregated in a checkout holding provenance files HEAD does not hold
  (`checkout_problem`): git diff, which compares a commit with the checkout,
  omits untracked files, such as a new `crates/ptr-*` crate the workspace
  takes as a member, whether written there, linked in or cloned in.

Records archived one commit at a time sit at different commits; they still
agree when no provenance file changed between those commits.

Each record's harness result must then be the run the wrapper selected it for
(`harness_results`): the same benchmark, seed and iteration count. Mutation
evidence (`results/mutations.json`, written by `scripts/mutation_check.py`) is
bound the same way (`mutation_evidence`): it is refused, not omitted, unless
it is this experiment's evidence for this benchmark and its commit has the
checkout's code, mutation checker, aggregator and mutation plan
(`mutation_record_paths`), which HEAD holds. A refused file is rerun or
removed; an aggregate never carries mutation counts of other code, and the
summary it carries names the SHA-256 of the whole file, so no outcome it
lists is replaced under the same counts (`aggregate_problems`).

`run_experiment.py` and `mutation_check.py` refuse to record from a working
tree with uncommitted or untracked provenance files (`uncommitted_files`,
which also counts a file only this clone's own ignore rules hide, a
`.gitignore` HEAD does not hold that hides one, a file git is told not to
look at, a symlink named as a provenance file, and a symlinked directory,
nested repository or submodule where the build reads), and a record names
the commit HEAD was at when its run started. A
run reads its sources while it runs (a `cargo run` entrypoint compiles them),
so after the run they look at the tree again (`ProvenanceWatch`) and write no
record when HEAD moved or a provenance file was written, created or removed
meanwhile, even if its content was put back; `ProvenanceWatch.changes` names
what that second look cannot see. Records made by an older recorder cannot
slip past this: the recorder is itself a provenance file, so a record
aggregates only while the recorder is the one that ran it.

Every record and aggregate is written whole or not at all (`write_atomically`,
`write_exclusively`). An aggregator publishes `run.json` and `metrics.json` as
one aggregate (`publish_aggregate`): run.json names the SHA-256 of the
metrics.json written with it, and an interrupted publish leaves no run.json
rather than one beside metrics it was not aggregated with.

Archived results of a completed experiment must describe HEAD's code, or carry
a `results/STALE.toml` marker saying since when they do not
(`staleness_errors`, run by `scripts/check_research_gates.py`). The marker
names a commit at which every stale file has the provenance files it ran at,
which `run.json` and `mutations.json` share even when they ran at different
commits, and the first commit descending from it that changed one; commits
merged in from another line of history, such as a pull request's base
branch, never count as that change, so one marker holds on the pull
request's head, on the merge CI checks and on the base branch after it.
The archived run.json must also be one aggregate with the metrics.json and
mutations.json beside it (`aggregate_errors`, run there too); one aggregated
before run.json bound its metrics passes only while it is stale and beside
the metrics.json committed with it.

An experiment's preregistration, the `[preregistration]` table of its
`config.toml`, has one canonical text and digest
(`preregistration_canonical`, `preregistration_digest`), which the research
gate compares with the manifest's `preregistration_sha256`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import posixpath
import re
import secrets
import stat
import subprocess
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path

# The files that decide what an experiment binary does, as git pathspecs. A
# pathspec without magic matches `*` across `/`, so `*.rs` is every Rust file.
CODE_PATHS = (
    "*.rs",
    "*.sql",
    # ptr-protocol's build script compiles these into every harness.
    "*.proto",
    "Cargo.toml",
    "*/Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    # The older names rustup and Cargo still read, and prefer when present.
    "rust-toolchain",
    ".cargo/config",
)
# The files that decide which untracked files git lists. HEAD's decide it for
# a clean tree; one HEAD does not hold, or a rule of this clone's own
# (`.git/info/exclude`, `core.excludesFile`), could hide a source.
IGNORE_FILE = ".gitignore"
# Git's modes for a symlink, whose blob is the target path rather than the
# content read through it, and for a submodule, whose entry names a commit of
# another repository. Git looks into neither.
SYMLINK_MODE = "120000"
GITLINK_MODE = "160000"
LINK_MODES = (SYMLINK_MODE, GITLINK_MODE)
EXCLUDE = ":(exclude)"
# A placeholder in an entrypoint, `<name>`, which the runner fills with a
# value (`scripts/run_experiment.py`); the gate reads a listed experiment's
# entrypoint by the same rule.
PLACEHOLDER = re.compile(r"<([A-Za-z][A-Za-z0-9_-]*)>")
WILDCARD = re.compile(r"[*?\[]")
# The scripts that decide what a record says and how it is judged.
JUDGE = "scripts/experiment_records.py"
RECORDER = "scripts/run_experiment.py"
MUTATOR = "scripts/mutation_check.py"
STALE_MARKER = "STALE.toml"
COMMIT = re.compile(r"[0-9a-f]{7,64}")
AGREED = (
    ("cargo_lock_sha256", "Cargo.lock"),
    ("parameters", "parameters"),
    ("entrypoint", "entrypoint"),
    # A listed experiment's run names the program it started, the Rust
    # toolchain and the environment it ran in, which lie outside the
    # commit: its seeds ran one of each, and a program the command starts
    # by name is found through that environment's PATH.
    ("executable", "program"),
    ("toolchain", "toolchain"),
    ("environment", "environment"),
)


class ProvenanceError(Exception):
    """The run records cannot be published as one result of the current code."""


class NotUTF8(ProvenanceError):
    """Git printed what is not UTF-8, such as a file name, which no check
    can read as a repository path."""


def program_name(token: str) -> str:
    """The name of the program `token` starts, as a command's first token or
    the program `rustup run` starts names it: its last path component,
    either slash a separator, in lower case and without an `.exe` suffix,
    as Windows runs `C:\\Rust\\Cargo.EXE` for `cargo`. A check of a command
    reads its programs so on every platform, so no spelling one of them
    accepts hides the program from it."""
    name = re.split(r"[\\/]", token)[-1].lower()
    return name[: -len(".exe")] if name.endswith(".exe") else name


def relative_to_root(path: Path, root: Path) -> str:
    """`path` as a git pathspec relative to the repository `root`."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def seed_record_paths(experiment_dir: Path, root: Path) -> tuple[str, ...]:
    """The provenance files of a seed record: `CODE_PATHS`, the recorder and
    this module, and the experiment's `aggregate.py`, which turns the records
    into a verdict."""
    experiment = relative_to_root(experiment_dir, root)
    return (*CODE_PATHS, JUDGE, RECORDER, f"{experiment}/aggregate.py")


def mutation_record_paths(experiment_dir: Path, root: Path) -> tuple[str, ...]:
    """The provenance files of mutation evidence: `CODE_PATHS`, the mutation
    checker and this module, the experiment's `aggregate.py` and the mutation
    plan (`tests/mutations.toml`) the evidence was planted from."""
    experiment = relative_to_root(experiment_dir, root)
    return (
        *CODE_PATHS,
        JUDGE,
        MUTATOR,
        f"{experiment}/aggregate.py",
        f"{experiment}/tests/mutations.toml",
    )


# The seed records the runner writes into a results directory, as a glob
# pattern within it.
SEED_RECORDS = "run-*.json"
# The files the runner, the aggregators and the mutation checker write into
# a results directory, as glob patterns within it: records accumulate there
# uncommitted while runs go on.
RESULT_OUTPUTS = (SEED_RECORDS, "run.json", "metrics.json", "mutations.json")


def listed_record_paths(results: str) -> tuple[str, ...]:
    """The provenance of a listed experiment's seed records, as pathspecs:
    the whole repository but the seed records in its results directory
    `results` (a repository path, `SEED_RECORDS`). A listed experiment's
    command may run or read any file of the repository (a script under
    `scripts/`, say, or an aggregate or mutation evidence committed in its
    results), so its seeds ran one tree only if their commits differ in
    nothing else: between them, only their seed records are committed, and
    the other outputs of the tools, which a later seed could read, once the
    seeds have run."""
    return (".", f":(exclude,glob){results}/{SEED_RECORDS}")


# The files completing a listed experiment changes once its seeds have run:
# its manifest and the registry, each in the status it names for it, moving
# forward. Its command may read either whole, so any other change to them is
# a change after its runs (`completion_changes`).
AFTER_THE_RUNS = ("experiment.toml", "experiments/registry.toml")
# A line naming a status, whose value is all completing an experiment
# changes in its manifest and in the registry's entry for it.
STATUS_LINE = re.compile(r'^(\s*status\s*=\s*)"[^"\\]*"')


def listed_staleness_paths(results: str, experiment: str) -> tuple[str, ...]:
    """The provenance of a listed experiment's archived results against the
    checkout, as pathspecs: the whole repository but what the tools write
    into its results directory `results` (a repository path,
    `RESULT_OUTPUTS`) and its stale marker (`STALE_MARKER`), which are
    committed once the seeds have run, and the files completing it changes
    (`AFTER_THE_RUNS`: the manifest in its directory `experiment` and the
    registry), which `completion_changes` compares but for the status it
    changes in them. Its command may run or read any other file of the
    repository, the list of preregistrations included, so a change to one
    after the results makes them stale (`staleness_errors`), and one in the
    checkout keeps them from being aggregated there (`source_revision`)."""
    return (
        ".",
        *(f":(exclude,glob){results}/{pattern}" for pattern in (*RESULT_OUTPUTS, STALE_MARKER)),
        f":(exclude,literal){experiment}/{AFTER_THE_RUNS[0]}",
        *(f":(exclude,literal){name}" for name in AFTER_THE_RUNS[1:]),
    )


def tree_pathspecs(
    experiment_dir: Path, results_dir: Path, root: Path, paths: tuple[str, ...], outputs: tuple[str, ...] = RESULT_OUTPUTS
) -> list[str]:
    """The pathspecs a run must find committed: the provenance `paths` and the
    whole experiment directory but the `outputs` the tools write into its
    results, as glob patterns within it (`RESULT_OUTPUTS`, or a run's own
    record). Every other file there, code or input a command could read,
    must be one HEAD holds, as anywhere in the experiment: one changed
    between runs would leave records naming one commit for different code."""
    results = relative_to_root(results_dir, root)
    return [
        *paths,
        relative_to_root(experiment_dir, root),
        *(f":(exclude,glob){results}/{pattern}" for pattern in outputs),
    ]


def without_status(manifest: dict) -> dict:
    """A manifest apart from its `status`, which completing an experiment changes."""
    return {key: value for key, value in manifest.items() if key != "status"}


def toml_time(value):
    """A TOML date, time or datetime as its ISO 8601 text (`json.dumps`'s
    `default`): JSON has no such type. A manifest holding one is refused
    (`unrecordable_keys`); this only keeps such a value from failing a write
    or a comparison that reports it."""
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def unrecordable_keys(value, where: str = "") -> list[tuple[str, str, str]]:
    """The key paths in `value`, a manifest as TOML reads it, that hold a
    value a run record cannot keep apart from another, in any table or
    array, each with what it holds and what the record would take it for.
    A run record holds the manifest as JSON, which has no date or time, so
    a TOML date, time or datetime and the string of its text would be one
    manifest; and which writes a NaN as `NaN` whatever its sign, so `nan`
    and `-nan`, which a run can tell apart (`math.copysign`), would be one
    manifest. The runner, `validate` and aggregation refuse a manifest that
    holds one (`manifest_problems`)."""
    if isinstance(value, (datetime.date, datetime.time)):
        return [(where, "a TOML date or time", "a string")]
    if isinstance(value, float) and math.isnan(value):
        return [(where, "a NaN", "a NaN of the other sign")]
    if isinstance(value, dict):
        return [
            found for key, item in value.items() for found in unrecordable_keys(item, f"{where}.{key}" if where else str(key))
        ]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in unrecordable_keys(item, f"{where}[{index}]")]
    return []


def manifest_problems(manifest: dict) -> list[str]:
    """Why a run record could not hold `manifest` apart from another
    manifest, one line for each value it holds that a record cannot keep
    apart (`unrecordable_keys`); empty when it can."""
    return [
        f"experiment.toml holds {held} at {key}, which a run record, holding the manifest as JSON, cannot tell from "
        f"{taken}; write it as a string"
        for key, held, taken in unrecordable_keys(manifest)
    ]


def recorded_text(value) -> str:
    """`value` as the JSON text a run record holds it as, keys sorted, so a
    manifest compares with the one a record names as the record holds it:
    `1`, `1.0` and `true` are three values, and so are `0.0` and `-0.0`. A
    date or time, which no manifest may hold, is written as its text
    (`toml_time`), and a NaN, which no manifest may hold either, as `NaN`
    (`unrecordable_keys`)."""
    return json.dumps(value, sort_keys=True, default=toml_time)


def agreement_errors(experiment_id: str, manifest: dict, records: dict[str, dict]) -> list[str]:
    """Why `records` (file name -> run record) cannot be seeds of one run of
    `experiment_id`, whose manifest is now `manifest`; empty when they can.
    This compares the records alone; `source_revision` also compares code.
    A manifest holding a TOML date or time, or a NaN, is refused
    (`manifest_problems`)."""
    errors = manifest_problems(manifest)
    for name, record in records.items():
        if record.get("experiment_id") != experiment_id:
            errors.append(f"{name} is a record of {record.get('experiment_id')!r}, not {experiment_id!r}")
        if not COMMIT.fullmatch(str(record.get("git_sha", ""))):
            errors.append(f"{name} names no commit it ran at: {record.get('git_sha')!r}")
    for key, label in AGREED:
        groups: dict[str, list[str]] = {}
        for name, record in records.items():
            groups.setdefault(json.dumps(record.get(key), sort_keys=True), []).append(name)
        if len(groups) > 1:
            errors.append(
                f"the records disagree on {label}: "
                + "; ".join(f"{value} in {', '.join(names)}" for value, names in groups.items())
            )
    current = recorded_text(without_status(manifest))
    for name, record in records.items():
        if recorded_text(without_status(record.get("manifest") or {})) != current:
            errors.append(f"{name} ran under an experiment.toml that differs from the current one")
        entrypoint = record.get("entrypoint")
        if not isinstance(entrypoint, str) or not str(manifest.get(entrypoint, "")).strip():
            errors.append(f"{name} ran {entrypoint!r}, which experiment.toml declares no command for")
    return errors


def git(
    root: Path,
    *args: str,
    stdin: str | bytes | None = None,
    env: dict[str, str] | None = None,
    binary: bool = False,
    work_tree: bool = True,
) -> subprocess.CompletedProcess:
    """Run git in `root`, given `stdin` and in `env` when named; raises
    `ProvenanceError` when git cannot start. Without `env`, git runs in this
    process's environment less every `GIT_*` variable, which could point it
    at another work tree, index or object store or change how it reads
    pathspecs, and with replacement objects off, so what it reports is the
    repository at `root` as its history holds it. Its work tree is `root`
    whatever `core.worktree` names, but for a command that makes a
    repository (`work_tree` false), which git refuses one without its own
    directory named. With `binary`, what git prints is kept as the bytes it
    wrote, as a file's content must be, and `stdin` is bytes too. Without
    it, what git prints is read as UTF-8 whatever the locale, and read as
    it is: a carriage return in a name git prints with `-z` stays one, as no
    translation of line endings would leave it. Output that is not UTF-8 (a
    file name, say) raises `NotUTF8`, a `ProvenanceError`, as what git
    cannot name as text no check can compare; so does an argument no
    process can be given (one holding a NUL character)."""
    if env is None:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        ran = subprocess.run(
            # Git reads the tree itself: no file system monitor a clone's own
            # configuration names answers for it, no hook of the clone's runs
            # when git refreshes its index, no name is taken for another that
            # differs from it in case, which a clone told to ignore case
            # would do on a file system that does not, and the work tree is
            # `root`, where commands run, not one `core.worktree` names.
            [
                "git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.ignoreCase=false",
                *((f"--work-tree={root}",) if work_tree else ()), *args,
            ],
            cwd=root, input=stdin.encode("utf-8") if isinstance(stdin, str) else stdin, env=env,
            capture_output=True, check=False,
        )
    except (OSError, ValueError) as error:
        raise ProvenanceError(f"cannot run git: {error}") from error
    if binary:
        return ran
    # Git prints paths as UTF-8 bytes. The locale's encoding could read bytes
    # that are not UTF-8 as text, or a UTF-8 name as another name, which
    # names no file of the tree, and text mode would turn a carriage return
    # into a newline.
    try:
        stdout = ran.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NotUTF8(
            f"git {' '.join(args[:2])} printed what is not UTF-8, such as a file name, which no check can read: {error}"
        ) from error
    return subprocess.CompletedProcess(ran.args, ran.returncode, stdout, ran.stderr.decode("utf-8", "replace"))


def code_changes(base: str, head: str | None, root: Path, paths: tuple[str, ...] = CODE_PATHS) -> list[str]:
    """The files under the pathspecs `paths` that differ between commit `base`
    and commit `head`, or the working tree of `root` when `head` is None,
    sorted, and the symlinks and submodules within reach of the build
    (`within_reach`) whose entry differs, whatever they point at: git holds
    a link's target path, which matches no file pathspec, but linking a
    crate in or pointing a link at another changes what Cargo builds.
    Raises `ProvenanceError` when git cannot compare them (say a commit is
    missing)."""
    revisions = [base] if head is None else [base, head]
    # Read as bytes: a name that is not UTF-8 keeps its bytes, so a file of
    # such a name that differs is a difference like any other, named by its
    # escapes, rather than one no comparison reports.
    files = git(root, "diff", "--name-only", "-z", "--no-renames", *revisions, "--", *paths, binary=True)
    entries = git(root, "diff", "--raw", "-z", "--no-renames", *revisions, binary=True)
    for diff in (files, entries):
        if diff.returncode != 0:
            against = "the checkout" if head is None else head
            problem = diff.stderr.decode("utf-8", "replace").strip()
            raise ProvenanceError(f"cannot compare {base} with {against}: {problem}")
    names = files.stdout.decode("utf-8", "surrogateescape")
    links = links_within_reach(link_paths(entries.stdout.decode("utf-8", "surrogateescape").split("\0")), root, paths)
    return sorted({*(name for name in names.split("\0") if name), *links})


def link_paths(fields: list[str]) -> list[str]:
    """The paths of the entries of `git diff --raw -z` or `git log --raw -z`
    output (`fields`, split at NUL) that are or were a symlink or a
    submodule."""
    names = []
    link = None
    for field in fields:
        if link is not None:
            if link:
                names.append(field)
            link = None
        elif field.startswith(":"):
            link = any(mode in LINK_MODES for mode in field[1:].split(" ")[:2])
    return names


def links_within_reach(names: list[str], root: Path, paths: tuple[str, ...]) -> list[str]:
    """Those of `names` that lie within reach of the build (`within_reach`),
    judged by the files under `paths` the index of `root` tracks."""
    if not names:
        return []
    directories = provenance_directories(listed_names(root, "ls-files", "-z", "--", *paths))
    return [name for name in names if within_reach(name, paths, directories)]


def provenance_directories(tracked: list[str]) -> set[str]:
    """Every directory below the root that holds, at any depth, one of the
    files `tracked` (paths relative to the root)."""
    directories: set[str] = set()
    for name in tracked:
        parent = posixpath.dirname(name)
        while parent and parent not in directories:
            directories.add(parent)
            parent = posixpath.dirname(parent)
    return directories


def within_reach(name: str, pathspecs, directories: set[str]) -> bool:
    """Whether the build can read through `name`, a path git does not look
    into (a symlink to a directory, a nested repository, a submodule): it
    sits in a directory below the root that holds tracked provenance files
    (`directories`, from `provenance_directories`), as a crate in `crates/`
    or a module directory beside sources does, or a literal pathspec of
    `pathspecs` names a path beneath it, as `.cargo/config.toml` does; and
    no `:(exclude)` pathspec covers it. Elsewhere it is not: Cargo reads a
    new top-level directory only once a manifest, itself a provenance file,
    names it, so a `target` or virtualenv linked in at the root holds no
    source of this checkout, and neither does a worktree kept in a directory
    that holds no tracked provenance file."""
    for spec in pathspecs:
        if spec.startswith(EXCLUDE):
            excluded = spec[len(EXCLUDE) :].rstrip("/")
            if name == excluded or name.startswith(f"{excluded}/"):
                return False
    if posixpath.dirname(name) in directories:
        return True
    literal = [spec for spec in pathspecs if not spec.startswith(":") and not WILDCARD.search(spec)]
    return any(spec.startswith(f"{name}/") for spec in literal)


def listed_names(root: Path, *args: str) -> list[str]:
    """The NUL-separated entries `git *args` prints in `root`. Raises
    `ProvenanceError` when git cannot list the tree."""
    listing = git(root, *args)
    if listing.returncode != 0:
        raise ProvenanceError(f"cannot list the working tree: {listing.stderr.strip()}")
    return [entry for entry in listing.stdout.split("\0") if entry]


PER_DIRECTORY = f"--exclude-per-directory={IGNORE_FILE}"


def untracked_files(root: Path, pathspecs: list[str]) -> list[str]:
    """The untracked files under `pathspecs` that the `.gitignore` files of
    the tree do not ignore. Rules of this clone's own (`.git/info/exclude`,
    `core.excludesFile`) hide nothing here: HEAD does not hold them, and a
    crate they hide is still one Cargo builds."""
    return listed_names(root, "ls-files", "-z", "--others", PER_DIRECTORY, "--", *pathspecs)


def uncommitted_files(root: Path, pathspecs: list[str]) -> list[str]:
    """The files under `pathspecs` whose working-tree state HEAD does not
    hold, sorted: modified, staged, deleted or untracked. A record made from
    such a tree would name a commit that is not the code it ran.

    Untracked means not ignored by the `.gitignore` files (`untracked_files`),
    so build output and caches are no sources but a file only this clone's
    own rules hide is. A file under `pathspecs` that the tree's `.gitignore`
    files hide but HEAD's would show counts as well, named by the rule file
    that hides it (`hiding_rules`): an untracked or edited `.gitignore`, even
    one that ignores itself; a rule that hides no such file, as the one an
    IDE writes into `.idea/`, does not count. A tracked file git is told to
    take as HEAD's (assume-unchanged, skip-worktree), which git status and
    git diff do not look at, counts, and so does a symlink named as a
    provenance file: git holds its target path, not the content read through
    it. So does a directory git does not look into where the build reads
    (`unseen_trees`): a symlink to a directory, a nested repository or a
    submodule, such as a `crates/ptr-*` crate linked or cloned in, which git
    lists as one path no file pathspec matches. And so does a tracked file
    whose bytes on disk are not HEAD's (`content_changes`), whatever git
    status says of it: a clone's own configuration (a file system monitor, a
    clean filter its own attributes name, a stat cache it trusts) could have
    it report a rewritten file as unchanged. Raises `ProvenanceError` when
    git cannot list the tree."""
    found = {
        entry[3:]
        for entry in listed_names(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=no", "--no-renames", "--", *pathspecs
        )
    }
    # A nested repository is listed as `name/`; `unseen_trees` names it too.
    found.update(name.rstrip("/") for name in untracked_files(root, pathspecs))
    found.update(hiding_rules(root, pathspecs))
    tracked = []
    for entry in listed_names(root, "ls-files", "-z", "--stage", "-v", "--", *pathspecs):
        fields, _, name = entry.partition("\t")
        tag, mode = fields.split(" ")[:2]
        tracked.append(name)
        if tag.islower() or tag.upper() == "S" or mode == SYMLINK_MODE:
            found.add(name)
    found.update(content_changes(root, tracked))
    found.update(unseen_trees(root, pathspecs, provenance_directories(tracked)))
    return sorted(found)


def content_changes(root: Path, names: list[str]) -> list[str]:
    """Those of the tracked files `names` whose bytes in the working tree of
    `root` are not the blob HEAD holds for them, sorted, read from disk as
    they are, so no clean filter, file system monitor or stat cache that a
    clone's own configuration names answers for them, and those whose
    executable bit is not HEAD's. The bytes are compared as they are, line
    endings included: a checkout that writes text with CRLF line endings
    (`core.autocrlf`, an `eol` attribute) holds other bytes than the blob,
    which a command reading them could tell apart, and two checkouts of one
    commit that converted differently would have run on different inputs
    under one commit; the repository's `.gitattributes` pin one form, LF. A
    name HEAD does not hold as a regular file, or the working tree does not
    hold as one, is git status's to report. Raises `ProvenanceError` when
    git cannot read HEAD."""
    wanted = set(names)
    if not wanted:
        return []
    held = {}
    for entry in listed_names(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD"):
        fields, _, name = entry.partition("\t")
        mode, kind, blob = fields.split(" ")
        if name in wanted and kind == "blob" and mode in ("100644", "100755"):
            held[name] = (mode, blob)
    changed = []
    for name, (mode, blob) in sorted(held.items()):
        path = root / name
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if os.name == "posix" and bool(info.st_mode & stat.S_IXUSR) != (mode == "100755"):
            changed.append(name)
            continue
        hashed = (hashlib.sha256 if len(blob) == 64 else hashlib.sha1)(b"blob %d\0" % len(data) + data).hexdigest()
        if hashed != blob:
            changed.append(name)
    return sorted(changed)


def hiding_rules(root: Path, pathspecs: list[str]) -> list[str]:
    """The rule files hiding untracked files under `pathspecs` that the
    `.gitignore` files as HEAD holds them would show, sorted. Each such file
    is named by the rule file that decides it (`git check-ignore`), as an
    untracked `.gitignore` of `*` beside a new build script is, which hides
    itself too, or by itself when that rule is none of the tree's
    `.gitignore` files. A rule that hides nothing under `pathspecs`, or only
    what HEAD's rules hide as well (build output, caches), is not named.
    Raises `ProvenanceError` when git cannot list the tree."""
    ignored = listed_names(root, "ls-files", "-z", "--others", "--ignored", PER_DIRECTORY, "--", *pathspecs)
    if not ignored:
        return []
    hidden = sorted(set(ignored) - ignored_by_head_rules(root, ignored))
    if not hidden:
        return []
    request = "".join(f"{name}\0" for name in hidden)
    deciding = git(root, "check-ignore", "-z", "-v", "--no-index", "--stdin", stdin=request)
    # Each match is its rule's source, line and pattern, and the path.
    fields = deciding.stdout.split("\0")
    sources = {fields[index + 3]: fields[index] for index in range(0, len(fields) - 3, 4)}
    found = set()
    for name in hidden:
        source = sources.get(name, "")
        in_tree = not os.path.isabs(source) and not source.startswith("../")
        found.add(source if posixpath.basename(source) == IGNORE_FILE and in_tree else name)
    return sorted(found)


def ignored_by_head_rules(root: Path, names: list[str]) -> set[str]:
    """Those of the untracked files `names` that the `.gitignore` files as
    HEAD holds them ignore, as git decides it: in a scratch repository
    holding only those rules and an empty file at each name. A top-level
    directory those rules ignore as a whole, as `/target/` does a build's
    output, decides every name below it at once, since git cannot show a
    file again below an ignored directory; only the others are placed in
    the scratch repository one by one. Raises `ProvenanceError` when git
    cannot read HEAD's rules or decide."""
    rules = head_rules(root)
    # The scratch repository is git's own, whatever repository the caller's
    # environment points git at.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    with tempfile.TemporaryDirectory() as scratch:
        mirror = Path(scratch)
        created = git(mirror, "init", "-q", env=env, work_tree=False)
        if created.returncode != 0:
            raise ProvenanceError(
                f"cannot create a repository to read HEAD's ignore rules in: {created.stderr.strip()}"
            )
        for name, rule in rules.items():
            path = mirror / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rule)
        tops = sorted({name.split("/", 1)[0] for name in names if "/" in name} - set(rules))
        for top in tops:
            (mirror / top).mkdir(parents=True, exist_ok=True)
        # Each named without a trailing slash, as an existing directory: a
        # rule such as `target/*` matches `target/` but not the directory
        # `target`, whose files it may show again (`!target/keep`).
        decided = git(
            mirror, "check-ignore", "-z", "--no-index", "--stdin", env=env,
            stdin="".join(f"{top}\0" for top in tops),
        )
        # check-ignore exits 1 when it ignores none of them.
        if decided.returncode not in (0, 1):
            raise ProvenanceError(f"cannot apply HEAD's ignore rules: {decided.stderr.strip()}")
        ignored_tops = {entry for entry in decided.stdout.split("\0") if entry}
        below = {name for name in names if name.split("/", 1)[0] in ignored_tops and "/" in name}
        rest = [name for name in names if name not in below]
        for name in rest:
            path = mirror / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name not in rules and not path.exists():
                path.touch()
        listing = git(mirror, "ls-files", "-z", "--others", "--ignored", PER_DIRECTORY, env=env)
        if listing.returncode != 0:
            raise ProvenanceError(f"cannot apply HEAD's ignore rules: {listing.stderr.strip()}")
        return below | ((set(listing.stdout.split("\0")) & set(rest)) - set(rules))


def head_rules(root: Path) -> dict[str, bytes]:
    """The content of every `.gitignore` file HEAD holds, by path. Raises
    `ProvenanceError` when git cannot read them."""
    blobs = {}
    for entry in listed_names(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD"):
        fields, _, name = entry.partition("\t")
        mode, kind, blob = fields.split(" ")
        if kind == "blob" and mode != SYMLINK_MODE and posixpath.basename(name) == IGNORE_FILE:
            blobs[name] = blob
    if not blobs:
        return {}
    # Read as the history holds them (`git`), so no replacement object
    # stands in for a rule.
    request = "".join(f"{blob}\n" for blob in blobs.values()).encode("ascii")
    shown = git(root, "cat-file", "--batch", stdin=request, binary=True)
    if shown.returncode != 0:
        problem = shown.stderr.decode(errors="replace").strip()
        raise ProvenanceError(f"cannot read HEAD's ignore rules: {problem}")
    # Each object is a header line, `<id> blob <size>`, its content and a newline.
    contents = {}
    output, offset = shown.stdout, 0
    for name in blobs:
        end = output.index(b"\n", offset)
        size = int(output[offset:end].split()[2])
        contents[name] = output[end + 1 : end + 1 + size]
        offset = end + 1 + size + 1
    return contents


def unseen_trees(root: Path, pathspecs: list[str], directories: set[str]) -> list[str]:
    """The paths git does not look into that `pathspecs` name themselves or
    that lie within reach of the build (`within_reach`, judged by
    `directories`): an untracked or tracked symlink to anything but a file
    (a directory, or nothing yet), an untracked nested repository and a
    submodule. Git lists each as one path and holds none of what is read
    through it. A file pathspec (`*.rs`) names no such path, but `.`, a
    listed experiment's whole repository (`listed_record_paths`), names
    every one, a submodule or a linked directory at the root included.
    Raises `ProvenanceError` when git cannot list the tree."""
    named = set(unseen_paths(root, pathspecs))
    return [
        name for name in unseen_paths(root, []) if name in named or within_reach(name, pathspecs, directories)
    ]


def unseen_paths(root: Path, pathspecs: list[str]) -> list[str]:
    """The paths under `pathspecs` (every path when there are none) that git
    does not look into, as `unseen_trees` names them, matched by git's own
    rules for pathspecs. Raises `ProvenanceError` when git cannot list the
    tree."""
    found = []
    for entry in listed_names(root, "ls-files", "-z", "--others", PER_DIRECTORY, "--", *pathspecs):
        # Without --directory git lists every untracked file, and only a
        # repository it will not enter as a directory of its own.
        name = entry.rstrip("/")
        if entry.endswith("/") or links_to_no_file(root / name):
            found.append(name)
    for entry in listed_names(root, "ls-files", "-z", "--stage", "--", *pathspecs):
        fields, _, name = entry.partition("\t")
        mode = fields.split(" ")[0]
        if mode == GITLINK_MODE or (mode == SYMLINK_MODE and links_to_no_file(root / name)):
            found.append(name)
    return found


def links_to_no_file(path: Path) -> bool:
    """Whether `path` is a symlink to something other than a file."""
    return path.is_symlink() and not path.is_file()


def head_commit(root: Path) -> str:
    """The full name of the commit HEAD is at in `root`. Raises
    `ProvenanceError` when git cannot name one (no repository, no commit)."""
    parsed = git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    name = parsed.stdout.strip()
    if parsed.returncode != 0 or not COMMIT.fullmatch(name):
        raise ProvenanceError(f"cannot name the commit HEAD is at: {parsed.stderr.strip() or 'no commit'}")
    return name


# A file's inode, size, modification time and status-change time (ns).
Stamp = tuple[int, int, int, int]


def file_stamp(path: Path) -> Stamp | None:
    """The stamp of `path`, or None when there is no such file, a directory
    on its way having been replaced by a file included."""
    try:
        status = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def file_stamps(root: Path, pathspecs: list[str]) -> dict[str, Stamp | None]:
    """The stamp of every file under `pathspecs` that git tracks or that is
    untracked (`untracked_files`), by path relative to `root`; None for a
    tracked file that is missing. On POSIX filesystems every write moves a
    file's status-change time, which ordinary tools cannot set back, so an
    edit that is undone still changes the stamp, unless it keeps the file's
    inode and size and lands within the filesystem's timestamp resolution of
    the stamp before it. Raises `ProvenanceError` when git cannot list the
    files."""
    tracked = listed_names(root, "ls-files", "-z", "--cached", "--", *pathspecs)
    return {name: file_stamp(root / name) for name in [*tracked, *untracked_files(root, pathspecs)]}


def directories_above(names) -> list[str]:
    """The repository root (`.`) and every directory that holds one of the
    paths `names`, or holds a directory that does, sorted."""
    found = {"."}
    for name in names:
        parent = posixpath.dirname(name)
        while parent and parent not in found:
            found.add(parent)
            parent = posixpath.dirname(parent)
    return sorted(found)


def directory_stamps(root: Path, names) -> dict[str, Stamp | None]:
    """The stamp of every directory on the way from `root` to the paths
    `names` (`directories_above`), by path relative to `root`; None for one
    that is gone. A directory's status-change time moves when it is renamed
    and when an entry is added to it, removed from it or renamed in it, so
    the stamps show a directory that was moved aside and put back, which the
    stamps of the files in it do not: it keeps its inode and every file in
    it keeps theirs. The directory it is in, stamped as well, moves too."""
    return {directory: file_stamp(root / directory) for directory in directories_above(names)}


class ProvenanceWatch:
    """The provenance tree of a run (the files under `pathspecs` in `root`)
    as it was when the run was about to start, to tell whether it stayed
    that way until the run's record is written.

    A record names one commit as the code it ran, but a run reads its sources
    while it runs (a `cargo run` entrypoint compiles them first), so that
    commit must be HEAD, with every provenance file as HEAD holds it, from
    before the run starts until the record is written. Creating a watch
    looks once: `head` is the commit HEAD is at and `uncommitted` lists the
    files HEAD does not hold (`uncommitted_files`), which the caller refuses
    before it runs anything. `changes` looks again and says what differs.
    Raises `ProvenanceError` when git cannot tell."""

    def __init__(self, root: Path, pathspecs: list[str]):
        self.root = root
        self.pathspecs = list(pathspecs)
        self.head = head_commit(root)
        self.stamps = file_stamps(root, self.pathspecs)
        self.uncommitted = uncommitted_files(root, self.pathspecs)
        self.foreign: set[str] = set()
        self.directories: dict[str, Stamp | None] = {}
        self.reserved: dict[str, Stamp | None] = {}

    def stamp_reserved(self, names) -> None:
        """Stamp, as they are now, the files `names` (paths relative to the
        root) that the caller wrote itself after the watch looked and that
        the pathspecs leave out, such as the record a run reserves before
        its command starts, for `changes` to compare. The command can read
        such a file, so an edit that is put back, or a file replaced by a
        copy of itself, is a change like one to any watched file. Call it
        right after the caller's last write to them."""
        self.reserved = {name: file_stamp(self.root / name) for name in names}

    def stamp_directories(self) -> None:
        """Stamp, as they are now, the directories on the way from the root
        to every file the watch holds and every file `stamp_reserved`
        stamped (`directory_stamps`), for `changes` to compare. A file's
        stamp does not show that the directory it is in was renamed aside, a
        prepared one put in its place for the run to read, and the original
        put back: every file in it is the one the watch stamped. Call it
        right before the run starts, once the caller has made every write
        of its own to the tree; from then on a run that adds an entry to one
        of them, or takes one out, even a file it removes again, is a change
        too, so it is for a run that writes nothing into the tree."""
        self.directories = directory_stamps(self.root, [*self.stamps, *self.reserved])

    @contextmanager
    def rewriting(self, files):
        """Let the run itself rewrite the watched `files` inside this block,
        as the mutation checker does when it plants a defect and removes it
        again. A write by anything else to one of them since the watch last
        stamped it is still a change; what the block writes is stamped
        afresh. Files the watch does not cover are ignored."""
        watched = [name for name in files if name in self.stamps]
        for name in watched:
            if file_stamp(self.root / name) != self.stamps[name]:
                self.foreign.add(name)
        try:
            yield
        finally:
            for name in watched:
                self.stamps[name] = file_stamp(self.root / name)

    def changes(self) -> list[str]:
        """Why the tree may no longer be the code `head` names; empty when
        the watch sees nothing that differs. It sees HEAD at another commit,
        a file under the pathspecs that HEAD does not hold, and a file
        written, replaced, created or removed since the watch looked (other
        than inside `rewriting`), even when its content was put back, and
        the same for a file `stamp_reserved` stamped, which the pathspecs
        leave out.

        Once `stamp_directories` has stamped them, it sees a directory on
        the way to a watched file that was renamed, or had an entry added or
        removed, since. Without that it does not see a file created and
        removed again between its two looks, nor a directory moved aside and
        put back; and it never sees a write that keeps a file's inode and
        size and lands within the filesystem's timestamp resolution of the
        stamp before it, a rename of a directory above the root, what
        `uncommitted_files` cannot see, or anything outside the pathspecs.
        Raises `ProvenanceError` when git cannot tell."""
        problems = []
        head = head_commit(self.root)
        if head != self.head:
            problems.append(f"HEAD moved from {self.head} to {head}")
        now = file_stamps(self.root, self.pathspecs)
        written = self.foreign | {
            name for name in self.stamps.keys() | now.keys() if self.stamps.get(name) != now.get(name)
        } | {name for name, stamp in self.reserved.items() if file_stamp(self.root / name) != stamp}
        if written:
            problems.append(f"{listed(sorted(written))} changed on disk")
        moved = [
            directory
            for directory, stamp in self.directories.items()
            if file_stamp(self.root / directory) != stamp
        ]
        if moved:
            problems.append(
                f"{listed(moved)} changed on disk, a directory holding watched files that was renamed, or had an "
                "entry added or removed, since the run started"
            )
        uncommitted = uncommitted_files(self.root, self.pathspecs)
        if uncommitted:
            problems.append(f"HEAD does not hold {listed(uncommitted)}")
        return problems


def resolve_commit(value: str, root: Path) -> str | None:
    """The full name of the commit `value` names in `root`, or None."""
    if not isinstance(value, str) or not COMMIT.fullmatch(value):
        return None
    parsed = git(root, "rev-parse", "--verify", "--quiet", f"{value}^{{commit}}")
    return parsed.stdout.strip() if parsed.returncode == 0 else None


def is_ancestor(ancestor: str, descendant: str, root: Path) -> bool:
    """Whether commit `ancestor` is `descendant` or on its history."""
    return git(root, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def changes_after(base: str, head: str, root: Path, paths: tuple[str, ...]) -> list[str]:
    """The commits that descend from `base`, are `head` or on its history, and
    changed a file under `paths`, or a symlink or submodule within reach of
    the build, as `code_changes` counts them, parents before children. Only
    descendants of `base` count (`--ancestry-path`): a commit merged in from
    another line of history, say the base branch of a pull request, whose
    merge CI checks, changed files the results at `base` never ran, but it
    is not a change after them; the merge that brings it to their line is.
    Raises `ProvenanceError` when git cannot list them."""
    # Read as bytes, so a name that is not UTF-8 a commit once held keeps its
    # bytes rather than failing every later check.
    history = git(
        root, "log", "-z", "--raw", "--no-renames", "-m", "--format=", "--ancestry-path", f"{base}..{head}",
        binary=True,
    )
    if history.returncode != 0:
        problem = history.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(f"cannot list the commits after {base}: {problem}")
    links = links_within_reach(link_paths(history.stdout.decode("utf-8", "surrogateescape").split("\0")), root, paths)
    literal = [f":(literal){name}" for name in sorted(set(links))]
    listed_commits = git(
        root, "rev-list", "--reverse", "--topo-order", "--ancestry-path", f"{base}..{head}",
        "--", *paths, *literal,
    )
    if listed_commits.returncode != 0:
        raise ProvenanceError(f"cannot list the commits after {base}: {listed_commits.stderr.strip()}")
    return listed_commits.stdout.split()


def changes_after_results(
    base: str, head: str, root: Path, paths: tuple[str, ...], experiment_id: str, experiment: str | None
) -> list[str]:
    """The commits `changes_after` names, and for a listed experiment (in the
    repository directory `experiment`, None for another) also those after
    `base` that left its manifest or the registry other than `base` holds
    them but for the status completing it moves (`completion_changes`,
    judged against `base`), in one order, parents before children. Raises
    `ProvenanceError` when git cannot list them."""
    found = changes_after(base, head, root, paths)
    if experiment is None:
        return found
    names = [f":(literal){experiment}/{AFTER_THE_RUNS[0]}", *(f":(literal){name}" for name in AFTER_THE_RUNS[1:])]
    touched = git(root, "rev-list", "--reverse", "--topo-order", "--ancestry-path", f"{base}..{head}", "--", *names)
    if touched.returncode != 0:
        raise ProvenanceError(f"cannot list the commits after {base}: {touched.stderr.strip()}")
    moved = {
        commit for commit in touched.stdout.split() if completion_changes(experiment_id, experiment, base, commit, root)
    }
    if not moved:
        return found
    every = git(root, "rev-list", "--reverse", "--topo-order", "--ancestry-path", f"{base}..{head}")
    if every.returncode != 0:
        raise ProvenanceError(f"cannot list the commits after {base}: {every.stderr.strip()}")
    counted = {*found, *moved}
    return [commit for commit in every.stdout.split() if commit in counted]


def completion_changes(
    experiment_id: str, experiment: str, base: str, head: str | None, root: Path
) -> list[str]:
    """The files completing the listed experiment `experiment_id` (in the
    repository directory `experiment`) changes (`AFTER_THE_RUNS`: its
    manifest and the registry) whose text at `head`, or in the checkout of
    `root` when None, is not their text at commit `base` but for the status
    completing it moves: the one its manifest names and the one the
    registry's entry for it names, each on its own line, every other line
    and value as it was (`status_moved_only`). Its command may read either
    file whole, so another experiment's entry, a comment or a key added is
    a change after its runs, as a change to any other file of the
    repository is. Raises `ProvenanceError` when git cannot read them."""
    changed = []
    for name in (f"{experiment}/{AFTER_THE_RUNS[0]}", *AFTER_THE_RUNS[1:]):
        before, after = text_at(root, base, name), text_at(root, head, name)
        if before == after:
            continue
        if before is None or after is None or not status_moved_only(
            before, after, experiment_id if name in AFTER_THE_RUNS[1:] else None
        ):
            changed.append(name)
    return changed


def text_at(root: Path, commit: str | None, name: str) -> str | None:
    """The UTF-8 text of the regular file `name` (a repository path) at
    `commit` in `root`, or in its checkout when None; None when there is
    none there, it is no regular file (a symlink, whose target path is all
    git holds of it), or it is not UTF-8. Raises `ProvenanceError` when git
    cannot read the commit."""
    if commit is None:
        path = root / name
        if path.is_symlink() or not path.is_file():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
    else:
        listing = git(root, "ls-tree", "-z", commit, "--", f":(literal){name}")
        if listing.returncode != 0:
            raise ProvenanceError(f"cannot read {name} at {commit}: {listing.stderr.strip()}")
        entry = listing.stdout.split("\0")[0]
        fields, _, _ = entry.partition("\t")
        if not entry or fields.split(" ")[:2] not in (["100644", "blob"], ["100755", "blob"]):
            return None
        shown = git(root, "cat-file", "blob", fields.split(" ")[2], binary=True)
        if shown.returncode != 0:
            raise ProvenanceError(f"cannot read {name} at {commit}: {shown.stderr.decode(errors='replace').strip()}")
        data = shown.stdout
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def status_moved_only(before: str, after: str, experiment_id: str | None) -> bool:
    """Whether the TOML text `after` is `before` with only a status moved:
    the registry's entry for `experiment_id`, or the manifest's own when
    None. Every line is as it was but for the value of a status line
    (`STATUS_LINE`), and read as TOML the two are equal once that status is
    set aside, so no other entry's status, no status in another table and
    no line of a multi-line string moved."""
    lines, moved = before.split("\n"), after.split("\n")
    if len(lines) != len(moved):
        return False
    for line, other in zip(lines, moved):
        if line != other and not (
            STATUS_LINE.match(line)
            and STATUS_LINE.match(other)
            and STATUS_LINE.sub(r'\1""', line) == STATUS_LINE.sub(r'\1""', other)
        ):
            return False
    try:
        documents = [tomllib.loads(before), tomllib.loads(after)]
    except tomllib.TOMLDecodeError:
        return False
    for document in documents:
        if experiment_id is None:
            document.pop("status", None)
            continue
        entries = document.get("experiment")
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("id") == experiment_id:
                entry.pop("status", None)
    return documents[0] == documents[1]


def listed(paths: list[str], limit: int = 5) -> str:
    more = f" and {len(paths) - limit} more" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def source_revision(
    experiment_id: str,
    manifest: dict,
    records: dict[str, dict],
    root: Path,
    experiment_dir: Path,
    paths: tuple[str, ...] | None = None,
    checkout_paths: tuple[str, ...] | None = None,
    listed_experiment: bool = False,
) -> str:
    """The commit the results in `records` were produced at: the earliest
    record's `git_sha`, once every record agrees (`agreement_errors`),
    every record's commit has the same provenance files (`paths`:
    `seed_record_paths` of `experiment_dir` unless named, and for a listed
    experiment the whole repository but its seed records,
    `listed_record_paths`), the checkout in `root` has them too
    (`checkout_paths`, `paths` unless named, and for a listed experiment
    `listed_staleness_paths`, which leaves out what the tools write once the
    seeds have run and what completing it changes), and HEAD holds each of
    them as the checkout does (`checkout_problem`); for a listed experiment
    (`listed_experiment`), its manifest and the registry, in the checkout
    and at HEAD, differ from their text at that commit in no more than the
    status completing it moves (`completion_changes`). Raises
    `ProvenanceError` naming what disagrees."""
    if not records:
        raise ProvenanceError("no run records")
    errors = agreement_errors(experiment_id, manifest, records)
    if errors:
        raise ProvenanceError("; ".join(errors))
    paths = seed_record_paths(experiment_dir, root) if paths is None else paths
    ordered = sorted(records.items(), key=lambda item: (str(item[1].get("started_at", "")), item[0]))
    revision = ordered[0][1]["git_sha"]
    for name, record in ordered[1:]:
        if record["git_sha"] != revision:
            changed = code_changes(revision, record["git_sha"], root, paths)
            if changed:
                errors.append(f"{name} ran at {record['git_sha']}, whose code differs in {listed(changed)}")
    if errors:
        raise ProvenanceError(f"the records ran different code from {revision}: " + "; ".join(errors))
    checkout_paths = paths if checkout_paths is None else checkout_paths
    changed = code_changes(revision, None, root, checkout_paths)
    if listed_experiment:
        experiment = relative_to_root(experiment_dir, root)
        for head in (None, "HEAD"):
            moved = completion_changes(experiment_id, experiment, revision, head, root)
            changed.extend(name for name in moved if name not in changed)
    if changed:
        raise ProvenanceError(
            f"the records ran at {revision}, and the checkout's code has changed since, "
            f"in {listed(changed)}; rerun the seeds or aggregate at that commit"
        )
    unheld = checkout_problem(root, checkout_paths)
    if unheld:
        raise ProvenanceError(
            f"the records ran at {revision}, but {unheld}; commit or remove them and aggregate again"
        )
    return revision


def checkout_problem(root: Path, paths: tuple[str, ...]) -> str:
    """Why the checkout in `root` may not be the code of a commit whose
    provenance files under `paths` `code_changes` found equal to it; empty
    when it is. git diff compares tracked files only, so an untracked file
    (a new `crates/ptr-*` crate the workspace takes as a member), one hidden
    from git, what a symlink points at, or a crate linked or cloned in, which
    git lists as one path, passes it unseen; `uncommitted_files`, the check a
    recorder runs before and after a run, sees them as far as its docstring
    says."""
    unheld = uncommitted_files(root, list(paths))
    if not unheld:
        return ""
    return f"HEAD does not hold {listed(unheld)} in the checkout, so it may not be that code"


def is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def last_result_line(stdout: str) -> dict:
    """The last line of `stdout` that starts a JSON object, parsed. Raises
    `ProvenanceError` when there is none or it is not a JSON object."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProvenanceError(f"its result line is not a JSON object: {error}") from error
            if not isinstance(result, dict):
                raise ProvenanceError("its result line is not a JSON object")
            return result
    raise ProvenanceError("it has no result line")


def harness_results(
    records: dict[str, dict], benchmark: str, counts: tuple[str, ...] = ()
) -> dict[str, dict]:
    """The harness result each of `records` (file name -> run record) carries,
    once every result is the run its wrapper was selected for: the wrapper's
    command ran `benchmark`, and the result reports that benchmark, the
    wrapper's seed and, when the wrapper set one, its iteration count. Every
    counter in `counts` (the hard counters an aggregator judges) must be
    reported as a nonnegative count, so a counter the harness stopped
    reporting cannot read as zero failures. Raises `ProvenanceError` naming
    every record that fails."""
    results: dict[str, dict] = {}
    errors = []
    for name, record in records.items():
        try:
            result = last_result_line(str(record.get("stdout") or ""))
        except ProvenanceError as error:
            errors.append(f"{name}: {error}")
            continue
        if benchmark not in (record.get("command") or []):
            errors.append(f"{name} did not run {benchmark!r}: {record.get('command')!r}")
        if result.get("benchmark") != benchmark:
            errors.append(f"{name} reports benchmark {result.get('benchmark')!r}, not {benchmark!r}")
        seed = record.get("seed")
        if not is_count(result.get("seed")) or result.get("seed") != seed:
            errors.append(f"{name} reports seed {result.get('seed')!r}, not {seed!r}")
        iterations = (record.get("parameters") or {}).get("iterations")
        if iterations is not None and str(result.get("iterations")) != str(iterations):
            errors.append(f"{name} reports {result.get('iterations')!r} iterations, not {iterations}")
        uncounted = [key for key in counts if not (is_count(result.get(key)) and result[key] >= 0)]
        if uncounted:
            errors.append(f"{name} reports no count of {listed(uncounted)}")
        results[name] = result
    if errors:
        raise ProvenanceError("; ".join(errors))
    return results


def mutation_evidence(
    experiment_id: str, path: Path, subcommand: str, root: Path, experiment_dir: Path
) -> dict | None:
    """The mutation check summary (`killed`, `total`, `git_sha` and the
    SHA-256 of the whole file, `sha256`, which binds each outcome it lists)
    of `path`, or None when there is no such file. The evidence must be of
    `experiment_id`, have mutated `subcommand`, count its own outcomes, and
    have run at a commit whose provenance files (`mutation_record_paths`) are
    the checkout's, which HEAD holds as the checkout does (`checkout_problem`);
    `source_revision` has already shown the checkout's code is the aggregated
    records'. Raises `ProvenanceError` otherwise."""
    if not path.exists():
        return None
    name = path.name
    try:
        data = path.read_bytes()
        evidence = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"{name} cannot be read: {error}") from error
    if not isinstance(evidence, dict):
        raise ProvenanceError(f"{name} is not a JSON object")
    errors = []
    if evidence.get("experiment_id") != experiment_id:
        errors.append(f"{name} is evidence of {evidence.get('experiment_id')!r}, not {experiment_id!r}")
    if evidence.get("subcommand") != subcommand:
        errors.append(f"{name} mutated {evidence.get('subcommand')!r}, not {subcommand!r}")
    killed, total, outcomes = evidence.get("killed"), evidence.get("total"), evidence.get("mutations")
    if not (is_count(killed) and is_count(total) and isinstance(outcomes, list)):
        errors.append(f"{name} has no killed and total counts and outcomes")
    elif not (
        0 < total == len(outcomes)
        and killed == sum(isinstance(item, dict) and item.get("result") == "killed" for item in outcomes)
    ):
        errors.append(f"{name} counts {killed} of {total} killed, but lists {len(outcomes)} outcomes")
    sha = evidence.get("git_sha")
    if not COMMIT.fullmatch(str(sha or "")):
        errors.append(f"{name} names no commit it ran at: {sha!r}")
    if errors:
        raise ProvenanceError("; ".join(errors))
    paths = mutation_record_paths(experiment_dir, root)
    changed = code_changes(sha, None, root, paths)
    if changed:
        raise ProvenanceError(
            f"{name} ran at {sha}, and the checkout's code has changed since, in {listed(changed)}; "
            f"rerun scripts/mutation_check.py {experiment_id} or remove {name}"
        )
    unheld = checkout_problem(root, paths)
    if unheld:
        raise ProvenanceError(f"{name} ran at {sha}, but {unheld}; commit or remove them, or remove {name}")
    return {"killed": killed, "total": total, "git_sha": sha, "sha256": hashlib.sha256(data).hexdigest()}


def clear_stale_marker(results_dir: Path) -> None:
    """Remove `results_dir`'s stale marker: results just aggregated from the
    checkout's code describe it."""
    (results_dir / STALE_MARKER).unlink(missing_ok=True)


def sync_directory(directory: Path) -> None:
    """Flush `directory`'s entries to disk where the platform lets a
    directory be opened, so a rename in it outlives a crash."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_temporary(path: Path, data: bytes, mode: int | None = None) -> Path:
    """A new file beside `path` holding `data`, flushed to disk and given
    `mode` when one is named; its name starts with a dot and ends in `.tmp`,
    which the results directories ignore. When the write fails the file is
    removed and the error raised."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_atomically(path: Path, text: str) -> None:
    """Replace `path` by `text` (UTF-8) in one step, keeping its permissions:
    whatever interrupts the write, `path` holds its old content or the new,
    never a part of either."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None
    temporary = write_temporary(path, text.encode("utf-8"), mode)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def write_exclusively(path: Path, text: str) -> None:
    """Create `path` holding `text` (UTF-8) in one step: it appears whole or
    not at all. Raises `FileExistsError`, and leaves the file as it is, when
    `path` exists."""
    temporary = write_temporary(path, text.encode("utf-8"))
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        except OSError:
            # A filesystem without hard links: refuse an existing file, then
            # rename, which leaves a moment for another writer to create one.
            if os.path.lexists(path):
                raise FileExistsError(f"{path} exists") from None
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sync_directory(path.parent)


def json_text(value) -> str:
    """`value` as the aggregators and recorders write JSON."""
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def is_printable_ascii(text: str) -> bool:
    """Every character of `text` is one of the 95 printable ASCII
    characters, space included."""
    return all(" " <= character <= "~" for character in text)


OUTSIDE_ASCII = "holds a character outside printable ASCII, whose escape depends on who serializes it"
# The largest magnitude an integer of the canonical text may have: RFC 8785
# writes numbers as IEEE-754 doubles, which hold every integer only up to it.
MAX_CANONICAL_INTEGER = 2**53 - 1
OUTSIDE_RANGE = f"is an integer beyond +-{MAX_CANONICAL_INTEGER}, which RFC 8785 cannot write exactly"


def canonical_value_problem(value) -> str | None:
    """Why `value` has no canonical preregistration text, or None when it
    has one: a boolean, an integer of magnitude at most
    `MAX_CANONICAL_INTEGER`, a string of printable ASCII, or a list holding
    only such integers or only such strings. A float, a date, a table, any
    other list, a larger integer, and a string holding a control or non-ASCII
    character (which one JSON writer escapes and another writes raw) have a
    text that depends on who serializes them."""
    if type(value) is bool:
        return None
    if type(value) is int:
        return None if abs(value) <= MAX_CANONICAL_INTEGER else OUTSIDE_RANGE
    if type(value) is str:
        return None if is_printable_ascii(value) else f"is a string that {OUTSIDE_ASCII}"
    if isinstance(value, list):
        kinds = {type(element) for element in value}
        if kinds <= {int}:
            for index, element in enumerate(value):
                if abs(element) > MAX_CANONICAL_INTEGER:
                    return f"has element {index} that {OUTSIDE_RANGE}"
            return None
        if kinds <= {str}:
            for index, element in enumerate(value):
                if not is_printable_ascii(element):
                    return f"has element {index} that {OUTSIDE_ASCII}"
            return None
        return "is a list that holds anything but only integers or only strings"
    return f"is a {type(value).__name__}, which has no canonical text"


def canonical_text(value) -> str:
    """The canonical text of a value `canonical_value_problem` accepts, or of
    a table of them: `true` or `false`; an integer in decimal, with a leading
    `-` when negative and no leading zeros; a string between double quotes,
    in which `\\` is written `\\\\` and `"` is written `\\"` and every other
    character as itself (no other escape, `/` included); a list as `[`, its
    elements joined by `,`, `]`; a table as `{`, its entries `"key":value` in
    ascending order of their keys' bytes joined by `,`, `}`. No whitespace
    anywhere. On this domain it is exactly the JSON Canonicalization Scheme of
    RFC 8785, and what Python's `json.dumps` writes with sorted keys and the
    separators `,` and `:`."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is int:
        return str(value)
    if type(value) is str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ",".join(canonical_text(element) for element in value) + "]"
    if isinstance(value, dict):
        entries = sorted(value.items(), key=lambda item: item[0].encode("ascii"))
        return "{" + ",".join(canonical_text(key) + ":" + canonical_text(item) for key, item in entries) + "}"
    raise ValueError(f"no canonical text for {type(value).__name__}")


def canonical_digest(value) -> str:
    """The SHA-256, in hex, of the UTF-8 bytes of `canonical_text(value)`."""
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def preregistration_canonical(table: dict) -> str:
    """The canonical text (`canonical_text`) of an experiment's
    `[preregistration]` table, which a harness in another language reproduces
    byte for byte. Raises `ValueError`, naming the key, for a key outside
    printable ASCII or a value `canonical_value_problem` refuses."""
    for key, value in table.items():
        if not is_printable_ascii(key):
            raise ValueError(f"preregistration key {key!r} {OUTSIDE_ASCII}")
        problem = canonical_value_problem(value)
        if problem:
            raise ValueError(f"preregistration key {key} {problem}")
    return canonical_text(table)


def preregistration_digest(table: dict) -> str:
    """The SHA-256, in hex, of the UTF-8 bytes of
    `preregistration_canonical(table)`: the `preregistration_sha256` an
    experiment's manifest names once it leaves `planned`
    (`scripts/check_research_gates.py`)."""
    preregistration_canonical(table)
    return canonical_digest(table)


def preregistered_bytes_digest(data: bytes) -> str:
    """The SHA-256, in hex, of `data`, the content of a file a
    preregistration names, byte for byte: line endings are content, so a
    checkout that converts them holds another file than the one frozen, and
    the gate names it as such rather than reading two forms alike."""
    return hashlib.sha256(data).hexdigest()


def preregistered_file_digest(path: Path) -> str:
    """`preregistered_bytes_digest` of the file at `path`, such as an
    adjudication protocol."""
    return preregistered_bytes_digest(path.read_bytes())


RUN = "run.json"
METRICS = "metrics.json"
MUTATIONS = "mutations.json"


def publish_aggregate(results_dir: Path, metrics: dict, run: dict) -> None:
    """Write `metrics` and `run` as `results_dir`'s metrics.json and
    run.json, one aggregate, and then clear the stale marker.

    run.json names the SHA-256 of the metrics.json written with it
    (`metrics_sha256`). Both are written to temporary files first, so a
    failure while writing leaves the previous aggregate as it was. Then the
    previous run.json is removed, and metrics.json and run.json are moved
    into place in that order: an interruption leaves the previous pair, or a
    metrics.json without run.json, which `check_research_gates.py` fails as a
    missing artifact, never a run.json beside metrics it was not aggregated
    with. The marker is cleared only once both files read back as written
    and the mutation evidence beside them is the one `run` carries
    (`aggregate_problems`), whole: mutation checks that name no SHA-256 of
    it (`mutation_evidence`) are refused before anything is written. Raises
    `ProvenanceError` otherwise."""
    carried = run.get("mutation_checks")
    if isinstance(carried, dict) and "sha256" not in carried:
        raise ProvenanceError(
            f"the mutation checks {RUN} would carry name no sha256 of {MUTATIONS}, so nothing would bind the "
            "outcomes it lists; read it with mutation_evidence"
        )
    metrics_data = json_text(metrics).encode("utf-8")
    run = {**run, "metrics_sha256": hashlib.sha256(metrics_data).hexdigest()}
    run_data = json_text(run).encode("utf-8")
    staged: dict[str, Path] = {}
    try:
        for name, data in ((METRICS, metrics_data), (RUN, run_data)):
            staged[name] = write_temporary(results_dir / name, data)
        (results_dir / RUN).unlink(missing_ok=True)
        sync_directory(results_dir)
        os.replace(staged[METRICS], results_dir / METRICS)
        sync_directory(results_dir)
        os.replace(staged[RUN], results_dir / RUN)
        sync_directory(results_dir)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    try:
        published = [(results_dir / name).read_bytes() for name in (METRICS, RUN)]
    except OSError as error:
        raise ProvenanceError(f"{METRICS} and {RUN} cannot be read back: {error}") from error
    if published != [metrics_data, run_data]:
        raise ProvenanceError(f"{METRICS} or {RUN} changed while they were published; aggregate again")
    problems = aggregate_problems(run, results_dir)
    if problems:
        raise ProvenanceError("; ".join(problems) + "; aggregate again")
    clear_stale_marker(results_dir)


def seed_record_digests(results_dir: Path) -> dict[str, str]:
    """The SHA-256 of each seed's run record in `results_dir`, by the seed as
    JSON writes it (`"17"`), which an aggregate of a listed experiment carries
    as `seed_records` (`check_research_gates.py` binds it to the records): a
    record of a command that failed to launch saw no outcome and is left out.
    Raises `ProvenanceError` when a record cannot be read or two are of one
    seed, which a listed experiment does not run twice."""
    digests: dict[str, str] = {}
    for path in sorted(results_dir.glob("run-*.json")):
        try:
            data = path.read_bytes()
            record = json.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ProvenanceError(f"{path.name} cannot be read: {error}") from error
        if not isinstance(record, dict) or "seed" not in record or record.get("status") == "failed-to-launch":
            continue
        seed = json.dumps(record["seed"], sort_keys=True)
        if seed in digests:
            raise ProvenanceError(f"seed {seed} has more than one run record in {results_dir}")
        digests[seed] = hashlib.sha256(data).hexdigest()
    return digests


def mutation_summary(path: Path) -> dict:
    """The summary of the mutation evidence at `path` that an aggregate
    carries (`mutation_evidence`): its counts, the commit it ran at and the
    SHA-256 of the whole file, which binds each outcome it lists. Raises
    OSError, ValueError or AttributeError when it cannot be read."""
    data = path.read_bytes()
    evidence = json.loads(data.decode("utf-8"))
    summary = {key: evidence.get(key) for key in ("killed", "total", "git_sha")}
    return {**summary, "sha256": hashlib.sha256(data).hexdigest()}


def aggregate_problems(run: dict, results_dir: Path) -> list[str]:
    """Why `run`, the run.json of `results_dir`, is not one aggregate with
    the metrics.json and mutations.json beside it; empty when it is. Each
    problem starts with the name of the file it is about.

    metrics.json must hash to the `metrics_sha256` run.json names, when it
    names one (`publish_aggregate`). mutations.json must be there exactly
    when run.json carries `mutation_checks`, and have that summary, the
    SHA-256 of the whole file included: the mutation checker rewrites it on
    its own, after or while the aggregate is written, and outcomes replaced
    under the same counts would be other evidence. A summary aggregated
    before it named that SHA-256 is compared by its counts and commit alone,
    which `aggregate_errors` accepts only while the aggregate is stale."""
    problems = []
    bound = run.get("metrics_sha256")
    if bound is not None:
        try:
            digest = hashlib.sha256((results_dir / METRICS).read_bytes()).hexdigest()
        except OSError as error:
            problems.append(f"{METRICS}, which {RUN} binds, cannot be read: {error}")
        else:
            if digest != bound:
                problems.append(
                    f"{METRICS} is not the metrics {RUN} was aggregated with: its SHA-256 is {digest}, "
                    f"{RUN} names {bound}"
                )
    carried = run.get("mutation_checks")
    evidence = results_dir / MUTATIONS
    if carried is None:
        if evidence.exists():
            problems.append(f"{RUN} carries no mutation checks, but {MUTATIONS} is there")
    else:
        try:
            summary = mutation_summary(evidence)
        except (OSError, ValueError, AttributeError) as error:
            problems.append(f"{MUTATIONS}, whose checks {RUN} carries, cannot be read: {error}")
        else:
            if isinstance(carried, dict) and "sha256" not in carried:
                summary.pop("sha256")
            if summary != carried:
                problems.append(
                    f"{MUTATIONS} is not the evidence {RUN} carries: it sums up to "
                    f"{json.dumps(summary, sort_keys=True)}, "
                    f"{RUN} carries {json.dumps(carried, sort_keys=True)}"
                )
    return problems


def aggregate_errors(experiment_id: str, experiment_dir: Path, results_dir: Path, root: Path) -> list[str]:
    """Why the archived run.json of `experiment_id` is not one aggregate with
    the metrics.json and mutations.json beside it (`aggregate_problems`);
    empty when it is, or when there is no readable run.json, which the
    required artifacts and `staleness_errors` report.

    A run.json that names no `metrics_sha256` was aggregated before run.json
    bound its metrics, and passes only while it is stale, since current
    results ran HEAD's aggregator, whose `publish_aggregate` binds them
    (every `aggregate.py` publishes through it), and only while it and the
    metrics.json beside it are the pair archived together
    (`unbound_pair_problems`). So does one whose mutation checks name no
    SHA-256 of mutations.json, only while it is stale and the mutations.json
    beside it is the one committed with it (`unbound_evidence_problems`):
    its counts alone would pass outcomes replaced under them."""
    shown = relative_to_root(results_dir, root)
    try:
        run = json.loads((results_dir / RUN).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(run, dict):
        return []
    problems = aggregate_problems(run, results_dir)
    if run.get("metrics_sha256") is None:
        if is_current(run.get("git_sha"), seed_record_paths(experiment_dir, root), root):
            problems.insert(
                0,
                f"{RUN} names no metrics_sha256, so nothing binds it to the {METRICS} beside it; "
                "rerun its aggregate.py",
            )
        else:
            problems[0:0] = unbound_pair_problems(results_dir, root)
    carried = run.get("mutation_checks")
    if isinstance(carried, dict) and "sha256" not in carried and os.path.lexists(results_dir / MUTATIONS):
        if is_current(run.get("git_sha"), seed_record_paths(experiment_dir, root), root):
            problems.append(
                f"{RUN} carries mutation checks that name no sha256 of {MUTATIONS}, so nothing binds the "
                "outcomes it lists; rerun its aggregate.py"
            )
        else:
            problems.extend(unbound_evidence_problems(results_dir, root))
    return [f"{experiment_id}: {shown}/{problem}" for problem in problems]


def unbound_pair_problems(results_dir: Path, root: Path) -> list[str]:
    """Why the run.json of `results_dir`, which names no `metrics_sha256`,
    and the metrics.json beside it are not the pair the last commit that
    changed run.json holds, as an aggregate archived before run.json bound
    its metrics is; empty when they are. New metrics beside the archived
    run.json, whether an interrupted aggregation left them, the old run.json
    was restored around them or they were committed on their own, are not
    that pair. Each problem starts with the name of the file it is about."""
    run_path, metrics_path = (relative_to_root(results_dir / name, root) for name in (RUN, METRICS))
    last = git(root, "log", "-1", "--format=%H", "--", run_path)
    commit = last.stdout.strip()
    if last.returncode != 0 or not COMMIT.fullmatch(commit):
        return [
            f"{RUN} names no metrics_sha256, and no commit holds it, so nothing binds it to the "
            f"{METRICS} beside it; rerun its aggregate.py"
        ]
    if not holds_committed(root, commit, run_path):
        return [
            f"{RUN} names no metrics_sha256 and is not the run.json committed at {commit}, so nothing "
            f"binds it to the {METRICS} beside it; restore it or rerun its aggregate.py"
        ]
    if os.path.lexists(root / metrics_path) and not holds_committed(root, commit, metrics_path):
        return [
            f"{METRICS} is not the metrics.json committed with {RUN}, which names no metrics_sha256, "
            f"at {commit}; restore it or rerun its aggregate.py"
        ]
    return []


def unbound_evidence_problems(results_dir: Path, root: Path) -> list[str]:
    """Why the mutations.json of `results_dir`, whose summary the run.json
    beside it carries without its SHA-256, is not the one the last commit
    that changed run.json holds, as evidence aggregated before an aggregate
    bound it whole is; empty when it is. Outcomes replaced under the same
    counts, committed or not, are not that evidence. Each problem starts
    with the name of the file it is about."""
    run_path, evidence_path = (relative_to_root(results_dir / name, root) for name in (RUN, MUTATIONS))
    last = git(root, "log", "-1", "--format=%H", "--", run_path)
    commit = last.stdout.strip()
    if last.returncode != 0 or not COMMIT.fullmatch(commit):
        return [
            f"{RUN} carries mutation checks that name no sha256 of {MUTATIONS}, and no commit holds it, so "
            f"nothing binds the outcomes {MUTATIONS} lists; rerun its aggregate.py"
        ]
    if not holds_committed(root, commit, evidence_path):
        return [
            f"{MUTATIONS} is not the evidence committed with {RUN}, whose mutation checks name no sha256 of "
            f"it, at {commit}; restore it or rerun its aggregate.py"
        ]
    return []


def holds_committed(root: Path, commit: str, relative: str) -> bool:
    """Whether the working tree of `root` holds, as a regular file at
    `relative`, the content `commit` holds there as one, its bytes read
    from disk as they are, so no clean filter or attribute of the clone's
    own answers for them as it could for git hash-object, and a copy that
    differs from the blob in its line endings is another file. False when
    either holds no regular file there or git cannot read the commit's."""
    listing = git(root, "--literal-pathspecs", "ls-tree", "-z", "--full-name", commit, "--", relative)
    held = None
    for entry in listing.stdout.split("\0") if listing.returncode == 0 else ():
        fields, _, name = entry.partition("\t")
        parts = fields.split(" ")
        if name == relative and len(parts) == 3 and parts[1] == "blob" and parts[0] in ("100644", "100755"):
            held = parts[2]
    if held is None:
        return False
    path = root / relative
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        data = path.read_bytes()
    except OSError:
        return False
    shown = git(root, "cat-file", "blob", held, binary=True)
    if shown.returncode != 0:
        return False
    content = shown.stdout
    return data == content


def is_current(sha, paths: tuple[str, ...], root: Path) -> bool:
    """Whether commit `sha` has HEAD's provenance files under `paths`; false
    when `sha` names no commit git can compare."""
    if not COMMIT.fullmatch(str(sha or "")):
        return False
    try:
        return not code_changes(sha, "HEAD", root, paths)
    except ProvenanceError:
        return False


# The archived evidence of a completed experiment and its provenance files.
ARCHIVED = (("run.json", seed_record_paths), ("mutations.json", mutation_record_paths))


def staleness_errors(
    experiment_id: str, experiment_dir: Path, results_dir: Path, root: Path, listed_experiment: bool = False
) -> list[str]:
    """Why the archived results of `experiment_id` misdescribe HEAD's code;
    empty when they do not. `results/run.json` and `results/mutations.json`
    are stale when HEAD's provenance files (`seed_record_paths`,
    `mutation_record_paths`) differ from those at their `git_sha`; for an
    experiment the list names (`listed_experiment`), whose command may run
    or read any file of the repository, when HEAD's repository does, but
    for the tools' outputs in its results and its stale marker
    (`listed_staleness_paths`), or when its manifest or the registry
    differs from its text there in more than the status completing it
    moves (`completion_changes`). Stale results pass only with a
    `results/STALE.toml` marker that names them, the
    first commit after them that changed a provenance file (`stale_since`)
    and a `reason` (`marker_errors`); a marker next to current results is an
    error too, so none outlives the rerun that replaces them."""
    results = relative_to_root(results_dir, root)
    experiment = relative_to_root(experiment_dir, root)
    errors = []
    stale: dict[str, tuple[str, tuple[str, ...], list[str]]] = {}
    for name, paths_of in ARCHIVED:
        path = results_dir / name
        if not path.exists():
            continue
        try:
            sha = json.loads(path.read_text(encoding="utf-8")).get("git_sha")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
            errors.append(f"{experiment_id}: {results}/{name} cannot be read: {error}")
            continue
        if not COMMIT.fullmatch(str(sha or "")):
            errors.append(f"{experiment_id}: {results}/{name} names no commit it ran at: {sha!r}")
            continue
        paths = listed_staleness_paths(results, experiment) if listed_experiment else paths_of(experiment_dir, root)
        try:
            changed = code_changes(sha, "HEAD", root, paths)
            if listed_experiment:
                changed = [*changed, *completion_changes(experiment_id, experiment, sha, "HEAD", root)]
        except ProvenanceError as error:
            errors.append(f"{experiment_id}: {results}/{name}: {error}")
            continue
        if changed:
            stale[name] = (sha, paths, changed)
    marker = results_dir / STALE_MARKER
    if not marker.exists():
        for name, (sha, _, changed) in stale.items():
            errors.append(
                f"{experiment_id}: {results}/{name} ran at {sha}, and the code has changed since, in "
                f"{listed(changed)}; rerun the experiment or record since when it is stale in "
                f"{results}/{STALE_MARKER}"
            )
        return errors
    if not stale:
        if not errors:
            errors.append(
                f"{experiment_id}: {results}/{STALE_MARKER} marks results stale that match HEAD's code; "
                "remove it"
            )
        return errors
    return errors + marker_errors(
        experiment_id, f"{results}/{STALE_MARKER}", marker, stale, root, experiment if listed_experiment else None
    )


def marker_errors(
    experiment_id: str,
    shown: str,
    marker: Path,
    stale: dict[str, tuple[str, tuple[str, ...], list[str]]],
    root: Path,
    experiment: str | None = None,
) -> list[str]:
    """Why the stale marker `marker` (shown as `shown`) does not honestly
    describe the `stale` results; empty when it does. For a listed
    experiment (in the repository directory `experiment`), a change to its
    manifest or the registry beyond the status completing it moves counts
    as a change of a provenance file (`completion_changes`).

    `results_git_sha` names the results: a commit on HEAD's history at which
    every stale file's provenance files are those at its own `git_sha`. When
    `run.json` and `mutations.json` ran at one commit that is it; when the
    seeds and the mutation check ran at different commits with the same
    code, either commit (or any other with that code) is. `stale_since` is a
    first change after it: a commit that descends from it, is HEAD or on
    HEAD's history, and changed a provenance file of a stale result, with no
    such change between the two. Commits that do not descend from the
    results (a base branch's, merged in) never count; the merge that brings
    their change to the results' line does."""
    try:
        fields = tomllib.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return [f"{experiment_id}: {shown} cannot be read: {error}"]
    errors = []
    for key in ("results_git_sha", "stale_since", "reason"):
        if key not in fields:
            errors.append(f"{experiment_id}: {shown} has no {key}")
    if errors:
        return errors
    if not isinstance(fields["reason"], str) or not fields["reason"].strip():
        errors.append(f"{experiment_id}: {shown} gives no reason")
    marked = resolve_commit(fields["results_git_sha"], root)
    since = resolve_commit(fields["stale_since"], root)
    for key, resolved in (("results_git_sha", marked), ("stale_since", since)):
        if resolved is None:
            errors.append(f"{experiment_id}: {shown}: {key} {fields[key]!r} is not a commit")
    if errors:
        return errors
    others = []
    for name, (sha, name_paths, _) in stale.items():
        differing = code_changes(sha, marked, root, name_paths)
        if experiment is not None:
            differing = [*differing, *completion_changes(experiment_id, experiment, sha, marked, root)]
        if differing:
            others.append(f"{name} ran at {sha}, whose provenance files differ there in {listed(differing)}")
    if others:
        return [f"{experiment_id}: {shown} names results of {marked}, but " + "; ".join(others)]
    if not is_ancestor(marked, "HEAD", root):
        return [f"{experiment_id}: {shown} names results of {marked}, which is not on HEAD's history"]
    paths: list[str] = []
    for _, name_paths, _ in stale.values():
        paths.extend(path for path in name_paths if path not in paths)
    changes = changes_after_results(marked, "HEAD", root, tuple(paths), experiment_id, experiment)
    # A first change is one no other change after the results precedes. Two
    # lines of history leaving the results each have their own; the marker
    # may name either.
    if since not in changes or changes_after_results(
        marked, since, root, tuple(paths), experiment_id, experiment
    ) != [since]:
        first = changes[0] if changes else None
        errors.append(
            f"{experiment_id}: {shown} says stale since {since}, but the first commit after {marked} "
            f"that changed a provenance file is {first}"
        )
    return errors
