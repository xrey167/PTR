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
removed; an aggregate never carries mutation counts of other code.

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
)


class ProvenanceError(Exception):
    """The run records cannot be published as one result of the current code."""


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


def tree_pathspecs(experiment_dir: Path, results_dir: Path, root: Path, paths: tuple[str, ...]) -> list[str]:
    """The pathspecs a run must find committed: the provenance `paths` and the
    whole experiment directory except its results, where records accumulate."""
    return [
        *paths,
        relative_to_root(experiment_dir, root),
        f":(exclude){relative_to_root(results_dir, root)}",
    ]


def without_status(manifest: dict) -> dict:
    """A manifest apart from its `status`, which completing an experiment changes."""
    return {key: value for key, value in manifest.items() if key != "status"}


def toml_time(value):
    """A TOML date, time or datetime as its ISO 8601 text (`json.dumps`'s
    `default`): JSON has no such type. A manifest holding one is refused
    (`temporal_keys`); this only keeps such a value from failing a write or
    a comparison that reports it."""
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def temporal_keys(value, where: str = "") -> list[str]:
    """The key paths in `value`, a manifest as TOML reads it, that hold a
    TOML date, time or datetime, in any table or array. A run record holds
    the manifest as JSON, which has no such type, so a date and the string
    of its text would be one manifest to the record: the runner, `validate`
    and aggregation refuse a manifest that holds one."""
    if isinstance(value, (datetime.date, datetime.time)):
        return [where]
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in temporal_keys(item, f"{where}.{key}" if where else str(key))]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in temporal_keys(item, f"{where}[{index}]")]
    return []


def recorded_text(value) -> str:
    """`value` as the JSON text a run record holds it as, keys sorted, so a
    manifest compares with the one a record names as the record holds it,
    NaN as itself. A date or time, which no manifest may hold
    (`temporal_keys`), is written as its text (`toml_time`)."""
    return json.dumps(value, sort_keys=True, default=toml_time)


def agreement_errors(experiment_id: str, manifest: dict, records: dict[str, dict]) -> list[str]:
    """Why `records` (file name -> run record) cannot be seeds of one run of
    `experiment_id`, whose manifest is now `manifest`; empty when they can.
    This compares the records alone; `source_revision` also compares code.
    A manifest holding a TOML date or time is refused (`temporal_keys`)."""
    errors = [
        f"experiment.toml holds a TOML date or time at {key}, which a run record, holding the manifest as JSON, "
        "cannot tell from a string; write it as a string"
        for key in temporal_keys(manifest)
    ]
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
    root: Path, *args: str, stdin: str | bytes | None = None, env: dict[str, str] | None = None, binary: bool = False
) -> subprocess.CompletedProcess:
    """Run git in `root`, given `stdin` and in `env` when named; raises
    `ProvenanceError` when git cannot start. Without `env`, git runs in this
    process's environment less every `GIT_*` variable, which could point it
    at another work tree, index or object store or change how it reads
    pathspecs, and with replacement objects off, so what it reports is the
    repository at `root` as its history holds it. With `binary`, what git
    prints is kept as the bytes it wrote, as a file's content must be, and
    `stdin` is bytes too."""
    if env is None:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", *args], cwd=root, input=stdin, env=env, capture_output=True, text=not binary, check=False
        )
    except OSError as error:
        raise ProvenanceError(f"cannot run git: {error}") from error


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
    files = git(root, "diff", "--name-only", "--no-renames", *revisions, "--", *paths)
    entries = git(root, "diff", "--raw", "-z", "--no-renames", *revisions)
    for diff in (files, entries):
        if diff.returncode != 0:
            against = "the checkout" if head is None else head
            raise ProvenanceError(f"cannot compare {base} with {against}: {diff.stderr.strip()}")
    links = links_within_reach(link_paths(entries.stdout.split("\0")), root, paths)
    return sorted({*files.stdout.splitlines(), *links})


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
    lists as one path no file pathspec matches. Raises `ProvenanceError` when
    git cannot list the tree."""
    found = {
        entry[3:]
        for entry in listed_names(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=no", "--no-renames", "--", *pathspecs
        )
    }
    found.update(untracked_files(root, pathspecs))
    found.update(hiding_rules(root, pathspecs))
    tracked = []
    for entry in listed_names(root, "ls-files", "-z", "--stage", "-v", "--", *pathspecs):
        fields, _, name = entry.partition("\t")
        tag, mode = fields.split(" ")[:2]
        tracked.append(name)
        if tag.islower() or tag.upper() == "S" or mode == SYMLINK_MODE:
            found.add(name)
    found.update(unseen_trees(root, pathspecs, provenance_directories(tracked)))
    return sorted(found)


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
    holding only those rules and an empty file at each name. Raises
    `ProvenanceError` when git cannot read HEAD's rules or decide."""
    rules = head_rules(root)
    # The scratch repository is git's own, whatever repository the caller's
    # environment points git at.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    with tempfile.TemporaryDirectory() as scratch:
        mirror = Path(scratch)
        created = git(mirror, "init", "-q", env=env)
        if created.returncode != 0:
            raise ProvenanceError(
                f"cannot create a repository to read HEAD's ignore rules in: {created.stderr.strip()}"
            )
        for name in [*rules, *names]:
            path = mirror / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in rules:
                path.write_bytes(rules[name])
            elif not path.exists():
                path.touch()
        listing = git(mirror, "ls-files", "-z", "--others", "--ignored", PER_DIRECTORY, env=env)
        if listing.returncode != 0:
            raise ProvenanceError(f"cannot apply HEAD's ignore rules: {listing.stderr.strip()}")
        return (set(listing.stdout.split("\0")) & set(names)) - set(rules)


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
    """The paths git does not look into that lie within reach of the build
    (`within_reach`, judged by `directories`): an untracked or tracked
    symlink to anything but a file (a directory, or nothing yet), an
    untracked nested repository and a submodule. Git lists each as one path,
    which no file pathspec matches, and holds none of what is read through
    it. Raises `ProvenanceError` when git cannot list the tree."""
    found = []
    for entry in listed_names(root, "ls-files", "-z", "--others", PER_DIRECTORY):
        # Without --directory git lists every untracked file, and only a
        # repository it will not enter as a directory of its own.
        name = entry.rstrip("/")
        if entry.endswith("/") or links_to_no_file(root / name):
            found.append(name)
    for entry in listed_names(root, "ls-files", "-z", "--stage"):
        fields, _, name = entry.partition("\t")
        mode = fields.split(" ")[0]
        if mode == GITLINK_MODE or (mode == SYMLINK_MODE and links_to_no_file(root / name)):
            found.append(name)
    return [name for name in found if within_reach(name, pathspecs, directories)]


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
    """The stamp of `path`, or None when there is no such file."""
    try:
        status = path.lstat()
    except FileNotFoundError:
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
        than inside `rewriting`), even when its content was put back.

        It does not see a file created and removed again between its two
        looks, a write that keeps a file's inode and size and lands within
        the filesystem's timestamp resolution of the stamp before it, what
        `uncommitted_files` cannot see, or anything outside the pathspecs.
        Raises `ProvenanceError` when git cannot tell."""
        problems = []
        head = head_commit(self.root)
        if head != self.head:
            problems.append(f"HEAD moved from {self.head} to {head}")
        now = file_stamps(self.root, self.pathspecs)
        written = self.foreign | {
            name for name in self.stamps.keys() | now.keys() if self.stamps.get(name) != now.get(name)
        }
        if written:
            problems.append(f"{listed(sorted(written))} changed on disk")
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
    history = git(
        root, "log", "-z", "--raw", "--no-renames", "-m", "--format=", "--ancestry-path", f"{base}..{head}"
    )
    if history.returncode != 0:
        raise ProvenanceError(f"cannot list the commits after {base}: {history.stderr.strip()}")
    links = links_within_reach(link_paths(history.stdout.split("\0")), root, paths)
    literal = [f":(literal){name}" for name in sorted(set(links))]
    listed_commits = git(
        root, "rev-list", "--reverse", "--topo-order", "--ancestry-path", f"{base}..{head}",
        "--", *paths, *literal,
    )
    if listed_commits.returncode != 0:
        raise ProvenanceError(f"cannot list the commits after {base}: {listed_commits.stderr.strip()}")
    return listed_commits.stdout.split()


def listed(paths: list[str], limit: int = 5) -> str:
    more = f" and {len(paths) - limit} more" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def source_revision(
    experiment_id: str, manifest: dict, records: dict[str, dict], root: Path, experiment_dir: Path
) -> str:
    """The commit the results in `records` were produced at: the earliest
    record's `git_sha`, once every record agrees (`agreement_errors`),
    every record's commit and the checkout in `root` have the same
    provenance files (`seed_record_paths` of `experiment_dir`), and HEAD
    holds each of them as the checkout does (`checkout_problem`). Raises
    `ProvenanceError` naming what disagrees."""
    if not records:
        raise ProvenanceError("no run records")
    errors = agreement_errors(experiment_id, manifest, records)
    if errors:
        raise ProvenanceError("; ".join(errors))
    paths = seed_record_paths(experiment_dir, root)
    ordered = sorted(records.items(), key=lambda item: (str(item[1].get("started_at", "")), item[0]))
    revision = ordered[0][1]["git_sha"]
    for name, record in ordered[1:]:
        if record["git_sha"] != revision:
            changed = code_changes(revision, record["git_sha"], root, paths)
            if changed:
                errors.append(f"{name} ran at {record['git_sha']}, whose code differs in {listed(changed)}")
    if errors:
        raise ProvenanceError(f"the records ran different code from {revision}: " + "; ".join(errors))
    changed = code_changes(revision, None, root, paths)
    if changed:
        raise ProvenanceError(
            f"the records ran at {revision}, and the checkout's code has changed since, "
            f"in {listed(changed)}; rerun the seeds or aggregate at that commit"
        )
    unheld = checkout_problem(root, paths)
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
    """The mutation check summary (`killed`, `total`, `git_sha`) of `path`,
    or None when there is no such file. The evidence must be of
    `experiment_id`, have mutated `subcommand`, count its own outcomes, and
    have run at a commit whose provenance files (`mutation_record_paths`) are
    the checkout's, which HEAD holds as the checkout does (`checkout_problem`);
    `source_revision` has already shown the checkout's code is the aggregated
    records'. Raises `ProvenanceError` otherwise."""
    if not path.exists():
        return None
    name = path.name
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
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
    return {"killed": killed, "total": total, "git_sha": sha}


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
    preregistration names: text (UTF-8 with no NUL byte) with every CRLF line
    ending read as LF, so a checkout that converts line endings digests the
    text the repository holds, and anything else byte for byte, whose
    carriage returns are content a converting checkout leaves alone."""
    if b"\0" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            data = data.replace(b"\r\n", b"\n")
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
    (`aggregate_problems`). Raises `ProvenanceError` otherwise."""
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


def mutation_summary(path: Path) -> dict:
    """The summary of the mutation evidence at `path` that an aggregate
    carries. Raises OSError, ValueError or AttributeError when it cannot be
    read."""
    evidence = json.loads(path.read_text(encoding="utf-8"))
    return {key: evidence.get(key) for key in ("killed", "total", "git_sha")}


def aggregate_problems(run: dict, results_dir: Path) -> list[str]:
    """Why `run`, the run.json of `results_dir`, is not one aggregate with
    the metrics.json and mutations.json beside it; empty when it is. Each
    problem starts with the name of the file it is about.

    metrics.json must hash to the `metrics_sha256` run.json names, when it
    names one (`publish_aggregate`). mutations.json must be there exactly
    when run.json carries `mutation_checks`, and have that summary: the
    mutation checker rewrites it on its own, after or while the aggregate is
    written."""
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
    (`unbound_pair_problems`)."""
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
    committed = {}
    for entry in listed_names(root, "ls-tree", "-z", "--full-name", commit, "--", run_path, metrics_path):
        fields, _, name = entry.partition("\t")
        committed[name] = fields.split(" ")[2]
    present = [name for name in (run_path, metrics_path) if (root / name).is_file()]
    hashed = git(root, "hash-object", "--", *present)
    current = dict(zip(present, hashed.stdout.split())) if hashed.returncode == 0 else {}
    if current.get(run_path) != committed.get(run_path):
        return [
            f"{RUN} names no metrics_sha256 and is not the run.json committed at {commit}, so nothing "
            f"binds it to the {METRICS} beside it; restore it or rerun its aggregate.py"
        ]
    if metrics_path in present and current.get(metrics_path) != committed.get(metrics_path):
        return [
            f"{METRICS} is not the metrics.json committed with {RUN}, which names no metrics_sha256, "
            f"at {commit}; restore it or rerun its aggregate.py"
        ]
    return []


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


def staleness_errors(experiment_id: str, experiment_dir: Path, results_dir: Path, root: Path) -> list[str]:
    """Why the archived results of `experiment_id` misdescribe HEAD's code;
    empty when they do not. `results/run.json` and `results/mutations.json`
    are stale when HEAD's provenance files (`seed_record_paths`,
    `mutation_record_paths`) differ from those at their `git_sha`. Stale
    results pass only with a `results/STALE.toml` marker that names them, the
    first commit after them that changed a provenance file (`stale_since`)
    and a `reason` (`marker_errors`); a marker next to current results is an
    error too, so none outlives the rerun that replaces them."""
    results = relative_to_root(results_dir, root)
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
        paths = paths_of(experiment_dir, root)
        try:
            changed = code_changes(sha, "HEAD", root, paths)
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
    return errors + marker_errors(experiment_id, f"{results}/{STALE_MARKER}", marker, stale, root)


def marker_errors(
    experiment_id: str,
    shown: str,
    marker: Path,
    stale: dict[str, tuple[str, tuple[str, ...], list[str]]],
    root: Path,
) -> list[str]:
    """Why the stale marker `marker` (shown as `shown`) does not honestly
    describe the `stale` results; empty when it does.

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
        if differing:
            others.append(f"{name} ran at {sha}, whose provenance files differ there in {listed(differing)}")
    if others:
        return [f"{experiment_id}: {shown} names results of {marked}, but " + "; ".join(others)]
    if not is_ancestor(marked, "HEAD", root):
        return [f"{experiment_id}: {shown} names results of {marked}, which is not on HEAD's history"]
    paths: list[str] = []
    for _, name_paths, _ in stale.values():
        paths.extend(path for path in name_paths if path not in paths)
    changes = changes_after(marked, "HEAD", root, tuple(paths))
    # A first change is one no other change after the results precedes. Two
    # lines of history leaving the results each have their own; the marker
    # may name either.
    if since not in changes or changes_after(marked, since, root, tuple(paths)) != [since]:
        first = changes[0] if changes else None
        errors.append(
            f"{experiment_id}: {shown} says stale since {since}, but the first commit after {marked} "
            f"that changed a provenance file is {first}"
        )
    return errors
