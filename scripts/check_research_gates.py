"""Research execution gates, run by CI.

A completed experiment must have its required artifacts, and its archived
results must describe HEAD's code: `results/run.json` and
`results/mutations.json` fail the gate once a provenance file (the Rust, SQL,
protobuf, Cargo and toolchain files, the recording scripts, the experiment's
`aggregate.py` and mutation plan) differs from the one at their `git_sha`,
and for a listed experiment, whose command may run or read any file of the
repository, once any file does but the tools' outputs and the stale marker
(`experiment_records.listed_staleness_paths`), its manifest and the registry
counting only where they differ in more than the status completing it moves
(`experiment_records.completion_changes`), unless `results/STALE.toml` names those results and the first commit that
made them stale (`experiment_records.staleness_errors`). `results/run.json`
must also be one aggregate with the `metrics.json` it names the SHA-256 of
and the `mutations.json` whose summary it carries; a run.json that binds no
metrics, aggregated before run.json bound them, passes only while it is stale
and beside the metrics.json committed with it (`experiment_records.aggregate_errors`).
Experiments that need a pinned baseline may run only once it is pinned.

An experiment listed in `experiments/preregistration.toml` leaves `planned`
(status `prepared`, `running`, `completed` or `failed`) only with a frozen
preregistration (`preregistration_errors`); a superseded experiment has been
replaced by another, whose own preregistration counts, and is not launched,
but one that was frozen or ran stays bound as below, as it was frozen:

- its manifest names its `entrypoint`, since the manifest is frozen from
  the first commit past `planned` and one named later could never be named,
  and the runner can build its command: it splits as a command, and each
  placeholder in it but `<seed>` is a key of the `[preregistration]` table
  holding an integer, a string or a boolean, the value the runner fills it
  with, it takes `<seed>` when the table preregisters more than one seed,
  and where it runs Cargo, it runs a built-in command of Cargo's with no
  configuration, directory or input from outside the repository on its
  command line, nor does the repository's Cargo configuration name a
  program, source or flags for it (`command_errors`), since a freeze no run
  can use cannot be repaired;
- its `config.toml` holds a `[preregistration]` table with every key the list
  requires, each a pinned value of its declared type (`int`, `str`, `bool`,
  a non-empty `int-list` or `str-list`, or `file`), and no other key of the
  table holds a placeholder either;
- every value of that table has a canonical text
  (`experiment_records.preregistration_canonical`), and the manifest's
  `preregistration_sha256` is its digest;
- a `file` key names a file in the repository, and the table's
  `<key>_sha256` is that file's digest
  (`experiment_records.preregistered_file_digest`), so the file's content is
  frozen with the table, not only its path;
- the table's `seeds`, which every entry of the list must require as an
  `int-list`, names the manifest's seeds, each once, so no seed is added
  after an outcome is seen and none is counted twice;
- every baseline the list names is pinned at each key path the list names
  for it, its status is pinned and not `blocked-*`, and the table's
  `baseline_<name>_sha256` is the digest of every file in its directory
  (`directory_digest`), so every setting of the baseline, not only the keys
  that must be pinned, and the implementation beside its configuration are
  frozen with the table;
- the manifest's `preregistration_rules_sha256` is the digest of the
  experiment's entry in the list, so its types and baselines are frozen too;
- every run record (`run-*.json`, written by `scripts/run_experiment.py`
  with the manifest it ran under) and aggregate (`run.json`) the experiment
  has committed, in any directory the registry has given it and whatever
  its results directory is now, and those in its results directory, name
  those digests and the commit they ran at, and that commit is on HEAD's
  history, where the registry placed the experiment in the directory it
  places it in now, and holds the same preregistration, entry, files and
  baselines,
  and the same manifest and `config.toml` but for the manifest's status,
  which has only moved forward since (`history_errors`), and no commit after
  a freeze holds it at an earlier status, whether or not a later commit puts
  it right again (`status_regressions`); a run record also
  names the SHA-256 of the manifest at that commit and has held the same
  content in every commit on every side of every merge since it was
  committed, an aggregate, which may be written again, holds to this in
  every version it was committed in, every one once committed is at HEAD
  where it was committed (one deleted and put back as it was, as a revert of
  its revert puts it back, is the record committed: while it is missing the
  gate fails and the runner launches nothing), no two run records are of
  one seed but for a command that failed to launch, and each is a regular
  file reached through no symlink. A preregistration rewritten after its
  runs fails even when the manifest and the records are rewritten to match,
  short of rewriting history;
- every commit that held the experiment frozen as the runner launches it
  (`launchable_at`) holds the same as a run's commit must: a run can be made
  there and its record discarded before it is committed, so the first
  committed freeze binds, whether or not a record of it was kept;
- no file the table freezes and no baseline's directory lies in the results
  directory, or holds it, and the results directory is a directory below the
  experiment's: the runner holds the experiment's files to HEAD except
  those, where runs write.

The list keeps every experiment it has named at a commit on HEAD's history
that the registry holds now or has held at any commit (`enrolled`), so
taking one out of it opens neither the gate nor the runner, and a listed
experiment that was committed frozen, or whose run records were committed,
cannot go back to `planned`; it can only be superseded, which keeps its
records and its preregistration as they were, and nothing after it goes back
to an earlier status on any line of history (`status_regressions`).

`scripts/run_experiment.py` refuses to run or prepare a listed experiment
until this holds for it (`launch_errors`), so no outcome is seen before the
preregistration is frozen, and holds the files that decision reads to HEAD
while it runs (`launch_inputs`). A file or baseline the list names, and every
file of a baseline's directory, must be a regular file inside the repository,
reached through no symlink (`repository_file`), and a path the list or the
table names must be one git reads as written: not starting with `:`, which
git reads as pathspec magic, and holding none of the glob characters `*`, `?`
and `[` (`is_repository_path`).

A value is pinned unless it is a placeholder: a `must-be-pinned-…` string for
a value still to be chosen, a `must-be-signed-…` string for an owner decision
still to be taken, an empty string, `unconfigured` or `none`. A placeholder is
a string, so it passes as no other type: an owner decision whose value is a
boolean cannot pass while it waits for the owner. The list itself must name
only registered experiments, known types and well-formed baselines, whatever
their status.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import math
import re
import shlex
import struct
import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
# The gate writes no bytecode cache into the tree: a `__pycache__` directory
# is one git ignores, and a listed experiment runs only from a checkout that
# holds none.
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_records  # noqa: E402

PREREGISTRATION="experiments/preregistration.toml"
REGISTRY="experiments/registry.toml"
FROZEN={"prepared","running","completed","failed"}
# How far a listed experiment has come: after a run its status only moves
# forward, and completed and failed are final short of being superseded.
RANK={"prepared":0,"running":1,"completed":2,"failed":2}
# The manifest keys a run does not bind to the manifest it ran under: the
# status, which moves on, and the digests, which are checked on their own.
UNBOUND={"status","preregistration_sha256","preregistration_rules_sha256"}
SCALARS={"int":int,"str":str,"bool":bool}
KINDS=set(SCALARS)|{f"{kind}-list" for kind in ("int","str")}|{"file"}
ENTRY_FIELDS={"required","baseline"}
BASELINE_FIELDS={"name","path","keys","status_key"}
BASELINE_NAME=re.compile(r"[a-z0-9_]+")
MISSING=object()
# The modes of a regular file in a git tree, executable or not.
REGULAR_MODES={"100644","100755"}
# The longest path a checkout holds (Linux's PATH_MAX).
MAX_PATH=4096
# A commit's full name, which names what it holds for good.
FULL_COMMIT=re.compile(r"[0-9a-f]{40}")

class Unreadable(Exception):
    """A file the gate reads that is not a readable TOML document."""

    def __init__(self, path: Path, error: Exception):
        super().__init__(f"{path}: {error}")
        self.path=path
        self.error=error

    def named(self, root: Path) -> str:
        """The error line naming the file from `root`."""
        try:
            shown=self.path.relative_to(root).as_posix()
        except ValueError:
            shown=str(self.path)
        return f"{shown} cannot be read as TOML: {self.error}"

class HistoryUnreadable(Exception):
    """Git could not read what HEAD's history holds, so the gate cannot tell
    what was frozen or recorded there: it refuses rather than read the
    history as empty."""

    def named(self, root: Path) -> str:
        """The error line, as `Unreadable.named` gives one."""
        return f"HEAD's history cannot be read in {root}: {self}"

def load(path: Path) -> dict:
    """The TOML file at `path`. Raises `Unreadable` when it cannot be read
    or does not parse, so the gate names it rather than crash."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeDecodeError,tomllib.TOMLDecodeError) as error:
        raise Unreadable(path,error) from error

def status_of(manifest: dict) -> str | None:
    """The manifest's status when it is a string, else None."""
    status=manifest.get("status")
    return status if isinstance(status,str) else None

def may_follow(then: str | None, now: str | None) -> bool:
    """Whether a manifest may hold the status `now` after it held `then`
    on the same line of history: a status moves from prepared to running to
    completed or failed (`RANK`), and from any of them to superseded, and
    never back; completed and failed are final, and superseded is final too.
    A status that is no string, or none of these (`planned`, say), follows
    nothing, and nothing follows it, superseded included."""
    if then=="superseded":
        return now=="superseded"
    if now=="superseded":
        return then in RANK
    return then in RANK and now in RANK and RANK[now]>=RANK[then] and (RANK[then]<2 or now==then)

def same_value(then, now) -> bool:
    """Whether two TOML values are the same: of one type, tables and lists
    the same key by key and element by element, NaN the same as NaN, which
    `==` denies, and floats of one sign. `1`, `1.0` and `true` are three
    values, which `==` would take as one, and so are `0.0` and `-0.0`: TOML
    holds the sign of a zero and of a NaN, and a run can read it. So are
    one instant at two offsets."""
    if type(then) is not type(now):
        return False
    if isinstance(then,dict):
        return then.keys()==now.keys() and all(same_value(then[key],now[key]) for key in then)
    if isinstance(then,list):
        return len(then)==len(now) and all(same_value(first,second) for first,second in zip(then,now))
    if isinstance(then,float):
        return (then==now or (then!=then and now!=now)) and math.copysign(1.0,then)==math.copysign(1.0,now)
    if isinstance(then,(datetime.datetime,datetime.time)):
        # `==` takes one instant at two offsets as one, but TOML holds the
        # offset and a run can read it. tomllib keeps no more than
        # microseconds, so a change below them reads as none.
        return then==now and then.utcoffset()==now.utcoffset()
    return then==now

def names_an_entrypoint(manifest: dict) -> bool:
    """Whether the manifest names the command its runs run."""
    entrypoint=manifest.get("entrypoint")
    return isinstance(entrypoint,str) and bool(entrypoint.strip())

def command_errors(exp_id: str, manifest: dict, table: dict, read=None, names=None) -> list[str]:
    """What keeps the runner from building the command of `exp_id`, a listed
    experiment, from its manifest `manifest` and its `[preregistration]`
    table `table`: it splits the manifest's `entrypoint` as a shell would
    (without running one), its first token names a program, no token holds
    a NUL character, which no process can be given, and it fills each
    placeholder in it but `<seed>` with the table's value of that name, an
    integer, a string holding no NUL character or a boolean
    (`run_experiment.command_parameters`). And where the table preregisters
    more than one seed, the command takes `<seed>`: one that does not runs
    the same command for every seed, while each record names the seed it
    was given. And a command that runs Cargo runs one of its built-in
    commands with arguments that give it no configuration, directory,
    target directory or input outside the repository
    (`cargo_command_errors`): a `--config` file or value can name a rustc
    wrapper, flags or sources outside what the commit holds. Nor does the
    repository's own Cargo configuration, a file of the commit, name any of
    them, or a program Cargo runs (`cargo_configuration_errors`): the record
    names the compiler rustup or the `PATH` resolves, not one Cargo is told
    to run instead, or a runner started in place of the built program. Nor
    does a `Cargo.toml` among the repository files `names` lists (a function
    returning their names, `read` giving each one's text) name a path outside
    the repository for Cargo to build from (`cargo_manifest_errors`). A commit
    whose command the runner cannot build could not launch and froze
    nothing (`launchable_at`); the tree check refuses it first."""
    if not names_an_entrypoint(manifest):
        return []
    try:
        tokens=shlex.split(manifest["entrypoint"])
    except ValueError as error:
        return [f"{exp_id}: entrypoint cannot be split into a command: {error}"]
    errors=[]
    if not tokens or not tokens[0].strip():
        errors.append(f"{exp_id}: entrypoint names no program: its first token is empty")
    if any("\0" in token for token in tokens):
        errors.append(f"{exp_id}: entrypoint holds a NUL character, which no command can be given")
    for name in sorted({name for token in tokens for name in experiment_records.PLACEHOLDER.findall(token)}-{"seed"}):
        if name not in table:
            errors.append(f"{exp_id}: entrypoint placeholder <{name}> is no key of the [preregistration] table, so the "
                          "runner has no frozen value for it")
        elif not isinstance(table[name],(bool,int,str)):
            errors.append(f"{exp_id}: entrypoint placeholder <{name}> is preregistered as a {type(table[name]).__name__}, "
                          "which no command token takes")
        elif isinstance(table[name],str) and "\0" in table[name]:
            errors.append(f"{exp_id}: entrypoint placeholder <{name}> is preregistered holding a NUL character, which no "
                          "command can be given")
    seeds=table.get("seeds")
    distinct=[]
    for seed in seeds if isinstance(seeds,list) else ():
        if not any(same_value(seed,other) for other in distinct):
            distinct.append(seed)
    if len(distinct)>1 and not any("seed" in experiment_records.PLACEHOLDER.findall(token) for token in tokens):
        errors.append(f"{exp_id}: entrypoint takes no <seed> placeholder, so each of its {len(distinct)} preregistered "
                      "seeds would run one command while its record named another seed")
    # Read as the runner fills it: a placeholder could name Cargo, or give
    # it --config, as well as the token it stands in.
    def filled(match):
        value=table.get(match.group(1)) if match.group(1)!="seed" else 0
        return ("true" if value else "false") if isinstance(value,bool) else (
            str(value) if isinstance(value,(int,str)) else match.group(0))
    command=[experiment_records.PLACEHOLDER.sub(filled,token) for token in tokens]
    invocation=cargo_invocation(command)
    if invocation is not None:
        errors.extend(cargo_command_errors(exp_id,invocation))
    # Cargo started from the root, by the command or by a program it runs,
    # reads the repository's configuration there.
    if read is not None:
        errors.extend(cargo_configuration_errors(exp_id,read))
        if names is not None:
            errors.extend(cargo_manifest_errors(exp_id,names(),read))
    return errors

# The Cargo commands a listed experiment's command may run: built into Cargo,
# so no alias the repository's configuration defines stands in for one
# (Cargo lets none shadow a built-in command), and none runs a program of
# another name, as an external command runs `cargo-<name>` from the PATH.
CARGO_COMMANDS=("bench","build","check","run","test")
# Cargo's options naming a path it reads the build from, and the one naming
# where it builds, and the short options that take a value, which a cluster
# of short options (`-vC dir`) ends with.
CARGO_PATH_OPTIONS=("--manifest-path","--lockfile-path","--target","--target-dir")
CARGO_SHORT_VALUES="CFjpZ"

def cargo_command_errors(exp_id: str, invocation: list[str]) -> list[str]:
    """Why the arguments a listed command gives Cargo (`invocation`,
    `cargo_invocation`) could have it build or run what no record binds;
    empty when they cannot. The first, after at most a rustup
    `+<toolchain>`, is one of `CARGO_COMMANDS`, so what follows are that
    command's arguments as written: an alias, an option before the command
    (`--`, `-C`, `-Zscript`) or an external command would have Cargo read
    them otherwise than as written, or run a program no record names. Up to
    a `--`, past which they are the built program's, they give Cargo no
    configuration (`--config`), which can name a rustc wrapper, flags or
    sources outside the commit, no directory to run in (`-C`, in any
    spelling a cluster of short options gives it), no target directory
    (`--target-dir`), which could hold a build made outside the commit, and
    no manifest, lockfile or target specification file outside what the
    repository's watch reads (`cargo_path_options`, `outside_repository`).
    Cargo takes a long option only by its full name."""
    rest=invocation[1:] if invocation and invocation[0].startswith("+") else invocation
    if not rest or rest[0] not in CARGO_COMMANDS:
        given=repr(rest[0]) if rest else "nothing"
        return [f"{exp_id}: entrypoint gives Cargo {given} where one of its built-in commands "
                f"{', '.join(CARGO_COMMANDS)} comes first (after at most a +toolchain): an alias, an option before "
                "the command or an external command could have Cargo build or run what no record binds"]
    arguments=rest[1:]
    arguments=arguments[:arguments.index("--")] if "--" in arguments else arguments
    errors=[]
    if any(argument=="--config" or argument.startswith("--config=") for argument in arguments):
        errors.append(f"{exp_id}: entrypoint gives Cargo configuration on its command line (--config), which can name "
                      "a rustc wrapper, flags or sources outside the commit; set what the build needs in the "
                      "repository's .cargo/config.toml")
    for index,argument in enumerate(arguments):
        following=arguments[index+1] if index+1<len(arguments) else ""
        if argument.startswith("-") and not argument.startswith("--"):
            short=cargo_short_value(argument,following)
            if short is not None and short[0]=="C":
                errors.append(f"{exp_id}: entrypoint gives Cargo a directory to run in (-C), whose Cargo "
                              "configuration no check reads; a listed experiment's Cargo runs from the root")
    for option,value in cargo_path_options(arguments):
        if option=="--target-dir":
            errors.append(f"{exp_id}: entrypoint gives Cargo a target directory (--target-dir), which could hold a "
                          "build made outside the commit; the runner builds a listed run into a fresh one")
        elif outside_repository(value):
            errors.append(f"{exp_id}: entrypoint gives Cargo {option} {value}, outside what the repository's watch "
                          "reads, whose sources no watch or record binds; name a path the repository holds")
    return errors

def cargo_path_options(arguments: list[str]) -> list[tuple[str,str]]:
    """The options among the arguments of a Cargo command
    (`cargo_command_errors`) that name a path it builds from or into, with
    their values: the manifest (`--manifest-path`), the lockfile
    (`--lockfile-path`), the target, which may be a specification file
    (`--target custom.json`), and the target directory (`--target-dir`),
    each given as one token (`--manifest-path=x`) or two; a value missing
    at the end reads as empty."""
    found=[]
    for index,argument in enumerate(arguments):
        following=arguments[index+1] if index+1<len(arguments) else ""
        for option in CARGO_PATH_OPTIONS:
            if argument==option:
                found.append((option,following))
            elif argument.startswith(f"{option}="):
                found.append((option,argument[len(option)+1:]))
    return found

def cargo_short_value(argument: str, following: str) -> tuple[str,str,bool] | None:
    """The short option taking a value that the cluster `argument` (`-Cx`,
    `-vC x`, `-C=x`) ends with, its value, joined or the `following`
    argument, and whether it was joined; None for a cluster that takes
    none."""
    for position,letter in enumerate(argument[1:],start=1):
        if letter in CARGO_SHORT_VALUES:
            joined=argument[position+1:]
            return (letter,joined,True) if joined else (letter,following,False)
    return None

def outside_repository(path: str) -> bool:
    """Whether `path`, as a command run from the repository's root reads it,
    may name something outside what the repository's watch reads: absolute
    on any platform (`/x`, `C:/x`, `\\\\host\\x`), climbing above the
    root (`../x`, `a/../../x`), or through the directory git keeps for
    itself and never lists (`.git/x/Cargo.toml`), however a platform names
    it (`is_git_administration`: `.GIT`, and on Windows `.git.`, `git~1`
    and `.git::$INDEX_ALLOCATION`), or through a component Windows reads as
    a step it is not written as (`is_windows_dot_name`: `.. `, `...`), or
    through a component with a colon (`targets/base:evil.json`), which
    Windows reads as an NTFS stream of the name before it, a file no scan of
    the tree lists and no watch binds (a colon is a name character
    elsewhere, but one entrypoint runs on every platform), either slash a
    separator. The runner starts no shell, so `~` is a name like any
    other."""
    normalized=path.replace("\\","/")
    if normalized.startswith("/") or names_a_drive(normalized):
        return True
    depth=0
    for part in normalized.split("/"):
        if is_git_administration(part) or is_windows_dot_name(part) or ":" in part:
            return True
        if part=="..":
            depth-=1
            if depth<0:
                return True
        elif part not in ("","."):
            depth+=1
    return False

def cargo_invocation(tokens: list[str]) -> list[str] | None:
    """The arguments the command `tokens` gives Cargo: those after `cargo`,
    as a rustup proxy or not, and after `rustup run <toolchain> cargo`, past
    rustup's own options and `+<toolchain>` and the options of `run`; None
    for another program. Each program is named as a platform runs it
    (`experiment_records.program_name`: `C:\\Rust\\rustup.exe` is
    rustup)."""
    program=experiment_records.program_name(tokens[0]) if tokens else ""
    rest=tokens[1:]
    if program=="rustup":
        while rest and rest[0].startswith(("-","+")):
            rest=rest[1:]
        if not rest or rest[0]!="run":
            return None
        rest=rest[1:]
        while rest and rest[0].startswith("-"):
            rest=rest[1:]
        rest=rest[1:]
        while rest and rest[0].startswith("-"):
            rest=rest[1:]
        program=experiment_records.program_name(rest[0]) if rest else ""
        rest=rest[1:]
    return rest if program=="cargo" else None

# The tables of a Cargo manifest whose entries can name a dependency by path.
CARGO_DEPENDENCY_TABLES=("dependencies","dev-dependencies","build-dependencies","dev_dependencies","build_dependencies")
# The array tables of a manifest that name a target by the source file it is
# built from.
CARGO_TARGET_TABLES=("bin","example","test","bench")

def cargo_manifest_paths(manifest: dict) -> list[str]:
    """Every path a Cargo manifest names for Cargo to read or build from,
    relative to its directory: a dependency's `path` (in `[dependencies]`,
    `[dev-dependencies]`, `[build-dependencies]`, under `[target.<cfg>]` and
    in `[workspace.dependencies]`), a `[patch]` or `[replace]` entry's, the
    workspace's `members`, `default-members` and `exclude`, the package's
    `workspace` root and `build` script, and the source file of `[lib]` and
    of each `[[bin]]`, `[[example]]`, `[[test]]` and `[[bench]]`. What is not
    a string or a table of the shape Cargo reads is skipped: Cargo refuses
    it itself."""
    found=[]

    def specs(table):
        for spec in table.values() if isinstance(table,dict) else ():
            if isinstance(spec,dict) and isinstance(spec.get("path"),str):
                found.append(spec["path"])

    def dependencies(table):
        for name in CARGO_DEPENDENCY_TABLES:
            specs(table.get(name))

    dependencies(manifest)
    platforms=manifest.get("target")
    for platform in platforms.values() if isinstance(platforms,dict) else ():
        if isinstance(platform,dict):
            dependencies(platform)
    workspace=manifest.get("workspace")
    if isinstance(workspace,dict):
        specs(workspace.get("dependencies"))
        for key in ("members","default-members","exclude"):
            listed=workspace.get(key)
            if isinstance(listed,list):
                found.extend(member for member in listed if isinstance(member,str))
    patches=manifest.get("patch")
    for source in patches.values() if isinstance(patches,dict) else ():
        specs(source)
    specs(manifest.get("replace"))
    package=manifest.get("package")
    if isinstance(package,dict):
        found.extend(value for value in (package.get("workspace"),package.get("build")) if isinstance(value,str))
    library=manifest.get("lib")
    if isinstance(library,dict) and isinstance(library.get("path"),str):
        found.append(library["path"])
    for kind in CARGO_TARGET_TABLES:
        targets=manifest.get(kind)
        for target in targets if isinstance(targets,list) else ():
            if isinstance(target,dict) and isinstance(target.get("path"),str):
                found.append(target["path"])
    return found

def resolves_outside(directory: str, path: str) -> bool:
    """Whether `path`, as a manifest in the repository `directory` (`.` for
    the root) names it, leaves what the repository's watch reads
    (`outside_repository`): absolute on any platform, or climbing above the
    root from `directory`, which a path like `../ptr-core` from `crates/x`
    does not."""
    normalized=path.replace("\\","/")
    if normalized.startswith("/") or names_a_drive(normalized):
        return True
    return outside_repository(normalized if directory=="." else f"{directory}/{normalized}")

def cargo_manifest_errors(exp_id: str, names, read) -> list[str]:
    """Why a `Cargo.toml` among the repository files `names` names a path
    Cargo builds from outside the repository (`cargo_manifest_paths`,
    `resolves_outside`): `external = { path = "../../external" }` is a
    source Cargo follows and that no watch or record binds, so it could
    change between seeds while every record named one commit. Each must
    parse as TOML, or the paths it names cannot be read. `read` returns a
    repository file's text, or None where there is none."""
    errors=[]
    for name in sorted(names):
        directory=PurePosixPath(name).parent.as_posix()
        if PurePosixPath(name).name!="Cargo.toml":
            continue
        text=read(name)
        if text is None:
            continue
        try:
            manifest=tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            errors.append(f"{exp_id}: {name} does not parse as TOML ({error}), so the paths it names cannot be read")
            continue
        for named in cargo_manifest_paths(manifest):
            if resolves_outside(directory,named):
                errors.append(f"{exp_id}: {name} names {named}, outside what the repository's watch reads, whose "
                              "sources no watch or record binds; name a path the repository holds")
    return errors

# The names Cargo reads its configuration from in a `.cargo` directory, the
# older first, which Cargo prefers where both are.
CARGO_CONFIGURATION_NAMES=("config","config.toml")

# The tables a listed run's Cargo configuration may set. None names a
# program Cargo runs (a compiler or its wrapper, rustdoc, a linker, a
# runner the built program is started through), a source or file it builds
# from (a path override, a patch, a source replacement, an included
# configuration, a target specification), flags it builds with, or
# variables it sets for what it runs (`[env]`: `LD_PRELOAD`, say, outside
# the environment the runner pins), which could lie outside what the record
# binds. An alias stands in for no built-in command (`CARGO_COMMANDS`).
CARGO_CONFIGURATION_TABLES=("alias","cargo-new","future-incompat-report","http","net","term")

def cargo_configuration_errors(exp_id: str, read) -> list[str]:
    """Why the repository's Cargo configuration a listed run's Cargo reads is
    not one it may build under: Cargo started from the root, by the command
    or by a program it runs, reads `.cargo/config` and `.cargo/config.toml`
    there (and those above the root, which the runner refuses), and each
    must parse and set nothing but `CARGO_CONFIGURATION_TABLES`. `read`
    returns a repository file's text, or None where there is none."""
    errors=[]
    for name in CARGO_CONFIGURATION_NAMES:
        path=f".cargo/{name}"
        text=read(path)
        if text is None:
            continue
        try:
            table=tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            errors.append(f"{exp_id}: {path} does not parse as TOML ({error}), so what it sets for Cargo cannot be "
                          "checked")
            continue
        others=sorted(set(table)-set(CARGO_CONFIGURATION_TABLES))
        if others:
            errors.append(f"{exp_id}: {path} sets {', '.join(others)} for Cargo, which can name a program Cargo "
                          "runs (a compiler, a wrapper, a linker or a runner), a source it builds from or flags it "
                          "builds with outside what a record binds; a listed experiment's Cargo configuration sets "
                          "only "+", ".join(f"[{name}]" for name in CARGO_CONFIGURATION_TABLES))
    return errors

def repeated_seeds(seeds) -> list:
    """The seeds a preregistered seed list names more than once, each once,
    in the order they first repeat; none for anything but a list. A listed
    experiment runs each seed once, so a seed named twice would be one run
    an aggregator counts twice."""
    seen,repeated=[],[]
    for seed in seeds if isinstance(seeds,list) else ():
        if any(same_value(seed,other) for other in seen):
            if not any(same_value(seed,other) for other in repeated):
                repeated.append(seed)
        else:
            seen.append(seed)
    return repeated

def is_placeholder(value: str) -> bool:
    """A value still to be chosen: empty, `must-be-pinned-…`, `unconfigured`
    or `none`, in any case and with any surrounding space."""
    value=value.strip().lower()
    return not value or "must-be-pinned" in value or value in {"unconfigured","none"}

def is_unsigned(value: str) -> bool:
    """An owner decision the owner has not taken yet."""
    return "must-be-signed" in value.strip().lower()

def is_unset(value) -> bool:
    """A string that is a placeholder or an owner decision not yet taken."""
    return isinstance(value,str) and (is_placeholder(value) or is_unsigned(value))

def type_name(value) -> str:
    """`value`'s TOML type, as an error names it."""
    if isinstance(value,bool):
        return "a boolean"
    if isinstance(value,int):
        return "an integer"
    if isinstance(value,str):
        return "a string"
    if isinstance(value,list):
        return "a list"
    if isinstance(value,dict):
        return "a table"
    return f"a {type(value).__name__}"

def article(kind: str) -> str:
    """The indefinite article before the type `kind` in an error."""
    return "an" if kind.startswith("int") else "a"

def kind_problem(value, kind: str) -> str | None:
    """Why `value` is not a pinned value of the declared `kind`, or None."""
    if is_unset(value):
        return f"is a placeholder ({value!r})"
    if kind.endswith("-list"):
        if not isinstance(value,list):
            return f"must be {article(kind)} {kind}, not {type_name(value)}"
        if not value:
            return "is an empty list"
        for index,element in enumerate(value):
            problem=kind_problem(element,kind.removesuffix("-list"))
            if problem:
                return f"has element {index} that {problem}"
        return None
    if type(value) is not SCALARS["str" if kind=="file" else kind]:
        return f"must be {article(kind)} {kind}, not {type_name(value)}"
    return None

def pinned_problem(value) -> str | None:
    """Why a baseline's `value` is not pinned, or None."""
    if is_unset(value):
        return f"is a placeholder ({value!r})"
    if isinstance(value,dict):
        return "is a table, not a value"
    if isinstance(value,list):
        if not value:
            return "is an empty list"
        for index,element in enumerate(value):
            problem=pinned_problem(element)
            if problem:
                return f"has element {index} that {problem}"
    return None

def unset_problem(value) -> str | None:
    """Why a preregistered value the list does not require is not pinned, or
    None: a placeholder, or a list holding one. Such a value may be an empty
    list."""
    if is_unset(value):
        return f"is a placeholder ({value!r})"
    if isinstance(value,list):
        for index,element in enumerate(value):
            if is_unset(element):
                return f"has element {index} that is a placeholder ({element!r})"
    return None

def lookup(table: dict, key_path: str):
    """The value at the dotted `key_path` of `table`, or MISSING."""
    value=table
    for part in key_path.split("."):
        if not isinstance(value,dict) or part not in value:
            return MISSING
        value=value[part]
    return value

def is_git_administration(part: str) -> bool:
    """Whether the path component `part` names the directory git keeps for
    itself, `.git`, as git refuses it in a path a commit holds: in any case,
    and as Windows reads it, with trailing dots or spaces, by its short name
    `git~1`, or with an NTFS stream (`.git::$INDEX_ALLOCATION`)."""
    return part.split(":",1)[0].lower().rstrip(". ") in {".git","git~1"}

def names_a_drive(path: str) -> bool:
    """Whether `path` begins with a drive letter and a colon (`C:x`, `C:/x`,
    `c:out`), which Windows reads as a path on that drive, or as that
    drive's working directory, whatever follows the colon, so a path that
    does leaves the repository. A colon further on (`x/C:y`) is an NTFS
    stream of a name and is not one."""
    return re.match(r"[A-Za-z]:",path) is not None

def is_windows_dot_name(part: str) -> bool:
    """Whether the path component `part` is spelled otherwise on Windows than
    it is written: with trailing dots or spaces before any NTFS stream, which
    Windows trims (`archive.` is `archive`; `. `, `.. ` and `...` are `.`,
    `..` and nothing, steps or no name at all), or with nothing before the
    stream (`:x`). What the runner writes there is under a spelling the
    history does not hold, and a step climbs out of the repository. The exact
    `.` and `..` and the empty name are not: every platform reads them as the
    steps they are."""
    stem=part.split(":",1)[0]
    return part not in ("",".","..") and (not stem or stem!=stem.rstrip(". "))

def is_repository_path(value) -> bool:
    """A relative path of printable ASCII that names no parent, so it stays
    inside the repository as written, and that git takes as the path it is
    wherever a pathspec names it: no leading `:`, which git reads as
    pathspec magic, and none of the glob characters `*`, `?` and `[`. It is
    written as git writes a path, with no `.` step and no doubled or
    trailing `/`, so the tree and a commit name one entry by it (git lists a
    directory's files for `dir/`), and no longer than `MAX_PATH` characters,
    as no checkout holds a longer path and git could not be handed one as
    an argument. No part of it is git's own directory
    (`is_git_administration`): git refuses to add a path through it and
    reports no file there as tracked or untracked, so a file frozen there
    would be one no commit made by `git add` holds. Nor is a part Windows
    spells otherwise (`is_windows_dot_name`: `.. `, `...`, `archive.`), which
    would climb out of the repository or name another entry.
    Nor does it begin with a drive letter and a colon (`C:x`, `C:/x`), which
    Windows reads as a path on that drive, outside the repository."""
    if not isinstance(value,str) or not value or len(value)>MAX_PATH or not experiment_records.is_printable_ascii(value):
        return False
    path=PurePosixPath(value)
    return (not path.is_absolute() and ".." not in path.parts and "\\" not in value and path.as_posix()==value
            and not value.startswith(":") and not names_a_drive(value)
            and not any(character in value for character in "*?[")
            and not any(is_git_administration(part) or is_windows_dot_name(part) for part in path.parts))

def repository_file(root: Path, relative: str) -> Path | None:
    """The regular file `relative` names under `root`, or None when it names
    none or reaches one through a symlink, in its own name or a directory's.
    A link out of the repository holds content no other checkout has, and git
    holds any link as its target's path, so the commit a run names could not
    show the content frozen through it."""
    if not is_repository_path(relative):
        return None
    path=root/relative
    reached=[root/PurePosixPath(*PurePosixPath(relative).parts[:depth]) for depth in range(1,len(PurePosixPath(relative).parts)+1)]
    if any(step.is_symlink() for step in reached) or not path.is_file():
        return None
    return path

def baseline_directory(path: str) -> str:
    """The directory of the baseline whose configuration file is `path`, as a
    repository path: its configuration and the implementation beside it."""
    return PurePosixPath(path).parent.as_posix()

def file_table_digest(files: dict[str, str]) -> str | None:
    """The canonical digest of `files`, each file's path within a directory
    mapped to its digest, or None when a path has no canonical text."""
    try:
        return experiment_records.canonical_digest(files)
    except ValueError:
        return None

def directory_digest(root: Path, directory: str) -> tuple[list[str], str | None]:
    """What keeps the repository `directory` under `root` from being frozen,
    and otherwise its digest: the canonical digest (`file_table_digest`) of
    every file in it that git tracks or would track (one the tree's
    `.gitignore` files do not ignore), each mapped from its path within the
    directory to its git mode and its
    `experiment_records.preregistered_file_digest` (`file_entry`), as git
    holds a file's mode with its content and a mode can change what runs. A
    file reached through a symlink is refused, as `repository_file` refuses
    it, and so is a file the tree's `.gitignore` files ignore: an
    interpreter can run one in place of a tracked file (a `__pycache__`
    entry, a bytecode file standing in for a module), and git holds none.
    The digest reads a file's bytes as they are, line endings included, so a
    checkout that converts line endings holds another baseline than the
    repository does; the repository's `.gitattributes` pin LF."""
    try:
        names=experiment_records.listed_names(
            root,"--literal-pathspecs","ls-files","-z","--cached","--others",experiment_records.PER_DIRECTORY,"--",directory)
        ignored=experiment_records.listed_names(
            root,"--literal-pathspecs","ls-files","-z","--others","--ignored",experiment_records.PER_DIRECTORY,"--",directory)
        staged=experiment_records.listed_names(root,"--literal-pathspecs","ls-files","-z","--stage","--",directory)
    except experiment_records.NotUTF8:
        return ["holds a file whose name is not UTF-8, which no repository path is"],None
    except experiment_records.ProvenanceError as error:
        return [f"cannot be listed: {error}"],None
    modes={}
    for entry in staged:
        fields,_,name=entry.partition("\t")
        modes[name]=fields.split(" ")[0]
    problems=[f"holds {name}, which git ignores; a baseline's directory holds only what git tracks or would track"
              for name in sorted(set(ignored))]
    files={}
    for name in sorted(set(names)):
        path=root/name
        if not path.exists() and not path.is_symlink():
            # Tracked, and removed from the tree: the tree does not hold it.
            continue
        if not is_repository_path(name):
            problems.append(f"holds {name}, whose name is not a repository path: printable ASCII, written as git "
                            "writes a path, with no glob character, leading `:` or step through .git")
            continue
        if repository_file(root,name) is None:
            problems.append(f"holds {name}, which is not a regular file reached through no symlink")
            continue
        # Git's mode for a tracked file, whatever the checkout can show of
        # it; the owner's execute bit, as git would take it, for another.
        mode=modes.get(name) or ("100755" if path.stat().st_mode & 0o100 else "100644")
        files[PurePosixPath(name).relative_to(directory).as_posix()]=file_entry(mode,experiment_records.preregistered_file_digest(path))
    if problems:
        return problems,None
    digest=file_table_digest(files)
    if digest is None:
        return ["holds a file whose path is not printable ASCII"],None
    return [],digest

def file_form_problem(mode: str) -> str | None:
    """What keeps a file with git mode `mode` from being a preregistered file
    as a commit holds it, or None: it is not executable, since its digest
    holds its content alone and the executable bit changes what a command
    does with it. Its content is digested as its bytes, line endings
    included, so no form of it is refused."""
    if mode!="100644":
        return "is executable; a preregistered file is frozen by its content, so it is a file no command runs as a program"
    return None

def preregistered_file_problem(root: Path, relative: str, file: Path) -> str | None:
    """`file_form_problem` of the preregistered file `relative` names, `file`
    in the tree under `root`: its mode as git holds it (the index's for a
    tracked file, the owner's execute bit for another)."""
    try:
        staged=experiment_records.listed_names(root,"--literal-pathspecs","ls-files","-z","--stage","--",relative)
    except experiment_records.ProvenanceError as error:
        return f"cannot be listed: {error}"
    mode=staged[0].split(" ")[0] if staged else ("100755" if file.stat().st_mode & 0o100 else "100644")
    return file_form_problem(mode)

def frozen_file_at(root: Path, commit: str, relative: str) -> bytes | None:
    """The content of the file `relative` names as `commit` holds it, or None
    when it holds none there in the form of a preregistered file
    (`file_form_problem`). Raises `HistoryUnreadable` when git cannot tell."""
    held=tree_entry(root,commit,relative)
    data=blob(root,commit,relative)
    if held is None or data is None or file_form_problem(held[0]):
        return None
    return data

def file_entry(mode: str, digest: str) -> str:
    """A file of a frozen directory as the digest of its directory holds it:
    its git mode (`100644`, or `100755` for an executable file) and the
    digest of its content, as `git ls-tree` writes a mode before an object."""
    return f"{mode} {digest}"

def entry_errors(exp_id: str, entry, registered: set[str]) -> list[str]:
    """What is wrong with the list's entry for `exp_id`."""
    where=f"{PREREGISTRATION}: {exp_id}"
    if not isinstance(entry,dict):
        return [f"{where} is not a table"]
    errors=[]
    if exp_id not in registered:
        errors.append(f"{where} is not a registered experiment")
    for field in sorted(set(entry)-ENTRY_FIELDS):
        errors.append(f"{where} has unknown field {field}")
    required=entry.get("required")
    if not isinstance(required,dict):
        errors.append(f"{where}: required must be a table of keys and their types")
    elif not required:
        errors.append(f"{where} requires no keys")
    else:
        for key,kind in required.items():
            if not experiment_records.is_printable_ascii(key):
                errors.append(f"{where}: key {key!r} is not printable ASCII")
            if not isinstance(kind,str) or kind not in KINDS:
                errors.append(f"{where}: key {key} has unknown type {kind!r}")
        if required.get("seeds")!="int-list":
            errors.append(f"{where} must require seeds as an int-list: every listed experiment preregisters the seeds it runs")
    baselines=entry.get("baseline",[])
    if not isinstance(baselines,list):
        return errors+[f"{where}: baseline must be an array of tables"]
    names=set()
    for index,baseline in enumerate(baselines):
        name=f"{where}: baseline {index}"
        if not isinstance(baseline,dict):
            errors.append(f"{name} is not a table")
            continue
        if isinstance(baseline.get("name"),str):
            if baseline["name"] in names:
                errors.append(f"{name} repeats the name {baseline['name']!r}")
            names.add(baseline["name"])
        for field in sorted(set(baseline)-BASELINE_FIELDS):
            errors.append(f"{name} has unknown field {field}")
        if not isinstance(baseline.get("name"),str) or not BASELINE_NAME.fullmatch(baseline["name"]):
            errors.append(f"{name} needs a name of lowercase letters, digits and underscores")
        if not is_repository_path(baseline.get("path")):
            errors.append(f"{name} needs a path inside the repository")
        elif len(PurePosixPath(baseline["path"]).parts)<2:
            # Its directory, which freezes with it, would be the repository.
            errors.append(f"{name} needs a configuration in a directory below the repository's root")
        keys=baseline.get("keys")
        if not isinstance(keys,list) or not keys or not all(isinstance(key,str) and key and experiment_records.is_printable_ascii(key) for key in keys):
            errors.append(f"{name} needs a non-empty list of key paths in printable ASCII")
        status_key=baseline.get("status_key")
        if not isinstance(status_key,str) or not status_key:
            errors.append(f"{name} needs a status_key")
    return errors

def baseline_errors(exp_id: str, baseline: dict, root: Path) -> tuple[list[str], str | None]:
    """What keeps `baseline` from being a pinned, unblocked baseline, and,
    when nothing does, the digest of every file in its directory
    (`directory_digest`): every setting of the baseline, not only the keys
    that must be pinned, and the implementation that runs it decide what the
    experiment compares against."""
    where=f"{exp_id}: baseline {baseline['path']}"
    path=repository_file(root,baseline["path"])
    if path is None:
        return [f"{where} is not a file in the repository"],None
    config=load(path)
    errors=[]
    for key in baseline["keys"]:
        value=lookup(config,key)
        problem="is missing" if value is MISSING else pinned_problem(value)
        if problem:
            errors.append(f"{where}: {key} {problem}")
    status_key=baseline["status_key"]
    status=lookup(config,status_key)
    if status is MISSING:
        errors.append(f"{where}: {status_key} is missing")
    elif not isinstance(status,str) or is_unset(status):
        errors.append(f"{where}: {status_key} is not pinned ({status!r})")
    elif status.strip().lower().startswith("blocked-"):
        errors.append(f"{where} is {status}")
    if errors:
        return errors,None
    directory=baseline_directory(baseline["path"])
    problems,digest=directory_digest(root,directory)
    if problems:
        return [f"{exp_id}: baseline {directory}/ {problem}" for problem in problems],None
    return [],digest

def frozen_value_errors(exp_id: str, table: dict, key: str, expected: str, what: str) -> list[str]:
    """What keeps the table's `key` from being `expected`, the digest of
    `what`. A placeholder there is already reported as one."""
    if key not in table:
        return [f"{exp_id}: preregistration key {key} is missing; it must be {expected}, the digest of {what}"]
    if is_unset(table[key]):
        return []
    if table[key]!=expected:
        return [f"{exp_id}: preregistration key {key} {table[key]!r} is not {expected}, the digest of {what}"]
    return []

def tree_entry(root: Path, commit: str, relative: str) -> tuple[str, str, str] | None:
    """The mode, kind and object name of the entry `relative` names in
    `commit`, or None when that commit holds no such entry. Raises
    `HistoryUnreadable` when git cannot tell. A commit named by its full
    name holds what it holds for good, so each of its entries is read once
    (`listed_entry`); a name such as HEAD is read each time."""
    if FULL_COMMIT.fullmatch(commit):
        return listed_entry(root,commit,relative)
    return listed_entry.__wrapped__(root,commit,relative)

@functools.lru_cache(maxsize=None)
def listed_entry(root: Path, commit: str, relative: str) -> tuple[str, str, str] | None:
    """`tree_entry`, as `git ls-tree` lists it."""
    try:
        listing=experiment_records.git(root,"--literal-pathspecs","ls-tree","-z",commit,"--",relative)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.strip() or f"git ls-tree exited {listing.returncode}")
    wanted=PurePosixPath(relative).as_posix()
    for entry in listing.stdout.split("\0"):
        meta,_,name=entry.partition("\t")
        fields=meta.split()
        if name==wanted and len(fields)==3:
            return fields[0],fields[1],fields[2]
    return None

@functools.lru_cache(maxsize=1024)
def object_bytes(root: Path, name: str) -> bytes:
    """The content of the blob `name` in the repository at `root`, as its
    history holds it (`experiment_records.git`); a blob's name is its
    content's, so each is read once. Raises `HistoryUnreadable` when git
    cannot read it."""
    try:
        shown=experiment_records.git(root,"cat-file","blob",name,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if shown.returncode!=0:
        raise HistoryUnreadable(shown.stderr.decode(errors="replace").strip() or f"git cat-file exited {shown.returncode}")
    return shown.stdout

def blob(root: Path, commit: str, relative: str) -> bytes | None:
    """The bytes of the regular file `relative` at `commit` in the repository
    at `root`, or None when that commit holds none there: no entry, a
    directory, a submodule, or a symlink, which git holds as its target's
    path and not as the content read through it, as `repository_file`
    refuses one in the tree."""
    entry=tree_entry(root,commit,relative)
    if entry is None or entry[1]!="blob" or entry[0] not in REGULAR_MODES:
        return None
    return object_bytes(root,entry[2])

def blob_text(root: Path, commit: str, relative: str) -> str | None:
    """The UTF-8 text of the regular file `relative` at `commit` (`blob`),
    or None when there is none or it is not UTF-8."""
    data=blob(root,commit,relative)
    try:
        return None if data is None else data.decode("utf-8")
    except UnicodeDecodeError:
        return None

def tree_text(root: Path, relative: str) -> str | None:
    """The UTF-8 text of the regular file `relative` in the tree at `root`,
    reached through no symlink, or None when there is none or it is not
    UTF-8."""
    path=root/relative
    if any((root/PurePosixPath(*PurePosixPath(relative).parts[:depth])).is_symlink()
           for depth in range(1,len(PurePosixPath(relative).parts)+1)) or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError,UnicodeDecodeError):
        return None

def commit_names(root: Path, commit: str) -> list[str]:
    """The names of the files `commit` holds, as git writes them. Raises
    `HistoryUnreadable` when git cannot tell."""
    try:
        listing=experiment_records.git(root,"--literal-pathspecs","ls-tree","-r","-z","--name-only",commit,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git ls-tree exited {listing.returncode}")
    return [name for name in listing.stdout.decode("utf-8","surrogateescape").split("\0") if name]

def commit_links(root: Path, commit: str) -> list[str]:
    """The names of the symlinks and gitlinks `commit` holds anywhere, as git
    writes them. Raises `HistoryUnreadable` when git cannot tell."""
    try:
        listing=experiment_records.git(root,"--literal-pathspecs","ls-tree","-r","-z",commit,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git ls-tree exited {listing.returncode}")
    found=[]
    for entry in listing.stdout.decode("utf-8","surrogateescape").split("\0"):
        fields,_,name=entry.partition("\t")
        if name and fields.split(" ")[0] in experiment_records.LINK_MODES:
            found.append(name)
    return found

def tree_names(root: Path) -> list[str]:
    """The names of the files in the tree at `root` that git tracks or lists
    as untracked and not ignored, as git writes them (a name that is not
    UTF-8 reads as one no repository path is, rather than failing every
    check). Raises `HistoryUnreadable` when git cannot list them."""
    try:
        listing=experiment_records.git(
            root,"ls-files","-z","--cached","--others",experiment_records.PER_DIRECTORY,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git ls-files exited {listing.returncode}")
    return [name for name in listing.stdout.decode("utf-8","surrogateescape").split("\0") if name]

def toml_at(root: Path, commit: str, relative: str) -> dict | None:
    """The TOML file `relative` at `commit`, or None when it is absent or
    does not parse."""
    if FULL_COMMIT.fullmatch(commit):
        return _toml_at_cached(root,commit,relative)
    data=blob(root,commit,relative)
    if data is None:
        return None
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError,tomllib.TOMLDecodeError):
        return None

@functools.lru_cache(maxsize=4096)
def _toml_at_cached(root: Path, commit: str, relative: str) -> dict | None:
    """Cached form for the immutable, full-name commit queries in a gate."""
    data=blob(root,commit,relative)
    if data is None:
        return None
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError,tomllib.TOMLDecodeError):
        return None

def has_head(root: Path) -> bool:
    """Whether HEAD in the repository at `root` names a commit, readable or
    not: only a branch with no commit yet has no history."""
    try:
        return experiment_records.git(root,"rev-parse","--verify","--quiet","HEAD").returncode==0
    except experiment_records.ProvenanceError:
        return False

@functools.lru_cache(maxsize=512)
def _history_cached(root: str, args: tuple[str, ...], literal: bool) -> tuple[str, ...]:
    """Run one immutable history query at most once per gate process.

    The gate asks the same path-scoped history questions from several
    independent checks.  Keeping the result as a tuple makes the cache safe
    to share between callers while still allowing ``history`` to preserve its
    historical list-returning API.
    """
    repository=Path(root)
    try:
        flags=("--literal-pathspecs",) if literal else ()
        listing=experiment_records.git(repository,*flags,"log","--full-history",*args,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        if not has_head(repository):
            return ()
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git log exited {listing.returncode}")
    # A name that is not UTF-8 keeps its bytes (and so names its file), and
    # fails every check that asks for a repository path, rather than
    # failing every later gate on the commit that once held it.
    names=listing.stdout.decode("utf-8","surrogateescape")
    return tuple(name for name in names.replace("\0","\n").split("\n") if name)

def history(root: Path, *args: str, literal: bool = True) -> list[str]:
    """The NUL- or newline-separated names `git log --full-history *args`
    prints in `root`, side branches merged into HEAD included; empty where
    HEAD has no commit yet. Every pathspec is a literal path unless `literal`
    is false, when each carries the magic it names (`:(literal)`, `:(glob)`).
    Raises `HistoryUnreadable` when git cannot read it otherwise: a history
    read as empty would hide every freeze and record in it."""
    return list(_history_cached(str(root), tuple(args), literal))

def versions(root: Path, relative: str, start: str = "HEAD") -> list[tuple[str, dict]]:
    """Every commit on the history of `start` (HEAD unless named) that
    changes the TOML file `relative`, newest first, with the file as that
    commit holds it; a commit where it is absent or does not parse is left
    out."""
    found=[]
    for commit in history(root,"--format=%H",start,"--",relative):
        held=toml_at(root,commit,relative)
        if isinstance(held,dict):
            found.append((commit,held))
    return found

def ever_registered(root: Path, start: str = "HEAD") -> set[str]:
    """Every experiment the registry has held at a commit on the history of
    `start` (HEAD unless named)."""
    found=set()
    for _,registry in versions(root,REGISTRY,start):
        items=registry.get("experiment")
        for item in items if isinstance(items,list) else ():
            if isinstance(item,dict) and isinstance(item.get("id"),str):
                found.add(item["id"])
    return found

def enrolled(root: Path, registered: set[str], start: str = "HEAD") -> dict[str, str]:
    """Every experiment the list has named at a commit on the history of
    `start` (HEAD unless named), with the newest such commit, of those the
    registry holds now (`registered`) or has held at any commit on that
    history, whether before, with or after the list named it. A name the
    registry has never held, such as a mistyped one, enrolled nothing."""
    known=registered|ever_registered(root,start)
    named={}
    for commit,listed in versions(root,PREREGISTRATION,start):
        entries=listed.get("experiment")
        for exp_id in entries if isinstance(entries,dict) else ():
            if exp_id in known:
                named.setdefault(exp_id,commit)
    return named

def delisted_error(exp_id: str, commit: str) -> str:
    """The error for `exp_id`, which the list named at `commit` and no longer
    names."""
    return (f"{PREREGISTRATION}: {exp_id} was listed at {commit[:12]} and no longer is; an experiment stays "
            "listed once it is, so neither its runs nor its preregistration leave the gate")

def registered_directories(registry, exp_id: str) -> list[str]:
    """The directories, sorted, in which the parsed `registry` (None or any
    other value where it is absent or unreadable) places `exp_id`, each as
    the repository path the runner reads: below `experiments`, without `.`
    steps or a trailing slash. A directory that is no repository path
    (`is_repository_path`) is named too; whoever asks decides what it means."""
    items=registry.get("experiment") if isinstance(registry,dict) else None
    return sorted({
        PurePosixPath("experiments",item["path"]).as_posix() for item in (items if isinstance(items,list) else ())
        if isinstance(item,dict) and item.get("id")==exp_id and isinstance(item.get("path"),str)
    })

def registered_statuses(registry, exp_id: str) -> list:
    """The statuses the parsed `registry` (None or any other value where it is
    absent or unreadable) gives `exp_id`, one for each entry naming it, in
    the registry's order."""
    items=registry.get("experiment") if isinstance(registry,dict) else None
    return [item.get("status") for item in (items if isinstance(items,list) else ())
            if isinstance(item,dict) and item.get("id")==exp_id]

def status_disagreement(exp_id: str, registry, manifest_status) -> str | None:
    """Why the registry's status for `exp_id` is not the manifest's, or None
    when it is: a harness reads the registry as it reads the manifest, so a
    run needs the two to say the same."""
    held=registered_statuses(registry,exp_id)
    if held==[manifest_status]:
        return None
    shown=", ".join(repr(status) for status in held) or "none"
    return (f"the registry holds status {shown} for it and experiment.toml {manifest_status!r}; a listed "
            "experiment holds the same status in both")

def placed_directories(root: Path, commit: str, exp_id: str) -> list[str]:
    """`registered_directories` of the registry as `commit` holds it."""
    return registered_directories(toml_at(root,commit,REGISTRY),exp_id)

def experiment_directories(root: Path, exp_id: str, current: str) -> list[str]:
    """The directories, as repository paths, that the registry has given
    `exp_id` at a commit on HEAD's history, and `current`, its directory now."""
    found={current}
    for _,registry in versions(root,REGISTRY):
        found.update(directory for directory in registered_directories(registry,exp_id) if is_repository_path(directory))
    return sorted(found)

def is_run_record(relative: str) -> bool:
    """Whether the repository path `relative` names a run record, which
    `scripts/run_experiment.py` writes as `run-*.json`."""
    name=PurePosixPath(relative).name
    return name.startswith("run-") and name.endswith(".json")

def is_aggregate(relative: str) -> bool:
    """Whether the repository path `relative` names an aggregate of runs,
    which an aggregator writes as `run.json`."""
    return PurePosixPath(relative).name=="run.json"

def results_directories(root: Path, directories: list[str]) -> set[str]:
    """Every results directory the experiment has had, as repository paths:
    in each of `directories`, `results` and every `results_dir` a version of
    its `experiment.toml` on HEAD's history names that is a directory below
    it. The runner writes a record only into the results directory of the
    manifest it runs, which a commit holds."""
    found=set()
    for directory in directories:
        found.add(f"{directory}/results")
        for _,manifest in versions(root,f"{directory}/experiment.toml"):
            named=manifest.get("results_dir")
            if is_repository_path(named) and PurePosixPath(named).parts:
                found.add(PurePosixPath(directory,named).as_posix())
    return found

def committed_records(root: Path, directories: list[str]) -> list[str]:
    """Every run record and aggregate committed on HEAD's history in a
    results directory the experiment has had in `directories`
    (`results_directories`), as repository paths, whether the tree still
    holds it or not, sorted; a file of that name elsewhere, such as a test's
    fixture, is none. Git reports a rename as the removal of the old path,
    so both are named."""
    results=results_directories(root,directories)
    # -m lists what a merge changes against each of its parents, so a record
    # a merge alone added or removed is named too.
    names=history(root,"-m","--no-renames","--name-only","-z","--format=","HEAD","--",*directories)
    return sorted({
        name for name in names
        if (is_run_record(name) or is_aggregate(name)) and PurePosixPath(name).parent.as_posix() in results
    })

def committed_blobs(root: Path, relative: str) -> set[tuple[str, str]]:
    """The distinct entries, as a mode and a blob name, that the repository
    path `relative` has had at the commits of HEAD's full history that change
    it, on every side of every merge. A symlink holds its target's text as
    its blob, so the link and the regular file with that text are two
    entries, told apart by mode."""
    found=set()
    for commit in history(root,"--format=%H","HEAD","--",relative):
        held=tree_entry(root,commit,relative)
        if held is not None:
            found.add((held[0],held[2]))
    return found

def launchable_at(root: Path, commit: str, exp_id: str, directory: str) -> tuple[str, str] | None:
    """The status at which `commit` holds the experiment in `directory`
    frozen as the runner launches it (`launch_errors`), with the digest of
    everything frozen there (the manifest, the configuration, the entry,
    the files' and baselines' digests and the directory), or None when it
    does not: the list names it with a well-formed entry, its manifest is past
    `planned` and holds no value a run record could not tell from another
    (`experiment_records.manifest_problems`), its `[preregistration]` table
    holds every required key,
    pinned and of its type, and no placeholder, the table's seeds are the
    manifest's, each named once, the manifest names the digests of the table and of the
    entry, every file and baseline the table freezes has the frozen
    content there, each baseline pinned and not blocked, the runner can
    build its command from the manifest and the table (`command_errors`),
    and the registry places the experiment in `directory` and holds the
    status the manifest does, and no symlink or gitlink lies anywhere in the
    commit, which the runner's watch of the whole repository refuses. A
    commit that held less could not launch it, and does not freeze it."""
    if FULL_COMMIT.fullmatch(commit):
        return _launchable_at_cached(root,commit,exp_id,directory)
    return _launchable_at_uncached(root,commit,exp_id,directory)

@functools.lru_cache(maxsize=4096)
def _launchable_at_cached(root: Path, commit: str, exp_id: str, directory: str) -> tuple[str, str] | None:
    return _launchable_at_uncached(root,commit,exp_id,directory)

def _launchable_at_uncached(root: Path, commit: str, exp_id: str, directory: str) -> tuple[str, str] | None:
    manifest=toml_at(root,commit,f"{directory}/experiment.toml")
    if not isinstance(manifest,dict) or status_of(manifest) not in FROZEN or not names_an_entrypoint(manifest):
        return None
    # The runner refuses a manifest a run record could not tell from another
    # (a TOML date or time, a NaN) before it launches anything.
    if experiment_records.manifest_problems(manifest):
        return None
    if placed_directories(root,commit,exp_id)!=[directory] or not is_repository_path(directory):
        return None
    # The runner watches the whole repository of a listed experiment and
    # refuses one that holds a symlink or a gitlink anywhere, which git holds
    # as a path and not as content: a commit that holds one could not launch.
    if commit_links(root,commit):
        return None
    if status_disagreement(exp_id,toml_at(root,commit,REGISTRY),status_of(manifest)) is not None:
        return None
    listed=toml_at(root,commit,PREREGISTRATION)
    entries=listed.get("experiment") if isinstance(listed,dict) else None
    entry=entries.get(exp_id) if isinstance(entries,dict) else None
    if entry_errors(exp_id,entry,{exp_id}):
        return None
    results_dir=manifest.get("results_dir","results")
    if not is_repository_path(results_dir) or not PurePosixPath(results_dir).parts:
        return None
    results=PurePosixPath(directory,results_dir)
    for depth in range(1,len(PurePosixPath(results_dir).parts)+1):
        # The runner writes no record through a file or a link on the way.
        held=tree_entry(root,commit,PurePosixPath(directory,*PurePosixPath(results_dir).parts[:depth]).as_posix())
        if held is not None and held[1]!="tree":
            return None
    config=toml_at(root,commit,f"{directory}/config.toml")
    table=config.get("preregistration") if isinstance(config,dict) else None
    if not isinstance(table,dict):
        return None
    # The runner builds the command from the manifest and the table as the
    # tree check reads them, and a run under a runner changed to let another
    # through was never confirmatory.
    if command_errors(exp_id,manifest,table,lambda name:blob_text(root,commit,name),
                      lambda:commit_names(root,commit)):
        return None
    for key,kind in entry["required"].items():
        if key not in table or kind_problem(table[key],kind):
            return None
        if kind=="file" and PurePosixPath(table[key]).is_relative_to(results):
            return None
        if kind=="file":
            data=frozen_file_at(root,commit,table[key]) if is_repository_path(table[key]) else None
            if data is None or experiment_records.preregistered_bytes_digest(data)!=table.get(f"{key}_sha256"):
                return None
    # The file and baseline digests are the table's, checked here and
    # below, so the table stands for them in the state.
    if (any(unset_problem(value) for value in table.values())
            or not same_value(table.get("seeds",MISSING),manifest.get("seeds",MISSING))
            or repeated_seeds(table.get("seeds"))):
        return None
    for baseline in entry.get("baseline",[]):
        held=PurePosixPath(baseline_directory(baseline["path"]))
        if held.is_relative_to(results) or results.is_relative_to(held):
            return None
        settings=toml_at(root,commit,baseline["path"])
        if not isinstance(settings,dict):
            return None
        status=lookup(settings,baseline["status_key"])
        if not isinstance(status,str) or is_unset(status) or status.strip().lower().startswith("blocked-"):
            return None
        for key in baseline["keys"]:
            value=lookup(settings,key)
            if value is MISSING or pinned_problem(value):
                return None
        # A directory the tree check refuses has no digest, and one with
        # none froze nothing, even where the table names none either.
        held=directory_digest_at(root,commit,baseline_directory(baseline["path"]))
        if held is None or held!=table.get(f"baseline_{baseline['name']}_sha256"):
            return None
    try:
        if (manifest.get("preregistration_sha256")!=experiment_records.preregistration_digest(table)
                or manifest.get("preregistration_rules_sha256")!=experiment_records.canonical_digest(entry)):
            return None
    except (ValueError,UnicodeEncodeError):
        return None
    state=json.dumps(typed({"directory":directory,"manifest":manifest,"config":config,"entry":entry}),sort_keys=True)
    return manifest["status"],hashlib.sha256(state.encode("utf-8")).hexdigest()

def typed(value):
    """`value` with every scalar tagged by its type and written exactly: a
    float by its bits, so the sign of a zero or a NaN is kept, a date or time
    by its ISO 8601 text with its offset. Values whose typed forms are equal
    are `same_value`, so states told apart by it are told apart here too."""
    if isinstance(value,dict):
        return {key:typed(item) for key,item in value.items()}
    if isinstance(value,list):
        return [typed(item) for item in value]
    if isinstance(value,float):
        return ["float",struct.pack(">d",value).hex()]
    if isinstance(value,(datetime.date,datetime.time)):
        return [type(value).__name__,value.isoformat()]
    return [type(value).__name__,value]

# The Cargo metadata `command_errors` reads at a commit (`cargo_configuration_errors`
# at the root, `cargo_manifest_errors` in every directory): a commit that changes
# one can be the first the runner could launch an experiment at, whatever else it
# holds.
CARGO_METADATA_PATHS=tuple(f".cargo/{name}" for name in CARGO_CONFIGURATION_NAMES)
CARGO_METADATA_GLOBS=("**/Cargo.toml",)

def launch_paths(root: Path, exp_id: str, directories: list[str]) -> set[str]:
    """Every repository path whose content can decide whether `exp_id` may
    launch at a commit on HEAD's history: its `directories`, the list, the
    registry, the directory of every baseline any version of its entry has
    named, every file any version of its table has named under a key of type
    `file`, and the Cargo configuration at the root (`CARGO_METADATA_PATHS`).
    So can any `Cargo.toml` (`CARGO_METADATA_GLOBS`), which is no path known
    beforehand and is named to git as a glob (`launch_listing`)."""
    paths=set(directories)|{PREREGISTRATION,REGISTRY,*CARGO_METADATA_PATHS}
    file_keys=set()
    for _,listed in versions(root,PREREGISTRATION):
        entries=listed.get("experiment")
        entry=entries.get(exp_id) if isinstance(entries,dict) else None
        if not isinstance(entry,dict):
            continue
        baselines=entry.get("baseline")
        for baseline in baselines if isinstance(baselines,list) else ():
            if isinstance(baseline,dict) and is_repository_path(baseline.get("path")):
                directory=baseline_directory(baseline["path"])
                if directory!=".":
                    paths.add(directory)
        required=entry.get("required")
        if isinstance(required,dict):
            file_keys.update(key for key,kind in required.items() if kind=="file")
    for directory in directories:
        for _,config in versions(root,f"{directory}/config.toml"):
            table=config.get("preregistration")
            for key in file_keys if isinstance(table,dict) else ():
                if is_repository_path(table.get(key)):
                    paths.add(table[key])
    return paths

@functools.lru_cache(maxsize=16)
def link_changes_at(root: str, head: str) -> frozenset[str]:
    """The commits on the history of `head` in the repository at `root` that
    add, remove or change a symlink or a gitlink anywhere, against any one of
    their parents. Raises `HistoryUnreadable` when git cannot tell."""
    try:
        listing=experiment_records.git(
            Path(root),"--literal-pathspecs","log","--full-history","-m","--raw","--no-abbrev","--no-renames",
            "--format=commit:%H",head,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git log exited {listing.returncode}")
    found=set()
    current=None
    for line in listing.stdout.decode("utf-8","surrogateescape").split("\n"):
        if line.startswith("commit:"):
            current=line[len("commit:"):]
        elif line.startswith(":") and current is not None:
            modes=line[1:].split(" ",2)[:2]
            if any(mode in experiment_records.LINK_MODES for mode in modes):
                found.add(current)
    return frozenset(found)

def launch_listing(root: Path, exp_id: str, directories: list[str]) -> list[tuple[str, tuple[str, str] | None]]:
    """Every commit on HEAD's history that changes a path that can decide
    whether the experiment may launch (`launch_paths`: its manifest and
    configuration in any of `directories`, the list, the registry, its
    baselines and its files, and the Cargo metadata the command check reads,
    `CARGO_METADATA_GLOBS` too) or adds or removes a symlink or gitlink
    anywhere (`link_changes_at`, which decides whether a commit could launch
    it at all), newest first, with what it holds there
    (`launchable_at`, in the first of `directories` that holds it frozen) or
    None where it does not hold it frozen as the runner launches it. A merge
    is listed where it differs from any one of its parents, so the state of
    the experiment on any line of history changes only at a listed commit."""
    listing=[]
    specs=[f":(literal){path}" for path in sorted(launch_paths(root,exp_id,directories))]
    specs+=[f":(glob){pattern}" for pattern in CARGO_METADATA_GLOBS]
    selected=history(root,"--format=%H","HEAD","--",*specs,literal=False)
    if has_head(root):
        moved=link_changes_at(str(root),experiment_records.head_commit(root))-set(selected)
        if moved:
            # In the order of the whole history, which the others keep.
            keep=moved|set(selected)
            selected=[commit for commit in history(root,"--format=%H","HEAD") if commit in keep]
    for commit in selected:
        held=None
        for directory in directories:
            held=launchable_at(root,commit,exp_id,directory)
            if held is not None:
                break
        listing.append((commit,held))
    return listing

def freezes(listing: list[tuple[str, tuple[str, str] | None]]) -> list[tuple[str, str]]:
    """The freezes among the commits of `launch_listing`, each with its status
    there, newest first. From such a commit on, the runner could launch the
    experiment, whether or not a record of that run was kept; the first such
    commit may change any one of those paths alone. Commits that hold the same
    frozen state, such as those that add only records, are one freeze, named
    by the oldest of them in the listing."""
    states={}
    for commit,held in listing:
        if held is not None:
            # Newest first: an older commit of the same state takes its
            # place, and moves it behind the states seen since.
            states.pop(held,None)
            states[held]=commit
    return [(commit,status) for (status,_),commit in states.items()]

def frozen_commits(root: Path, exp_id: str, relative: str) -> list[tuple[str, str]]:
    """The freezes on HEAD's history (`freezes`) of the experiment, whose
    directory now is `relative`, in every directory the registry has given
    it (`experiment_directories`), with their status there, newest first."""
    return freezes(launch_listing(root,exp_id,experiment_directories(root,exp_id,relative)))

def statuses_at(root: Path, commit: str, exp_id: str) -> list[str | None]:
    """The status the manifest of `exp_id` holds at `commit` in each
    directory the registry places it in there: None for a status that is no
    string. A directory that is no repository path, or holds no readable
    manifest, adds none; what the commit does not hold says nothing of a
    status."""
    if FULL_COMMIT.fullmatch(commit):
        return list(_statuses_at_cached(root,commit,exp_id))
    return _statuses_at_uncached(root,commit,exp_id)

@functools.lru_cache(maxsize=4096)
def _statuses_at_cached(root: Path, commit: str, exp_id: str) -> tuple[str | None, ...]:
    return tuple(_statuses_at_uncached(root,commit,exp_id))

def _statuses_at_uncached(root: Path, commit: str, exp_id: str) -> list[str | None]:
    found=[]
    for directory in placed_directories(root,commit,exp_id):
        if not is_repository_path(directory):
            continue
        manifest=toml_at(root,commit,f"{directory}/experiment.toml")
        if isinstance(manifest,dict):
            found.append(status_of(manifest))
    return found

def descendants_of(root: Path, commit: str) -> set[str]:
    """Every commit on HEAD's history that descends from `commit`, by full
    name, `commit` itself not among them: one listing of git's for the
    commit, and none for each commit that might descend from it. Raises
    `HistoryUnreadable` when git cannot list them."""
    if FULL_COMMIT.fullmatch(commit):
        return set(_descendants_of_cached(root,commit))
    return _descendants_of_uncached(root,commit)

@functools.lru_cache(maxsize=4096)
def _descendants_of_cached(root: Path, commit: str) -> frozenset[str]:
    return frozenset(_descendants_of_uncached(root,commit))

def _descendants_of_uncached(root: Path, commit: str) -> set[str]:
    try:
        listing=experiment_records.git(root,"rev-list","--ancestry-path",f"{commit}..HEAD","--")
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.strip() or f"git rev-list exited {listing.returncode}")
    return set(listing.stdout.split())

def status_regressions(root: Path, exp_id: str, listing: list[tuple[str, tuple[str, str] | None]]) -> list[str]:
    """The commits of `listing` (`launch_listing`) that hold the manifest at a
    status it may not hold after a status it held on their own line of
    history (`may_follow`), one error for each. A freeze holds a status that
    was frozen there, and a commit that holds `superseded` with a freeze
    behind it holds it for good; whichever commit descends from one
    (`descendants_of`), on any side of any merge, may not go back, whether the
    status it goes back to was frozen or not, and whether a later commit puts
    the status back or not: a manifest that once was prepared and ran, and is
    prepared again at a commit between two freezes, is one whose runs could
    be chosen. A commit that is no descendant of the freeze, such as one on a
    side branch that never saw it, is not one that went back."""
    held={commit:statuses_at(root,commit,exp_id) for commit,_ in listing}
    anchors=[(commit,frozen[0]) for commit,frozen in listing if frozen is not None]
    below={}
    def after(anchor: str, commit: str) -> bool:
        if anchor not in below:
            below[anchor]=descendants_of(root,anchor)
        return commit in below[anchor]
    # A superseded status is final only where the experiment had been frozen.
    frozen_names=tuple(commit for commit,_ in anchors)
    for commit,frozen in listing:
        if frozen is None and "superseded" in held[commit] and any(after(name,commit) for name in frozen_names):
            anchors.append((commit,"superseded"))
    errors=[]
    for commit,_ in listing:
        broken=[
            (anchor,then,now) for anchor,then in anchors if anchor!=commit
            for now in dict.fromkeys(held[commit]) if not may_follow(then,now)
        ]
        for anchor,then,now in broken:
            if after(anchor,commit):
                errors.append(f"{exp_id} was {then!r} at {anchor[:12]} and is {now!r} at {commit[:12]}, a commit "
                              "after it; a status moves only from prepared to running to completed or failed, or to "
                              "superseded, and never back, whether or not a later commit puts it there again")
                break
    return errors

def directory_digest_at(root: Path, commit: str, directory: str) -> str | None:
    """`directory_digest` of the repository `directory` as `commit` holds it,
    or None when the tree check (`directory_digest`) would refuse what it
    holds there: anything but a regular file (a symlink, a submodule), a
    file named as no repository path (`is_repository_path`), a name that is
    not UTF-8 included. None is no digest, so it matches no frozen one.
    Raises `HistoryUnreadable` when git cannot tell."""
    if FULL_COMMIT.fullmatch(commit):
        return _directory_digest_at_cached(root,commit,directory)
    return _directory_digest_at_uncached(root,commit,directory)

@functools.lru_cache(maxsize=4096)
def _directory_digest_at_cached(root: Path, commit: str, directory: str) -> str | None:
    return _directory_digest_at_uncached(root,commit,directory)

def _directory_digest_at_uncached(root: Path, commit: str, directory: str) -> str | None:
    try:
        listing=experiment_records.git(root,"--literal-pathspecs","ls-tree","-r","-z",commit,"--",directory,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git ls-tree exited {listing.returncode}")
    files={}
    # Git writes a name's bytes as they are; one that is not UTF-8 reads as
    # a name no repository path is, rather than failing every later gate.
    for entry in listing.stdout.decode("utf-8","surrogateescape").split("\0"):
        if not entry:
            continue
        meta,_,name=entry.partition("\t")
        mode,kind,held=meta.split()
        if kind!="blob" or mode not in REGULAR_MODES or not is_repository_path(name):
            # A file the tree check refuses, by its kind or its name, makes
            # a directory no launch could have frozen.
            return None
        data=object_bytes(root,held)
        files[PurePosixPath(name).relative_to(directory).as_posix()]=file_entry(mode,experiment_records.preregistered_bytes_digest(data))
    return file_table_digest(files)

def changed_keys(then: dict, now: dict, unbound: set[str]) -> list[str]:
    """The keys other than `unbound` whose values differ between `then` and
    `now`, sorted; a key only one of them holds differs."""
    return sorted(key for key in (then.keys()|now.keys())-unbound if not same_value(then.get(key,MISSING),now.get(key,MISSING)))

def history_errors(exp_id: str, name: str, named, root: Path, experiment: Path, entry: dict, table: dict,
                   frozen: tuple[str, str], current: tuple[dict, dict], at: str | None = None) -> tuple[list[str], str | None]:
    """What keeps the commit a record names (`named`) from holding the
    preregistration frozen now and the experiment as it is now, and that
    commit. The commit is on HEAD's history; the experiment's
    `[preregistration]` table and `experiment.toml` there name the frozen
    digests (`frozen`: the table's and the rules'), the list holds the same
    entry for it, and every file and baseline the table freezes has the
    frozen content there. The manifest and the rest of `config.toml` there
    are the ones now (`current`: the manifest and the configuration), but
    for the manifest's status, which has only moved forward since (`RANK`).
    A record's own fields can be edited; the commit it ran at cannot, short
    of rewriting history. `at`, when given, names what is checked at the
    commit in each error, in place of the record that ran there."""
    digest,rules=frozen
    now_manifest,now_config=current
    where=f"{exp_id}: {name}"
    commit=experiment_records.resolve_commit(named,root) if isinstance(named,str) else None
    if commit is None:
        return [f"{where} names no commit this repository holds ({named!r})"],None
    at=f"{where} ran at {commit[:12]}" if at is None else at
    errors=[]
    if not experiment_records.is_ancestor(commit,"HEAD",root):
        errors.append(f"{at}, which is not on HEAD's history")
    relative=experiment.relative_to(root).as_posix()
    # The files read at the commit are the experiment's only where the
    # registry placed it there as it does now; another directory, even one
    # holding a twin of it, ran something else.
    placed=placed_directories(root,commit,exp_id)
    if placed!=[relative]:
        errors.append(f"{at}, where the registry placed it in {', '.join(placed) or 'no directory'}, not in {relative}")
    config=toml_at(root,commit,f"{relative}/config.toml")
    manifest=toml_at(root,commit,f"{relative}/experiment.toml")
    listed=toml_at(root,commit,PREREGISTRATION)
    then=config.get("preregistration") if isinstance(config,dict) else None
    try:
        then_digest=experiment_records.preregistration_digest(then) if isinstance(then,dict) else None
    except ValueError:
        then_digest=None
    if then_digest!=digest:
        errors.append(f"{at}, whose config.toml holds another [preregistration] than the frozen one")
    if isinstance(config,dict):
        changed=changed_keys(config,now_config,{"preregistration"})
        if changed:
            errors.append(f"{at}, whose config.toml differs from the current one in {', '.join(changed)}; "
                          "after a run the configuration stays as it ran")
    if not isinstance(manifest,dict) or manifest.get("preregistration_sha256")!=digest or manifest.get("preregistration_rules_sha256")!=rules:
        errors.append(f"{at}, whose experiment.toml names other preregistration digests than the frozen ones")
    if isinstance(manifest,dict):
        changed=changed_keys(manifest,now_manifest,UNBOUND)
        if changed:
            errors.append(f"{at}, whose experiment.toml differs from the current one in {', '.join(changed)}; "
                          "after a run only its status changes")
        then_status,now_status=status_of(manifest),status_of(now_manifest)
        problem=status_disagreement(exp_id,toml_at(root,commit,REGISTRY),then_status)
        if problem is not None:
            errors.append(f"{at}, where {problem}")
        if then_status not in RANK:
            errors.append(f"{at}, where it was {then_status!r}; a listed experiment runs only once it is prepared")
        elif not may_follow(then_status,now_status):
            errors.append(f"{at}, where it was {then_status!r}; it cannot be {now_status!r} after that, "
                          "since a status moves only from prepared to running to completed or failed, or to superseded")
    then_entry=listed.get("experiment",{}).get(exp_id) if isinstance(listed,dict) and isinstance(listed.get("experiment"),dict) else None
    try:
        then_rules=experiment_records.canonical_digest(then_entry) if isinstance(then_entry,dict) else None
    except (ValueError,UnicodeEncodeError):
        then_rules=None
    if then_rules!=rules:
        errors.append(f"{at}, whose {PREREGISTRATION} lists other rules for it than the frozen ones")
    for key,kind in entry["required"].items():
        if kind!="file" or not is_repository_path(table.get(key)):
            continue
        data=frozen_file_at(root,commit,table[key])
        held=None if data is None else experiment_records.preregistered_bytes_digest(data)
        if held!=table.get(f"{key}_sha256"):
            errors.append(f"{at}, where {table[key]} is not the file frozen as {key}")
    for baseline in entry.get("baseline",[]):
        held=directory_digest_at(root,commit,baseline_directory(baseline["path"])) if is_repository_path(baseline["path"]) else None
        if held is None or held!=table.get(f"baseline_{baseline['name']}_sha256"):
            errors.append(f"{at}, where baseline {baseline['name']} is not the frozen one: its directory holds other files")
    return errors,commit

def committed_seed_records(root: Path, commit: str, directory: str, shown=lambda name:name) -> tuple[dict, dict]:
    """The run records `commit` holds directly in `directory`, as
    `archived_errors` reads the ones on disk: by the seed as JSON writes it,
    the names of the records that run it (a command that failed to launch saw
    no outcome and is left out), and the SHA-256 of the record of each seed
    whose run finished (`experiment_records.finished_run`); `shown` writes a
    record's path as an error names it."""
    runs={}
    digests={}
    for name in sorted(commit_names(root,commit)):
        if PurePosixPath(name).parent.as_posix()!=directory or not is_run_record(name):
            continue
        data=blob(root,commit,name)
        try:
            record=json.loads(data.decode("utf-8")) if data is not None else None
        except (UnicodeDecodeError,json.JSONDecodeError):
            continue
        if not isinstance(record,dict) or "seed" not in record or record.get("status")=="failed-to-launch":
            continue
        seed=json.dumps(record["seed"],sort_keys=True)
        runs.setdefault(seed,[]).append(shown(name))
        if experiment_records.finished_run(record):
            digests[seed]=hashlib.sha256(data).hexdigest()
    return runs,digests

def aggregate_version_errors(exp_id: str, version: str, earlier: dict, root: Path, commit: str, path: str) -> list[str]:
    """Why `earlier`, the aggregate the repository path `path` held at
    `commit` (named `version`), is not bound to the `metrics.json` and
    `mutations.json` that commit holds beside it: it names the SHA-256 of
    the metrics (`metrics_sha256`) and, where it carries mutation checks,
    that of the whole evidence (`mutation_checks.sha256`), which every
    aggregator of a listed experiment writes (`experiment_records.
    publish_aggregate`), and what it reports of that evidence (`killed`,
    `total`, `git_sha`) is what the evidence says, as `experiment_records.
    aggregate_problems` asks of the aggregate on disk. A version whose
    artifacts do not match, replaced with a valid pair later, is one whose
    outcome nothing bound."""
    directory=PurePosixPath(path).parent
    errors=[]

    def check(field: str, name: str, named, subject: str) -> None:
        held=blob(root,commit,(directory/name).as_posix())
        if not isinstance(named,str):
            errors.append(f"{exp_id}: {version} names no {field}, so nothing binds it to the {name} committed beside it")
        elif held is None:
            errors.append(f"{exp_id}: {version} names {field} {named}, but that commit holds no {name} beside it")
        elif hashlib.sha256(held).hexdigest()!=named:
            errors.append(f"{exp_id}: {version} names {field} {named}, not {hashlib.sha256(held).hexdigest()}, the "
                          f"SHA-256 of the {name} committed beside it")

    check("metrics_sha256",experiment_records.METRICS,earlier.get("metrics_sha256"),"metrics")
    carried=earlier.get("mutation_checks")
    evidence=blob(root,commit,(directory/experiment_records.MUTATIONS).as_posix())
    if isinstance(carried,dict):
        check("mutation_checks.sha256",experiment_records.MUTATIONS,carried.get("sha256"),"mutation evidence")
        # The digest binds the evidence a version names, and what it reports
        # of it is what that evidence says, as `aggregate_problems` asks of
        # the aggregate on disk: counts the evidence does not hold, replaced
        # later by the right summary, were still reported where committed.
        if evidence is not None and hashlib.sha256(evidence).hexdigest()==carried.get("sha256"):
            try:
                summary=experiment_records.mutation_summary_of(evidence)
            except (ValueError,AttributeError) as error:
                errors.append(f"{exp_id}: {version} names mutation checks over the {experiment_records.MUTATIONS} committed "
                              f"beside it, which cannot be read: {error}")
            else:
                if summary!=carried:
                    errors.append(f"{exp_id}: {version} carries mutation_checks {json.dumps(carried,sort_keys=True)}, but the "
                                  f"{experiment_records.MUTATIONS} committed beside it sums up to "
                                  f"{json.dumps(summary,sort_keys=True)}")
    elif carried is not None:
        errors.append(f"{exp_id}: {version} carries mutation_checks that is no object, so nothing binds it to the "
                      f"{experiment_records.MUTATIONS} committed beside it")
    elif evidence is not None:
        # `aggregate_problems` refuses the same pair for the aggregate on disk.
        errors.append(f"{exp_id}: {version} carries no mutation_checks, but that commit holds a "
                      f"{experiment_records.MUTATIONS} beside it")
    return errors

def recorded_command_errors(where: str, record: dict, manifest, table, commit: str) -> list[str]:
    """Why the `command` and `parameters` a run record names are not those
    the manifest and the `[preregistration]` table `table` (as `commit`
    holds them) give the seed it names, as the runner builds them
    (`run_experiment.command_parameters`, `build_command`): a record of
    another command (a program chosen from a directory that was replaced while
    the launch read it, and put back) is no run of what was frozen. Empty for
    a record naming no seed, and where the frozen command cannot be built
    (`command_errors` names that)."""
    seed=record.get("seed")
    if isinstance(seed,bool) or not isinstance(seed,int):
        return []
    if not isinstance(manifest,dict) or not isinstance(table,dict) or not names_an_entrypoint(manifest):
        return []
    try:
        tokens=shlex.split(manifest["entrypoint"])
    except ValueError:
        return []
    values={}
    for placeholder in sorted({name for token in tokens for name in experiment_records.PLACEHOLDER.findall(token)}-{"seed"}):
        value=table.get(placeholder,MISSING)
        if isinstance(value,bool):
            values[placeholder]="true" if value else "false"
        elif isinstance(value,(int,str)):
            values[placeholder]=str(value)
        else:
            return []
    filled={**values,"seed":str(seed)}
    command=[experiment_records.PLACEHOLDER.sub(lambda match:filled[match.group(1)],token) for token in tokens]
    errors=[]
    if record.get("command")!=command:
        errors.append(f"{where} names command {record.get('command')!r}, not {command!r}, the command the manifest and "
                      f"[preregistration] table at {commit[:12]} give seed {seed}")
    if record.get("parameters")!=values:
        errors.append(f"{where} names parameters {record.get('parameters')!r}, not {values!r}, the values the "
                      f"[preregistration] table at {commit[:12]} freezes for the command's placeholders")
    return errors

def record_errors(exp_id: str, name: str, record, aggregate: bool, root: Path, experiment: Path, entry: dict,
                  table: dict, frozen: tuple[str, str], current: tuple[dict, dict]) -> list[str]:
    """What keeps `record`, the parsed run record or aggregate the error
    names as `name`, from having run under the preregistration frozen now
    (`frozen`) and the experiment as it is now (`current`): it names both
    digests (a run record in the manifest it carries) and a commit that
    holds the same (`history_errors`), and a run record names the SHA-256 of
    the manifest at that commit."""
    digest,rules=frozen
    where=f"{exp_id}: {name}"
    if not isinstance(record,dict):
        return [f"{where} is not a JSON object"]
    errors=[]
    named=record if aggregate else record.get("manifest") if isinstance(record.get("manifest"),dict) else {}
    for field,expected in (("preregistration_sha256",digest),("preregistration_rules_sha256",rules)):
        if named.get(field)!=expected:
            errors.append(f"{where} names {field} {named.get(field)!r}, not {expected}, the digest the experiment is frozen at")
    problems,commit=history_errors(exp_id,name,record.get("git_sha"),root,experiment,entry,table,frozen,current)
    errors.extend(problems)
    if not aggregate and commit is not None:
        held=blob(root,commit,f"{experiment.relative_to(root).as_posix()}/experiment.toml")
        # run_experiment.py hashes the manifest as the checkout holds it, and
        # the watch holds every file it reads to HEAD's bytes, line endings
        # included, so that is the committed manifest as it is.
        named_manifest=record.get("manifest_sha256")
        if held is None or not isinstance(named_manifest,str) or named_manifest!=hashlib.sha256(held).hexdigest():
            errors.append(f"{where} names manifest_sha256 {record.get('manifest_sha256')!r}, not the SHA-256 of experiment.toml at {commit[:12]}")
        # The command it ran is the one the manifest and the table frozen at
        # that commit give its seed.
        relative=experiment.relative_to(root).as_posix()
        then=toml_at(root,commit,f"{relative}/config.toml")
        errors.extend(recorded_command_errors(
            where,record,toml_at(root,commit,f"{relative}/experiment.toml"),
            then.get("preregistration") if isinstance(then,dict) else None,commit))
    return errors

def archived_errors(exp_id: str, manifest: dict, config: dict, experiment: Path, root: Path, entry: dict, table: dict,
                    frozen: tuple[str, str]) -> list[str]:
    """What keeps the experiment's archived runs from having run under the
    preregistration frozen now (`frozen`: the digests of the table and of
    the list's rules for it), with the manifest and configuration as they
    are now (`manifest`, `config`). The records are every run record
    (`run-*.json`) and aggregate (`run.json`) committed under any directory
    the registry has given the experiment (`committed_records`), whatever
    its results directory is now, and the run records and aggregate in its
    results directory. Every one once committed must be at HEAD where it was
    committed; one deleted and put back as it was, as a revert of its revert
    puts it back, is the record committed, and while it is missing the gate
    fails and the runner, which asks it, launches nothing. Each is a regular
    file reached through no symlink. Every run record and aggregate must
    pass `record_errors`; a run record must also be unchanged since it was
    committed, and an aggregate, which an aggregator may write again, must
    pass `record_errors` in every version committed and, in each, name the
    SHA-256 of the record of each preregistered seed's finished run
    (`seed_records`, `experiment_records.finished_run`): an earlier version
    that named none saw an outcome the runs were not bound to, which a later
    one that does hides no more. No two run records are
    of one seed, bar those whose command failed to launch: a seed run again
    after its outcome was seen could keep whichever run came out best. And
    they all name one program, one Rust toolchain and one environment
    (`executable`, `toolchain`, `environment`), which lie outside the
    commit: one replaced between seeds would leave records naming one
    commit for different code, and a program the command starts by name is
    found through the environment's PATH. And the
    seeds ran one tree: the repository at each run's commit is the first
    run's, but for the seed records of the results directory
    (`experiment_records.listed_record_paths`), since the command may run
    or read any file of it, an aggregate or mutation evidence committed
    there included."""
    relative=experiment.relative_to(root).as_posix()
    current=(manifest,config)

    def shown(path: str) -> str:
        """A record's path as an error names it: within the experiment's
        directory, or from the repository's root when outside it."""
        held=PurePosixPath(path)
        return held.relative_to(relative).as_posix() if held.is_relative_to(relative) else path

    directories=experiment_directories(root,exp_id,relative)
    committed=committed_records(root,directories)
    errors=[]
    # A record, once committed, stays where it was recorded: one deleted,
    # renamed or moved after its outcome was seen could be replaced by a
    # record of another preregistration under a new name or in a new place.
    for path in committed:
        if not (root/path).exists() and not (root/path).is_symlink():
            errors.append(f"{exp_id}: {shown(path)} was committed and has since been deleted or renamed; a run record stays as it was recorded")
    records={path for path in committed if (root/path).exists() or (root/path).is_symlink()}
    results_dir=manifest.get("results_dir","results")
    if is_repository_path(results_dir) and (experiment/results_dir).is_dir():
        for path in (experiment/results_dir).glob("run*.json"):
            held=path.relative_to(root).as_posix()
            if is_aggregate(held) or is_run_record(held):
                records.add(held)
    # A commit that held the experiment completed declared it complete: what
    # it was completed with, the aggregate that binds a finished run of each
    # preregistered seed, stays after it is superseded, whatever its manifest
    # lists (a completed one is asked for by `gate_errors`).
    if status_of(manifest)!="completed" and isinstance(results_dir,str) and is_repository_path(results_dir):
        completed=[commit for directory in directories
                   for commit,held in versions(root,f"{directory}/experiment.toml") if status_of(held)=="completed"]
        if completed:
            errors.extend(artifact_errors(
                exp_id,root,experiment/results_dir,STANDARD_ARTIFACTS,
                f"experiment, completed at {completed[0][:12]} and now {status_of(manifest)!r},"))
            # What it was completed with stays what it was: the metrics and
            # the mutation evidence are the ones its aggregate names, and the
            # results still describe the code (`staleness_errors`, which a
            # superseding status moves nothing of): its command may have read
            # any file of the repository.
            errors.extend(experiment_records.aggregate_errors(exp_id,experiment,experiment/results_dir,root))
            errors.extend(experiment_records.staleness_errors(exp_id,experiment,experiment/results_dir,root,True))
    runs={}
    programs={}
    trees=[]
    digests={}
    aggregates=[]
    seeds=table.get("seeds")
    preregistered=[json.dumps(seed,sort_keys=True) for seed in seeds] if isinstance(seeds,list) else []
    for path in sorted(records):
        name=shown(path)
        where=f"{exp_id}: {name}"
        aggregate=is_aggregate(path)
        # A record read through a symlink is another file's content, whose
        # history the checks below would not see.
        if repository_file(root,path) is None:
            errors.append(f"{where} is not a regular file reached through no symlink")
            continue
        try:
            data=(root/path).read_bytes()
            record=json.loads(data.decode("utf-8"))
        except OSError as error:
            errors.append(f"{where} cannot be read: {error}")
            continue
        except (UnicodeDecodeError,json.JSONDecodeError) as error:
            errors.append(f"{where} cannot be read: {error}")
            record=MISSING
        if record is not MISSING:
            errors.extend(record_errors(exp_id,name,record,aggregate,root,experiment,entry,table,frozen,current))
            if aggregate and isinstance(record,dict):
                aggregates.append((name,record,None))
            # A prepared record names no seed, and a command that failed to
            # launch saw no outcome.
            if not aggregate and isinstance(record,dict) and "seed" in record and record.get("status")!="failed-to-launch":
                runs.setdefault(json.dumps(record["seed"],sort_keys=True),[]).append(name)
                # A reservation nothing finished (`started`, left by a runner
                # that died or was refused) is a run of its seed all the same,
                # which no other may follow, but it holds no outcome for an
                # aggregate to report on.
                if experiment_records.finished_run(record):
                    digests[json.dumps(record["seed"],sort_keys=True)]=hashlib.sha256(data).hexdigest()
                # A seed outside the frozen list could be added once an outcome
                # is seen, and count towards what is reported.
                if json.dumps(record["seed"],sort_keys=True) not in preregistered:
                    errors.append(f"{where} ran seed {json.dumps(record['seed'])}, which is not one of the "
                                  "preregistered seeds; a listed experiment runs only those it froze")
                # The program, the toolchain and the environment lie outside
                # the commit: the seeds of one experiment ran one of each.
                programs.setdefault(json.dumps(
                    [record.get("executable"),record.get("toolchain"),record.get("environment")],sort_keys=True),[]
                ).append(name)
                trees.append((str(record.get("started_at","")),name,record.get("git_sha")))
        if aggregate:
            # Written again, an aggregate keeps every version it was
            # committed in: each saw the outcome of the runs it names. It is
            # run.json with the metrics.json and mutations.json beside it, so
            # a commit that changes only one of the two is a state it was in
            # too, and a corruption restored by the next commit is seen.
            beside=PurePosixPath(path).parent
            for commit in history(root,"--format=%H","HEAD","--",path,
                                  *((beside/name).as_posix() for name in (experiment_records.METRICS,experiment_records.MUTATIONS))):
                if tree_entry(root,commit,path) is None:
                    # The commit removed it.
                    continue
                version=f"{name} as committed at {commit[:12]}"
                held=blob(root,commit,path)
                if held is None:
                    # A link, say, which a checkout reads another file through.
                    errors.append(f"{exp_id}: {version} is not a regular file")
                    continue
                if held==data:
                    # The aggregate now on disk is checked as such; what that
                    # commit holds beside it is its own, and may differ from
                    # what is beside it now.
                    if isinstance(record,dict):
                        errors.extend(aggregate_version_errors(exp_id,version,record,root,commit,path))
                        aggregates.append((version,record,(commit,PurePosixPath(path).parent.as_posix())))
                    continue
                try:
                    earlier=json.loads(held.decode("utf-8"))
                except (UnicodeDecodeError,json.JSONDecodeError) as error:
                    errors.append(f"{exp_id}: {version} cannot be read: {error}")
                    continue
                errors.extend(record_errors(exp_id,version,earlier,True,root,experiment,entry,table,frozen,current))
                # Each version saw the runs it names: it is bound to their
                # records below as the aggregate now on disk is, and to the
                # metrics and mutation evidence committed beside it.
                if isinstance(earlier,dict):
                    aggregates.append((version,earlier,(commit,PurePosixPath(path).parent.as_posix())))
                    errors.extend(aggregate_version_errors(exp_id,version,earlier,root,commit,path))
            continue
        # Every commit that holds the record, on every side of every merge,
        # holds the same content: a record rewritten on one side of a merge
        # and kept by it was changed all the same.
        blobs=committed_blobs(root,path)
        if len(blobs)>1:
            errors.append(f"{where} was changed after it was committed ({len(blobs)} versions of it were committed)")
        # Its bytes on disk against HEAD's blob, not git diff, which takes a
        # file the index marks skip-worktree or assume-unchanged as HEAD's
        # whatever it holds; a copy that differs in line endings only is
        # another file, which the repository's `.gitattributes` keeps a
        # checkout from writing.
        if blobs:
            head=blob(root,"HEAD",path)
            if head is None or data!=head:
                errors.append(f"{where} differs from the record committed as it")
    for seed,names in sorted(runs.items()):
        if len(names)>1:
            errors.append(f"{exp_id}: seed {seed} ran more than once ({', '.join(names)}); a listed experiment runs each "
                          "seed once, so no run of it is chosen by its outcome")
    # An aggregate reports on the runs, so it names the SHA-256 of the record
    # of each preregistered seed (`seed_records`, as
    # `experiment_records.seed_record_digests` writes it): outcomes committed
    # beside no run, or beside records rewritten since, would pass on the
    # digests it carries alone.
    for name,report,version_of in aggregates:
        # The aggregate now on disk names the records of the checkout; an
        # earlier version, those its own commit held beside it, which a
        # record committed after it does not stand in for.
        if version_of is None:
            seen_runs,seen_digests=runs,digests
        else:
            seen_runs,seen_digests=committed_seed_records(root,*version_of,shown)
        bound=report.get("seed_records")
        if not isinstance(bound,dict):
            errors.append(f"{exp_id}: {name} names no seed_records, the SHA-256 of the run record of each preregistered "
                          "seed, so nothing binds the outcome it reports to the runs")
            continue
        for seed in preregistered:
            if seed not in seen_runs:
                errors.append(f"{exp_id}: {name} reports on seed {seed}, which has no run record")
            elif seed not in seen_digests:
                errors.append(f"{exp_id}: {name} reports on seed {seed}, whose run record ({', '.join(seen_runs[seed])}) holds no "
                              "outcome; a run that finished has status completed or failed with its exit_code, finished_at, "
                              "stdout and stderr, and a reservation nothing finished saw none")
            elif len(seen_runs[seed])>1:
                # Two records of one seed are named as such: none is the record.
                continue
            elif bound.get(seed)!=seen_digests[seed]:
                errors.append(f"{exp_id}: {name} names seed_records[{seed}] {bound.get(seed)!r}, not {seen_digests[seed]}, "
                              f"the SHA-256 of the run record of seed {seed}")
        for seed in sorted(set(bound)-set(preregistered)):
            errors.append(f"{exp_id}: {name} names a record for seed {seed}, which is not preregistered")
    if len(programs)>1:
        errors.append(f"{exp_id}: its runs name {len(programs)} programs, toolchains or environments ("
                      + "; ".join(", ".join(names) for _,names in sorted(programs.items()))
                      + "); a program, toolchain or environment changed between seeds lies outside the commit every "
                      "record names")
    # The seeds ran one tree: the command may run or read any file of the
    # repository, and each run's commit is the tree it launched from, so
    # between the first run's commit and each other's only the seed records
    # of the results directory may differ.
    if trees and is_repository_path(results_dir):
        paths=experiment_records.listed_record_paths(f"{relative}/{results_dir}")
        _,first,base=min(trees)
        for _,name,commit in sorted(trees):
            if commit==base or not isinstance(commit,str) or not isinstance(base,str):
                continue
            try:
                changed=experiment_records.code_changes(base,commit,root,paths)
            except experiment_records.NotUTF8 as error:
                errors.append(f"{exp_id}: {name} ran at {commit[:12]}, which cannot be compared with {base[:12]}, where "
                              f"{first} ran: {error}")
                continue
            except experiment_records.ProvenanceError:
                # record_errors names a commit the history does not hold.
                continue
            if changed:
                errors.append(f"{exp_id}: {name} ran at {commit[:12]}, whose repository differs from {base[:12]}'s, "
                              f"where {first} ran, in {experiment_records.listed(changed)}; the seeds of a listed "
                              "experiment run one tree, only their seed records committed between them")
    # A commit that holds the experiment past planned froze it, whether or
    # not a record of a run there was kept: the runner could launch it, and
    # a record can be discarded before it is committed.
    listing=launch_listing(root,exp_id,directories)
    for commit,_ in freezes(listing):
        problems,_=history_errors(exp_id,"",commit,root,experiment,entry,table,frozen,current,
                                  at=f"{exp_id} was frozen at {commit[:12]}")
        errors.extend(problems)
    # And a status never goes back on any line of history, whether or not the
    # commit it goes back to was frozen, or a later commit puts it there again.
    errors.extend(status_regressions(root,exp_id,listing))
    return errors

def frozen_errors(exp_id: str, entry: dict, manifest: dict, experiment: Path, root: Path) -> list[str]:
    """What keeps a listed experiment that left `planned` from having a frozen
    preregistration."""
    relative=experiment.relative_to(root).as_posix() if experiment.is_relative_to(root) else str(experiment)
    if not is_repository_path(relative):
        # The gate searches the history of the directories the registry has
        # given it (`experiment_directories`) only where they are such paths.
        return [f"{exp_id}: {REGISTRY} places it in {relative!r}, which is not a repository path, so its runs and "
                "freezes there could not be found"]
    parts=PurePosixPath(relative).parts
    if any((root/PurePosixPath(*parts[:depth])).is_symlink() for depth in range(1,len(parts)+1)):
        # Git holds a link as its target's path, so the history of the
        # directory would hold none of the files read through it.
        return [f"{exp_id}: {REGISTRY} places it in {relative}, which is reached through a symlink, so its runs and "
                "freezes there could not be found"]
    config=experiment/"config.toml"
    if not config.is_file() and not config.is_symlink():
        return [f"{exp_id}: config.toml does not exist"]
    for name in ("experiment.toml","config.toml"):
        # Read through a link, either would be a file no commit holds as the
        # experiment's, and the runner could not launch from it.
        if repository_file(root,f"{relative}/{name}") is None:
            return [f"{exp_id}: {name} is not a regular file reached through no symlink"]
    settings=load(config)
    table=settings.get("preregistration")
    if not isinstance(table,dict):
        return [f"{exp_id}: config.toml has no [preregistration] table"]
    errors=[]
    problem=status_disagreement(exp_id,load(root/REGISTRY),status_of(manifest))
    if problem is not None:
        errors.append(f"{exp_id}: {problem}")
    if not names_an_entrypoint(manifest):
        errors.append(f"{exp_id}: experiment.toml names no entrypoint; a listed experiment names what it runs before "
                      "it leaves planned, since its manifest is frozen from then on")
    errors.extend(command_errors(exp_id,manifest,table,lambda name:tree_text(root,name),
                                 lambda:tree_names(root)))
    results_dir=manifest.get("results_dir","results")
    results=None
    if isinstance(results_dir,str) and any(is_git_administration(part) for part in PurePosixPath(results_dir).parts):
        errors.append(f"{exp_id}: results_dir {results_dir!r} passes through git's own directory, where no record can be committed")
    elif not is_repository_path(results_dir) or not PurePosixPath(results_dir).parts:
        errors.append(f"{exp_id}: results_dir {results_dir!r} is not a directory below the experiment's directory")
    else:
        results=PurePosixPath(experiment.relative_to(root).as_posix(),results_dir)

    def in_results(relative: str) -> bool:
        """Whether the repository path `relative` lies in the experiment's
        results directory, whose files the runner does not hold to HEAD: a
        run writes its records there."""
        return results is not None and PurePosixPath(relative).is_relative_to(results)

    for key,kind in entry["required"].items():
        if key not in table:
            errors.append(f"{exp_id}: preregistration key {key} is missing")
            continue
        problem=kind_problem(table[key],kind)
        if problem:
            errors.append(f"{exp_id}: preregistration key {key} {problem}")
        elif kind=="file":
            file=repository_file(root,table[key])
            if in_results(table[key]):
                errors.append(f"{exp_id}: preregistration key {key} names {table[key]!r}, which lies in the results "
                              "directory, whose files the runner does not hold to HEAD")
            elif file is None:
                errors.append(f"{exp_id}: preregistration key {key} names {table[key]!r}, which is not a file in the repository")
            else:
                form=preregistered_file_problem(root,table[key],file)
                if form:
                    errors.append(f"{exp_id}: preregistration key {key} names {table[key]!r}, which {form}")
                errors.extend(frozen_value_errors(
                    exp_id,table,f"{key}_sha256",experiment_records.preregistered_file_digest(file),table[key]))
    for key,value in table.items():
        problem=None if key in entry["required"] else unset_problem(value)
        if problem:
            errors.append(f"{exp_id}: preregistration key {key} {problem}")
    if "seeds" in table and not same_value(table["seeds"],manifest.get("seeds",MISSING)):
        errors.append(f"{exp_id}: preregistered seeds {table['seeds']!r} are not the manifest's seeds {manifest.get('seeds')!r}")
    for seed in repeated_seeds(table.get("seeds")):
        errors.append(f"{exp_id}: preregistered seeds name seed {seed!r} more than once; each seed runs once, so an "
                      "aggregate would count its one run twice")
    for baseline in entry.get("baseline",[]):
        directory=baseline_directory(baseline["path"])
        if in_results(directory) or (results is not None and results.is_relative_to(directory)):
            errors.append(f"{exp_id}: baseline {directory}/ shares files with the results directory {results}/, "
                          "whose files the runner does not hold to HEAD")
            continue
        problems,selection=baseline_errors(exp_id,baseline,root)
        errors.extend(problems)
        if selection is not None:
            errors.extend(frozen_value_errors(
                exp_id,table,f"baseline_{baseline['name']}_sha256",selection,
                f"every file in {baseline_directory(baseline['path'])}/"))
    try:
        digest=experiment_records.preregistration_digest(table)
    except ValueError as error:
        errors.append(f"{exp_id}: {error}")
        return errors
    rules=experiment_records.canonical_digest(entry)
    for field,expected,what in (
        ("preregistration_sha256",digest,"config.toml's [preregistration]"),
        ("preregistration_rules_sha256",rules,f"its entry in {PREREGISTRATION}"),
    ):
        recorded=manifest.get(field)
        if recorded is None:
            errors.append(f"{exp_id}: experiment.toml names no {field}")
        elif recorded!=expected:
            errors.append(f"{exp_id}: {field} {recorded!r} is not {expected}, the digest of {what}")
    errors.extend(archived_errors(exp_id,manifest,settings,experiment,root,entry,table,(digest,rules)))
    return errors

def preregistration_errors(root: Path, manifests: dict[str, dict], experiments: dict[str, Path]) -> list[str]:
    """What keeps the listed experiments that left `planned` from having
    frozen preregistrations, and what is wrong with the list itself: it
    still names every experiment it has named (`enrolled`), and a listed
    experiment that has left `planned` in a commit, or whose runs were
    committed, has not gone back to it."""
    if not (root/PREREGISTRATION).is_file():
        return [f"{PREREGISTRATION} does not exist"]
    try:
        listed=load(root/PREREGISTRATION)
    except Unreadable as error:
        return [error.named(root)]
    errors=[]
    entries=listed.get("experiment",{})
    if not isinstance(entries,dict):
        return [f"{PREREGISTRATION}: experiment must be a table of experiments"]
    for field in sorted(set(listed)-{"version","experiment"}):
        errors.append(f"{PREREGISTRATION} has unknown field {field}")
    for exp_id,entry in entries.items():
        problems=entry_errors(exp_id,entry,set(manifests))
        if problems:
            errors.extend(problems)
            continue
        manifest=manifests[exp_id]
        status=status_of(manifest)
        if status is None:
            errors.append(f"{exp_id}: status {manifest.get('status')!r} is not a status")
        elif status in FROZEN:
            try:
                errors.extend(frozen_errors(exp_id,entry,manifest,experiments[exp_id],root))
            except Unreadable as error:
                errors.append(error.named(root))
        else:
            relative=experiments[exp_id].relative_to(root).as_posix()
            committed=committed_records(root,experiment_directories(root,exp_id,relative))
            left=frozen_commits(root,exp_id,relative)
            if status=="superseded":
                # Superseding stops the runner, not the binding: an experiment
                # that was frozen or ran keeps its records and its
                # preregistration, so no outcome is erased by superseding it.
                if committed or left:
                    try:
                        errors.extend(frozen_errors(exp_id,entry,manifest,experiments[exp_id],root))
                    except Unreadable as error:
                        errors.append(error.named(root))
            elif committed:
                errors.append(
                    f"{exp_id} is {status!r}, but run records of it were committed ({committed[0]}); a listed "
                    "experiment that has run stays prepared, running, completed or failed, or is superseded"
                )
            elif left:
                errors.append(
                    f"{exp_id} is {status!r}, but it was {left[0][1]!r} at {left[0][0][:12]}; a listed experiment "
                    "that has left planned stays prepared, running, completed or failed, or is superseded"
                )
    for exp_id,commit in sorted(enrolled(root,set(manifests)).items()):
        if exp_id not in entries:
            errors.append(delisted_error(exp_id,commit))
    return errors

def launch_errors(root: Path, exp_id: str) -> list[str]:
    """Why `exp_id` may not be run or prepared now (`run_experiment.py`),
    empty when it may. An experiment the list names runs only once it has
    left `planned` with a frozen preregistration (`frozen_errors`), so none of
    its outcomes can be seen before its preregistration is fixed, and one
    the list named once and no longer names (`enrolled`) does not run at
    all; one the list has never named runs as before. A file it cannot read
    refuses the launch, named."""
    try:
        return launch_decision(root,exp_id)
    except (Unreadable,HistoryUnreadable) as error:
        return [error.named(root)]

def launch_decision(root: Path, exp_id: str) -> list[str]:
    """`launch_errors`, raising `Unreadable` for a file it cannot read."""
    if not (root/PREREGISTRATION).is_file():
        return [f"{PREREGISTRATION} does not exist, so whether {exp_id} preregisters is unknown"]
    entries=load(root/PREREGISTRATION).get("experiment",{})
    if not isinstance(entries,dict):
        return [f"{PREREGISTRATION}: experiment must be a table of experiments"]
    items={
        item["id"]:item for item in load(root/REGISTRY).get("experiment",[])
        if isinstance(item,dict) and isinstance(item.get("id"),str)
    }
    if exp_id not in entries:
        commit=enrolled(root,set(items)).get(exp_id)
        return [] if commit is None else [delisted_error(exp_id,commit)]
    twice=duplicate_registrations(load(root/REGISTRY),exp_id)
    if twice:
        return twice
    problems=entry_errors(exp_id,entries[exp_id],set(items))
    if problems:
        return problems
    if not isinstance(items[exp_id].get("path"),str):
        return [f"{REGISTRY}: {exp_id} names no path"]
    experiment=root/"experiments"/items[exp_id]["path"]
    manifest=load(experiment/"experiment.toml")
    if status_of(manifest) not in FROZEN:
        return [
            f"{exp_id} preregisters ({PREREGISTRATION}) and is {manifest.get('status')!r}: it runs only once its "
            "preregistration is frozen and it is prepared, running, completed or failed"
        ]
    return frozen_errors(exp_id,entries[exp_id],manifest,experiment,root)

def launch_commit_errors(root: Path, exp_id: str, commit: str) -> list[str]:
    """Why `commit`, the commit a run of `exp_id` launched from the tree
    names as what it ran, does not hold the experiment frozen as the tree
    launches it (`launchable_at`), empty when it does or when the list as
    `commit` holds it does not name `exp_id`. `scripts/run_experiment.py`
    asks this once the tree is known to hold `commit`'s files
    (`experiment_records.ProvenanceWatch`); the list and the registry are
    read from `commit`, so no write to the tree after that look can change
    the answer. The gate finds a freeze by what a commit holds
    (`frozen_commits`), so a run launched from a tree whose inputs that
    commit does not hold as regular files (a file git ignores, one below
    git's own directory, a symlink, which git holds as its target's path)
    would leave a freeze the gate cannot find, and a preregistration that
    could be rewritten once the run's record was discarded; so would any
    other input `launchable_at` finds wanting at the commit and the tree
    check does not, such as a registry path git names otherwise. A list the
    commit does not hold readable, or a history git cannot read, refuses the
    launch."""
    try:
        listed=toml_at(root,commit,PREREGISTRATION)
        if not isinstance(listed,dict) or not isinstance(listed.get("experiment",{}),dict):
            return [f"{PREREGISTRATION} cannot be read as {commit[:12]} holds it, so whether {exp_id} preregisters is unknown"]
        if exp_id not in listed.get("experiment",{}):
            return []
        placed=placed_directories(root,commit,exp_id)
        if len(placed)==1 and launchable_at(root,commit,exp_id,placed[0]) is not None:
            return []
    except HistoryUnreadable as error:
        return [error.named(root)]
    return [
        f"{exp_id}: {commit[:12]}, the commit its run would name, does not hold it frozen as the tree launches it; "
        "every file that decides its launch must be a regular file that commit holds, not a file git ignores or "
        "keeps in its own directory, nor a symlink"
    ]

def launch_mode_errors(root: Path, exp_id: str, commit: str, listed: bool) -> list[str]:
    """Why a run of `exp_id` launched as a listed experiment's run (`listed`)
    or as an unlisted one's may not name `commit`, the commit the tree was
    found to hold (`experiment_records.ProvenanceWatch`); empty when it may.
    `scripts/run_experiment.py` reads whether the list names `exp_id` from
    the tree before that look, so a commit made in between could have named
    or dropped it; and an unlisted run holds no lock, reserves no seed and
    takes no frozen value, so its outcome could be seen and its record
    discarded. So the list as `commit` holds it names `exp_id` exactly when
    the run is listed, and an unlisted run's experiment is one the list has
    named at no commit on `commit`'s history (`enrolled`). Read from
    `commit`, no write to the tree after that look changes the answer. A
    list the commit does not hold readable, or a history git cannot read,
    refuses the launch."""
    try:
        held=toml_at(root,commit,PREREGISTRATION)
        entries=held.get("experiment",{}) if isinstance(held,dict) else None
        if not isinstance(entries,dict):
            return [f"{PREREGISTRATION} cannot be read as {commit[:12]} holds it, so whether {exp_id} preregisters is unknown"]
        if (exp_id in entries)!=listed:
            return [f"{PREREGISTRATION} changed while the launch of {exp_id} was checked: {commit[:12]}, the commit its "
                    f"run would name, {'does not name' if listed else 'names'} it; rerun from a tree that holds HEAD"]
        if listed:
            return []
        registry=toml_at(root,commit,REGISTRY)
        items=registry.get("experiment") if isinstance(registry,dict) else None
        registered={
            item["id"] for item in (items if isinstance(items,list) else ())
            if isinstance(item,dict) and isinstance(item.get("id"),str)
        }
        named=enrolled(root,registered,commit).get(exp_id)
    except HistoryUnreadable as error:
        return [error.named(root)]
    return [] if named is None else [delisted_error(exp_id,named)]

def launch_inputs(root: Path, exp_id: str) -> list[str]:
    """The files besides the experiment's own that decide whether `exp_id`
    may launch (`launch_errors`), as repository paths: the list, the
    registry, this gate, the directories of the baselines a listed
    experiment's entry names (their configuration and implementation), and
    the files its table preregisters. `scripts/run_experiment.py` holds them
    to HEAD before and while it runs, as it holds the experiment's own files,
    so an edit made to launch cannot be put back after the outcome is seen."""
    inputs=[PREREGISTRATION,REGISTRY,"scripts/check_research_gates.py"]
    try:
        entries=load(root/PREREGISTRATION).get("experiment",{})
        entry=entries.get(exp_id) if isinstance(entries,dict) else None
        if not isinstance(entry,dict):
            return inputs
        baselines=entry.get("baseline",[])
        for baseline in baselines if isinstance(baselines,list) else []:
            if isinstance(baseline,dict) and is_repository_path(baseline.get("path")):
                inputs.append(baseline_directory(baseline["path"]))
        items={item.get("id"):item for item in load(root/REGISTRY).get("experiment",[])}
        table=load(root/"experiments"/items[exp_id]["path"]/"config.toml").get("preregistration",{})
        required=entry.get("required",{})
        for key,kind in required.items() if isinstance(required,dict) and isinstance(table,dict) else []:
            if kind=="file" and is_repository_path(table.get(key)):
                inputs.append(table[key])
    except (OSError,KeyError,TypeError,tomllib.TOMLDecodeError,Unreadable):
        pass
    return inputs

def main(root: Path = ROOT) -> int:
    """Check every gate on the repository at `root`, print each error, and
    return the exit status: 0 when every gate holds, 1 otherwise. A file the
    gate cannot read is an error that names it."""
    try:
        errors=gate_errors(root)
    except (Unreadable,HistoryUnreadable) as error:
        errors=[error.named(root)]
    if errors:
        # A name that is not UTF-8 is printed as its escapes.
        print("\n".join("ERROR: "+error for error in errors).encode("utf-8","backslashreplace").decode("utf-8"))
        return 1
    print("OK: research execution gates satisfied")
    return 0

def duplicate_registrations(registry: dict, exp_id: str | None = None) -> list[str]:
    """Why the registry `registry` registers an experiment more than once,
    of every experiment, or of `exp_id` alone when named: the registry
    places an experiment in one directory, and of two entries of one id the
    gate would check one and the runner, which finds both
    (`launchable_at`), launch neither."""
    items=registry.get("experiment",[])
    places={}
    for item in items if isinstance(items,list) else ():
        if isinstance(item,dict) and isinstance(item.get("id"),str):
            places.setdefault(item["id"],[]).append(item.get("path"))
    return [
        f"{REGISTRY} registers {name} {len(paths)} times ({', '.join(map(repr,paths))}); an experiment is registered "
        "once, in one directory"
        for name,paths in sorted(places.items()) if len(paths)>1 and exp_id in (None,name)
    ]

def listed_now(root: Path) -> set[str]:
    """The experiments the list names now; none when it cannot be read,
    which `preregistration_errors` names."""
    try:
        entries=load(root/PREREGISTRATION).get("experiment",{})
    except Unreadable:
        return set()
    return set(entries) if isinstance(entries,dict) else set()

STANDARD_ARTIFACTS=(experiment_records.RUN,experiment_records.METRICS)

def reached_through_symlink(root: Path, path: Path) -> bool:
    """Whether `path` is a symlink or lies below one, from `root` down, which
    a commit holds as its target's path and not as the content read through
    it; a path not under `root` is reached through one."""
    try:
        parts=path.relative_to(root).parts
    except ValueError:
        return True
    return any(root.joinpath(*parts[:depth]).is_symlink() for depth in range(1,len(parts)+1))

def artifact_errors(exp_id: str, root: Path, results: Path, required, what: str) -> list[str]:
    """Why an artifact of `required` is not there in `results`, the results
    directory of the experiment `what` describes ("completed experiment"), or
    is not the file a commit holds at its own path: a link reads another
    file's content, which no commit holds there."""
    errors=[]
    for artifact in required:
        if not (results/artifact).exists():
            errors.append(f"{exp_id}: {what} missing {artifact}")
        elif reached_through_symlink(root,results/artifact):
            errors.append(f"{exp_id}: {what} {artifact} is a symlink or lies below one; an artifact is the file a "
                          "commit holds at its own path")
    return errors

def gate_errors(root: Path) -> list[str]:
    """Every error of every gate on the repository at `root` (`main`),
    raising `Unreadable` for a file it cannot read outside the listed
    experiments, whose unreadable files are errors of their own."""
    # A gate run must observe one repository snapshot.  Clear the process
    # cache at its boundary so a caller that reuses a temporary checkout path
    # after changing or replacing that checkout cannot receive old history.
    _history_cached.cache_clear()
    listed_entry.cache_clear()
    object_bytes.cache_clear()
    _toml_at_cached.cache_clear()
    _launchable_at_cached.cache_clear()
    _directory_digest_at_cached.cache_clear()
    _statuses_at_cached.cache_clear()
    _descendants_of_cached.cache_clear()
    link_changes_at.cache_clear()
    errors=[]
    experiments={}
    directories={}
    registry=load(root/REGISTRY)
    errors.extend(duplicate_registrations(registry))
    listed=listed_now(root)
    for item in registry.get("experiment",[]):
        if not isinstance(item,dict) or not isinstance(item.get("id"),str) or not isinstance(item.get("path"),str):
            errors.append(f"{REGISTRY}: every experiment names an id and a path, not {item!r}")
            continue
        experiment=root/"experiments"/item["path"]
        manifest=load(experiment/"experiment.toml")
        experiments[item["id"]]=manifest
        directories[item["id"]]=experiment
        status=manifest.get("status")
        if status=="completed":
            results_dir=manifest.get("results_dir","results")
            if not isinstance(results_dir,str):
                errors.append(f'{item["id"]}: results_dir {results_dir!r} is not a path')
                continue
            results=experiment/results_dir
            required=manifest.get("required_artifacts",[])
            if item["id"] in listed:
                # What a listed experiment reports is its aggregate, bound to
                # the finished run of each preregistered seed
                # (`archived_errors`); that it is there does not rest on the
                # list its manifest names, which is written by its author.
                required=[*required,*(name for name in STANDARD_ARTIFACTS if name not in required)]
            errors.extend(artifact_errors(item["id"],root,results,required,"completed experiment"))
            # A listed experiment's command may run or read any file of the
            # repository: its results are stale once any of them changes.
            errors.extend(experiment_records.staleness_errors(item["id"],experiment,results,root,item["id"] in listed))
            errors.extend(experiment_records.aggregate_errors(item["id"],experiment,results,root))

    plain=load(root/"research/baselines/plain_model/config.toml")
    for exp_id in ["M001","M002","M003","M004","M005"]:
        if status_of(experiments.get(exp_id,{})) in {"running","completed"}:
            model=plain.get("model",{})
            if is_placeholder(str(model.get("backend",""))) or is_placeholder(str(model.get("model",""))):
                errors.append(f"{exp_id}: matched plain-model baseline is not pinned")

    if status_of(experiments.get("E002",{})) in {"running","completed"}:
        rag=load(root/"research/baselines/strong_rag/config.toml")
        required=[
            rag.get("dense",{}).get("revision",""),
            rag.get("reranker",{}).get("revision",""),
            rag.get("answer",{}).get("revision",""),
        ]
        if any(is_placeholder(str(value)) for value in required):
            errors.append("E002: strong RAG model/generator revisions are not pinned")
        if str(rag.get("status","")).startswith("blocked-"):
            errors.append("E002: strong RAG baseline is still blocked")

    errors.extend(preregistration_errors(root,experiments,directories))
    return errors

if __name__=="__main__":
    raise SystemExit(main())
