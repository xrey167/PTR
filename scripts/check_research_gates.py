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
  and it gives Cargo no configuration on its command line, nor does the
  repository's Cargo configuration name a program, source or flags for it
  (`command_errors`), since a freeze no run can use cannot be repaired;
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
  which has only moved forward since (`history_errors`); a run record also
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
records and its preregistration as they were.

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

def command_errors(exp_id: str, manifest: dict, table: dict, read=None) -> list[str]:
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
    was given. And it gives Cargo no configuration on its command line
    (`cargo_arguments`): a `--config` file or value can name a rustc
    wrapper, flags or sources outside what the commit holds. Nor does the
    repository's own Cargo configuration, a file of the commit, name any of
    them, or a program Cargo runs (`cargo_configuration_errors`): the record
    names the compiler rustup or the `PATH` resolves, not one Cargo is told
    to run instead, or a runner started in place of the built program. A commit
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
    arguments=cargo_arguments(command,read)
    if any(argument=="--config" or argument.startswith("--config=") for argument in arguments):
        errors.append(f"{exp_id}: entrypoint gives Cargo configuration on its command line (--config), which can name "
                      "a rustc wrapper, flags or sources outside the commit; set what the build needs in the "
                      "repository's .cargo/config.toml")
    for option,value in cargo_path_options(arguments):
        if option=="--target-dir":
            errors.append(f"{exp_id}: entrypoint gives Cargo a target directory (--target-dir), which could hold a "
                          "build made outside the commit; the runner builds a listed run into a fresh one")
        elif outside_repository(value):
            errors.append(f"{exp_id}: entrypoint gives Cargo {option} {value}, outside what the repository's watch "
                          "reads, whose sources no watch or record binds; name a path the repository holds")
    if read is not None:
        directory=next((value for option,value in cargo_path_options(arguments) if option=="-C"),"")
        errors.extend(cargo_configuration_errors(exp_id,read,directory))
    return errors

# Cargo's options naming a path it reads the build from, and the one naming
# where it builds, with the short options that take a value, which a cluster
# of short options (`-vC dir`) ends with, and the long ones that take the
# next argument as theirs.
CARGO_PATH_OPTIONS=("--manifest-path","--lockfile-path","--target","--target-dir")
CARGO_SHORT_VALUES="CFjpZ"
CARGO_LONG_VALUES=("--config","--manifest-path","--lockfile-path","--target-dir","--color","--explain","--package",
                   "--jobs","--features","--target","--bin","--example","--test","--bench","--profile",
                   "--message-format","--registry","--index")

def cargo_path_options(arguments: list[str]) -> list[tuple[str,str]]:
    """The options among Cargo's own `arguments` (`cargo_arguments`) that
    name a path it builds from or into, with their values: the manifest
    (`--manifest-path`), the lockfile (`--lockfile-path`), the target,
    which may be a specification file (`--target custom.json`), the
    directory it runs in (`-C`, alone, joined or ending a cluster of short
    options) and the target directory (`--target-dir`), each given as one
    token
    (`--manifest-path=x`, `-Cx`) or two; a value missing at the end reads
    as empty. In Cargo's script mode (`-Zscript`, `-Z script`), whose
    manifest is a file an argument names, every argument that is no option
    or option's value counts as one (`-Zscript`), the script's own after it
    included."""
    found=[]
    for index,argument in enumerate(arguments):
        following=arguments[index+1] if index+1<len(arguments) else ""
        for option in CARGO_PATH_OPTIONS:
            if argument==option:
                found.append((option,following))
            elif argument.startswith(f"{option}="):
                found.append((option,argument[len(option)+1:]))
        if argument.startswith("-") and not argument.startswith("--"):
            short=cargo_short_value(argument,following)
            if short is not None and short[0]=="C":
                found.append(("-C",short[1]))
    if cargo_script_mode(arguments):
        found.extend(("-Zscript",argument) for argument in cargo_positionals(arguments))
    return found

def cargo_short_value(argument: str, following: str) -> tuple[str,str,bool] | None:
    """The short option taking a value that the cluster `argument` (`-Zx`,
    `-vZ x`) ends with, its value, joined or the `following` argument, and
    whether it was joined; None for a cluster that takes none."""
    for position,letter in enumerate(argument[1:],start=1):
        if letter in CARGO_SHORT_VALUES:
            joined=argument[position+1:]
            return (letter,joined,True) if joined else (letter,following,False)
    return None

def cargo_script_mode(arguments: list[str]) -> bool:
    """Whether Cargo's own `arguments` turn on its script mode, `-Z script`
    in any spelling (`-Zscript`, `-Z script`, `-vZscript`)."""
    for index,argument in enumerate(arguments):
        following=arguments[index+1] if index+1<len(arguments) else ""
        if argument.startswith("-") and not argument.startswith("--"):
            short=cargo_short_value(argument,following)
            if short is not None and short[:2]==("Z","script"):
                return True
    return False

def cargo_positionals(arguments: list[str]) -> list[str]:
    """Those of Cargo's own `arguments` that are neither an option nor the
    value an option takes as the next argument, a rustup `+<toolchain>`
    being no argument of Cargo's."""
    positionals=[]
    taken=False
    for index,argument in enumerate(arguments):
        following=arguments[index+1] if index+1<len(arguments) else ""
        if taken:
            taken=False
        elif argument in CARGO_LONG_VALUES:
            taken=True
        elif argument.startswith("-") and not argument.startswith("--"):
            short=cargo_short_value(argument,following)
            taken=short is not None and not short[2]
        elif not argument.startswith(("-","+")):
            positionals.append(argument)
    return positionals

def outside_repository(path: str) -> bool:
    """Whether `path`, as a command run from the repository's root reads it,
    may name something outside what the repository's watch reads: absolute
    on any platform (`/x`, `C:/x`, `\\\\host\\x`), climbing above the
    root (`../x`, `a/../../x`), or through a directory named `.git`, in any
    case, which git keeps for itself and never lists (`.git/x/Cargo.toml`),
    either slash a separator. The runner starts no shell, so `~` is a name
    like any other."""
    normalized=path.replace("\\","/")
    if normalized.startswith("/") or re.match(r"[A-Za-z]:",normalized):
        return True
    depth=0
    for part in normalized.split("/"):
        if part.lower()==".git":
            return True
        if part=="..":
            depth-=1
            if depth<0:
                return True
        elif part not in ("","."):
            depth+=1
    return False

def cargo_arguments(tokens: list[str], read=None) -> list[str]:
    """The arguments the command `tokens` gives Cargo itself, up to a `--`
    past which they are the built program's: those after `cargo`, as a
    rustup proxy or not, and after `rustup run <toolchain> cargo`, past
    rustup's own options and `+<toolchain>` and the options of `run`; none
    for another program. Each program is named as a platform runs it
    (`experiment_records.program_name`: `C:\\Rust\\rustup.exe` is rustup).
    Given `read`, which returns the text of a repository file or None,
    Cargo's subcommand is read through the aliases the repository's Cargo
    configuration defines for it (`cargo_aliases`), as Cargo expands them
    before it parses the rest."""
    program=experiment_records.program_name(tokens[0]) if tokens else ""
    rest=tokens[1:]
    if program=="rustup":
        while rest and rest[0].startswith(("-","+")):
            rest=rest[1:]
        if not rest or rest[0]!="run":
            return []
        rest=rest[1:]
        while rest and rest[0].startswith("-"):
            rest=rest[1:]
        rest=rest[1:]
        program=experiment_records.program_name(rest[0]) if rest else ""
        rest=rest[1:]
    if program!="cargo":
        return []
    if read is not None:
        rest=expand_cargo_aliases(rest,read)
    return rest[:rest.index("--")] if "--" in rest else rest

def expand_cargo_aliases(arguments: list[str], read) -> list[str]:
    """Cargo's own `arguments` with the subcommand, the first argument that
    is no option or option's value, replaced by what the alias of its name
    stands for (`cargo_aliases`, read from the directory Cargo runs in,
    `-C` given), again for the subcommand that yields, up to sixteen times
    and never through a name twice, as Cargo expands aliases of aliases."""
    seen=set()
    for _ in range(16):
        before=arguments[:arguments.index("--")] if "--" in arguments else arguments
        index=cargo_subcommand_index(before)
        if index is None or before[index] in seen:
            return arguments
        directory=next((value for option,value in cargo_path_options(before) if option=="-C"),"")
        aliases=cargo_aliases(read,directory)
        name=before[index]
        if name not in aliases:
            return arguments
        seen.add(name)
        arguments=[*arguments[:index],*aliases[name],*arguments[index+1:]]
    return arguments

def cargo_subcommand_index(arguments: list[str]) -> int | None:
    """The index among Cargo's own `arguments` of the first that is neither
    an option nor the value an option takes as the next argument, nor a
    rustup `+<toolchain>`: Cargo's subcommand; None when there is none."""
    taken=False
    for index,argument in enumerate(arguments):
        following=arguments[index+1] if index+1<len(arguments) else ""
        if taken:
            taken=False
        elif argument in CARGO_LONG_VALUES:
            taken=True
        elif argument.startswith("-") and not argument.startswith("--"):
            short=cargo_short_value(argument,following)
            taken=short is not None and not short[2]
        elif not argument.startswith(("-","+")):
            return index
    return None

# The names Cargo reads its configuration from in a `.cargo` directory, the
# older first, which Cargo prefers where both are.
CARGO_CONFIGURATION_NAMES=("config","config.toml")

def cargo_configuration_directories(directory: str="") -> list[str]:
    """The repository directories whose `.cargo` configuration Cargo run
    from the repository's `directory` (`-C`, the root when empty) reads:
    that directory and each above it up to the root, nearest first, the
    root as the empty path; none for a directory outside the repository,
    which the path check refuses."""
    if outside_repository(directory or "."):
        return []
    parts=[part for part in directory.replace("\\","/").split("/") if part not in ("",".")]
    directories=[]
    for part in parts:
        if part=="..":
            if directories:
                directories.pop()
        else:
            directories.append(part)
    return ["/".join(directories[:depth]) for depth in range(len(directories),-1,-1)]

# The tables a listed run's Cargo configuration may set. None names a
# program Cargo runs (a compiler or its wrapper, rustdoc, a linker, a
# runner the built program is started through), a source or file it builds
# from (a path override, a patch, a source replacement, an included
# configuration, a target specification) or flags it builds with, which
# could lie outside what the record binds; the aliases are read through
# (`expand_cargo_aliases`).
CARGO_CONFIGURATION_TABLES=("alias","cargo-new","env","future-incompat-report","http","net","term")

def cargo_configuration_errors(exp_id: str, read, directory: str="") -> list[str]:
    """Why the repository's Cargo configuration a listed run's Cargo reads is
    not one it may build under: Cargo started from the root, by the command
    or by a program it runs, and from the directory the command gives it
    with `-C`, reads `.cargo/config` and `.cargo/config.toml` there and in
    each directory above (`cargo_configuration_directories`), and each of
    those files must parse and set nothing but `CARGO_CONFIGURATION_TABLES`.
    `read` returns a repository file's text, or None where there is none."""
    errors=[]
    paths=[]
    for base in [*cargo_configuration_directories(directory),*cargo_configuration_directories("")]:
        for name in CARGO_CONFIGURATION_NAMES:
            path=f"{base}/.cargo/{name}" if base else f".cargo/{name}"
            if path not in paths:
                paths.append(path)
    for path in paths:
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

def cargo_aliases(read, directory: str="") -> dict[str,list[str]]:
    """The aliases Cargo run from the repository's `directory` (`-C`, the
    root when empty) reads from the repository's Cargo configuration, each
    as the arguments it stands for: the `[alias]` tables of `.cargo/config`,
    or `.cargo/config.toml` where there is none, in that directory and each
    above it up to the root, a nearer one's alias of a name winning; a
    string alias split at whitespace, as Cargo splits it, a list of strings
    as it is. `read` returns a repository file's text, or None where there
    is none; text that does not parse defines none (Cargo refuses to run
    from it). A directory outside the repository, which the path check
    refuses, holds none."""
    aliases={}
    for base in cargo_configuration_directories(directory):
        for name in CARGO_CONFIGURATION_NAMES:
            text=read(f"{base}/.cargo/{name}" if base else f".cargo/{name}")
            if text is None:
                continue
            try:
                table=tomllib.loads(text).get("alias")
            except tomllib.TOMLDecodeError:
                table=None
            for alias,value in (table.items() if isinstance(table,dict) else ()):
                expansion=value.split() if isinstance(value,str) else value if (
                    isinstance(value,list) and all(isinstance(item,str) for item in value)) else None
                if expansion is not None:
                    aliases.setdefault(alias,expansion)
            break
    return aliases

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
    and as Windows reads it, with trailing dots or spaces, or by its short
    name `git~1`."""
    return part.lower().rstrip(". ") in {".git","git~1"}

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
    would be one no commit made by `git add` holds."""
    if not isinstance(value,str) or not value or len(value)>MAX_PATH or not experiment_records.is_printable_ascii(value):
        return False
    path=PurePosixPath(value)
    return (not path.is_absolute() and ".." not in path.parts and "\\" not in value and path.as_posix()==value
            and not value.startswith(":") and not any(character in value for character in "*?[")
            and not any(is_git_administration(part) for part in path.parts))

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
    entry, a bytecode file standing in for a module), and git holds none;
    and so is text git holds with CRLF line endings, which the digest reads
    as LF where a command would tell them apart."""
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
    modes,blobs={},{}
    for entry in staged:
        fields,_,name=entry.partition("\t")
        modes[name],blobs[name]=fields.split(" ")[:2]
    problems=[f"holds {name}, which git ignores; a baseline's directory holds only what git tracks or would track"
              for name in sorted(set(ignored))]
    # Its digest reads CRLF as LF, so a checkout that converts line endings
    # holds the baseline the repository does; text committed with CRLF would
    # read alike with LF, where a command reading it would tell them apart.
    for name,held in sorted(blobs.items()):
        try:
            data=object_bytes(root,held)
        except HistoryUnreadable as error:
            problems.append(f"holds {name}, which git cannot read: {error}")
            continue
        if experiment_records.is_preregistered_text(data) and b"\r\n" in data:
            problems.append(f"holds {name}, which is committed with CRLF line endings; its digest reads them as LF, so "
                            "a baseline's text files are committed with LF line endings")
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

def file_form_problem(data: bytes, mode: str) -> str | None:
    """What keeps content `data` with git mode `mode` from being a
    preregistered file as a commit holds it, or None: it is not executable,
    since its digest holds its content alone and the executable bit changes
    what a command does with it, and as text it holds no CRLF line ending,
    since its digest reads CRLF as LF and a command reading it would read
    two such versions apart."""
    if mode!="100644":
        return "is executable; a preregistered file is frozen by its content, so it is a file no command runs as a program"
    if experiment_records.is_preregistered_text(data) and b"\r\n" in data:
        return ("is committed with CRLF line endings; its digest reads them as LF, so a preregistered text file is "
                "committed with LF line endings")
    return None

def preregistered_file_problem(root: Path, relative: str, file: Path) -> str | None:
    """`file_form_problem` of the preregistered file `relative` names, `file`
    in the tree under `root`: its mode as git holds it (the index's for a
    tracked file, the owner's execute bit for another), and its content as
    HEAD holds it where HEAD holds it, a checkout that converts line endings
    holding it with CRLF, or as the tree holds it where HEAD does not."""
    try:
        staged=experiment_records.listed_names(root,"--literal-pathspecs","ls-files","-z","--stage","--",relative)
    except experiment_records.ProvenanceError as error:
        return f"cannot be listed: {error}"
    mode=staged[0].split(" ")[0] if staged else ("100755" if file.stat().st_mode & 0o100 else "100644")
    held=blob(root,"HEAD",relative) if has_head(root) else None
    return file_form_problem(file.read_bytes() if held is None else held,mode)

def frozen_file_at(root: Path, commit: str, relative: str) -> bytes | None:
    """The content of the file `relative` names as `commit` holds it, or None
    when it holds none there in the form of a preregistered file
    (`file_form_problem`). Raises `HistoryUnreadable` when git cannot tell."""
    held=tree_entry(root,commit,relative)
    data=blob(root,commit,relative)
    if held is None or data is None or file_form_problem(data,held[0]):
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

def toml_at(root: Path, commit: str, relative: str) -> dict | None:
    """The TOML file `relative` at `commit`, or None when it is absent or
    does not parse."""
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

def history(root: Path, *args: str) -> list[str]:
    """The NUL- or newline-separated names `git log --full-history *args`
    prints in `root`, side branches merged into HEAD included; empty where
    HEAD has no commit yet. Raises `HistoryUnreadable` when git cannot read
    it otherwise: a history read as empty would hide every freeze and record
    in it."""
    try:
        listing=experiment_records.git(root,"--literal-pathspecs","log","--full-history",*args,binary=True)
    except experiment_records.ProvenanceError as error:
        raise HistoryUnreadable(str(error)) from error
    if listing.returncode!=0:
        if not has_head(root):
            return []
        raise HistoryUnreadable(listing.stderr.decode("utf-8","replace").strip() or f"git log exited {listing.returncode}")
    # A name that is not UTF-8 keeps its bytes (and so names its file), and
    # fails every check that asks for a repository path, rather than
    # failing every later gate on the commit that once held it.
    names=listing.stdout.decode("utf-8","surrogateescape")
    return [name for name in names.replace("\0","\n").split("\n") if name]

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

def experiment_directories(root: Path, exp_id: str, current: str) -> list[str]:
    """The directories, as repository paths, that the registry has given
    `exp_id` at a commit on HEAD's history, and `current`, its directory now."""
    found={current}
    for _,registry in versions(root,REGISTRY):
        items=registry.get("experiment")
        for item in items if isinstance(items,list) else ():
            if isinstance(item,dict) and item.get("id")==exp_id and isinstance(item.get("path"),str):
                # As a path names it, without `.` steps or a trailing slash,
                # as the runner and `launchable_at` read it.
                directory=PurePosixPath("experiments",item["path"]).as_posix()
                if is_repository_path(directory):
                    found.add(directory)
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

def committed_blobs(root: Path, relative: str) -> set[str]:
    """The distinct contents, as blob names, that the repository path
    `relative` has had at the commits of HEAD's full history that change it,
    on every side of every merge."""
    found=set()
    for commit in history(root,"--format=%H","HEAD","--",relative):
        held=experiment_records.git(root,"rev-parse","--verify","--quiet",f"{commit}:{relative}")
        if held.returncode==0:
            found.add(held.stdout.strip())
    return found

def launchable_at(root: Path, commit: str, exp_id: str, directory: str) -> tuple[str, str] | None:
    """The status at which `commit` holds the experiment in `directory`
    frozen as the runner launches it (`launch_errors`), with the digest of
    everything frozen there (the manifest, the configuration, the entry,
    the files' and baselines' digests and the directory), or None when it
    does not: the list names it with a well-formed entry, its manifest is past
    `planned`, its `[preregistration]` table holds every required key,
    pinned and of its type, and no placeholder, the table's seeds are the
    manifest's, each named once, the manifest names the digests of the table and of the
    entry, every file and baseline the table freezes has the frozen
    content there, each baseline pinned and not blocked, the runner can
    build its command from the manifest and the table (`command_errors`),
    and the registry places the experiment in `directory`. A commit that
    held less could not launch it, and does not freeze it."""
    manifest=toml_at(root,commit,f"{directory}/experiment.toml")
    if not isinstance(manifest,dict) or status_of(manifest) not in FROZEN or not names_an_entrypoint(manifest):
        return None
    registry=toml_at(root,commit,REGISTRY)
    items=registry.get("experiment") if isinstance(registry,dict) else None
    placed={
        PurePosixPath("experiments",item["path"]).as_posix() for item in (items if isinstance(items,list) else ())
        if isinstance(item,dict) and item.get("id")==exp_id and isinstance(item.get("path"),str)
    }
    if placed!={directory} or not is_repository_path(directory):
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
    if command_errors(exp_id,manifest,table,lambda name:blob_text(root,commit,name)):
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

def launch_paths(root: Path, exp_id: str, directories: list[str]) -> set[str]:
    """Every repository path whose content can decide whether `exp_id` may
    launch at a commit on HEAD's history: its `directories`, the list, the
    registry, the directory of every baseline any version of its entry has
    named, and every file any version of its table has named under a key
    of type `file`."""
    paths=set(directories)|{PREREGISTRATION,REGISTRY}
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

def frozen_commits(root: Path, exp_id: str, relative: str) -> list[tuple[str, str]]:
    """Every commit on HEAD's history that changes a path that can decide
    whether the experiment may launch (`launch_paths`: its manifest and
    configuration in any directory the registry has given it, `relative`
    being its directory now, the list, the registry, its baselines and its
    files) and holds it frozen as the runner launches it (`launchable_at`),
    with its status there, newest first. From such a commit on, the runner
    could launch the experiment, whether or not a record of that run was
    kept; the first such commit may change any one of those paths alone.
    Commits that hold the same frozen state, such as those that add only
    records, are one freeze, named by the oldest of them."""
    directories=experiment_directories(root,exp_id,relative)
    states={}
    for commit in history(root,"--format=%H","HEAD","--",*sorted(launch_paths(root,exp_id,directories))):
        for directory in directories:
            frozen=launchable_at(root,commit,exp_id,directory)
            if frozen is not None:
                # Newest first: an older commit of the same state takes its
                # place, and moves it behind the states seen since.
                states.pop(frozen,None)
                states[frozen]=commit
                break
    return [(commit,status) for (status,_),commit in states.items()]

def directory_digest_at(root: Path, commit: str, directory: str) -> str | None:
    """`directory_digest` of the repository `directory` as `commit` holds it,
    or None when the tree check (`directory_digest`) would refuse what it
    holds there: anything but a regular file (a symlink, a submodule), a
    file named as no repository path (`is_repository_path`), a name that is
    not UTF-8 included, or text committed with CRLF line endings, which its
    digest reads as LF. None is no digest, so it matches no frozen one.
    Raises `HistoryUnreadable` when git cannot tell."""
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
        if experiment_records.is_preregistered_text(data) and b"\r\n" in data:
            # Text committed with CRLF, which its digest reads as LF.
            return None
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
    registry=toml_at(root,commit,REGISTRY)
    items=registry.get("experiment") if isinstance(registry,dict) else None
    placed=sorted({
        PurePosixPath("experiments",item["path"]).as_posix() for item in (items if isinstance(items,list) else ())
        if isinstance(item,dict) and item.get("id")==exp_id and isinstance(item.get("path"),str)
    })
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
        if then_status not in RANK:
            errors.append(f"{at}, where it was {then_status!r}; a listed experiment runs only once it is prepared")
        elif now_status!="superseded" and (now_status not in RANK or RANK[now_status]<RANK[then_status]
                                            or (RANK[then_status]==2 and now_status!=then_status)):
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
        # run_experiment.py hashes the manifest as the checkout holds it,
        # which is the committed text, or that text with CRLF line endings
        # where the checkout converts them.
        spellings=set() if held is None else {held,re.sub(rb"(?<!\r)\n",b"\r\n",held)}
        named_manifest=record.get("manifest_sha256")
        if not isinstance(named_manifest,str) or named_manifest not in {hashlib.sha256(spelling).hexdigest() for spelling in spellings}:
            errors.append(f"{where} names manifest_sha256 {record.get('manifest_sha256')!r}, not the SHA-256 of experiment.toml at {commit[:12]}")
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
    pass `record_errors` in every version committed. No two run records are
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

    committed=committed_records(root,experiment_directories(root,exp_id,relative))
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
    runs={}
    programs={}
    trees=[]
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
            # A prepared record names no seed, and a command that failed to
            # launch saw no outcome.
            if not aggregate and isinstance(record,dict) and "seed" in record and record.get("status")!="failed-to-launch":
                runs.setdefault(json.dumps(record["seed"],sort_keys=True),[]).append(name)
                # The program, the toolchain and the environment lie outside
                # the commit: the seeds of one experiment ran one of each.
                programs.setdefault(json.dumps(
                    [record.get("executable"),record.get("toolchain"),record.get("environment")],sort_keys=True),[]
                ).append(name)
                trees.append((str(record.get("started_at","")),name,record.get("git_sha")))
        if aggregate:
            # Written again, an aggregate keeps every version it was
            # committed in: each saw the outcome of the runs it names.
            for commit in history(root,"--format=%H","HEAD","--",path):
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
                    continue
                try:
                    earlier=json.loads(held.decode("utf-8"))
                except (UnicodeDecodeError,json.JSONDecodeError) as error:
                    errors.append(f"{exp_id}: {version} cannot be read: {error}")
                    continue
                errors.extend(record_errors(exp_id,version,earlier,True,root,experiment,entry,table,frozen,current))
            continue
        # Every commit that holds the record, on every side of every merge,
        # holds the same content: a record rewritten on one side of a merge
        # and kept by it was changed all the same.
        blobs=committed_blobs(root,path)
        if len(blobs)>1:
            errors.append(f"{where} was changed after it was committed ({len(blobs)} versions of it were committed)")
        # Its bytes on disk against HEAD's blob, not git diff, which takes a
        # file the index marks skip-worktree or assume-unchanged as HEAD's
        # whatever it holds; a copy that differs from a blob holding no
        # carriage return only in CRLF line endings, as a converting
        # checkout writes one, holds HEAD's content.
        if blobs:
            head=blob(root,"HEAD",path)
            if head is None or not (data==head or (b"\r" not in head and data.replace(b"\r\n",b"\n")==head)):
                errors.append(f"{where} differs from the record committed as it")
    for seed,names in sorted(runs.items()):
        if len(names)>1:
            errors.append(f"{exp_id}: seed {seed} ran more than once ({', '.join(names)}); a listed experiment runs each "
                          "seed once, so no run of it is chosen by its outcome")
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
    for commit,_ in frozen_commits(root,exp_id,relative):
        problems,_=history_errors(exp_id,"",commit,root,experiment,entry,table,frozen,current,
                                  at=f"{exp_id} was frozen at {commit[:12]}")
        errors.extend(problems)
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
    if not names_an_entrypoint(manifest):
        errors.append(f"{exp_id}: experiment.toml names no entrypoint; a listed experiment names what it runs before "
                      "it leaves planned, since its manifest is frozen from then on")
    errors.extend(command_errors(exp_id,manifest,table,lambda name:tree_text(root,name)))
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
        registry=toml_at(root,commit,REGISTRY)
        items=registry.get("experiment") if isinstance(registry,dict) else None
        placed=sorted({
            PurePosixPath("experiments",item["path"]).as_posix() for item in (items if isinstance(items,list) else ())
            if isinstance(item,dict) and item.get("id")==exp_id and isinstance(item.get("path"),str)
        })
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

def gate_errors(root: Path) -> list[str]:
    """Every error of every gate on the repository at `root` (`main`),
    raising `Unreadable` for a file it cannot read outside the listed
    experiments, whose unreadable files are errors of their own."""
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
            for artifact in manifest.get("required_artifacts",[]):
                if not (results/artifact).exists():
                    errors.append(f'{item["id"]}: completed experiment missing {artifact}')
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
