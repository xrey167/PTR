"""Research execution gates, run by CI.

A completed experiment must have its required artifacts, and its archived
results must describe HEAD's code: `results/run.json` and
`results/mutations.json` fail the gate once a provenance file (the Rust, SQL,
protobuf, Cargo and toolchain files, the recording scripts, the experiment's
`aggregate.py` and mutation plan) differs from the one at their `git_sha`,
unless `results/STALE.toml` names those results and the first commit that
made them stale (`experiment_records.staleness_errors`). `results/run.json`
must also be one aggregate with the `metrics.json` it names the SHA-256 of
and the `mutations.json` whose summary it carries; a run.json that binds no
metrics, aggregated before run.json bound them, passes only while it is stale
and beside the metrics.json committed with it (`experiment_records.aggregate_errors`).
Experiments that need a pinned baseline may run only once it is pinned.

An experiment listed in `experiments/preregistration.toml` leaves `planned`
(status `prepared`, `running`, `completed` or `failed`) only with a frozen
preregistration (`preregistration_errors`); a superseded experiment has been
replaced by another, whose own preregistration counts:

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
  `int-list`, names the manifest's seeds, so no seed is added after an
  outcome is seen;
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
  history and holds the same preregistration, entry, files and baselines,
  and the same manifest and `config.toml` but for the manifest's status,
  which has only moved forward since (`history_errors`); a run record also
  names the SHA-256 of the manifest at that commit and is unchanged since it
  was committed, an aggregate, which may be written again, holds to this in
  every version it was committed in, none once committed is deleted, renamed
  or moved, and each is a regular file reached through no symlink. A
  preregistration rewritten after its runs fails even when the manifest and
  the records are rewritten to match, short of rewriting history;
- no file the table freezes and no baseline's directory lies in the results
  directory, or holds it: the runner holds the experiment's files to HEAD
  except those, where runs write.

The list keeps every experiment it has named at a commit on HEAD's history
(`enrolled`), so taking one out of it opens neither the gate nor the
runner, and a listed experiment whose run records were committed cannot go
back to `planned`; it can only be superseded.

`scripts/run_experiment.py` refuses to run or prepare a listed experiment
until this holds for it (`launch_errors`), so no outcome is seen before the
preregistration is frozen, and holds the files that decision reads to HEAD
while it runs (`launch_inputs`). A file or baseline the list names, and every
file of a baseline's directory, must be a regular file inside the repository,
reached through no symlink (`repository_file`).

A value is pinned unless it is a placeholder: a `must-be-pinned-…` string for
a value still to be chosen, a `must-be-signed-…` string for an owner decision
still to be taken, an empty string, `unconfigured` or `none`. A placeholder is
a string, so it passes as no other type: an owner decision whose value is a
boolean cannot pass while it waits for the owner. The list itself must name
only registered experiments, known types and well-formed baselines, whatever
their status.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
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

def load(path: Path) -> dict:
    """The TOML file at `path`."""
    return tomllib.loads(path.read_text(encoding="utf-8"))

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

def is_repository_path(value) -> bool:
    """A relative path of printable ASCII that names no parent, so it stays
    inside the repository as written."""
    if not isinstance(value,str) or not value or not experiment_records.is_printable_ascii(value):
        return False
    path=PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value

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
    `.gitignore` files do not ignore, so no cache or build output), each
    mapped from its path within the directory to its
    `experiment_records.preregistered_file_digest`. A file reached through a
    symlink is refused, as `repository_file` refuses it."""
    try:
        names=experiment_records.listed_names(
            root,"ls-files","-z","--cached","--others",experiment_records.PER_DIRECTORY,"--",directory)
    except experiment_records.ProvenanceError as error:
        return [f"cannot be listed: {error}"],None
    problems=[]
    files={}
    for name in sorted(set(names)):
        path=root/name
        if not path.exists() and not path.is_symlink():
            # Tracked, and removed from the tree: the tree does not hold it.
            continue
        if repository_file(root,name) is None:
            problems.append(f"holds {name}, which is not a regular file reached through no symlink")
            continue
        files[PurePosixPath(name).relative_to(directory).as_posix()]=experiment_records.preregistered_file_digest(path)
    if problems:
        return problems,None
    digest=file_table_digest(files)
    if digest is None:
        return ["holds a file whose path is not printable ASCII"],None
    return [],digest

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

def blob(root: Path, commit: str, relative: str) -> bytes | None:
    """The bytes of `relative` at `commit` in the repository at `root`, or
    None when that commit holds no such file."""
    try:
        shown=subprocess.run(["git","show",f"{commit}:{relative}"],cwd=root,capture_output=True,check=False)
    except OSError:
        return None
    return shown.stdout if shown.returncode==0 else None

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

def history(root: Path, *args: str) -> list[str]:
    """The NUL- or newline-separated names `git log --full-history *args`
    prints in `root`, side branches merged into HEAD included; empty where
    there is no such history."""
    try:
        listing=experiment_records.git(root,"log","--full-history",*args)
    except experiment_records.ProvenanceError:
        return []
    if listing.returncode!=0:
        return []
    return [name for name in listing.stdout.replace("\0","\n").split("\n") if name]

def versions(root: Path, relative: str) -> list[tuple[str, dict]]:
    """Every commit on HEAD's history that changes the TOML file `relative`,
    newest first, with the file as that commit holds it; a commit where it
    is absent or does not parse is left out."""
    found=[]
    for commit in history(root,"--format=%H","HEAD","--",relative):
        held=toml_at(root,commit,relative)
        if isinstance(held,dict):
            found.append((commit,held))
    return found

def registered_at(root: Path, commit: str) -> set[str]:
    """The experiments the registry holds at `commit`."""
    registry=toml_at(root,commit,REGISTRY)
    items=registry.get("experiment") if isinstance(registry,dict) else None
    return {item.get("id") for item in items if isinstance(item,dict)} if isinstance(items,list) else set()

def enrolled(root: Path) -> dict[str, str]:
    """Every experiment the list has named at a commit on HEAD's history
    where the registry held it too, with the newest such commit. A name the
    registry did not hold there, such as a mistyped one, enrolled nothing."""
    named={}
    for commit,listed in versions(root,PREREGISTRATION):
        entries=listed.get("experiment")
        if not isinstance(entries,dict):
            continue
        registered=registered_at(root,commit)
        for exp_id in entries:
            if exp_id in registered:
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
            if isinstance(item,dict) and item.get("id")==exp_id and is_repository_path(item.get("path")):
                # As a path names it, without `.` steps or a trailing slash.
                found.add(PurePosixPath("experiments",item["path"]).as_posix())
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

def committed_records(root: Path, directories: list[str]) -> list[str]:
    """Every run record and aggregate committed under `directories` on
    HEAD's history, as repository paths, whether the tree still holds it or
    not, sorted. Git reports a rename as the removal of the old path, so
    both are named."""
    names=history(root,"--no-renames","--name-only","-z","--format=","HEAD","--",*directories)
    return sorted({name for name in names if is_run_record(name) or is_aggregate(name)})

def directory_digest_at(root: Path, commit: str, directory: str) -> str | None:
    """`directory_digest` of the repository `directory` as `commit` holds it,
    or None when it holds anything but regular files there (a symlink, a
    submodule) or git cannot tell."""
    try:
        listing=experiment_records.git(root,"ls-tree","-r","-z",commit,"--",directory)
    except experiment_records.ProvenanceError:
        return None
    if listing.returncode!=0:
        return None
    files={}
    for entry in listing.stdout.split("\0"):
        if not entry:
            continue
        meta,_,name=entry.partition("\t")
        mode,kind,_=meta.split()
        if kind!="blob" or mode not in {"100644","100755"}:
            return None
        data=blob(root,commit,name)
        if data is None:
            return None
        files[PurePosixPath(name).relative_to(directory).as_posix()]=hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()
    return file_table_digest(files)

def changed_keys(then: dict, now: dict, unbound: set[str]) -> list[str]:
    """The keys other than `unbound` whose values differ between `then` and
    `now`, sorted; a key only one of them holds differs."""
    return sorted(key for key in (then.keys()|now.keys())-unbound if then.get(key,MISSING)!=now.get(key,MISSING))

def history_errors(exp_id: str, name: str, named, root: Path, experiment: Path, entry: dict, table: dict,
                   frozen: tuple[str, str], current: tuple[dict, dict]) -> tuple[list[str], str | None]:
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
    of rewriting history."""
    digest,rules=frozen
    now_manifest,now_config=current
    where=f"{exp_id}: {name}"
    commit=experiment_records.resolve_commit(named,root) if isinstance(named,str) else None
    if commit is None:
        return [f"{where} names no commit this repository holds ({named!r})"],None
    at=f"{where} ran at {commit[:12]}"
    errors=[]
    if not experiment_records.is_ancestor(commit,"HEAD",root):
        errors.append(f"{at}, which is not on HEAD's history")
    relative=experiment.relative_to(root).as_posix()
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
        then_status,now_status=manifest.get("status"),now_manifest.get("status")
        if then_status not in RANK:
            errors.append(f"{at}, where it was {then_status!r}; a listed experiment runs only once it is prepared")
        elif now_status not in RANK or RANK[now_status]<RANK[then_status] or (RANK[then_status]==2 and now_status!=then_status):
            errors.append(f"{at}, where it was {then_status!r}; it cannot be {now_status!r} after that, "
                          "since a status moves only from prepared to running to completed or failed")
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
        data=blob(root,commit,table[key])
        held=None if data is None else hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()
        if held!=table.get(f"{key}_sha256"):
            errors.append(f"{at}, where {table[key]} is not the file frozen as {key}")
    for baseline in entry.get("baseline",[]):
        held=directory_digest_at(root,commit,baseline_directory(baseline["path"])) if is_repository_path(baseline["path"]) else None
        if held!=table.get(f"baseline_{baseline['name']}_sha256"):
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
        if record.get("manifest_sha256") not in {hashlib.sha256(spelling).hexdigest() for spelling in spellings}:
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
    results directory. None once committed may be deleted, renamed or moved,
    and each is a regular file reached through no symlink. Every run record
    and aggregate must pass `record_errors`; a run record must also be
    unchanged since it was committed, and an aggregate, which an aggregator
    may write again, must pass `record_errors` in every version committed."""
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
        if aggregate:
            # Written again, an aggregate keeps every version it was
            # committed in: each saw the outcome of the runs it names.
            for commit in history(root,"--format=%H","HEAD","--",path):
                held=blob(root,commit,path)
                if held is None or held==data:
                    continue
                version=f"{name} as committed at {commit[:12]}"
                try:
                    earlier=json.loads(held.decode("utf-8"))
                except (UnicodeDecodeError,json.JSONDecodeError) as error:
                    errors.append(f"{exp_id}: {version} cannot be read: {error}")
                    continue
                errors.extend(record_errors(exp_id,version,earlier,True,root,experiment,entry,table,frozen,current))
            continue
        touched=experiment_records.git(root,"rev-list","HEAD","--",path).stdout.split()
        if len(touched)>1:
            errors.append(f"{where} was changed after it was committed ({len(touched)} commits touch it)")
        if touched and experiment_records.git(root,"diff","--quiet","HEAD","--",path).returncode!=0:
            errors.append(f"{where} differs from the record committed as it")
    return errors

def frozen_errors(exp_id: str, entry: dict, manifest: dict, experiment: Path, root: Path) -> list[str]:
    """What keeps a listed experiment that left `planned` from having a frozen
    preregistration."""
    config=experiment/"config.toml"
    if not config.is_file():
        return [f"{exp_id}: config.toml does not exist"]
    settings=load(config)
    table=settings.get("preregistration")
    if not isinstance(table,dict):
        return [f"{exp_id}: config.toml has no [preregistration] table"]
    errors=[]
    results_dir=manifest.get("results_dir","results")
    results=None
    if not is_repository_path(results_dir):
        errors.append(f"{exp_id}: results_dir {results_dir!r} is not a path inside the experiment's directory")
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
                errors.extend(frozen_value_errors(
                    exp_id,table,f"{key}_sha256",experiment_records.preregistered_file_digest(file),table[key]))
    for key,value in table.items():
        problem=None if key in entry["required"] else unset_problem(value)
        if problem:
            errors.append(f"{exp_id}: preregistration key {key} {problem}")
    if "seeds" in table and table["seeds"]!=manifest.get("seeds"):
        errors.append(f"{exp_id}: preregistered seeds {table['seeds']!r} are not the manifest's seeds {manifest.get('seeds')!r}")
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
    experiment whose runs were committed has not gone back to `planned`."""
    if not (root/PREREGISTRATION).is_file():
        return [f"{PREREGISTRATION} does not exist"]
    listed=load(root/PREREGISTRATION)
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
        status=manifest.get("status")
        if status in FROZEN:
            errors.extend(frozen_errors(exp_id,entry,manifest,experiments[exp_id],root))
        elif status!="superseded":
            relative=experiments[exp_id].relative_to(root).as_posix()
            committed=committed_records(root,experiment_directories(root,exp_id,relative))
            if committed:
                errors.append(
                    f"{exp_id} is {status!r}, but run records of it were committed ({committed[0]}); a listed "
                    "experiment that has run stays prepared, running, completed or failed, or is superseded"
                )
    for exp_id,commit in sorted(enrolled(root).items()):
        if exp_id not in entries:
            errors.append(delisted_error(exp_id,commit))
    return errors

def launch_errors(root: Path, exp_id: str) -> list[str]:
    """Why `exp_id` may not be run or prepared now (`run_experiment.py`),
    empty when it may. An experiment the list names runs only once it has
    left `planned` with a frozen preregistration (`frozen_errors`), so none of
    its outcomes can be seen before its preregistration is fixed, and one
    the list named once and no longer names (`enrolled`) does not run at
    all; one the list has never named runs as before."""
    if not (root/PREREGISTRATION).is_file():
        return [f"{PREREGISTRATION} does not exist, so whether {exp_id} preregisters is unknown"]
    entries=load(root/PREREGISTRATION).get("experiment",{})
    if not isinstance(entries,dict):
        return [f"{PREREGISTRATION}: experiment must be a table of experiments"]
    if exp_id not in entries:
        commit=enrolled(root).get(exp_id)
        return [] if commit is None else [delisted_error(exp_id,commit)]
    items={item.get("id"):item for item in load(root/REGISTRY).get("experiment",[])}
    problems=entry_errors(exp_id,entries[exp_id],set(items))
    if problems:
        return problems
    experiment=root/"experiments"/items[exp_id]["path"]
    manifest=load(experiment/"experiment.toml")
    status=manifest.get("status")
    if status not in FROZEN:
        return [
            f"{exp_id} preregisters ({PREREGISTRATION}) and is {status!r}: it runs only once its preregistration "
            "is frozen and it is prepared, running, completed or failed"
        ]
    return frozen_errors(exp_id,entries[exp_id],manifest,experiment,root)

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
    except (OSError,KeyError,TypeError,tomllib.TOMLDecodeError):
        pass
    return inputs

def main(root: Path = ROOT) -> int:
    """Check every gate on the repository at `root`, print each error, and
    return the exit status: 0 when every gate holds, 1 otherwise."""
    errors=[]
    experiments={}
    directories={}
    registry=load(root/"experiments/registry.toml")
    for item in registry.get("experiment",[]):
        experiment=root/"experiments"/item["path"]
        manifest=load(experiment/"experiment.toml")
        experiments[item["id"]]=manifest
        directories[item["id"]]=experiment
        status=manifest.get("status")
        if status=="completed":
            results=experiment/manifest.get("results_dir","results")
            for artifact in manifest.get("required_artifacts",[]):
                if not (results/artifact).exists():
                    errors.append(f'{item["id"]}: completed experiment missing {artifact}')
            errors.extend(experiment_records.staleness_errors(item["id"],experiment,results,root))
            errors.extend(experiment_records.aggregate_errors(item["id"],experiment,results,root))

    plain=load(root/"research/baselines/plain_model/config.toml")
    for exp_id in ["M001","M002","M003","M004","M005"]:
        if experiments.get(exp_id,{}).get("status") in {"running","completed"}:
            model=plain.get("model",{})
            if is_placeholder(str(model.get("backend",""))) or is_placeholder(str(model.get("model",""))):
                errors.append(f"{exp_id}: matched plain-model baseline is not pinned")

    if experiments.get("E002",{}).get("status") in {"running","completed"}:
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

    if errors:
        print("\n".join("ERROR: "+error for error in errors))
        return 1
    print("OK: research execution gates satisfied")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
