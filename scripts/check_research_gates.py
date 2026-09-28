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
- a `seeds` key in the table, where there is one, names the manifest's seeds;
- every baseline the list names is pinned at each key path the list names
  for it, its status is pinned and not `blocked-*`, and the table's
  `baseline_<name>_sha256` is the digest of its whole configuration file, so
  every setting of the baseline is frozen with the table, not only the keys
  that must be pinned;
- the manifest's `preregistration_rules_sha256` is the digest of the
  experiment's entry in the list, so its types and baselines are frozen too;
- every run record in the experiment's results (`run-*.json`, written by
  `scripts/run_experiment.py` with the manifest it ran under) and its
  aggregate (`run.json`) name those digests and the commit they ran at, and
  that commit holds the same preregistration, entry, files and baseline
  configurations (`history_errors`); a run record also names the SHA-256 of the
  manifest at that commit and is unchanged since it was committed. A
  preregistration rewritten after its runs fails even when the manifest and
  the records are rewritten to match, short of rewriting history.

`scripts/run_experiment.py` refuses to run or prepare a listed experiment
until this holds for it (`launch_errors`), so no outcome is seen before the
preregistration is frozen. A file or baseline the list names must be a file
inside the repository, not a link out of it (`repository_file`).

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
FROZEN={"prepared","running","completed","failed"}
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
    """The file `relative` names under `root`, or None when it names none, or
    names one only through a link out of the repository: a symlink stored in
    the repository whose target lies elsewhere holds content no other
    checkout has."""
    if not is_repository_path(relative):
        return None
    path=root/relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        return None
    return path

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
    when nothing does, the digest of its whole configuration file
    (`experiment_records.preregistered_file_digest`): every setting of the
    baseline, not only the keys that must be pinned, decides what the
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
    return [],experiment_records.preregistered_file_digest(path)

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

def history_errors(exp_id: str, name: str, named, root: Path, experiment: Path, entry: dict, table: dict,
                   frozen: tuple[str, str]) -> tuple[list[str], str | None]:
    """What keeps the commit a record names (`named`) from holding the
    preregistration frozen now, and that commit: the experiment's
    `[preregistration]` table and `experiment.toml` there name the frozen
    digests (`frozen`: the table's and the rules'), the list holds the same
    entry for it, and every file and baseline the table freezes has the
    frozen content there. A record's own fields can be edited; the commit it
    ran at cannot, short of rewriting history."""
    digest,rules=frozen
    where=f"{exp_id}: {name}"
    commit=experiment_records.resolve_commit(named,root) if isinstance(named,str) else None
    if commit is None:
        return [f"{where} names no commit this repository holds ({named!r})"],None
    at=f"{where} ran at {commit[:12]}"
    relative=experiment.relative_to(root).as_posix()
    config=toml_at(root,commit,f"{relative}/config.toml")
    manifest=toml_at(root,commit,f"{relative}/experiment.toml")
    listed=toml_at(root,commit,PREREGISTRATION)
    errors=[]
    then=config.get("preregistration") if isinstance(config,dict) else None
    try:
        then_digest=experiment_records.preregistration_digest(then) if isinstance(then,dict) else None
    except ValueError:
        then_digest=None
    if then_digest!=digest:
        errors.append(f"{at}, whose config.toml holds another [preregistration] than the frozen one")
    if not isinstance(manifest,dict) or manifest.get("preregistration_sha256")!=digest or manifest.get("preregistration_rules_sha256")!=rules:
        errors.append(f"{at}, whose experiment.toml names other preregistration digests than the frozen ones")
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
        data=blob(root,commit,baseline["path"]) if is_repository_path(baseline["path"]) else None
        held=None if data is None else hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()
        if held!=table.get(f"baseline_{baseline['name']}_sha256"):
            errors.append(f"{at}, where baseline {baseline['name']} is not the frozen configuration")
    return errors,commit

def archived_errors(exp_id: str, manifest: dict, experiment: Path, root: Path, entry: dict, table: dict,
                    frozen: tuple[str, str]) -> list[str]:
    """What keeps the experiment's archived runs from having run under the
    preregistration frozen now (`frozen`: the digests of the table and of
    the list's rules for it). Every run record (`run-*.json`) must carry a
    manifest naming both digests, name the commit it ran at, which must hold
    that preregistration (`history_errors`) and the manifest the record
    names by its SHA-256, and be unchanged since it was committed; the
    aggregate (`run.json`) must name both digests and a commit that holds
    that preregistration."""
    digest,rules=frozen
    results=experiment/manifest.get("results_dir","results")
    if not results.is_dir():
        return []
    errors=[]
    for path in sorted(results.glob("run*.json")):
        if path.name!="run.json" and not path.name.startswith("run-"):
            continue
        name=path.relative_to(experiment).as_posix()
        where=f"{exp_id}: {name}"
        try:
            record=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,UnicodeDecodeError,json.JSONDecodeError) as error:
            errors.append(f"{where} cannot be read: {error}")
            continue
        if not isinstance(record,dict):
            errors.append(f"{where} is not a JSON object")
            continue
        aggregate=path.name=="run.json"
        named=record if aggregate else record.get("manifest") if isinstance(record.get("manifest"),dict) else {}
        for field,expected in (("preregistration_sha256",digest),("preregistration_rules_sha256",rules)):
            if named.get(field)!=expected:
                errors.append(f"{where} names {field} {named.get(field)!r}, not {expected}, the digest the experiment is frozen at")
        problems,commit=history_errors(exp_id,name,record.get("git_sha"),root,experiment,entry,table,frozen)
        errors.extend(problems)
        if aggregate:
            continue
        relative=path.relative_to(root).as_posix()
        if commit is not None:
            held=blob(root,commit,f"{experiment.relative_to(root).as_posix()}/experiment.toml")
            if held is None or hashlib.sha256(held).hexdigest()!=record.get("manifest_sha256"):
                errors.append(f"{where} names manifest_sha256 {record.get('manifest_sha256')!r}, not the SHA-256 of experiment.toml at {commit[:12]}")
        touched=experiment_records.git(root,"rev-list","HEAD","--",relative).stdout.split()
        if len(touched)>1:
            errors.append(f"{where} was changed after it was committed ({len(touched)} commits touch it)")
        if touched and experiment_records.git(root,"diff","--quiet","HEAD","--",relative).returncode!=0:
            errors.append(f"{where} differs from the record committed as it")
    return errors

def frozen_errors(exp_id: str, entry: dict, manifest: dict, experiment: Path, root: Path) -> list[str]:
    """What keeps a listed experiment that left `planned` from having a frozen
    preregistration."""
    config=experiment/"config.toml"
    if not config.is_file():
        return [f"{exp_id}: config.toml does not exist"]
    table=load(config).get("preregistration")
    if not isinstance(table,dict):
        return [f"{exp_id}: config.toml has no [preregistration] table"]
    errors=[]
    for key,kind in entry["required"].items():
        if key not in table:
            errors.append(f"{exp_id}: preregistration key {key} is missing")
            continue
        problem=kind_problem(table[key],kind)
        if problem:
            errors.append(f"{exp_id}: preregistration key {key} {problem}")
        elif kind=="file":
            file=repository_file(root,table[key])
            if file is None:
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
        problems,selection=baseline_errors(exp_id,baseline,root)
        errors.extend(problems)
        if selection is not None:
            errors.extend(frozen_value_errors(
                exp_id,table,f"baseline_{baseline['name']}_sha256",selection,baseline["path"]))
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
    errors.extend(archived_errors(exp_id,manifest,experiment,root,entry,table,(digest,rules)))
    return errors

def preregistration_errors(root: Path, manifests: dict[str, dict], experiments: dict[str, Path]) -> list[str]:
    """What keeps the listed experiments that left `planned` from having
    frozen preregistrations, and what is wrong with the list itself."""
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
        if manifest.get("status") in FROZEN:
            errors.extend(frozen_errors(exp_id,entry,manifest,experiments[exp_id],root))
    return errors

def launch_errors(root: Path, exp_id: str) -> list[str]:
    """Why `exp_id` may not be run or prepared now (`run_experiment.py`),
    empty when it may. An experiment the list names runs only once it has
    left `planned` with a frozen preregistration (`frozen_errors`), so none of
    its outcomes can be seen before its preregistration is fixed; one the
    list does not name runs as before."""
    if not (root/PREREGISTRATION).is_file():
        return [f"{PREREGISTRATION} does not exist, so whether {exp_id} preregisters is unknown"]
    entries=load(root/PREREGISTRATION).get("experiment",{})
    if not isinstance(entries,dict):
        return [f"{PREREGISTRATION}: experiment must be a table of experiments"]
    if exp_id not in entries:
        return []
    items={item.get("id"):item for item in load(root/"experiments/registry.toml").get("experiment",[])}
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
