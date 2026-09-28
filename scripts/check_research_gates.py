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
(status `prepared`, `running` or `completed`) only with a frozen
preregistration (`preregistration_errors`):

- its `config.toml` holds a `[preregistration]` table with every key the list
  requires, each a pinned value of its declared type (`int`, `str`, `bool`,
  or a non-empty `int-list` or `str-list`);
- every value of that table has a canonical text
  (`experiment_records.preregistration_canonical`), and the manifest's
  `preregistration_sha256` is its digest;
- a `seeds` key in the table, where there is one, names the manifest's seeds;
- every baseline the list names is pinned at each key path the list names
  for it, and its status is pinned and not `blocked-*`.

A value is pinned unless it is a placeholder: a `must-be-pinned-…` string for
a value still to be chosen, a `must-be-signed-…` string for an owner decision
still to be taken, an empty string, `unconfigured` or `none`. A placeholder is
a string, so it passes as no other type: an owner decision whose value is a
boolean cannot pass while it waits for the owner. The list itself must name
only registered experiments, known types and well-formed baselines, whatever
their status.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_records  # noqa: E402

PREREGISTRATION="experiments/preregistration.toml"
FROZEN={"prepared","running","completed"}
SCALARS={"int":int,"str":str,"bool":bool}
KINDS=set(SCALARS)|{f"{kind}-list" for kind in ("int","str")}
ENTRY_FIELDS={"required","baseline"}
BASELINE_FIELDS={"path","keys","status_key"}
MISSING=object()

def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))

def is_placeholder(value: str) -> bool:
    value=value.strip().lower()
    return not value or "must-be-pinned" in value or value in {"unconfigured","none"}

def is_unsigned(value: str) -> bool:
    """An owner decision the owner has not taken yet."""
    return "must-be-signed" in value.strip().lower()

def is_unset(value) -> bool:
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
    if type(value) is not SCALARS[kind]:
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

def lookup(table: dict, key_path: str):
    """The value at the dotted `key_path` of `table`, or MISSING."""
    value=table
    for part in key_path.split("."):
        if not isinstance(value,dict) or part not in value:
            return MISSING
        value=value[part]
    return value

def is_repository_path(value) -> bool:
    """A relative path that stays inside the repository."""
    if not isinstance(value,str) or not value:
        return False
    path=PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value

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
    if not isinstance(required,dict) or not required:
        errors.append(f"{where} requires no keys")
    else:
        for key,kind in required.items():
            if kind not in KINDS:
                errors.append(f"{where}: key {key} has unknown type {kind!r}")
    baselines=entry.get("baseline",[])
    if not isinstance(baselines,list):
        return errors+[f"{where}: baseline must be an array of tables"]
    for index,baseline in enumerate(baselines):
        name=f"{where}: baseline {index}"
        if not isinstance(baseline,dict):
            errors.append(f"{name} is not a table")
            continue
        for field in sorted(set(baseline)-BASELINE_FIELDS):
            errors.append(f"{name} has unknown field {field}")
        if not is_repository_path(baseline.get("path")):
            errors.append(f"{name} needs a path inside the repository")
        keys=baseline.get("keys")
        if not isinstance(keys,list) or not keys or not all(isinstance(key,str) and key for key in keys):
            errors.append(f"{name} needs a non-empty list of key paths")
        status_key=baseline.get("status_key")
        if not isinstance(status_key,str) or not status_key:
            errors.append(f"{name} needs a status_key")
    return errors

def baseline_errors(exp_id: str, baseline: dict, root: Path) -> list[str]:
    """What keeps `baseline` from being a pinned, unblocked baseline."""
    where=f"{exp_id}: baseline {baseline['path']}"
    path=root/baseline["path"]
    if not path.is_file():
        return [f"{where} does not exist"]
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
    try:
        digest=experiment_records.preregistration_digest(table)
    except ValueError as error:
        errors.append(f"{exp_id}: {error}")
    else:
        recorded=manifest.get("preregistration_sha256")
        if recorded is None:
            errors.append(f"{exp_id}: experiment.toml names no preregistration_sha256")
        elif recorded!=digest:
            errors.append(f"{exp_id}: preregistration_sha256 {recorded!r} is not {digest}, the digest of config.toml's [preregistration]")
    if "seeds" in table and table["seeds"]!=manifest.get("seeds"):
        errors.append(f"{exp_id}: preregistered seeds {table['seeds']!r} are not the manifest's seeds {manifest.get('seeds')!r}")
    for baseline in entry.get("baseline",[]):
        errors.extend(baseline_errors(exp_id,baseline,root))
    return errors

def preregistration_errors(root: Path, manifests: dict[str, dict], experiments: dict[str, Path]) -> list[str]:
    """What keeps the listed experiments that left `planned` from having
    frozen preregistrations, and what is wrong with the list itself."""
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

def main(root: Path = ROOT) -> int:
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
