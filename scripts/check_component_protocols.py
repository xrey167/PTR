"""Check component-evaluation protocols against the evidence and statuses that cite them.

`evaluations/PROTOCOL_FORMAT.md` defines the per-component `protocol.toml`. A
protocol is frozen before the first evidence run, so evidence is only worth
reading if it names the protocol it was produced under. This checker enforces:

* a component's `protocol.toml` exists once any candidate has a benchmark record
  or a status past `to-evaluate`/`evaluating` (`preferred`, `rejected`), parses,
  and matches the schema;
* every benchmark record (a JSON evidence file with a `bench` object) names the
  SHA-256 of the current `protocol.toml`, covers exactly the protocol's gates and
  was measured with at least the protocol's repetitions;
* the protocol is committed no later than the evidence that cites it (git);
* status rules: a candidate with a failed gate is `rejected`; `preferred` needs
  at least two candidates that passed every gate and ran the benchmark, decision
  grade; a lone survivor stays `evaluating` and carries a `note`; `rejected`
  without gate-failure evidence carries a `note` saying why.

It also provides `automated_score`, the scoring rule of the format document, so
the rule is executable and tested rather than prose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The fields scripts/run_component_eval.py stores for a run (base_record, then the
# run fields). The Rust bench kit writes the same list (evidence.rs RECORD_FIELDS).
RECORD_FIELDS = (
    "schema_version", "component", "candidate", "criteria", "git_sha", "platform",
    "python", "candidate_manifest_sha256", "cargo_lock_sha256", "status", "started_at",
    "command", "finished_at", "exit_code", "stdout", "stderr", "launch_error", "duration_ns",
)
EXTENSION_FIELDS = ("protocol_sha256", "bench")

CRITERIA = (
    "correctness", "failure_semantics", "latency", "throughput", "memory", "portability",
    "operability", "integration_cost", "license_fit", "research_flexibility",
)
OWNER_SCORED = ("research_flexibility",)
GATE_CRITERIA = ("correctness", "failure_semantics")
PROXY_CRITERIA = ("license_fit", "integration_cost", "portability")
INTEGRATION_PARTS = ("adapter_loc", "transitive_crates", "build_seconds")
CANDIDATE_KINDS = ("crate", "binary", "distro-package", "python-package", "incumbent")
DIRECTIONS = ("lower", "higher")
MIN_WARMUP = 1
MIN_MEASURED = 5
MIN_LATENCY_SAMPLES = 10_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_toml(path: Path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- protocol schema


def lock_versions(lock: Path) -> dict[str, set[str]]:
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for package in data.get("package", []):
        found.setdefault(package["name"], set()).add(package["version"])
    return found


def validate_protocol(protocol: dict, component: str, root: Path = ROOT) -> list[str]:
    """Schema errors of one parsed protocol (no evidence involved)."""
    errors: list[str] = []

    def need(table: dict, key: str, kind: type, where: str):
        value = table.get(key)
        if not isinstance(value, kind) or isinstance(value, bool) != (kind is bool):
            errors.append(f"{where}: `{key}` must be {kind.__name__}")
            return None
        return value

    if protocol.get("version") != 1:
        errors.append("version must be 1")
    if protocol.get("component") != component:
        errors.append(f"component must be {component!r}, found {protocol.get('component')!r}")
    protocol_id = need(protocol, "protocol_id", str, "protocol")
    if protocol_id is not None and not ID.match(protocol_id):
        errors.append(f"protocol_id {protocol_id!r} is not a lowercase id")
    profile = need(protocol, "hardware_profile", str, "protocol")
    if profile is not None and not (root / profile).is_file():
        errors.append(f"hardware_profile {profile} does not exist")
    workspace = need(protocol, "bench_workspace", str, "protocol")
    if workspace is not None and not (root / workspace / "Cargo.toml").is_file():
        errors.append(f"bench_workspace {workspace} has no Cargo.toml")
    lock = need(protocol, "lockfile", str, "protocol")
    versions: dict[str, set[str]] | None = None
    if lock is not None:
        if not (root / lock).is_file():
            errors.append(f"lockfile {lock} does not exist")
        else:
            versions = lock_versions(root / lock)

    repetitions = protocol.get("repetitions", {})
    warmup, measured = repetitions.get("warmup"), repetitions.get("measured")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < MIN_WARMUP:
        errors.append(f"repetitions.warmup must be an integer >= {MIN_WARMUP}")
    if not isinstance(measured, int) or isinstance(measured, bool) or measured < MIN_MEASURED:
        errors.append(f"repetitions.measured must be an integer >= {MIN_MEASURED}")

    workload = protocol.get("workload", {})
    if not isinstance(workload.get("description"), str) or not workload.get("description", "").strip():
        errors.append("workload.description must be a non-empty string")
    seeds = workload.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)):
        errors.append("workload.seeds must be a non-empty list of integers")

    gates = protocol.get("gate", [])
    if not gates:
        errors.append("at least one [[gate]] is required")
    gate_ids: set[str] = set()
    for gate in gates:
        gid = gate.get("id")
        if not isinstance(gid, str) or not ID.match(gid):
            errors.append(f"gate id {gid!r} is not a lowercase id")
            continue
        if gid in gate_ids:
            errors.append(f"duplicate gate id {gid}")
        gate_ids.add(gid)
        if gate.get("criterion") not in GATE_CRITERIA:
            errors.append(f"gate {gid}: criterion must be one of {GATE_CRITERIA}")
        if not isinstance(gate.get("description"), str) or not gate["description"].strip():
            errors.append(f"gate {gid}: description is required")

    metrics = protocol.get("metric", [])
    metric_ids: set[str] = set()
    samples = workload.get("samples")
    for metric in metrics:
        mid = metric.get("id")
        if not isinstance(mid, str) or not ID.match(mid):
            errors.append(f"metric id {mid!r} is not a lowercase id")
            continue
        if mid in metric_ids:
            errors.append(f"duplicate metric id {mid}")
        metric_ids.add(mid)
        criterion = metric.get("criterion")
        if criterion in OWNER_SCORED:
            errors.append(f"metric {mid}: {criterion} is owner-scored and cannot have a metric")
        elif criterion in GATE_CRITERIA or criterion in PROXY_CRITERIA:
            errors.append(f"metric {mid}: {criterion} is scored by gates/proxies, not a metric")
        elif criterion not in CRITERIA:
            errors.append(f"metric {mid}: unknown criterion {criterion!r}")
        if metric.get("direction") not in DIRECTIONS:
            errors.append(f"metric {mid}: direction must be one of {DIRECTIONS}")
        if not isinstance(metric.get("unit"), str) or not metric.get("unit"):
            errors.append(f"metric {mid}: unit is required")
        if criterion == "latency" and (not isinstance(samples, int) or samples < MIN_LATENCY_SAMPLES):
            errors.append(
                f"metric {mid}: a latency metric needs workload.samples >= {MIN_LATENCY_SAMPLES}"
            )
    if not metrics:
        errors.append("at least one [[metric]] is required")

    scoring = protocol.get("scoring", {})
    if scoring.get("weights_from") != "config.toml":
        errors.append('scoring.weights_from must be "config.toml"')
    if sorted(scoring.get("excluded", [])) != sorted(OWNER_SCORED):
        errors.append(f"scoring.excluded must be exactly {list(OWNER_SCORED)}")
    if scoring.get("method") != "ratio-to-best":
        errors.append('scoring.method must be "ratio-to-best"')
    proxies = scoring.get("proxies", {})
    if proxies.get("license_fit", {}).get("source") != "cargo-deny":
        errors.append('scoring.proxies.license_fit.source must be "cargo-deny"')
    if list(proxies.get("integration_cost", {}).get("parts", [])) != list(INTEGRATION_PARTS):
        errors.append(f"scoring.proxies.integration_cost.parts must be {list(INTEGRATION_PARTS)}")
    targets = proxies.get("portability", {}).get("targets")
    if not isinstance(targets, list) or len(targets) < 2 or not all(isinstance(t, str) for t in targets):
        errors.append("scoring.proxies.portability.targets must list at least two target triples")

    candidates = protocol.get("candidate", [])
    if len(candidates) < 2:
        errors.append("a protocol compares at least two candidates")
    seen: set[str] = set()
    for item in candidates:
        cid = item.get("id")
        if not isinstance(cid, str) or not ID.match(cid):
            errors.append(f"candidate id {cid!r} is not a lowercase id")
            continue
        if cid in seen:
            errors.append(f"duplicate candidate {cid}")
        seen.add(cid)
        kind = item.get("kind")
        version = item.get("version")
        if kind not in CANDIDATE_KINDS:
            errors.append(f"candidate {cid}: kind must be one of {CANDIDATE_KINDS}")
        if not isinstance(version, str) or not version:
            errors.append(f"candidate {cid}: version is required")
        if kind == "crate":
            package = item.get("lock_package")
            if not isinstance(package, str):
                errors.append(f"candidate {cid}: crate candidates name their lock_package")
            elif versions is not None and version not in versions.get(package, set()):
                errors.append(
                    f"candidate {cid}: {package} {version} is not in {lock} "
                    f"(locked: {sorted(versions.get(package, []))})"
                )
        elif kind in ("binary", "distro-package", "python-package"):
            if not SHA256.match(str(item.get("sha256", ""))):
                errors.append(f"candidate {cid}: {kind} candidates pin a 64-hex sha256")
            if kind == "binary" and not str(item.get("url", "")).startswith("https://"):
                errors.append(f"candidate {cid}: binary candidates pin an https url")
    return errors


# ---------------------------------------------------------------- scoring rule


def automated_score(weights: dict[str, float], protocol: dict, measured: dict[str, dict]) -> dict:
    """Score survivors by the format document's rule.

    `measured[candidate]` is {"metrics": {id: value}, "proxies": {"license_fit": bool,
    "integration_cost": {part: value}, "portability": {target: bool}}}. Every candidate
    given is a survivor (gates passed, benchmark ran). Returns {candidate:
    {"score": float, "criteria": {criterion: score}, "unscored": [criterion]}}.

    * research_flexibility is excluded (owner-scored);
    * correctness and failure_semantics are gate criteria: a survivor scores 1.0;
    * a metric scores ratio-to-best (best/value for lower, value/best for higher);
      a criterion with several metrics takes their mean;
    * license_fit is 1.0 on a clean cargo-deny verdict, else 0.0;
    * integration_cost is the mean ratio-to-best (lower is better) of the parts;
    * portability is the fraction of targets where cargo check passed;
    * the total renormalises the weights of the criteria that have a basis, and
      names the ones that do not in `unscored`.
    """
    direction = {m["id"]: m["direction"] for m in protocol.get("metric", [])}
    metric_criterion = {m["id"]: m["criterion"] for m in protocol.get("metric", [])}
    names = list(measured)

    def ratio(values: dict[str, float], lower: bool) -> dict[str, float]:
        usable = {k: v for k, v in values.items() if v is not None}
        if not usable:
            return {}
        if lower:
            best = min(usable.values())
            return {k: (1.0 if v == best else (best / v if v > 0 else 0.0)) for k, v in usable.items()}
        best = max(usable.values())
        return {k: (v / best if best > 0 else 0.0) for k, v in usable.items()}

    per: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    for mid, dirn in direction.items():
        values = {n: measured[n].get("metrics", {}).get(mid) for n in names}
        for n, score in ratio(values, dirn == "lower").items():
            per[n].setdefault(metric_criterion[mid], []).append(score)
    for part in INTEGRATION_PARTS:
        values = {n: measured[n].get("proxies", {}).get("integration_cost", {}).get(part) for n in names}
        for n, score in ratio(values, True).items():
            per[n].setdefault("integration_cost", []).append(score)
    for n in names:
        proxies = measured[n].get("proxies", {})
        if "license_fit" in proxies:
            per[n]["license_fit"] = [1.0 if proxies["license_fit"] else 0.0]
        targets = proxies.get("portability")
        if targets:
            per[n]["portability"] = [sum(1 for ok in targets.values() if ok) / len(targets)]
        for criterion in GATE_CRITERIA:
            per[n][criterion] = [1.0]

    out = {}
    for n in names:
        criteria = {c: sum(v) / len(v) for c, v in per[n].items() if v and c not in OWNER_SCORED}
        total_weight = sum(weights[c] for c in criteria)
        score = sum(weights[c] * s for c, s in criteria.items()) / total_weight if total_weight else 0.0
        unscored = sorted(c for c in weights if c not in criteria)
        out[n] = {"score": score, "criteria": criteria, "unscored": unscored}
    return out


# ---------------------------------------------------------------- evidence rules


def git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def first_commit(root: Path, path: Path) -> str | None:
    """The commit that first added `path`, or None when it is untracked."""
    rel = path.relative_to(root).as_posix()
    if git(root, "ls-files", "--error-unmatch", rel) is None:
        return None
    log = git(root, "log", "--diff-filter=A", "--format=%H", "--", rel)
    return log.splitlines()[-1] if log else None


def protocol_precedes(root: Path, protocol: Path, evidence: Path) -> str | None:
    """An error text when the protocol was not committed before/with the evidence."""
    evidence_commit = first_commit(root, evidence)
    if evidence_commit is None:
        return None  # uncommitted evidence cannot predate anything
    protocol_commit = first_commit(root, protocol)
    if protocol_commit is None:
        return "evidence is committed but its protocol.toml is not"
    if protocol_commit == evidence_commit:
        return "protocol.toml and its evidence are in the same commit; freeze the protocol first"
    if git(root, "merge-base", "--is-ancestor", protocol_commit, evidence_commit) is None:
        return "protocol.toml was committed after (or off the history of) the evidence"
    return None


def read_record(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_benchmark_record(record: dict | None) -> bool:
    return bool(record) and ("bench" in record or "protocol_sha256" in record)


def candidate_state(record: dict, protocol: dict) -> tuple[str, list[str]]:
    """('gate_failed' | 'passed' | 'incomplete', problems) for one benchmark record."""
    problems: list[str] = []
    bench = record.get("bench")
    if not isinstance(bench, dict):
        return "incomplete", ["record has no `bench` object"]
    expected = {g["id"] for g in protocol.get("gate", [])}
    gates = bench.get("gates")
    if not isinstance(gates, dict) or not all(isinstance(v, bool) for v in gates.values()):
        return "incomplete", ["bench.gates must map gate ids to booleans"]
    if set(gates) != expected:
        problems.append(f"bench.gates {sorted(gates)} differ from the protocol's {sorted(expected)}")
    if any(v is False for v in gates.values()):
        return "gate_failed", problems
    if problems:
        return "incomplete", problems
    reps = bench.get("repetitions") or {}
    wanted = protocol.get("repetitions", {})
    if not bench.get("benchmark_ran"):
        return "incomplete", ["benchmark did not run"]
    if reps.get("warmup", 0) < wanted.get("warmup", MIN_WARMUP) or reps.get("measured", 0) < wanted.get("measured", MIN_MEASURED):
        return "incomplete", [f"repetitions {reps} below the protocol's {wanted}"]
    return "passed", []


def check_component(root: Path, name: str, use_git: bool = True) -> list[str]:
    errors: list[str] = []
    base = root / "evaluations/components" / name
    candidates_path = base / "candidates.toml"
    if not candidates_path.is_file():
        return [f"{name}: no candidates.toml"]
    data = load_toml(candidates_path)
    candidates = data.get("candidate", [])
    protocol_path = base / "protocol.toml"
    protocol = None
    protocol_hash = None
    profile_grade = None
    if protocol_path.is_file():
        try:
            protocol = load_toml(protocol_path)
        except tomllib.TOMLDecodeError as error:
            return [f"{name}: protocol.toml does not parse: {error}"]
        protocol_hash = sha256_file(protocol_path)
        errors += [f"{name}: protocol.toml: {e}" for e in validate_protocol(protocol, name, root)]
        profile = root / str(protocol.get("hardware_profile", ""))
        if profile.is_file():
            try:
                profile_grade = load_toml(profile).get("decision_grade")
            except tomllib.TOMLDecodeError:
                profile_grade = None
        ids = {c.get("id") for c in candidates}
        for item in protocol.get("candidate", []):
            if item.get("id") not in ids:
                errors.append(f"{name}: protocol candidate {item.get('id')} is not in candidates.toml")
        weights = {k: float(v) for k, v in data.get("criteria", {}).items()}
        if set(weights) != set(CRITERIA):
            errors.append(f"{name}: config criteria differ from the standard ten: {sorted(weights)}")

    # Collect benchmark records: listed in `evidence` or present in evidence/.
    records: dict[str, list[tuple[Path, dict]]] = {}
    seen_paths: set[Path] = set()
    for item in candidates:
        cid = item.get("id")
        paths = [(base / rel).resolve() for rel in item.get("evidence", []) if str(rel).endswith(".json")]
        paths += [
            p.resolve() for p in sorted((base / "evidence").glob(f"*-{cid}-run.json"))
        ]
        for path in dict.fromkeys(paths):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not path.is_file():
                errors.append(f"{name}/{cid}: evidence {path.relative_to(root.resolve()) if root.resolve() in path.parents else path} is missing")
                continue
            record = read_record(path)
            if record is None:
                errors.append(f"{name}/{cid}: evidence {path.name} is not a JSON object")
                continue
            if not is_benchmark_record(record):
                continue  # legacy runner records and smoke files predate protocols
            records.setdefault(cid, []).append((path, record))

    states: dict[str, str] = {}
    for item in candidates:
        cid = item["id"]
        status = item.get("status")
        mine = records.get(cid, [])
        if mine and protocol is None:
            errors.append(f"{name}/{cid}: benchmark evidence exists but the component has no protocol.toml")
            continue
        latest_state = None
        for path, record in mine:
            missing = [f for f in RECORD_FIELDS + EXTENSION_FIELDS if f not in record]
            if missing:
                errors.append(f"{name}/{cid}: {path.name} lacks fields {missing}")
            if record.get("protocol_sha256") != protocol_hash:
                errors.append(
                    f"{name}/{cid}: {path.name} names protocol "
                    f"{str(record.get('protocol_sha256'))[:12]}, current is {str(protocol_hash)[:12]}"
                )
            if (record.get("candidate") or {}).get("id") != cid:
                errors.append(f"{name}/{cid}: {path.name} records a different candidate")
            if use_git and protocol_path.is_file():
                problem = protocol_precedes(root, protocol_path, path)
                if problem:
                    errors.append(f"{name}/{cid}: {path.name}: {problem}")
        if mine:
            path, record = max(mine, key=lambda pr: pr[0].name)
            latest_state, problems = candidate_state(record, protocol)
            errors += [f"{name}/{cid}: {path.name}: {p}" for p in problems]
            claimed = (record.get("bench") or {}).get("decision_grade", False)
            if claimed and profile_grade is not True:
                errors.append(
                    f"{name}/{cid}: {path.name} claims decision_grade but the protocol's hardware "
                    f"profile is not decision_grade (decision_grade = {profile_grade!r})"
                )
                claimed = False
            if latest_state == "passed" and not claimed:
                states[cid] = "passed-not-decision-grade"
            else:
                states[cid] = latest_state
        if states.get(cid) == "gate_failed" and status != "rejected":
            errors.append(f"{name}/{cid}: a gate failed, so the status must be rejected (is {status})")
        if status == "rejected" and states.get(cid) != "gate_failed" and not str(item.get("note", "")).strip():
            errors.append(f"{name}/{cid}: rejected without gate-failure evidence needs a `note` with the reason")
        if status == "preferred" and protocol is None:
            errors.append(f"{name}/{cid}: preferred needs a protocol.toml and benchmark evidence")

    survivors = [c for c, s in states.items() if s in ("passed", "passed-not-decision-grade")]
    decision_grade = [c for c, s in states.items() if s == "passed"]
    for item in candidates:
        cid, status = item["id"], item.get("status")
        if status == "preferred" and protocol is not None:
            if states.get(cid) != "passed":
                reason = "is not decision-grade" if states.get(cid) == "passed-not-decision-grade" else "has not passed every gate and run the benchmark"
                errors.append(f"{name}/{cid}: preferred but its latest benchmark record {reason}")
            if len(decision_grade) < 2:
                errors.append(
                    f"{name}/{cid}: preferred needs >= 2 candidates that passed the gates and ran "
                    f"the benchmark (decision grade), found {len(decision_grade)}"
                )
        if (len(survivors) == 1 and survivors[0] == cid and status == "evaluating"
                and not str(item.get("note", "")).strip()):
            errors.append(f"{name}/{cid}: lone survivor stays evaluating and needs a `note` saying so")
    return errors


def components(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / "evaluations/components").glob("*/candidates.toml"))


def check(root: Path = ROOT, use_git: bool = True) -> list[str]:
    errors: list[str] = []
    for name in components(root):
        errors += check_component(root, name, use_git)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-git", action="store_true", help="skip the protocol-before-evidence commit order")
    args = parser.parse_args()
    errors = check(args.root, use_git=not args.no_git)
    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        return 1
    count = len(list((args.root / "evaluations/components").glob("*/protocol.toml")))
    print(f"OK: {count} component protocols, {len(components(args.root))} components checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
