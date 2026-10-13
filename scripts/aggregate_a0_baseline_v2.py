"""Decider of the A0 baseline study v2 (M001-v6, M002-v11, M003-v3 against plain).

v2 replaces study v1 for a procedural defect of its decider, not for an outcome.
The v1 decider (scripts/aggregate_a0_baseline.py, unchanged and pinned) demands
that the five run records of a study name the same `git_sha`. The hardened
runner (scripts/run_experiment.py) commits each record before the next run, so
the records of a correctly executed study name five different commits and the v1
decider refuses before it computes anything. v2 keeps the v1 statistics, gates,
verdict rules and refusals (it imports and reuses them) and changes exactly two
things, both preregistered in research/falsification/A0-baseline-v2/criteria.toml:

1. Record identity. The records may name different commits iff all of them lie
   on one line of HEAD's history (each, in run order, an ancestor of the next or
   equal to it) and `git diff` between the earliest and the latest record commit
   touches only files directly inside the results/ directories of this study's own
   experiments. Every other identity field (manifest_sha256, parameters, rustc,
   toolchain, environment, executable, host, lock digests, the manifest's
   preregistration digests) must still be identical across the five records.
   Identical commits are accepted without consulting git, as in v1.
2. Interruption. A confirmatory seed whose run record has status "started" and
   neither output nor exit code (the process or its host was killed) makes the
   study INCONCLUSIVE-INTERRUPTED: a named, reported outcome that computes no
   statistic and is never rerun, instead of an unhandled refusal.

Everything else is v1: the decision rules (SUPPORTS, FALSIFIES/HARMFUL,
NO-GO-CALIBRATION, INCONCLUSIVE), the nine gates, exact float64 comparisons, the
refusal of evidence that cannot support a decision (missing seed, fold, arm or
split, non-finite value, protocol or digest mismatch, mixed builds, a failed
run), exit code 0 for any decision and 2 for a refusal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "scripts/aggregate_a0_baseline.py"
CRITERIA_PATH = ROOT / "research/falsification/A0-baseline-v2/criteria.toml"
NAME = "A0-baseline-v2"
INTERRUPTED_LABEL = "INCONCLUSIVE-INTERRUPTED"
IDENTITY_MODE = "single-commit-or-linear-history-results-only"
COMMIT = re.compile(r"[0-9a-f]{40}")
# The identity fields of the core other than git_sha, plus the manifest digests
# the v1 decider binds; criteria.toml [record_identity] lists exactly these.
IDENTICAL_FIELDS = (
    "manifest_sha256", "parameters", "rustc", "toolchain", "environment", "executable", "host",
    "cargo_lock_sha256", "uv_lock_sha256", "manifest.preregistration_sha256",
    "manifest.preregistration_rules_sha256",
)


def load_v1():
    spec = importlib.util.spec_from_file_location("_a0_baseline_v1_decider", V1_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the v1 decider")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = load_v1()
CORE = V1.CORE
EvidenceError = V1.EvidenceError
Table = V1.Table
expected_meta = V1.expected_meta
parse_lr_file = V1.parse_lr_file
toml_lines = V1.toml_lines
toml_scalar = V1.toml_scalar
sha256_bytes = V1.sha256_bytes
json_text = V1.json_text
write_exclusive = V1.write_exclusive


def script_sha256() -> str:
    return sha256_bytes(Path(__file__).read_bytes())


# ------------------------------------------------------------------ criteria


def load_criteria(path: Path = CRITERIA_PATH, root: Path = ROOT) -> dict:
    """The v1 loader and validation on v2's criteria, plus the v2 tables: the
    pinned v1 decider, the interrupted label and the record identity rule."""
    criteria = V1.load_criteria(path, root)
    data = criteria["raw"]
    if data.get("name") != NAME:
        raise EvidenceError(f"criteria: name is not {NAME}")
    decision = V1.need(data, "decision", dict, "root")
    if V1.need(decision, "base", str, "decision") != "scripts/aggregate_a0_baseline.py":
        raise EvidenceError("criteria: the pinned v1 decider must be scripts/aggregate_a0_baseline.py")
    if sha256_bytes(V1_PATH.read_bytes()) != V1.need(decision, "base_sha256", str, "decision"):
        raise EvidenceError("criteria: the v1 decider's SHA-256 differs from the pin")
    rules = V1.need(data, "decision_rules", dict, "root")
    if V1.need(rules, "interrupted_label", str, "decision_rules") != INTERRUPTED_LABEL:
        raise EvidenceError(f"criteria: interrupted_label is not {INTERRUPTED_LABEL}")
    identity = V1.need(data, "record_identity", dict, "root")
    if V1.need(identity, "mode", str, "record_identity") != IDENTITY_MODE:
        raise EvidenceError(f"criteria: record_identity.mode is not {IDENTITY_MODE}")
    if tuple(V1.need(identity, "identical_fields", list, "record_identity")) != IDENTICAL_FIELDS:
        raise EvidenceError("criteria: record_identity.identical_fields are not the fields the decider checks")
    return criteria


# ------------------------------------------------------------------ record identity


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    except OSError as error:
        raise EvidenceError(f"run records disagree in git_sha: git cannot run: {error}") from None


def git_text(repo: Path, *args: str) -> str:
    done = run_git(repo, *args)
    if done.returncode != 0:
        raise EvidenceError(
            f"run records disagree in git_sha: git {' '.join(args[:2])} failed: {done.stderr.decode('utf-8', 'replace').strip()}"
        )
    return done.stdout.decode("utf-8", "surrogateescape")


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    done = run_git(repo, "merge-base", "--is-ancestor", older, newer)
    if done.returncode not in (0, 1):
        raise EvidenceError(
            f"run records disagree in git_sha: cannot compare {older[:12]} with {newer[:12]}: "
            f"{done.stderr.decode('utf-8', 'replace').strip()}"
        )
    return done.returncode == 0


def results_path_pattern(criteria: dict) -> re.Pattern:
    """The files a record commit may add: directly inside `results/` of an
    experiment directory of this study (`experiments/model/<study id>-<slug>/`,
    sweeps included)."""
    ids = "|".join(re.escape(study_id) for study_id in sorted(criteria["studies"]))
    return re.compile(rf"experiments/model/(?:{ids})-[A-Za-z0-9][A-Za-z0-9-]*/results/[^/]+")


def run_order(records: list[tuple[str, dict]], need_time: bool) -> list[tuple[str, dict]]:
    """The records in the order they ran: by `started_at` (the runner's compact
    UTC stamp), else (identical commits only) by seed."""
    stamps = [record.get("started_at") for _, record in records]
    if all(isinstance(stamp, str) and stamp for stamp in stamps):
        if len(set(stamps)) != len(stamps):
            raise EvidenceError("run records disagree in git_sha: two records share a started_at")
        return sorted(records, key=lambda item: item[1]["started_at"])
    if need_time:
        raise EvidenceError("run records disagree in git_sha: a record has no started_at, so the run order is unknown")
    return sorted(records, key=lambda item: item[1].get("seed", 0))


def check_record_commits(records: list[tuple[str, dict]], criteria: dict, repo: Path | None) -> list[str]:
    """The record-identity rule for git_sha (criteria.toml [record_identity]).
    Returns the commits in run order; raises EvidenceError otherwise."""
    for name, record in records:
        if not isinstance(record.get("git_sha"), str) or not COMMIT.fullmatch(record["git_sha"]):
            raise EvidenceError(f"run records disagree in git_sha: {name} names no commit: {record.get('git_sha')!r}")
    if len({record["git_sha"] for _, record in records}) == 1:
        return [records[0][1]["git_sha"]] * len(records)
    ordered = run_order(records, need_time=True)
    commits = [record["git_sha"] for _, record in ordered]
    if repo is None:
        raise EvidenceError("run records disagree in git_sha: different commits cannot be verified without a repository")
    head = git_text(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    for name, record in ordered:
        sha = record["git_sha"]
        if run_git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            raise EvidenceError(f"run records disagree in git_sha: {name} names commit {sha}, which the repository does not hold")
        if not is_ancestor(repo, sha, head):
            raise EvidenceError(f"run records disagree in git_sha: {name} names commit {sha}, which is not in HEAD's history")
    for (_, before), (name, after) in zip(ordered, ordered[1:]):
        if not is_ancestor(repo, before["git_sha"], after["git_sha"]):
            raise EvidenceError(
                f"run records disagree in git_sha: {before['git_sha'][:12]} is not an ancestor of {after['git_sha'][:12]} "
                f"({name}): the records are not on one line of history"
            )
    earliest, latest = commits[0], commits[-1]
    raw = git_text(repo, "diff", "--name-only", "-z", "--no-renames", earliest, latest)
    allowed = results_path_pattern(criteria)
    outside = sorted(path for path in raw.split("\0") if path and not allowed.fullmatch(path))
    if outside:
        shown = ", ".join(outside[:5]) + (f" and {len(outside) - 5} more" if len(outside) > 5 else "")
        raise EvidenceError(
            f"run records disagree in git_sha: between {earliest[:12]} and {latest[:12]} the repository changed outside "
            f"the results/ directories of the study's experiments: {shown}"
        )
    return commits


def build_table(
    sources: list[tuple[str, object]], criteria: dict, study: dict, allow_rows: bool = False, repo: Path | None = ROOT
) -> tuple[dict[int, dict], str, list[dict]]:
    """V1's `build_table` with the v2 record-identity rule: admit all sources,
    check cross-record identity and consistency, return the cell table."""
    if len(sources) != len(CORE.SEEDS):
        raise EvidenceError(f"expected exactly {len(CORE.SEEDS)} run records, found {len(sources)}")
    kinds = set()
    by_seed: dict[int, dict] = {}
    record_fields = [field for field in CORE.IDENTITY_FIELDS if field != "git_sha"]
    identities: dict[str, set] = {field: set() for field in record_fields}
    records: list[tuple[str, dict]] = []
    for name, source in sources:
        seed, kind, record = V1.admit_record(name, source, study, allow_rows)
        kinds.add(kind)
        if len(kinds) != 1:
            raise EvidenceError("run records and bare row files cannot be mixed")
        rows = V1.split_rows(record["stdout"], name) if kind == "run-record" else V1.row_file_rows(name, record)
        run_seed, cells = V1.index_rows(name, rows, seed, criteria, study)
        if run_seed in by_seed:
            raise EvidenceError(f"seed {run_seed} has duplicate records")
        by_seed[run_seed] = cells
        if kind == "run-record":
            records.append((name, record))
            for field in record_fields:
                identities[field].add(CORE.canonical(record[field]))
            manifest = record["manifest"]
            for field in ("preregistration_sha256", "preregistration_rules_sha256"):
                if not isinstance(manifest.get(field), str) or len(manifest[field]) != 64:
                    raise EvidenceError(f"{name}: recorded manifest has no bound {field}")
                identities.setdefault(f"manifest.{field}", set()).add(manifest[field])
    missing = sorted(set(CORE.SEEDS) - set(by_seed))
    if missing:
        raise EvidenceError(f"missing seeds: {missing}")
    if kinds == {"run-record"}:
        for field, values in identities.items():
            if len(values) != 1:
                raise EvidenceError(f"run records disagree in {field}")
        check_record_commits(records, criteria, repo)
    V1.check_cross_run_consistency(by_seed, criteria, study)
    digest = [
        {"path": name, "canonical_sha256": sha256_bytes(CORE.canonical(source).encode("utf-8"))}
        for name, source in sorted(sources, key=lambda item: item[0])
    ]
    return by_seed, kinds.pop(), digest


# ------------------------------------------------------------------ interruption


def is_interrupted(source: object) -> bool:
    """A run record the runner reserved and nothing finished: status "started",
    no stdout, no exit code."""
    return (
        isinstance(source, dict)
        and "stdout" not in source
        and source.get("status") == "started"
        and source.get("exit_code") is None
    )


def interrupted_result(sources: list[tuple[str, object]], criteria: dict, study: dict) -> dict:
    """The study's outcome when a confirmatory seed was interrupted: named,
    reported, no statistic computed (the completed seeds' rows are not read).
    Refuses records that are not this study's hardened records."""
    if not 1 <= len(sources) <= len(CORE.SEEDS):
        raise EvidenceError(f"expected between 1 and {len(CORE.SEEDS)} run records, found {len(sources)}")
    seeds: dict[int, str] = {}
    for name, source in sources:
        if not isinstance(source, dict) or "rows" in source:
            raise EvidenceError(f"{name}: an interrupted study admits hardened run records only")
        if source.get("experiment_id") != study["id"]:
            raise EvidenceError(f"{name}: experiment_id is not {study['id']}")
        manifest = source.get("manifest")
        if source.get("entrypoint") != "entrypoint" or not isinstance(manifest, dict) or manifest.get("id") != study["id"]:
            raise EvidenceError(f"{name}: not a record of the frozen entrypoint of {study['id']}")
        seed = source.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in CORE.SEEDS:
            raise EvidenceError(f"{name}: seed {seed!r} is not confirmatory")
        if seed in seeds:
            raise EvidenceError(f"seed {seed} has duplicate records")
        if is_interrupted(source):
            seeds[seed] = "interrupted"
        elif source.get("status") == "completed" and source.get("exit_code") == 0:
            seeds[seed] = "completed"
        else:
            raise EvidenceError(f"{name}: run did not complete successfully")
    contrasts = []
    for item in study["contrast"]:
        contrasts.append({
            "id": item["id"], "role": item["role"], "isolating": item["isolating"],
            "treatment": item["treatment"], "control": item["control"], "compute": item["compute"],
            "claim": item["claim"], "verdict": "INCONCLUSIVE", "label": INTERRUPTED_LABEL,
            "reason": "a confirmatory seed was interrupted (record started, no output, no exit code); "
                      "no statistic is computed and the study is never rerun",
            "failing_gates": [], "gates": {}, "gate_details": {}, "learnable": {},
            "quality_claim_allowed": False,
            "scope_note": "no outcome: the study was interrupted", "endpoints": [],
        })
    headline = next(item for item in contrasts if item["id"] == study["headline"])
    return {
        "schema_version": V1.SCHEMA_VERSION,
        "experiment_id": study["id"],
        "study": criteria["raw"]["name"],
        "evidence_kind": "run-record",
        "outcome": INTERRUPTED_LABEL,
        "seeds": list(CORE.SEEDS),
        "folds": [fold for fold, _ in criteria["folds"]],
        "splits": {"iid": criteria["data"]["iid_split"], "ood": criteria["data"]["ood_split"]},
        "t_critical_used": CORE.T_CRITICAL_95_DF4,
        "interrupted": {
            "seeds_interrupted": sorted(s for s, kind in seeds.items() if kind == "interrupted"),
            "seeds_completed": sorted(s for s, kind in seeds.items() if kind == "completed"),
            "seeds_not_run": sorted(set(CORE.SEEDS) - set(seeds)),
        },
        "headline": {"contrast": headline["id"], "verdict": headline["verdict"], "label": headline["label"]},
        "contrasts": contrasts,
        "arms": {},
        "mask_share": None,
        "diagnostics": {"note": "interrupted study: no statistic was computed and no completed seed's rows were read"},
        "provenance": provenance(sources, criteria),
    }


# ------------------------------------------------------------------ decision


def provenance(sources: list[tuple[str, object]], criteria: dict) -> dict:
    records = [
        (name, source) for name, source in sources
        if isinstance(source, dict) and isinstance(source.get("git_sha"), str)
    ]
    ordered = run_order(records, need_time=False) if records else []
    shas = [source["git_sha"] for _, source in ordered]
    return {
        "git_sha": shas[0] if shas else None,
        "record_commits": [{"seed": source.get("seed"), "git_sha": source["git_sha"]} for _, source in ordered],
        "record_identity": criteria["raw"]["record_identity"]["mode"],
        "criteria_sha256": criteria["sha256"],
        "decision_script_sha256": script_sha256(),
        "decision_base_sha256": sha256_bytes(V1_PATH.read_bytes()),
        "decision_core_sha256": sha256_bytes(V1.CORE_PATH.read_bytes()),
        "dataset_lock_sha256": criteria["raw"]["data"]["lock_sha256"],
        "source_records": [
            {"path": name, "canonical_sha256": sha256_bytes(CORE.canonical(source).encode("utf-8"))}
            for name, source in sorted(sources, key=lambda item: item[0])
        ],
    }


def decide(
    sources: list[tuple[str, object]],
    study_id: str,
    criteria: dict | None = None,
    *,
    allow_rows: bool = False,
    lr_text: str | None = None,
    repo: Path | None = ROOT,
) -> dict:
    """Decide every contrast of one study (v1 rules), or name the study
    INCONCLUSIVE-INTERRUPTED; malformed evidence is a hard refusal."""
    criteria = load_criteria() if criteria is None else criteria
    study = V1.study_of(criteria, study_id)
    if any(is_interrupted(source) for _, source in sources):
        return interrupted_result(sources, criteria, study)
    by_seed, kind, _ = build_table(sources, criteria, study, allow_rows, repo)
    if lr_text is not None:
        V1.check_lr_file(lr_text, by_seed, criteria, study)
    table = Table(by_seed, criteria)
    contrasts = [V1.decide_contrast(table, item, criteria) for item in study["contrast"]]
    headline = next(item for item in contrasts if item["id"] == study["headline"])
    first = next(iter(by_seed.values()))
    evidence = provenance(sources, criteria)
    return {
        "schema_version": V1.SCHEMA_VERSION,
        "experiment_id": study["id"],
        "study": criteria["raw"]["name"],
        "evidence_kind": kind,
        "seeds": list(CORE.SEEDS),
        "folds": [fold for fold, _ in criteria["folds"]],
        "splits": {"iid": table.iid, "ood": table.ood},
        "t_critical_used": CORE.T_CRITICAL_95_DF4,
        "headline": {"contrast": headline["id"], "verdict": headline["verdict"], "label": headline["label"]},
        "contrasts": contrasts,
        "arms": V1.arm_table(table, study, criteria),
        "mask_share": V1.mask_share(table, study),
        "diagnostics": {
            "temperature_cap": criteria["columns"]["temperature_cap"],
            "note": (
                "fitted_* columns are labelled diagnostics; the fit grid ends at the cap and may saturate. "
                "FLOP columns are descriptive (A0 and plain helpers are not comparable)."
            ),
            "data_fnv64": {fold: first[("data-v2", fold, None, None)]["data_fnv64"] for fold in table.folds},
        },
        "provenance": evidence,
    }


# ------------------------------------------------------------------ outputs


def paired_ci_document(result: dict) -> dict:
    document = V1.paired_ci_document(result)
    if result.get("outcome") == INTERRUPTED_LABEL:
        document["status"] = "not-computed-interrupted"
        document["interrupted"] = result["interrupted"]
    return document


def record_identity_lines(result: dict) -> list[str]:
    evidence = result["provenance"]
    lines = ["", "[record_identity]"]
    toml_lines("record_identity", {
        "mode": evidence["record_identity"],
        "decision_base_sha256": evidence["decision_base_sha256"],
        "commits": [item["git_sha"] for item in evidence["record_commits"]],
    }, lines)
    return lines


def decision_toml(result: dict, paired_ci_sha256: str) -> str:
    evidence = result["provenance"]
    if result.get("outcome") != INTERRUPTED_LABEL:
        text = V1.decision_toml(result, paired_ci_sha256)
        return text.rstrip("\n") + "\n" + "\n".join(record_identity_lines(result)) + "\n"
    headline = result["headline"]
    top = {
        "version": 1,
        "experiment_id": result["experiment_id"],
        "study": result["study"],
        "headline_contrast": headline["contrast"],
        "verdict": headline["verdict"],
        "label": headline["label"],
        "go": False,
        "claim_scope": "none: the study was interrupted and produced no outcome",
        "evidence_kind": result["evidence_kind"],
        "records_immutable": True,
        "paired_ci_sha256": paired_ci_sha256,
        "criteria_sha256": evidence["criteria_sha256"],
        "decision_script_sha256": evidence["decision_script_sha256"],
        "decision_core_sha256": evidence["decision_core_sha256"],
        "git_sha": evidence["git_sha"],
    }
    lines: list[str] = []
    toml_lines("", top, lines)
    lines += ["", "[reporting]"]
    toml_lines("reporting", {
        "required_label": headline["label"],
        "allow_quality_claim": False,
        "allow_bundle_claim_as_mechanism_evidence": False,
        "rerun_allowed": False,
    }, lines)
    lines += ["", "[interrupted]"]
    toml_lines("interrupted", result["interrupted"], lines)
    for item in result["contrasts"]:
        lines += ["", "[[contrast]]"]
        toml_lines("contrast", {
            key: item[key] for key in (
                "id", "role", "isolating", "treatment", "control", "compute", "verdict", "label", "reason",
                "quality_claim_allowed", "scope_note", "claim",
            )
        }, lines)
    for record in evidence["source_records"]:
        lines += ["", "[[source_record]]"]
        lines += [f"path = {toml_scalar(record['path'])}", f"canonical_sha256 = {toml_scalar(record['canonical_sha256'])}"]
    lines += record_identity_lines(result)
    return "\n".join(lines) + "\n"


def write_outputs(result: dict, paired_ci: Path, decision: Path) -> str:
    """Write paired-ci-95.json and DECISION.toml (exclusively); return the paired-CI SHA-256."""
    for path in (paired_ci, decision):
        if path.exists():
            raise EvidenceError(f"{path} exists; outputs are immutable")
    ci_text = json_text(paired_ci_document(result))
    digest = sha256_bytes(ci_text.encode("utf-8"))
    text = decision_toml(result, digest)
    tomllib.loads(text)
    write_exclusive(paired_ci, ci_text)
    write_exclusive(decision, text)
    return digest


def publish_results(result: dict, results_dir: Path) -> None:
    """Publish metrics.json and run.json (v1 path). An interrupted study has no
    aggregate to publish: its outcome is DECISION.toml."""
    if result.get("outcome") == INTERRUPTED_LABEL:
        raise EvidenceError("an interrupted study publishes no aggregate; its outcome is DECISION.toml")
    V1.publish_results(result, results_dir)


# ------------------------------------------------------------------ CLI


def load_source(path: Path) -> tuple[str, object]:
    """V1's reader, but a run record without stdout (a reservation the runner
    left) stays a record."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict) and "stdout" not in value and "experiment_id" in value and "status" in value:
        try:
            name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            name = path.name
        return name, value
    return V1.load_source(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=Path, help="the five per-seed run records of the study")
    parser.add_argument("--study", required=True, help="M001-v6, M002-v11 or M003-v3")
    parser.add_argument("--criteria", type=Path, default=CRITERIA_PATH)
    parser.add_argument("--criteria-sha256", help="refuse unless criteria.toml has this SHA-256")
    parser.add_argument("--lr-file", type=Path, help="check the trained rates against this frozen lr file")
    parser.add_argument("--repo", type=Path, default=ROOT, help="git repository that holds the record commits")
    parser.add_argument("--allow-row-files", action="store_true", help="accept bare JSON-row files (rehearsal only)")
    parser.add_argument("--paired-ci", type=Path, help="write paired-ci-95.json here (exclusive)")
    parser.add_argument("--decision", type=Path, help="write DECISION.toml here (exclusive)")
    parser.add_argument("--publish-results", type=Path, help="publish metrics.json and run.json into this results directory")
    args = parser.parse_args(argv)
    try:
        criteria = load_criteria(args.criteria)
        if args.criteria_sha256 and args.criteria_sha256 != criteria["sha256"]:
            raise EvidenceError("criteria.toml SHA-256 differs from --criteria-sha256")
        lr_text = None
        if args.lr_file is not None:
            try:
                lr_text = args.lr_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                raise EvidenceError(f"{args.lr_file}: unreadable: {error}") from None
        sources = [load_source(path) for path in args.records]
        result = decide(sources, args.study, criteria, allow_rows=args.allow_row_files, lr_text=lr_text, repo=args.repo)
        if (args.paired_ci is None) != (args.decision is None):
            raise EvidenceError("--paired-ci and --decision are written together")
        if args.publish_results is not None and result.get("outcome") == INTERRUPTED_LABEL:
            raise EvidenceError("an interrupted study publishes no aggregate; drop --publish-results (nothing was written)")
        if args.paired_ci is not None:
            write_outputs(result, args.paired_ci, args.decision)
        if args.publish_results is not None:
            publish_results(result, args.publish_results)
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    summary = {
        "experiment_id": result["experiment_id"],
        "headline": result["headline"],
        "contrasts": {
            item["id"]: {"verdict": item["verdict"], "label": item["label"], "failing_gates": item["failing_gates"]}
            for item in result["contrasts"]
        },
    }
    if result.get("outcome") == INTERRUPTED_LABEL:
        summary["interrupted"] = result["interrupted"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
