"""Decider of the A0 baseline study v1 (M001-v5, M002-v10, M003-v2 against plain).

Reads the five per-seed run records of one study (or, for tests and rehearsals,
plain files of the same JSON rows), the preregistered criteria of
research/falsification/A0-baseline-v1/criteria.toml (the only source of every
threshold), and decides each contrast of the study separately:

- SUPPORTS: every gate holds, the primary endpoint included;
- FALSIFIES: the OOD accuracy delta's interval lies below the minimum effect with
  both arms learnable and the compute match intact (label HARMFUL when the upper
  bound is below zero);
- INCONCLUSIVE with label NO-GO-CALIBRATION: the accuracy endpoint holds but a
  calibration gate fails (this forbids any quality claim);
- INCONCLUSIVE otherwise, naming every failing gate.

Paired statistics reuse the immutable core scripts/aggregate_m002_v5.py by import
(its SHA-256 is pinned in the criteria and checked; it is never modified): the
paired Student-t interval over five seeds, finite-number checks, the row parser
and the dataset-digest reader. Folds are never replicates; a pooled endpoint
averages the folds inside a seed first, a per-fold endpoint pairs over seeds.

Evidence that cannot support any decision (a missing seed, fold, arm or split, a
non-finite value, a protocol or arm-binding mismatch, a changed dataset digest,
records from different builds) is a hard refusal (EvidenceError, exit code 2),
never a verdict. A complete negative result is a valid decision (exit code 0).

Outputs: paired-ci-95.json (every paired endpoint, the descriptive arm table, the
mask share), DECISION.toml (verdict and gates per contrast, bound to the SHA-256
of the paired-CI file), and optionally the metrics.json/run.json aggregate
(experiment_records.publish_aggregate).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/aggregate_m002_v5.py"
CRITERIA_PATH = ROOT / "research/falsification/A0-baseline-v1/criteria.toml"
SCHEMA_VERSION = 1
COMPUTE_MODES = ("plain-params", "equal-params-flops", "equal-params")
ROLES = ("primary", "secondary", "bundle")
SPLIT_KEYS = ("iid_split", "ood_split")
PLAIN = "plain"


def load_core():
    spec = importlib.util.spec_from_file_location("_a0_baseline_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the immutable M002-v5 decision core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_core()
EvidenceError = CORE.EvidenceError


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def script_sha256() -> str:
    return sha256_bytes(Path(__file__).read_bytes())


# ------------------------------------------------------------------ criteria


def need(table: object, key: str, kind: type | tuple, where: str):
    """A required key of the right type; booleans only where a bool is asked for."""
    if not isinstance(table, dict) or key not in table:
        raise EvidenceError(f"criteria: {where} lacks {key}")
    value = table[key]
    ok = isinstance(value, bool) if kind is bool else isinstance(value, kind) and not isinstance(value, bool)
    if not ok:
        raise EvidenceError(f"criteria: {where}.{key} has the wrong type")
    return value


def load_criteria(path: Path = CRITERIA_PATH, root: Path = ROOT) -> dict:
    """Load and validate the preregistered criteria; the result is read-only data."""
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise EvidenceError(f"criteria: cannot load {path}: {error}") from None
    if data.get("version") != 1:
        raise EvidenceError("criteria: unsupported version")
    if data.get("seeds") != list(CORE.SEEDS):
        raise EvidenceError("criteria: seeds are not the confirmatory seeds of the core")
    t_value = need(data, "t_critical", float, "root")
    if abs(t_value - CORE.T_CRITICAL_95_DF4) > 1e-3:
        raise EvidenceError("criteria: t_critical differs from the core's 4-df quantile")
    decision = need(data, "decision", dict, "root")
    if need(decision, "core", str, "decision") != "scripts/aggregate_m002_v5.py":
        raise EvidenceError("criteria: the decision core must be scripts/aggregate_m002_v5.py")
    if sha256_bytes(CORE_PATH.read_bytes()) != need(decision, "core_sha256", str, "decision"):
        raise EvidenceError("criteria: the immutable decision core's SHA-256 differs from the pin")

    block = need(data, "data", dict, "root")
    lock = root / need(block, "lock", str, "data")
    try:
        lock_sha = sha256_bytes(lock.read_bytes())
    except OSError as error:
        raise EvidenceError(f"criteria: cannot read the dataset lock: {error}") from None
    if lock_sha != need(block, "lock_sha256", str, "data"):
        raise EvidenceError("criteria: the dataset lock's SHA-256 differs from the pin")
    folds = []
    for token in need(block, "folds", str, "data").split(","):
        name, separator, digest = token.partition("=")
        if not separator or not name or not digest:
            raise EvidenceError(f"criteria: malformed fold token {token!r}")
        folds.append((name, digest))
    if tuple(name for name, _ in folds) != CORE.FOLDS:
        raise EvidenceError("criteria: folds are not the three confirmatory folds in order")
    locked = CORE.expected_fold_digests(root)
    if any(locked[name] != digest for name, digest in folds):
        raise EvidenceError("criteria: a fold token differs from the dataset lock")
    for key in SPLIT_KEYS:
        need(block, key, str, "data")

    protocol = need(data, "protocol", dict, "root")
    for key in ("d_model", "steps", "batch_size", "rank", "plain_latent_steps", "sweep_seed"):
        need(protocol, key, int, "protocol")
    for key in ("bias_limit", "metadata_dropout", "label_smoothing", "consistency_weight"):
        need(protocol, key, (int, float), "protocol")
    need(protocol, "router_mode", str, "protocol")
    need(protocol, "latent_nonlinearity", bool, "protocol")
    grid = need(protocol, "lr_grid", list, "protocol")
    if not grid or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in grid):
        raise EvidenceError("criteria: protocol.lr_grid must be positive numbers")

    columns = need(data, "columns", dict, "root")
    for key in (
        "accuracy", "nll", "ece", "coverage", "selective_error",
        "diagnostic_nll", "diagnostic_ece", "temperature",
    ):
        need(columns, key, str, "columns")
    need(columns, "temperature_cap", (int, float), "columns")

    thresholds = need(data, "thresholds", dict, "root")
    for key in (
        "delta_min", "ood_accuracy_ci_lower_gt", "positive_accuracy_each_fold_gt",
        "ood_nll_ci_upper_le", "ood_ece_mean_le", "ood_ece_ci_upper_le",
        "iid_accuracy_ci_lower_ge", "coverage_iid_ge", "coverage_ood_ge",
        "selective_error_mean_le", "plain_param_tolerance", "learnable_min_iid_accuracy",
    ):
        need(thresholds, key, (int, float), "thresholds")

    rules = need(data, "decision_rules", dict, "root")
    order = need(rules, "gate_order", list, "decision_rules")
    calibration = need(rules, "calibration_gates", list, "decision_rules")
    if len(set(order)) != len(order) or not set(calibration) <= set(order):
        raise EvidenceError("criteria: gate_order/calibration_gates are inconsistent")
    if "complete_finite_bound_evidence" not in order:
        raise EvidenceError("criteria: complete_finite_bound_evidence must be a gate")

    studies = {}
    for study in need(data, "study", list, "root"):
        sid = need(study, "id", str, "study")
        if sid in studies:
            raise EvidenceError(f"criteria: duplicate study {sid}")
        need(study, "runner_study", str, sid)
        arms = need(study, "arms", list, sid)
        if len(set(arms)) != len(arms) or PLAIN not in arms:
            raise EvidenceError(f"criteria: {sid} arms must be unique and include plain")
        pins = need(study, "pins", dict, sid)
        for arm in arms:
            if arm != PLAIN and not isinstance(pins.get(arm, {}).get("latent_steps"), int):
                raise EvidenceError(f"criteria: {sid}/{arm} needs a latent_steps pin")
        if not set(pins) <= set(arms) - {PLAIN}:
            raise EvidenceError(f"criteria: {sid} pins name an arm outside the study")
        need(study, "require_mask_share", bool, sid)
        contrasts = need(study, "contrast", list, sid)
        ids = [need(item, "id", str, sid) for item in contrasts]
        if len(set(ids)) != len(ids):
            raise EvidenceError(f"criteria: {sid} has duplicate contrast ids")
        if need(study, "headline", str, sid) not in ids:
            raise EvidenceError(f"criteria: {sid} headline is not a contrast")
        for item in contrasts:
            if need(item, "role", str, sid) not in ROLES:
                raise EvidenceError(f"criteria: {item['id']} has an unknown role")
            need(item, "isolating", bool, sid)
            need(item, "claim", str, sid)
            if need(item, "compute", str, sid) not in COMPUTE_MODES:
                raise EvidenceError(f"criteria: {item['id']} has an unknown compute mode")
            for key in ("treatment", "control"):
                if need(item, key, str, sid) not in arms:
                    raise EvidenceError(f"criteria: {item['id']} {key} is not an arm of {sid}")
            if item["treatment"] == item["control"]:
                raise EvidenceError(f"criteria: {item['id']} compares an arm with itself")
            if item["compute"] == "plain-params" and item["control"] != PLAIN:
                raise EvidenceError(f"criteria: {item['id']} plain-params needs control plain")
        if sum(1 for item in contrasts if item["role"] == "primary") != 1:
            raise EvidenceError(f"criteria: {sid} needs exactly one primary contrast")
        headline = next(item for item in contrasts if item["id"] == study["headline"])
        if headline["role"] != "primary":
            raise EvidenceError(f"criteria: {sid} headline must be the primary contrast")
        studies[sid] = study
    deferred = {item["id"]: item for item in data.get("deferred", [])}
    if set(deferred) & set(studies):
        raise EvidenceError("criteria: a study cannot be both run and deferred")
    return {
        "raw": data,
        "sha256": sha256_bytes(raw),
        "folds": folds,
        "protocol": protocol,
        "columns": columns,
        "thresholds": thresholds,
        "rules": rules,
        "data": block,
        "studies": studies,
        "deferred": deferred,
    }


def study_of(criteria: dict, study_id: str) -> dict:
    if study_id in criteria["deferred"]:
        raise EvidenceError(f"{study_id} is deferred ({criteria['deferred'][study_id]['reason']}); nothing to decide")
    study = criteria["studies"].get(study_id)
    if study is None:
        raise EvidenceError(f"unknown study {study_id!r}; expected one of {sorted(criteria['studies'])}")
    return study


# ------------------------------------------------------------------ evidence


def number(row: dict, key: str, where: str) -> float:
    return CORE.finite_number(row.get(key), f"{where}/{key}")


def integer(row: dict, key: str, where: str, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{where}/{key} must be an integer >= {minimum}")
    return value


def unit_interval(row: dict, key: str, where: str) -> float:
    value = number(row, key, where)
    if not 0.0 <= value <= 1.0:
        raise EvidenceError(f"{where}/{key} is outside [0, 1]")
    return value


def agree(actual: object, expected: object, where: str) -> None:
    if isinstance(expected, bool):
        ok = isinstance(actual, bool) and actual == expected
    elif isinstance(expected, (int, float)):
        ok = (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        )
    else:
        ok = actual == expected
    if not ok:
        raise EvidenceError(f"{where} is {actual!r}, expected {expected!r}")


def expected_meta(criteria: dict, study: dict, arm: str) -> dict:
    protocol = criteria["protocol"]
    common = {
        "steps": protocol["steps"],
        "batch_size": protocol["batch_size"],
        "d_model": protocol["d_model"],
    }
    if arm == PLAIN:
        return {**common, "kind": "plain", "latent_steps": protocol["plain_latent_steps"]}
    pinned = {
        "kind": "a0",
        "typed_attention_rank": protocol["rank"],
        "typed_attention_limit": protocol["bias_limit"],
        "metadata_dropout": protocol["metadata_dropout"],
        "router_mode": protocol["router_mode"],
        "label_smoothing": protocol["label_smoothing"],
        "consistency_weight": protocol["consistency_weight"],
        "latent_nonlinearity": protocol["latent_nonlinearity"],
    }
    return {**common, **pinned, **study["pins"][arm]}


FINAL_NUMBERS = (
    "raw_nll", "raw_ece15", "raw_coverage", "raw_selective_error", "raw_confidence_threshold",
    "fitted_temperature", "fitted_nll", "fitted_ece15", "total_params", "flops_per_example",
    "latency_p95_ms", "ms_per_step", "explicit_token_mean",
)


def split_rows(stdout: object, name: str) -> list[dict]:
    return CORE.metric_rows(stdout, name)


def index_rows(name: str, rows: list[dict], seed: int | None, criteria: dict, study: dict) -> tuple[int, dict]:
    """Validate one run's rows and return (seed, cells keyed by (kind, fold, arm, split))."""
    runs = [row for row in rows if row.get("row") == "run-baseline-v1"]
    if len(runs) != 1:
        raise EvidenceError(f"{name}: expected one run-baseline-v1 row, found {len(runs)}")
    run = runs[0]
    run_seed = run.get("seed")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed not in CORE.SEEDS:
        raise EvidenceError(f"{name}: seed {run_seed!r} is not confirmatory")
    if seed is not None and seed != run_seed:
        raise EvidenceError(f"{name}: record seed {seed!r} differs from the run row seed {run_seed!r}")
    protocol = criteria["protocol"]
    if run.get("study") != study["runner_study"]:
        raise EvidenceError(f"{name}: run row study {run.get('study')!r} is not {study['runner_study']!r}")
    for key, want in (("d_model", protocol["d_model"]), ("batch_size", protocol["batch_size"]), ("steps", protocol["steps"])):
        agree(run.get(key), want, f"{name}: run row {key}")
    announced = run.get("arms")
    if not isinstance(announced, str) or sorted(announced.split(",")) != sorted(study["arms"]):
        raise EvidenceError(f"{name}: run row arms {announced!r} are not the study's arms")

    folds = {fold for fold, _ in criteria["folds"]}
    arms = set(study["arms"])
    splits = {criteria["data"]["iid_split"], criteria["data"]["ood_split"]}
    cells: dict[tuple, dict] = {}
    for row in rows:
        kind = row.get("row")
        if kind not in ("data-v2", "meta-baseline-v1", "final-baseline-v1", "mask-share-baseline-v1"):
            continue
        fold = row.get("fold")
        if fold not in folds:
            raise EvidenceError(f"{name}: {kind} row has unknown fold {fold!r}")
        if kind != "data-v2" and row.get("study") != study["runner_study"]:
            raise EvidenceError(f"{name}: {kind} row of study {row.get('study')!r} inside {study['runner_study']!r}")
        arm = row.get("arm") if kind in ("meta-baseline-v1", "final-baseline-v1") else None
        if kind in ("meta-baseline-v1", "final-baseline-v1") and arm not in arms:
            raise EvidenceError(f"{name}: {kind} row has arm {arm!r}, which is not in the study table")
        split = row.get("split") if kind in ("final-baseline-v1", "mask-share-baseline-v1") else None
        if kind in ("final-baseline-v1", "mask-share-baseline-v1") and split not in splits:
            raise EvidenceError(f"{name}: {kind} row has unknown split {split!r}")
        if kind in ("meta-baseline-v1", "final-baseline-v1"):
            agree(row.get("seed"), run_seed, f"{name}: {kind}/{fold}/{arm} seed")
        key = (kind, fold, arm, split)
        if key in cells:
            raise EvidenceError(f"{name}: duplicate evidence row {key}")
        cells[key] = row

    for fold, digest in criteria["folds"]:
        data = cells.get(("data-v2", fold, None, None))
        if data is None:
            raise EvidenceError(f"{name}: missing data identity for {fold}")
        if data.get("data_fnv64") != digest:
            raise EvidenceError(f"{name}: {fold} dataset digest mismatch")
        for arm in study["arms"]:
            where = f"{name}: {fold}/{arm}"
            meta = cells.get(("meta-baseline-v1", fold, arm, None))
            if meta is None:
                raise EvidenceError(f"{where}: missing meta row")
            if meta.get("nan") not in (0, 0.0) or isinstance(meta.get("nan"), bool):
                raise EvidenceError(f"{where}: reported NaN training")
            for key in ("lr", "final_train_loss", "ms_per_step", "total_params", "flops_per_example"):
                number(meta, key, f"{where}/meta")
            for key, want in expected_meta(criteria, study, arm).items():
                agree(meta.get(key), want, f"{where}/meta/{key}")
            if not any(math.isclose(meta["lr"], grid, rel_tol=1e-12) for grid in protocol["lr_grid"]):
                raise EvidenceError(f"{where}: lr {meta['lr']!r} is not on the preregistered grid")
            for split in sorted(splits):
                final = cells.get(("final-baseline-v1", fold, arm, split))
                if final is None:
                    raise EvidenceError(f"{where}/{split}: missing final row")
                check_final(final, meta, f"{where}/{split}")
    if study["require_mask_share"]:
        check_mask_share(name, cells, criteria, study)
    return run_seed, cells


def check_final(final: dict, meta: dict, where: str) -> None:
    n = integer(final, "n", where, 1)
    correct = integer(final, "correct", where)
    if correct > n:
        raise EvidenceError(f"{where}: correct exceeds n")
    accuracy = unit_interval(final, "accuracy", where)
    if not math.isclose(accuracy, correct / n, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceError(f"{where}: accuracy disagrees with counts")
    for key in FINAL_NUMBERS:
        number(final, key, where)
    if final["raw_nll"] < 0:
        raise EvidenceError(f"{where}: raw_nll is negative")
    for key in ("raw_ece15", "raw_coverage", "raw_selective_error"):
        unit_interval(final, key, where)
    if final["fitted_temperature"] <= 0:
        raise EvidenceError(f"{where}: fitted temperature is not positive")
    integer(final, "explicit_token_count", where, 1)
    for key in ("total_params", "flops_per_example", "latent_steps"):
        agree(final.get(key), meta.get(key), f"{where}/{key} (final vs meta)")


def check_mask_share(name: str, cells: dict, criteria: dict, study: dict) -> None:
    masked, unmasked = "no-semantic-slots-masked", "no-semantic-slots"
    for fold, _ in criteria["folds"]:
        for split in (criteria["data"]["iid_split"], criteria["data"]["ood_split"]):
            where = f"{name}: {fold}/{split} mask share"
            row = cells.get(("mask-share-baseline-v1", fold, None, split))
            if row is None:
                raise EvidenceError(f"{where}: row missing")
            for key, arm in (("masked_accuracy", masked), ("unmasked_accuracy", unmasked)):
                number(row, key, where)
                agree(row[key], cells[("final-baseline-v1", fold, arm, split)]["accuracy"], f"{where}/{key}")
            if not math.isclose(number(row, "mask_share", where), row["masked_accuracy"] - row["unmasked_accuracy"], rel_tol=0.0, abs_tol=1e-12):
                raise EvidenceError(f"{where}: mask_share is not masked - unmasked")
            if row.get("mask_share_of_full_gap") is not None:
                number(row, "mask_share_of_full_gap", where)
            unit_interval(row, "nonlive_slot_share", where)


def admit_record(name: str, record: object, study: dict, allow_rows: bool) -> tuple[int | None, str, dict]:
    """Admit one source: a hardened run record, or (only if allowed) a bare row file."""
    if not isinstance(record, dict):
        raise EvidenceError(f"{name}: record is not an object")
    if "stdout" not in record:
        if not allow_rows:
            raise EvidenceError(f"{name}: a bare row file is not admissible evidence")
        return None, "row-file", record
    if record.get("experiment_id") != study["id"]:
        raise EvidenceError(f"{name}: experiment_id is not {study['id']}")
    if record.get("entrypoint") != "entrypoint":
        raise EvidenceError(f"{name}: only the frozen entrypoint is admissible")
    if record.get("status") != "completed" or record.get("exit_code") != 0:
        raise EvidenceError(f"{name}: run did not complete successfully")
    if record.get("git_dirty") is not False:
        raise EvidenceError(f"{name}: confirmatory evidence must come from a clean commit")
    manifest = record.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("id") != study["id"]:
        raise EvidenceError(f"{name}: recorded manifest identity is not {study['id']}")
    for field in CORE.IDENTITY_FIELDS:
        if field not in record:
            raise EvidenceError(f"{name}: missing bound identity field {field}")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in CORE.SEEDS:
        raise EvidenceError(f"{name}: seed {seed!r} is not confirmatory")
    return seed, "run-record", record


def build_table(
    sources: list[tuple[str, object]], criteria: dict, study: dict, allow_rows: bool = False
) -> tuple[dict[int, dict], str, list[dict]]:
    """Admit all sources, check cross-record identity and consistency, return the cell table."""
    if len(sources) != len(CORE.SEEDS):
        raise EvidenceError(f"expected exactly {len(CORE.SEEDS)} run records, found {len(sources)}")
    kinds = set()
    by_seed: dict[int, dict] = {}
    identities: dict[str, set] = {field: set() for field in CORE.IDENTITY_FIELDS}
    for name, source in sources:
        seed, kind, record = admit_record(name, source, study, allow_rows)
        kinds.add(kind)
        if len(kinds) != 1:
            raise EvidenceError("run records and bare row files cannot be mixed")
        rows = split_rows(record["stdout"], name) if kind == "run-record" else row_file_rows(name, record)
        run_seed, cells = index_rows(name, rows, seed, criteria, study)
        if run_seed in by_seed:
            raise EvidenceError(f"seed {run_seed} has duplicate records")
        by_seed[run_seed] = cells
        if kind == "run-record":
            for field in CORE.IDENTITY_FIELDS:
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
    check_cross_run_consistency(by_seed, criteria, study)
    digest = [
        {"path": name, "canonical_sha256": sha256_bytes(CORE.canonical(source).encode("utf-8"))}
        for name, source in sorted(sources, key=lambda item: item[0])
    ]
    return by_seed, kinds.pop(), digest


def row_file_rows(name: str, record: dict) -> list[dict]:
    rows = record.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise EvidenceError(f"{name}: a row file holds no rows")
    return rows


def check_cross_run_consistency(by_seed: dict[int, dict], criteria: dict, study: dict) -> None:
    """Same arm => same lr, parameters and flops; same data => same n and token counts."""
    first = by_seed[CORE.SEEDS[0]]
    for arm in study["arms"]:
        lrs = {
            cells[("meta-baseline-v1", fold, arm, None)]["lr"]
            for cells in by_seed.values()
            for fold, _ in criteria["folds"]
        }
        if len(lrs) != 1:
            raise EvidenceError(f"{arm}: learning rate differs across seeds or folds: {sorted(lrs)}")
    for seed, cells in by_seed.items():
        for fold, _ in criteria["folds"]:
            for arm in study["arms"]:
                key = ("meta-baseline-v1", fold, arm, None)
                for field in ("total_params", "flops_per_example"):
                    if cells[key][field] != first[key][field]:
                        raise EvidenceError(f"seed {seed} {fold}/{arm}: {field} differs from the first seed")
            for split in (criteria["data"]["iid_split"], criteria["data"]["ood_split"]):
                reference = first[("final-baseline-v1", fold, study["arms"][0], split)]
                for arm in study["arms"]:
                    row = cells[("final-baseline-v1", fold, arm, split)]
                    for field in ("n", "explicit_token_count"):
                        if row[field] != reference[field]:
                            raise EvidenceError(f"seed {seed} {fold}/{arm}/{split}: {field} differs across arms or seeds")


# ------------------------------------------------------------------ statistics

DIRECTIONS = {
    "accuracy": "higher_is_better",
    "coverage": "higher_is_better",
    "nll": "lower_is_better",
    "ece": "lower_is_better",
    "selective_error": "lower_is_better",
    "diagnostic_nll": "lower_is_better",
    "diagnostic_ece": "lower_is_better",
}
DIAGNOSTIC = ("diagnostic_nll", "diagnostic_ece")


class Table:
    """Typed access to final-row values of the evidence table."""

    def __init__(self, by_seed: dict[int, dict], criteria: dict):
        self.by_seed = by_seed
        self.criteria = criteria
        self.folds = [fold for fold, _ in criteria["folds"]]
        self.iid = criteria["data"]["iid_split"]
        self.ood = criteria["data"]["ood_split"]

    def final(self, seed: int, fold: str, arm: str, split: str, metric: str) -> float:
        column = self.criteria["columns"][metric]
        return float(self.by_seed[seed][("final-baseline-v1", fold, arm, split)][column])

    def meta(self, seed: int, fold: str, arm: str, field: str) -> float:
        return float(self.by_seed[seed][("meta-baseline-v1", fold, arm, None)][field])

    def deltas(self, treatment: str, control: str, split: str, metric: str, fold: str | None) -> list[float]:
        def one(seed: int, name: str) -> float:
            if fold is not None:
                return self.final(seed, fold, name, split, metric)
            return statistics.fmean(self.final(seed, f, name, split, metric) for f in self.folds)

        return [one(seed, treatment) - one(seed, control) for seed in CORE.SEEDS]


def endpoint(table: Table, treatment: str, control: str, split: str, metric: str, fold: str | None) -> dict:
    interval = CORE.interval(table.deltas(treatment, control, split, metric, fold))
    return {
        "scope": fold if fold is not None else "pooled-over-folds",
        "split": split,
        "metric": table.criteria["columns"][metric],
        "key": metric,
        "direction": DIRECTIONS[metric],
        "diagnostic": metric in DIAGNOSTIC,
        "seeds": list(CORE.SEEDS),
        "deltas": interval["values"],
        "mean": interval["mean"],
        "sd": interval["sd"],
        "ci95": interval["ci95"],
        "positive_seeds": sum(1 for value in interval["values"] if value > 0),
    }


def contrast_endpoints(table: Table, treatment: str, control: str) -> dict[tuple, dict]:
    out = {}
    for split, metrics in (
        (table.ood, ("accuracy", "nll", "ece", "selective_error", "coverage", "diagnostic_nll", "diagnostic_ece")),
        (table.iid, ("accuracy", "nll", "ece", "selective_error", "coverage")),
    ):
        for metric in metrics:
            for fold in (None, *table.folds):
                item = endpoint(table, treatment, control, split, metric, fold)
                out[(item["scope"], split, metric)] = item
    return out


def compute_check(table: Table, contrast: dict, thresholds: dict) -> dict:
    mode = contrast["compute"]
    treatment, control = contrast["treatment"], contrast["control"]
    worst_relative = 0.0
    flops_ratios = []
    ok = True
    for seed in CORE.SEEDS:
        for fold in table.folds:
            tp, cp = table.meta(seed, fold, treatment, "total_params"), table.meta(seed, fold, control, "total_params")
            tf, cf = table.meta(seed, fold, treatment, "flops_per_example"), table.meta(seed, fold, control, "flops_per_example")
            relative = abs(tp - cp) / cp if cp else math.inf
            worst_relative = max(worst_relative, relative)
            flops_ratios.append(tf / cf if cf else math.inf)
            if mode == "plain-params":
                ok = ok and relative <= thresholds["plain_param_tolerance"]
            elif mode == "equal-params-flops":
                ok = ok and tp == cp and tf == cf
            else:
                ok = ok and tp == cp
    return {
        "mode": mode,
        "pass": ok,
        "max_relative_param_difference": worst_relative,
        "flops_ratio_min": min(flops_ratios),
        "flops_ratio_max": max(flops_ratios),
        "flops_gate": mode == "equal-params-flops",
        "note": "FLOP columns are descriptive unless the mode is equal-params-flops (A0 and plain helpers are not comparable)",
    }


def coverage_check(table: Table, arm: str, thresholds: dict) -> dict:
    cells = []
    for seed in CORE.SEEDS:
        for fold in table.folds:
            cells.append({
                "seed": seed,
                "fold": fold,
                "iid": table.final(seed, fold, arm, table.iid, "coverage"),
                "ood": table.final(seed, fold, arm, table.ood, "coverage"),
            })
    ok = all(c["iid"] >= thresholds["coverage_iid_ge"] and c["ood"] >= thresholds["coverage_ood_ge"] for c in cells)
    return {
        "arm": arm,
        "pass": ok,
        "min_iid": min(c["iid"] for c in cells),
        "min_ood": min(c["ood"] for c in cells),
    }


def mean_accuracy(table: Table, arm: str, split: str) -> float:
    return statistics.fmean(
        table.final(seed, fold, arm, split, "accuracy") for seed in CORE.SEEDS for fold in table.folds
    )


def evaluate_gates(table: Table, contrast: dict, ends: dict[tuple, dict], thresholds: dict) -> tuple[dict, dict]:
    pooled = "pooled-over-folds"
    ood_accuracy = ends[(pooled, table.ood, "accuracy")]
    iid_accuracy = ends[(pooled, table.iid, "accuracy")]
    selective = ends[(pooled, table.ood, "selective_error")]
    per_fold = {fold: {m: ends[(fold, table.ood, m)] for m in ("accuracy", "nll", "ece")} for fold in table.folds}
    compute = compute_check(table, contrast, thresholds)
    coverage = coverage_check(table, contrast["treatment"], thresholds)
    gates = {
        "complete_finite_bound_evidence": True,
        "ood_accuracy": ood_accuracy["mean"] >= thresholds["delta_min"]
        and ood_accuracy["ci95"][0] > thresholds["ood_accuracy_ci_lower_gt"],
        "positive_accuracy_each_fold": all(
            item["accuracy"]["mean"] > thresholds["positive_accuracy_each_fold_gt"] for item in per_fold.values()
        ),
        "ood_nll_each_fold": all(
            item["nll"]["ci95"][1] <= thresholds["ood_nll_ci_upper_le"] for item in per_fold.values()
        ),
        "ood_ece_each_fold": all(
            item["ece"]["mean"] <= thresholds["ood_ece_mean_le"] and item["ece"]["ci95"][1] <= thresholds["ood_ece_ci_upper_le"]
            for item in per_fold.values()
        ),
        "iid_accuracy": iid_accuracy["ci95"][0] >= thresholds["iid_accuracy_ci_lower_ge"],
        "coverage": coverage["pass"],
        "selective_error": selective["mean"] <= thresholds["selective_error_mean_le"],
        "matched_compute": compute["pass"],
    }
    details = {
        "ood_accuracy": f"mean {ood_accuracy['mean']:.6g}, CI {ood_accuracy['ci95'][0]:.6g}..{ood_accuracy['ci95'][1]:.6g}",
        "positive_accuracy_each_fold": {fold: item["accuracy"]["mean"] for fold, item in per_fold.items()},
        "ood_nll_each_fold": {fold: item["nll"]["ci95"][1] for fold, item in per_fold.items()},
        "ood_ece_each_fold": {
            fold: [item["ece"]["mean"], item["ece"]["ci95"][1]] for fold, item in per_fold.items()
        },
        "coverage": coverage,
        "matched_compute": compute,
    }
    return gates, details


def decide_contrast(table: Table, contrast: dict, criteria: dict) -> dict:
    thresholds = criteria["thresholds"]
    ends = contrast_endpoints(table, contrast["treatment"], contrast["control"])
    gates, details = evaluate_gates(table, contrast, ends, thresholds)
    order = criteria["rules"]["gate_order"]
    calibration = criteria["rules"]["calibration_gates"]
    failing = [name for name in order if not gates[name]]
    ood = ends[("pooled-over-folds", table.ood, "accuracy")]
    learnable = {
        arm: mean_accuracy(table, arm, table.iid) >= thresholds["learnable_min_iid_accuracy"]
        for arm in (contrast["treatment"], contrast["control"])
    }
    label_reason = ""
    if gates["ood_accuracy"]:
        if not failing:
            verdict, label = "SUPPORTS", "SUPPORTS"
            label_reason = "every gate holds, the primary accuracy endpoint included"
        elif any(name in calibration for name in failing):
            verdict, label = "INCONCLUSIVE", "NO-GO-CALIBRATION"
            label_reason = "OOD accuracy improves but a per-fold calibration gate fails; no quality claim"
        else:
            verdict, label = "INCONCLUSIVE", "INCONCLUSIVE"
            label_reason = "OOD accuracy improves but a non-calibration gate fails"
    elif ood["ci95"][1] < thresholds["delta_min"] and all(learnable.values()) and gates["matched_compute"]:
        if ood["ci95"][1] < 0:
            verdict, label = "FALSIFIES", "HARMFUL"
            label_reason = "CI upper < 0: the control is better"
        else:
            verdict, label = "FALSIFIES", "FALSIFIES"
            label_reason = "CI upper < delta_min with both arms learnable and compute matched"
    else:
        verdict, label = "INCONCLUSIVE", "INCONCLUSIVE"
        if ood["ci95"][1] < thresholds["delta_min"]:
            blockers = [arm for arm, ok in learnable.items() if not ok]
            if blockers:
                label_reason = f"falsification blocked: not learnable: {blockers}"
            else:
                label_reason = "falsification blocked: compute match failed"
        else:
            label_reason = "neither SUPPORTS nor FALSIFIES holds"
    return {
        "id": contrast["id"],
        "role": contrast["role"],
        "isolating": contrast["isolating"],
        "treatment": contrast["treatment"],
        "control": contrast["control"],
        "compute": contrast["compute"],
        "claim": contrast["claim"],
        "verdict": verdict,
        "label": label,
        "reason": label_reason,
        "failing_gates": failing,
        "gates": gates,
        "gate_details": details,
        "learnable": learnable,
        "quality_claim_allowed": verdict == "SUPPORTS",
        "scope_note": (
            "isolating contrast of one mechanism inside A0"
            if contrast["isolating"]
            else "bundle vs baseline, non-isolating: more than one mechanism differs"
        ),
        "endpoints": list(ends.values()),
    }


def arm_table(table: Table, study: dict, criteria: dict) -> dict:
    cap = criteria["columns"]["temperature_cap"]
    out = {}
    for arm in study["arms"]:
        cells = [(seed, fold) for seed in CORE.SEEDS for fold in table.folds]
        item = {
            "lr": table.meta(CORE.SEEDS[0], table.folds[0], arm, "lr"),
            "total_params": table.meta(CORE.SEEDS[0], table.folds[0], arm, "total_params"),
            "flops_per_example": table.meta(CORE.SEEDS[0], table.folds[0], arm, "flops_per_example"),
            "latent_steps": expected_meta(criteria, study, arm)["latent_steps"],
            "mean_ms_per_step": statistics.fmean(table.meta(s, f, arm, "ms_per_step") for s, f in cells),
        }
        for split in (table.iid, table.ood):
            item[split] = {
                "accuracy": mean_accuracy(table, arm, split),
                "raw_nll": statistics.fmean(table.final(s, f, arm, split, "nll") for s, f in cells),
                "raw_ece15": statistics.fmean(table.final(s, f, arm, split, "ece") for s, f in cells),
                "fitted_temperature_mean": statistics.fmean(table.final(s, f, arm, split, "temperature") for s, f in cells),
                "fitted_temperature_saturated_cells": sum(
                    1 for s, f in cells if table.final(s, f, arm, split, "temperature") >= cap
                ),
                "cells": len(cells),
                "explicit_token_count": {
                    f: table.by_seed[CORE.SEEDS[0]][("final-baseline-v1", f, arm, split)]["explicit_token_count"]
                    for f in table.folds
                },
            }
        item["mean_latency_p95_ms"] = statistics.fmean(
            float(table.by_seed[s][("final-baseline-v1", f, arm, table.ood)]["latency_p95_ms"]) for s, f in cells
        )
        out[arm] = item
    return out


def mask_share(table: Table, study: dict) -> dict | None:
    if not study["require_mask_share"]:
        return None
    out = {}
    for split in (table.iid, table.ood):
        def rows(seed: int) -> list[dict]:
            return [table.by_seed[seed][("mask-share-baseline-v1", fold, None, split)] for fold in table.folds]

        share = CORE.interval([statistics.fmean(r["mask_share"] for r in rows(s)) for s in CORE.SEEDS])
        gaps = [r["mask_share_of_full_gap"] for s in CORE.SEEDS for r in rows(s) if r["mask_share_of_full_gap"] is not None]
        out[split] = {
            "definition": "masked accuracy - unmasked accuracy (A0-v2); descriptive, no verdict",
            "mask_share": {"mean": share["mean"], "sd": share["sd"], "ci95": share["ci95"], "deltas": share["values"]},
            "mask_share_of_full_gap_mean": statistics.fmean(gaps) if gaps else None,
            "mask_share_of_full_gap_defined_cells": len(gaps),
            "nonlive_slot_share_mean": statistics.fmean(r["nonlive_slot_share"] for s in CORE.SEEDS for r in rows(s)),
        }
    return out


def decide(
    sources: list[tuple[str, object]],
    study_id: str,
    criteria: dict | None = None,
    *,
    allow_rows: bool = False,
    lr_text: str | None = None,
) -> dict:
    """Decide every contrast of one study; malformed evidence is a hard refusal."""
    criteria = load_criteria() if criteria is None else criteria
    study = study_of(criteria, study_id)
    by_seed, kind, source_records = build_table(sources, criteria, study, allow_rows)
    if lr_text is not None:
        check_lr_file(lr_text, by_seed, criteria, study)
    table = Table(by_seed, criteria)
    contrasts = [decide_contrast(table, item, criteria) for item in study["contrast"]]
    headline = next(item for item in contrasts if item["id"] == study["headline"])
    first = next(iter(by_seed.values()))
    git_shas = sorted({
        source["git_sha"] for _, source in sources if isinstance(source, dict) and "git_sha" in source
    }) if kind == "run-record" else []
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": study["id"],
        "study": criteria["raw"]["name"],
        "evidence_kind": kind,
        "seeds": list(CORE.SEEDS),
        "folds": [fold for fold, _ in criteria["folds"]],
        "splits": {"iid": table.iid, "ood": table.ood},
        "t_critical_used": CORE.T_CRITICAL_95_DF4,
        "headline": {"contrast": headline["id"], "verdict": headline["verdict"], "label": headline["label"]},
        "contrasts": contrasts,
        "arms": arm_table(table, study, criteria),
        "mask_share": mask_share(table, study),
        "diagnostics": {
            "temperature_cap": criteria["columns"]["temperature_cap"],
            "note": (
                "fitted_* columns are labelled diagnostics; the fit grid ends at the cap and may saturate. "
                "FLOP columns are descriptive (A0 and plain helpers are not comparable)."
            ),
            "data_fnv64": {fold: first[("data-v2", fold, None, None)]["data_fnv64"] for fold in table.folds},
        },
        "provenance": {
            "git_sha": git_shas[0] if len(git_shas) == 1 else None,
            "criteria_sha256": criteria["sha256"],
            "decision_script_sha256": script_sha256(),
            "decision_core_sha256": sha256_bytes(CORE_PATH.read_bytes()),
            "dataset_lock_sha256": criteria["raw"]["data"]["lock_sha256"],
            "source_records": source_records,
        },
    }


# ------------------------------------------------------------------ learning-rate file


def parse_lr_file(text: str) -> dict[str, float]:
    rates: dict[str, float] = {}
    for number_, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.split("\t")[0] == "arm":
            continue
        fields = stripped.split("\t")
        if len(fields) < 2:
            raise EvidenceError(f"lr file line {number_}: expected arm<TAB>lr")
        try:
            rate = float(fields[1])
        except ValueError:
            raise EvidenceError(f"lr file line {number_}: lr is not a number") from None
        if not math.isfinite(rate) or rate <= 0:
            raise EvidenceError(f"lr file line {number_}: lr must be positive and finite")
        if fields[0] in rates:
            raise EvidenceError(f"lr file line {number_}: duplicate arm {fields[0]}")
        rates[fields[0]] = rate
    return rates


def check_lr_file(text: str, by_seed: dict[int, dict], criteria: dict, study: dict) -> None:
    rates = parse_lr_file(text)
    first = by_seed[CORE.SEEDS[0]]
    fold = criteria["folds"][0][0]
    for arm in study["arms"]:
        if arm not in rates:
            raise EvidenceError(f"lr file has no entry for {arm}")
        if not math.isclose(rates[arm], first[("meta-baseline-v1", fold, arm, None)]["lr"], rel_tol=1e-12):
            raise EvidenceError(f"{arm}: trained lr differs from the frozen lr file")


# ------------------------------------------------------------------ outputs


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def paired_ci_document(result: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": result["experiment_id"],
        "study": result["study"],
        "status": "complete",
        "evidence_kind": result["evidence_kind"],
        "seeds": result["seeds"],
        "folds": result["folds"],
        "splits": result["splits"],
        "t_critical_used": result["t_critical_used"],
        "contrasts": [
            {
                key: item[key]
                for key in ("id", "role", "isolating", "treatment", "control", "compute", "endpoints")
            }
            for item in result["contrasts"]
        ],
        "arms": result["arms"],
        "mask_share": result["mask_share"],
        "diagnostics": result["diagnostics"],
        "provenance": result["provenance"],
    }


def toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError("cannot write a non-finite number to DECISION.toml")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value {value!r}")


def toml_lines(prefix: str, table: dict, out: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if v is not None and not isinstance(v, dict)}
    out.extend(f"{key} = {toml_scalar(value)}" for key, value in scalars.items())
    for key, value in table.items():
        if isinstance(value, dict):
            out.append("")
            out.append(f"[{prefix}.{key}]" if prefix else f"[{key}]")
            toml_lines(f"{prefix}.{key}" if prefix else key, value, out)


def decision_toml(result: dict, paired_ci_sha256: str) -> str:
    headline = result["headline"]
    top = {
        "version": 1,
        "experiment_id": result["experiment_id"],
        "study": result["study"],
        "headline_contrast": headline["contrast"],
        "verdict": headline["verdict"],
        "label": headline["label"],
        "go": headline["verdict"] == "SUPPORTS",
        "claim_scope": "A0 on the synthetic operator-routing v2 benchmark only (ADR-0021); no language-model claim",
        "evidence_kind": result["evidence_kind"],
        "records_immutable": True,
        "paired_ci_sha256": paired_ci_sha256,
        "criteria_sha256": result["provenance"]["criteria_sha256"],
        "decision_script_sha256": result["provenance"]["decision_script_sha256"],
        "decision_core_sha256": result["provenance"]["decision_core_sha256"],
        "git_sha": result["provenance"]["git_sha"],
    }
    lines: list[str] = []
    toml_lines("", top, lines)
    lines += ["", "[reporting]"]
    toml_lines(
        "reporting",
        {
            "required_label": headline["label"],
            "allow_quality_claim": headline["verdict"] == "SUPPORTS",
            "allow_bundle_claim_as_mechanism_evidence": False,
        },
        lines,
    )
    for item in result["contrasts"]:
        ood = next(
            e for e in item["endpoints"]
            if e["scope"] == "pooled-over-folds" and e["key"] == "accuracy" and e["split"] == result["splits"]["ood"]
        )
        lines += ["", "[[contrast]]"]
        toml_lines(
            "contrast",
            {
                "id": item["id"],
                "role": item["role"],
                "isolating": item["isolating"],
                "treatment": item["treatment"],
                "control": item["control"],
                "compute": item["compute"],
                "verdict": item["verdict"],
                "label": item["label"],
                "reason": item["reason"],
                "failing_gates": item["failing_gates"],
                "quality_claim_allowed": item["quality_claim_allowed"],
                "scope_note": item["scope_note"],
                "claim": item["claim"],
            },
            lines,
        )
        lines += ["", "[contrast.gates]"]
        lines += [f"{name} = {toml_scalar(value)}" for name, value in item["gates"].items()]
        lines += ["", "[contrast.ood_accuracy]"]
        lines += [
            f"mean_delta = {toml_scalar(ood['mean'])}",
            f"ci95 = {toml_scalar(ood['ci95'])}",
        ]
    for record in result["provenance"]["source_records"]:
        lines += ["", "[[source_record]]"]
        lines += [f"path = {toml_scalar(record['path'])}", f"canonical_sha256 = {toml_scalar(record['canonical_sha256'])}"]
    return "\n".join(lines) + "\n"


def write_exclusive(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)


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
    """Publish metrics.json and run.json as one aggregate (experiment_records.publish_aggregate)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import experiment_records  # noqa: PLC0415

    provenance = result["provenance"]
    if result["evidence_kind"] != "run-record" or not provenance["git_sha"]:
        raise EvidenceError("only hardened run records of one commit can be published as an aggregate")
    metrics = {
        "experiment_id": result["experiment_id"],
        "study": result["study"],
        "verdict": result["headline"]["verdict"],
        "label": result["headline"]["label"],
        "contrasts": [
            {key: item[key] for key in ("id", "role", "isolating", "verdict", "label", "failing_gates", "gates")}
            for item in result["contrasts"]
        ],
        "arms": result["arms"],
        "mask_share": result["mask_share"],
    }
    run = {
        "experiment_id": result["experiment_id"],
        "status": "completed",
        "git_sha": provenance["git_sha"],
        "aggregated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "criteria_sha256": provenance["criteria_sha256"],
        "decision_script_sha256": provenance["decision_script_sha256"],
        "decision_core_sha256": provenance["decision_core_sha256"],
        "verdict": result["headline"]["verdict"],
        "label": result["headline"]["label"],
        "seeds": [{"seed": seed} for seed in result["seeds"]],
        "source_records": provenance["source_records"],
    }
    try:
        experiment_records.publish_aggregate(results_dir, metrics, run)
    except experiment_records.ProvenanceError as error:
        raise EvidenceError(str(error)) from None


# ------------------------------------------------------------------ CLI


def load_source(path: Path) -> tuple[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"{path}: unreadable: {error}") from None
    try:
        name = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        name = path.name
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and "stdout" in value:
        return name, value
    rows = []
    for number_, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{path}: line {number_} is invalid JSON: {error}") from None
    return name, {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=Path, help="the five per-seed run records of the study")
    parser.add_argument("--study", required=True, help="M001-v5, M002-v10 or M003-v2")
    parser.add_argument("--criteria", type=Path, default=CRITERIA_PATH)
    parser.add_argument("--criteria-sha256", help="refuse unless criteria.toml has this SHA-256")
    parser.add_argument("--lr-file", type=Path, help="check the trained rates against this frozen lr file")
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
        result = decide(sources, args.study, criteria, allow_rows=args.allow_row_files, lr_text=lr_text)
        if (args.paired_ci is None) != (args.decision is None):
            raise EvidenceError("--paired-ci and --decision are written together")
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
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
