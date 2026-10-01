"""Strict paired decision for the confirmatory M002-v5 study.

Each hardened runner record contains both arms over all three confirmatory
folds for one seed. Folds are averaged inside a seed before the five paired
seed values enter a Student-t interval; folds are never treated as independent
replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43, 71, 101)
FOLDS = ("evidence-temporal", "evidence-tabular", "claim-interventional")
TREATMENT = "factorized-v2"
CONTROL = "factorized-v2-off"
ARMS = (TREATMENT, CONTROL)
T_CRITICAL_95_DF4 = 2.7764451051977987
IDENTITY_FIELDS = (
    "git_sha",
    "manifest_sha256",
    "parameters",
    "rustc",
    "toolchain",
    "environment",
    "executable",
    "host",
    "cargo_lock_sha256",
    "uv_lock_sha256",
)


class EvidenceError(ValueError):
    """The records cannot support any statistical decision."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def finite_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{where} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"{where} is not finite")
    return number


def metric_rows(stdout: object, record: str) -> list[dict]:
    if not isinstance(stdout, str):
        raise EvidenceError(f"{record}: stdout is not text")
    rows = []
    for line_no, line in enumerate(stdout.splitlines(), 1):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{record}: stdout line {line_no} is invalid JSON: {error}") from None
        if not isinstance(row, dict):
            raise EvidenceError(f"{record}: stdout line {line_no} is not an object")
        rows.append(row)
    return rows


def interval(values: list[float]) -> dict:
    if len(values) != len(SEEDS):
        raise EvidenceError(f"a confirmatory interval needs {len(SEEDS)} paired seeds")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {"values": values, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def expected_fold_digests(root: Path = ROOT) -> dict[str, str]:
    lock = json.loads((root / "benchmarks/operator-routing-v2/splits.lock.json").read_text(encoding="utf-8"))
    return {fold: lock["folds"][fold]["data_fnv1a64"] for fold in FOLDS}


def index_records(records: list[tuple[str, dict]], root: Path = ROOT) -> dict:
    if len(records) != len(SEEDS):
        raise EvidenceError(f"expected exactly {len(SEEDS)} run records, found {len(records)}")
    by_seed: dict[int, tuple[str, dict]] = {}
    identities: dict[str, set[str]] = {field: set() for field in IDENTITY_FIELDS}
    candidates = set()
    for name, record in records:
        if not isinstance(record, dict):
            raise EvidenceError(f"{name}: record is not an object")
        if record.get("experiment_id") != "M002-v5":
            raise EvidenceError(f"{name}: experiment_id is not M002-v5")
        if record.get("entrypoint") != "entrypoint":
            raise EvidenceError(f"{name}: only the frozen entrypoint is admissible")
        if record.get("status") != "completed" or record.get("exit_code") != 0:
            raise EvidenceError(f"{name}: run did not complete successfully")
        if record.get("git_dirty") is not False:
            raise EvidenceError(f"{name}: confirmatory evidence must come from a clean commit")
        manifest = record.get("manifest")
        if not isinstance(manifest, dict):
            raise EvidenceError(f"{name}: missing recorded manifest")
        if manifest.get("id") != "M002-v5" or not isinstance(manifest.get("entrypoint"), str):
            raise EvidenceError(f"{name}: recorded manifest identity is not M002-v5")
        for field in ("preregistration_sha256", "preregistration_rules_sha256"):
            if not isinstance(manifest.get(field), str) or len(manifest[field]) != 64:
                raise EvidenceError(f"{name}: recorded manifest has no bound {field}")
            identities.setdefault(f"manifest.{field}", set()).add(canonical(manifest[field]))
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            raise EvidenceError(f"{name}: seed {seed!r} is not confirmatory")
        if seed in by_seed:
            raise EvidenceError(f"seed {seed} has duplicate records")
        by_seed[seed] = (name, record)
        for field in IDENTITY_FIELDS:
            if field not in record:
                raise EvidenceError(f"{name}: missing bound identity field {field}")
            identities[field].add(canonical(record[field]))
    missing = sorted(set(SEEDS) - set(by_seed))
    if missing:
        raise EvidenceError(f"missing seeds: {missing}")
    for field, values in identities.items():
        if len(values) != 1:
            raise EvidenceError(f"run records disagree in {field}")

    expected_digests = expected_fold_digests(root)
    indexed: dict[int, dict] = {}
    for seed in SEEDS:
        name, record = by_seed[seed]
        rows = metric_rows(record.get("stdout"), name)
        protocols = [row for row in rows if row.get("row") == "run-v5"]
        if len(protocols) != 1 or protocols[0].get("seed") != seed:
            raise EvidenceError(f"{name}: expected one matching run-v5 protocol row")
        protocol = protocols[0]
        candidate = (
            int(finite_number(protocol.get("rank"), f"{name}: rank")),
            finite_number(protocol.get("bias_limit"), f"{name}: bias_limit"),
            finite_number(protocol.get("metadata_dropout"), f"{name}: metadata_dropout"),
        )
        candidates.add(candidate)
        cells: dict[tuple, dict] = {}
        for row in rows:
            kind = row.get("row")
            if kind not in {"data-v2", "meta", "calibration", "performance-v5", "final-v5"}:
                continue
            fold = row.get("fold")
            if fold not in FOLDS:
                raise EvidenceError(f"{name}: evidence row has unknown fold {fold!r}")
            arm = row.get("arm")
            split = row.get("split")
            key = (kind, fold, arm, split)
            if key in cells:
                raise EvidenceError(f"{name}: duplicate evidence row {key}")
            cells[key] = row

        for fold in FOLDS:
            data = cells.get(("data-v2", fold, None, None))
            if data is None:
                raise EvidenceError(f"{name}: missing data identity for {fold}")
            if data.get("data_fnv64") != expected_digests[fold]:
                raise EvidenceError(f"{name}: {fold} dataset digest mismatch")
            if data.get("hard_validity_violations") != 0:
                raise EvidenceError(f"{name}: {fold} has a hard-validity violation")
            for arm in ARMS:
                meta = cells.get(("meta", fold, arm, None))
                performance = cells.get(("performance-v5", fold, arm, None))
                calibration = cells.get(("calibration", fold, arm, None))
                if meta is None or performance is None or calibration is None:
                    raise EvidenceError(f"{name}: incomplete metadata for {fold}/{arm}")
                if meta.get("nan") not in (0, 0.0):
                    raise EvidenceError(f"{name}: {fold}/{arm} reported NaN training")
                for field in ("total_params", "estimated_flops_per_example", "final_train_loss"):
                    finite_number(meta.get(field), f"{name}: {fold}/{arm}/{field}")
                finite_number(performance.get("latency_p95_ms"), f"{name}: {fold}/{arm}/latency_p95_ms")
                finite_number(calibration.get("temperature"), f"{name}: {fold}/{arm}/temperature")
                finite_number(calibration.get("confidence_threshold"), f"{name}: {fold}/{arm}/threshold")
                for split in ("test_iid", "test_ood"):
                    final = cells.get(("final-v5", fold, arm, split))
                    if final is None:
                        raise EvidenceError(f"{name}: missing {fold}/{arm}/{split}")
                    for field in ("accuracy", "nll", "ece15", "coverage", "selective_error"):
                        finite_number(final.get(field), f"{name}: {fold}/{arm}/{split}/{field}")
                    for field in ("n", "correct", "covered", "covered_correct", "abstained"):
                        value = final.get(field)
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise EvidenceError(f"{name}: {fold}/{arm}/{split}/{field} must be a non-negative integer")
                    n = final["n"]
                    covered = final["covered"]
                    if n <= 0 or final["correct"] > n or covered > n or final["covered_correct"] > covered:
                        raise EvidenceError(f"{name}: {fold}/{arm}/{split} has inconsistent abstention counts")
                    if final["abstained"] != n - covered:
                        raise EvidenceError(f"{name}: {fold}/{arm}/{split} abstained count is inconsistent")
                    expected = {
                        "accuracy": final["correct"] / n,
                        "coverage": covered / n,
                        "selective_error": 0.0 if covered == 0 else 1.0 - final["covered_correct"] / covered,
                    }
                    for field, computed in expected.items():
                        if not math.isclose(float(final[field]), computed, rel_tol=0.0, abs_tol=1e-12):
                            raise EvidenceError(f"{name}: {fold}/{arm}/{split}/{field} disagrees with counts")
            on_meta = cells[("meta", fold, TREATMENT, None)]
            off_meta = cells[("meta", fold, CONTROL, None)]
            if on_meta.get("typed_attention_mode") != "factorized-v2" or off_meta.get("typed_attention_mode") != "off":
                raise EvidenceError(f"{name}: {fold} is not the FactorizedV2/Off contrast")
            meta_candidate = (
                int(finite_number(on_meta.get("typed_attention_rank"), f"{name}: {fold}/rank")),
                finite_number(on_meta.get("typed_attention_limit"), f"{name}: {fold}/bias_limit"),
                finite_number(on_meta.get("metadata_dropout"), f"{name}: {fold}/metadata_dropout"),
            )
            if meta_candidate != candidate:
                raise EvidenceError(f"{name}: {fold} treatment does not match the protocol candidate")
            for field in (
                "typed_attention_rank", "typed_attention_limit", "typed_query", "latent_steps",
                "latent_nonlinearity", "frozen_router", "router_mode", "router_logit_scale",
                "label_smoothing", "metadata_dropout", "consistency_weight", "batch", "lr", "steps",
            ):
                if on_meta.get(field) != off_meta.get(field):
                    raise EvidenceError(f"{name}: {fold} arms differ in matched field {field}")
        indexed[seed] = cells
    if len(candidates) != 1:
        raise EvidenceError("run records disagree on the frozen architecture candidate")
    return indexed


def decide(records: list[tuple[str, dict]], root: Path = ROOT) -> dict:
    """Apply every preregistered gate; malformed evidence is a hard refusal."""
    table = index_records(records, root)

    def value(seed: int, fold: str, arm: str, split: str, metric: str) -> float:
        return float(table[seed][("final-v5", fold, arm, split)][metric])

    def seed_delta(split: str, metric: str) -> list[float]:
        return [
            statistics.fmean(value(seed, fold, TREATMENT, split, metric) - value(seed, fold, CONTROL, split, metric) for fold in FOLDS)
            for seed in SEEDS
        ]

    ood_accuracy = interval(seed_delta("test_ood", "accuracy"))
    ood_nll = interval(seed_delta("test_ood", "nll"))
    ood_ece = interval(seed_delta("test_ood", "ece15"))
    iid_accuracy = interval(seed_delta("test_iid", "accuracy"))
    selective_error = interval(seed_delta("test_ood", "selective_error"))
    fold_accuracy = {
        fold: statistics.fmean(
            value(seed, fold, TREATMENT, "test_ood", "accuracy")
            - value(seed, fold, CONTROL, "test_ood", "accuracy")
            for seed in SEEDS
        )
        for fold in FOLDS
    }

    compute_checks = []
    coverage_checks = []
    for seed in SEEDS:
        for fold in FOLDS:
            on_meta = table[seed][("meta", fold, TREATMENT, None)]
            off_meta = table[seed][("meta", fold, CONTROL, None)]
            on_params = float(on_meta["total_params"])
            off_params = float(off_meta["total_params"])
            on_flops = float(on_meta["estimated_flops_per_example"])
            off_flops = float(off_meta["estimated_flops_per_example"])
            on_latency = float(table[seed][("performance-v5", fold, TREATMENT, None)]["latency_p95_ms"])
            off_latency = float(table[seed][("performance-v5", fold, CONTROL, None)]["latency_p95_ms"])
            compute_checks.append({
                "seed": seed,
                "fold": fold,
                "same_parameters": on_params == off_params,
                "flops_ratio": max(on_flops, off_flops) / min(on_flops, off_flops),
                "latency_ratio": on_latency / off_latency,
            })
            for arm in ARMS:
                coverage_checks.append({
                    "seed": seed,
                    "fold": fold,
                    "arm": arm,
                    "iid": value(seed, fold, arm, "test_iid", "coverage"),
                    "ood": value(seed, fold, arm, "test_ood", "coverage"),
                })

    coverage_summary = {
        arm: {
            split: statistics.fmean(
                value(seed, fold, arm, split, "coverage")
                for seed in SEEDS for fold in FOLDS
            )
            for split in ("test_iid", "test_ood")
        }
        for arm in ARMS
    }
    treatment_latency = statistics.fmean(
        float(table[seed][("performance-v5", fold, TREATMENT, None)]["latency_p95_ms"])
        for seed in SEEDS for fold in FOLDS
    )
    control_latency = statistics.fmean(
        float(table[seed][("performance-v5", fold, CONTROL, None)]["latency_p95_ms"])
        for seed in SEEDS for fold in FOLDS
    )
    latency_ratio = treatment_latency / control_latency

    gates = {
        "complete_finite_bound_evidence": True,
        "ood_accuracy": ood_accuracy["mean"] >= 0.02 and ood_accuracy["ci95"][0] > 0.0,
        "positive_accuracy_each_fold": all(delta > 0.0 for delta in fold_accuracy.values()),
        "ood_nll": ood_nll["ci95"][1] <= 0.0,
        "ood_ece": ood_ece["mean"] <= 0.0 and ood_ece["ci95"][1] <= 0.02,
        "iid_accuracy": iid_accuracy["ci95"][0] >= -0.01,
        "coverage": all(item["iid"] >= 0.90 and item["ood"] >= 0.80 for item in coverage_checks),
        "selective_error": selective_error["mean"] <= 0.0,
        "matched_compute": (
            all(
                item["same_parameters"]
                and item["flops_ratio"] <= 1.01
                and item["latency_ratio"] <= 1.10
                for item in compute_checks
            )
        ),
    }
    passed = all(gates.values())
    source_records = [
        {
            "path": name,
            "canonical_sha256": hashlib.sha256(canonical(record).encode("utf-8")).hexdigest(),
        }
        for name, record in sorted(records)
    ]
    first_record = records[0][1]
    manifest = first_record["manifest"]
    return {
        "schema_version": 1,
        "experiment_id": "M002-v5",
        "decision": "PASS" if passed else "INCONCLUSIVE/NO-GO",
        "claim": (
            "Factorized Typed Pair Attention improves on operator_routing_v2 compositional routing accuracy "
            "at matched compute without measurable calibration degradation."
            if passed else None
        ),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "provenance": {
            "freeze_git_sha": first_record["git_sha"],
            "manifest_sha256": first_record["manifest_sha256"],
            "preregistration_sha256": manifest["preregistration_sha256"],
            "preregistration_rules_sha256": manifest["preregistration_rules_sha256"],
            "dataset_lock_sha256": hashlib.sha256(
                (root / "benchmarks/operator-routing-v2/splits.lock.json").read_bytes()
            ).hexdigest(),
            "decision_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_records": source_records,
        },
        "statistics": {
            "ood_accuracy_delta": ood_accuracy,
            "ood_nll_delta": ood_nll,
            "ood_ece15_delta": ood_ece,
            "iid_accuracy_delta": iid_accuracy,
            "ood_selective_error_delta": selective_error,
            "ood_accuracy_delta_by_fold": fold_accuracy,
        },
        "compute": {
            "cells": compute_checks,
            "mean_treatment_latency_p95_ms": treatment_latency,
            "mean_control_latency_p95_ms": control_latency,
            "latency_ratio": latency_ratio,
        },
        "coverage": {"cells": coverage_checks, "means": coverage_summary},
        "gates": gates,
    }


def load_records(paths: list[Path]) -> list[tuple[str, dict]]:
    records = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceError(f"{path}: unreadable run record: {error}") from None
        try:
            name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            name = path.name
        records.append((name, value))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = decide(load_records(args.records))
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    # A complete negative result is a valid research decision, not an execution
    # failure. Evidence errors still return 2 above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
