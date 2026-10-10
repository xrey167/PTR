"""Decider of the A0 baseline study v3-m003 (M003-v4 against plain).

M003-v4 is the successor study of M003-v3 (A0 baseline study v2), whose seed 71 was
killed by a container restart (INCONCLUSIVE-INTERRUPTED, an interruption and not an
outcome). Its protocol, arms, contrasts, gates, thresholds, seeds and rules are those
of v2 value for value (research/falsification/A0-baseline-v3-m003/criteria.toml), so
this decider is the v2 decider (scripts/aggregate_a0_baseline_v2.py: record identity of
a correct run loop, the INCONCLUSIVE-INTERRUPTED outcome, and through it the v1
decider and the immutable core) and nothing else. The v2 file is pinned by
`decision.parent_sha256` of criteria.toml and by the experiment's `decision_base`; it
is imported unchanged into a private module object, and only the names that tie it to
its own study are rebound there: the criteria name it accepts, the default criteria
path, the criteria loader (this one, which adds the pin of the v2 file) and the script
digest it reports (this file's). No statistic, gate or rule is re-implemented here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = "scripts/aggregate_a0_baseline_v2.py"
PARENT_PATH = ROOT / PARENT
CRITERIA_PATH = ROOT / "research/falsification/A0-baseline-v3-m003/criteria.toml"
NAME = "A0-baseline-v3-m003"


def load_parent():
    spec = importlib.util.spec_from_file_location("_a0_baseline_v2_decider", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the v2 decider")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_parent()
V1 = V2.V1
CORE = V2.CORE
EvidenceError = V2.EvidenceError
_parent_load_criteria = V2.load_criteria


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_criteria(path: Path = CRITERIA_PATH, root: Path = ROOT) -> dict:
    """The v2 loader (itself the v1 validation plus the v2 rules) on this study's
    criteria, plus the pin of the v2 decider and the learning-rate reuse table."""
    criteria = _parent_load_criteria(path, root)  # checks `name` against the rebound NAME
    data = criteria["raw"]
    decision = V1.need(data, "decision", dict, "root")
    if V1.need(decision, "parent", str, "decision") != PARENT:
        raise EvidenceError(f"criteria: the pinned parent decider must be {PARENT}")
    if V1.sha256_bytes(PARENT_PATH.read_bytes()) != V1.need(decision, "parent_sha256", str, "decision"):
        raise EvidenceError("criteria: the v2 decider's SHA-256 differs from the pin")
    reuse = V1.need(data, "sweep_reuse", dict, "root")
    for key in ("study", "experiment", "lr_file"):
        V1.need(reuse, key, str, "sweep_reuse")
    if reuse["study"] not in criteria["studies"]:
        raise EvidenceError("criteria: sweep_reuse.study is not a study of this criteria file")
    return criteria


# Tie the private copy of the v2 decider to this study; everything it does is then
# the v2 decider's, on this study's criteria.
V2.NAME = NAME
V2.CRITERIA_PATH = CRITERIA_PATH
V2.script_sha256 = script_sha256
V2.load_criteria = load_criteria

def publish_results(result: dict, results_dir: Path) -> None:
    """Publish metrics.json and run.json in the form the research gate requires of a
    listed experiment: v1's metrics and run, plus the experiment's
    `preregistration_sha256` and `preregistration_rules_sha256` (copied from the
    experiment.toml beside `results_dir`) and `seed_records` (the SHA-256 of each
    finished seed record in `results_dir`). An interrupted study publishes nothing."""
    if result.get("outcome") == V2.INTERRUPTED_LABEL:
        raise EvidenceError("an interrupted study publishes no aggregate; its outcome is DECISION.toml")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import experiment_records  # noqa: PLC0415

    provenance = result["provenance"]
    if result["evidence_kind"] != "run-record" or not provenance["git_sha"]:
        raise EvidenceError("only hardened run records of one commit can be published as an aggregate")
    try:
        manifest = tomllib.loads((results_dir.parent / "experiment.toml").read_text(encoding="utf-8"))
        digests = {key: manifest[key] for key in ("preregistration_sha256", "preregistration_rules_sha256")}
        seed_records = experiment_records.seed_record_digests(results_dir)
    except (OSError, ValueError, KeyError, experiment_records.ProvenanceError) as error:
        raise EvidenceError(f"cannot bind the aggregate to the experiment and its seed records: {error}") from None
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
        **digests,
        "seed_records": seed_records,
    }
    try:
        experiment_records.publish_aggregate(results_dir, metrics, run)  # recomputes metrics_sha256
    except experiment_records.ProvenanceError as error:
        raise EvidenceError(str(error)) from None


expected_meta = V2.expected_meta
decide = V2.decide
interrupted_result = V2.interrupted_result
write_outputs = V2.write_outputs
V2.publish_results = publish_results  # main() publishes through the rebound name
main = V2.main


if __name__ == "__main__":
    raise SystemExit(main())
