"""Aggregate L001 ledger-recovery results into results/metrics.json and results/run.json.

metrics.json and run.json are published as one aggregate
(`experiment_records.publish_aggregate`): run.json names the SHA-256 of the
metrics.json written with it, a failure while writing leaves the previous pair,
and an interruption while publishing leaves no run.json rather than the previous
one beside new metrics. `scripts/check_research_gates.py` asks a completed
experiment's current run.json for that binding.
"""

from __future__ import annotations
import argparse, datetime as dt, json, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RESULTS=HERE/"results"

sys.path.insert(0,str(ROOT/"scripts"))
import experiment_records  # noqa: E402

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs",nargs="+",type=Path)
    a=ap.parse_args()

    records=[json.loads(p.read_text(encoding="utf-8")) for p in a.inputs]
    totals={
      "cases":sum(r["iterations"] for r in records),
      "false_accepts":sum(r["false_accepts"] for r in records),
      "recovery_errors":sum(r["recovery_errors"] for r in records),
      "tail_trim_errors":sum(r["tail_trim_errors"] for r in records),
    }
    metrics={"scope":"reference FileLedger partial-tail crash injection","seeds":records,"totals":totals}
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:
        sha="unknown"
    run={
      "experiment_id":"L001",
      "status":"running",
      "scope":metrics["scope"],
      "git_sha":sha,
      "recorded_at":dt.datetime.now(dt.timezone.utc).isoformat(),
      "hard_pass":totals["false_accepts"]==0 and totals["recovery_errors"]==0 and totals["tail_trim_errors"]==0,
      "limitations":[
        "partial trailing-record crash model only",
        "not yet process-kill/fail-rs/raft-engine/partition coverage",
      ],
    }
    try:
        experiment_records.publish_aggregate(RESULTS,metrics,run)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"L001: {error}")

if __name__=="__main__":
    main()
