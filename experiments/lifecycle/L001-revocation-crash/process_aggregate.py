from __future__ import annotations
import argparse, datetime as dt, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]

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
      "child_exit_errors":sum(r["child_exit_errors"] for r in records),
    }
    results=Path(__file__).resolve().parent/"results"
    metrics={
      "scope":"reference FileLedger real child-process abort after durable revocation plus incomplete next record",
      "seeds":records,
      "totals":totals,
    }
    (results/"process_crash_metrics.json").write_text(
      json.dumps(metrics,indent=2)+"\n",encoding="utf-8"
    )
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
      "hard_pass":all(value==0 for key,value in totals.items() if key!="cases"),
      "limitations":[
        "reference FileLedger only",
        "child process abort occurs after revocation fsync and while writing a subsequent incomplete record",
        "does not yet cover kill during revocation write, raft-engine, leader change, or network partition",
      ],
    }
    (results/"process_crash_run.json").write_text(
      json.dumps(run,indent=2)+"\n",encoding="utf-8"
    )

if __name__=="__main__":
    main()
