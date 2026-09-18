from __future__ import annotations
import argparse, datetime as dt, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs",nargs="+",type=Path)
    a=ap.parse_args()
    records=[json.loads(path.read_text(encoding="utf-8")) for path in a.inputs]

    typed=sum(r["typed_accuracy"] for r in records)/len(records)
    ablated=sum(r["ablated_accuracy"] for r in records)/len(records)
    delta=typed-ablated
    metrics={
      "scope":"synthetic mechanism sanity check; not M001 OOD evidence",
      "seeds":records,
      "mean_typed_accuracy":typed,
      "mean_ablated_accuracy":ablated,
      "mean_accuracy_delta":delta,
    }
    results=Path(__file__).resolve().parent/"results"
    (results/"pilot_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:
        sha="unknown"
    run={
      "experiment_id":"M001-pilot",
      "status":"completed-pilot-only",
      "git_sha":sha,
      "recorded_at":dt.datetime.now(dt.timezone.utc).isoformat(),
      "scope":metrics["scope"],
      "reporting_restriction":"Do not use this pilot as evidence for M001 superiority or OOD generalization.",
    }
    (results/"pilot_run.json").write_text(json.dumps(run,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
