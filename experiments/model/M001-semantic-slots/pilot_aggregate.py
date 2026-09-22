from __future__ import annotations
import argparse, datetime as dt, hashlib, json, subprocess
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
    # A commit id alone can describe code that was never measured: a pilot is
    # normally run from the worktree that is about to become the next commit, so
    # HEAD is its parent, and new files are not in `git diff` at all. Fingerprint
    # the measured code as it was on disk instead — the model and the type kernel
    # it embeds, deliberately not this directory, whose contents are the results.
    measured=["model/burn-a0","crates/ptr-types"]
    def measured_digest():
        listing=subprocess.check_output(
            ["git","ls-files","--cached","--others","--exclude-standard","--"]+measured,
            cwd=ROOT,text=True).split()
        digest=hashlib.sha256()
        for rel in sorted(listing):
            path=ROOT/rel
            if not path.is_file():
                continue
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()
    try:
        content_sha=measured_digest()
    except Exception:
        content_sha=None
    run={
      "experiment_id":"M001-pilot",
      "status":"completed-pilot-only",
      "git_sha":sha,
      "measured_paths":measured,
      "measured_content_sha256":content_sha,
      "recorded_at":dt.datetime.now(dt.timezone.utc).isoformat(),
      "scope":metrics["scope"],
      "reporting_restriction":"Do not use this pilot as evidence for M001 superiority or OOD generalization.",
    }
    (results/"pilot_run.json").write_text(json.dumps(run,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
