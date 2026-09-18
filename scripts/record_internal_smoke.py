from __future__ import annotations
import argparse, datetime as dt, json, platform, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def first_json(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if line:
            return json.loads(line)
    raise ValueError(f"no JSON record in {path}")

def rustc():
    try:
        return subprocess.check_output(["rustc","+stable","--version"],text=True).strip()
    except Exception:
        return "unknown"

def git_sha():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:
        return "unknown"

def write(component: str, candidate: str, metric: dict):
    record={
      "schema_version":1,
      "evidence_kind":"internal-smoke-microbenchmark",
      "component":component,
      "candidate":candidate,
      "git_sha":git_sha(),
      "recorded_at":dt.datetime.now(dt.timezone.utc).isoformat(),
      "platform":platform.platform(),
      "machine":platform.machine(),
      "python":platform.python_version(),
      "rustc":rustc(),
      "metric":metric,
      "decision_scope":"smoke evidence only; insufficient for candidate preference",
    }
    out=ROOT/"evaluations"/"components"/component/"evidence"/"internal-smoke-linux-x86_64.json"
    out.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(out.relative_to(ROOT))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--semdb",type=Path,required=True)
    ap.add_argument("--mailbox",type=Path,required=True)
    a=ap.parse_args()
    write("semantic-db","ptr-incremental-semdb",first_json(a.semdb))
    write("execution-runtime","ptr-isolates",first_json(a.mailbox))

if __name__=="__main__":
    main()
