from pathlib import Path
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("component"); ap.add_argument("candidate"); ap.add_argument("--source",default="external"); a=ap.parse_args()
root=Path(__file__).resolve().parents[1]
p=root/"evaluations"/"components"/a.component/"candidates.toml"
if not p.exists(): raise SystemExit(f"unknown component: {a.component}")
with p.open("a",encoding="utf-8") as f:
    f.write(f"\n[[candidate]]\nid = \"{a.candidate}\"\nsource = \"{a.source}\"\nstatus = \"to-evaluate\"\nevidence = []\n")
print(p)
