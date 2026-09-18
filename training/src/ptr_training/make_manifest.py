from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--out',type=Path,default=Path('manifest.json')); a=ap.parse_args()
    rows=[]
    for p in sorted(x for x in a.root.rglob('*') if x.is_file()):
        h=hashlib.sha256(p.read_bytes()).hexdigest(); rows.append({'path':str(p.relative_to(a.root)),'bytes':p.stat().st_size,'sha256':h})
    a.out.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8'); print(a.out)
if __name__=='__main__': main()
