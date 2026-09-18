from __future__ import annotations
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'REPO_MANIFEST.sha256'
rows=[]
for p in sorted(x for x in ROOT.rglob('*') if x.is_file() and '.git' not in x.parts and x.name!='REPO_MANIFEST.sha256'):
    rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
out.write_text('\n'.join(rows)+'\n',encoding='utf-8')
print(f'wrote {len(rows)} entries to {out}')
