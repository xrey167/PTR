from __future__ import annotations
import argparse, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load(path): return tomllib.loads(Path(path).read_text(encoding='utf-8'))

def validate():
    schema=load(ROOT/'evaluations/schema.toml')
    req=set(schema['required_candidate']); allowed=set(schema['allowed_status'])
    errors=[]; candidates=0
    for p in sorted((ROOT/'evaluations/components').glob('*/candidates.toml')):
        data=load(p)
        if data.get('component')!=p.parent.name: errors.append(f'{p}: component mismatch')
        seen=set()
        weights=sum(float(v) for v in data.get('criteria',{}).values())
        if abs(weights-1.0)>1e-6: errors.append(f'{p}: criteria weights sum to {weights}')
        for c in data.get('candidate',[]):
            candidates+=1; missing=req-set(c)
            if missing: errors.append(f'{p}:{c.get("id","?")}: missing {sorted(missing)}')
            if c.get('id') in seen: errors.append(f'{p}: duplicate candidate {c.get("id")}')
            seen.add(c.get('id'))
            if c.get('status') not in allowed: errors.append(f'{p}:{c.get("id")}: invalid status')
        if not (p.parent/'config.toml').exists(): errors.append(f'{p.parent}: missing config.toml')
        if not (p.parent/'tests').exists(): errors.append(f'{p.parent}: missing tests/')
    if errors:
        print('\n'.join('ERROR: '+e for e in errors)); return 1
    print(f'OK: validated {candidates} component candidates'); return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['validate']); a=ap.parse_args()
    raise SystemExit(validate())
if __name__=='__main__': main()
