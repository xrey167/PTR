from __future__ import annotations
import argparse, datetime as dt, json, platform, subprocess, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load(path): return tomllib.loads(Path(path).read_text(encoding='utf-8'))

def components():
    return {p.parent.name:p for p in sorted((ROOT/'evaluations/components').glob('*/candidates.toml'))}

def validate():
    schema=load(ROOT/'evaluations/schema.toml')
    req=set(schema['required_candidate']); allowed=set(schema['allowed_status'])
    errors=[]; candidates=0
    for name,p in components().items():
        data=load(p)
        if data.get('component')!=name: errors.append(f'{p}: component mismatch')
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

def prepare(component: str, candidate: str) -> int:
    p=components().get(component)
    if p is None:
        print(f'ERROR: unknown component {component}'); return 1
    data=load(p)
    match=next((c for c in data.get('candidate',[]) if c.get('id')==candidate),None)
    if match is None:
        print(f'ERROR: unknown candidate {candidate} for {component}'); return 1
    try:
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:
        git_sha='unknown'
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record={
      'component':component,'candidate':match,'criteria':data.get('criteria',{}),
      'git_sha':git_sha,'platform':platform.platform(),'prepared_at':stamp,
      'status':'prepared-no-benchmark-evidence',
    }
    out=p.parent/'evidence'/f'{stamp}-{candidate}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(out.relative_to(ROOT)); return 0

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('validate')
    sub.add_parser('list')
    prep=sub.add_parser('prepare'); prep.add_argument('component'); prep.add_argument('candidate')
    a=ap.parse_args()
    if a.command=='validate': raise SystemExit(validate())
    if a.command=='list':
        for name,p in components().items():
            data=load(p)
            for c in data.get('candidate',[]): print(name,c['id'],c['status'])
        return
    if a.command=='prepare': raise SystemExit(prepare(a.component,a.candidate))
if __name__=='__main__': main()
