from __future__ import annotations
import argparse, datetime as dt, hashlib, json, platform, subprocess, sys, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'experiments/registry.toml'

def load(path: Path):
    return tomllib.loads(path.read_text(encoding='utf-8'))

def registry():
    return {e['id']: e for e in load(REGISTRY).get('experiment',[])}

def resolve(exp_id: str):
    item=registry().get(exp_id)
    if not item: raise SystemExit(f'unknown experiment: {exp_id}')
    root=ROOT/'experiments'/item['path']
    return item, root, load(root/'experiment.toml')

def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def validate():
    schema=load(ROOT/'experiments/schema.toml')
    required=set(schema['required']); allowed=set(schema['allowed_status'])
    errors=[]
    for exp_id,item in registry().items():
        root=ROOT/'experiments'/item['path']; p=root/'experiment.toml'
        if not p.exists(): errors.append(f'{exp_id}: missing experiment.toml'); continue
        data=load(p)
        missing=required-set(data)
        if missing: errors.append(f'{exp_id}: missing {sorted(missing)}')
        if data.get('id')!=exp_id: errors.append(f'{exp_id}: id mismatch')
        if data.get('status') not in allowed: errors.append(f'{exp_id}: invalid status')
        if item.get('status')!=data.get('status'): errors.append(f'{exp_id}: registry/manifest status mismatch')
        if not (root/'config.toml').exists(): errors.append(f'{exp_id}: missing config.toml')
        if not (root/'tests').exists(): errors.append(f'{exp_id}: missing tests/')
    if errors:
        print('\n'.join('ERROR: '+e for e in errors)); return 1
    print(f'OK: validated {len(registry())} experiments'); return 0

def prepare(exp_id: str):
    item,root,data=resolve(exp_id)
    results=root/data.get('results_dir','results'); results.mkdir(exist_ok=True)
    timestamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    try: git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception: git_sha='unknown'
    record={
      'experiment_id':exp_id,'status':'prepared','prepared_at':timestamp,
      'git_sha':git_sha,'python':sys.version,'platform':platform.platform(),
      'manifest':data,'manifest_sha256':sha(root/'experiment.toml'),
      'cargo_lock_sha256':sha(ROOT/'Cargo.lock'),
      'uv_lock_sha256':sha(ROOT/'training/uv.lock'),
      'hardware_profile':data.get('hardware_profile'),
    }
    out=results/f'run-{timestamp}.json'
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(out.relative_to(ROOT)); return 0

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('list'); sub.add_parser('validate')
    p=sub.add_parser('show'); p.add_argument('id')
    p=sub.add_parser('prepare'); p.add_argument('id')
    a=ap.parse_args()
    if a.cmd=='list':
        for k,v in registry().items(): print(k,v['status'],v['path'])
        return
    if a.cmd=='validate': raise SystemExit(validate())
    if a.cmd=='show':
        _,_,data=resolve(a.id); print(json.dumps(data,indent=2)); return
    if a.cmd=='prepare': raise SystemExit(prepare(a.id))
if __name__=='__main__': main()
