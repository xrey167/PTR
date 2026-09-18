from __future__ import annotations
import json, sys, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
errors=[]
required=[
    'Cargo.toml','README.md','docs/DEFINITION_OF_DONE.md','experiments/registry.toml',
    'datasets/registry.toml','crates/ptr-types/src/lib.rs','crates/ptr-semdb/src/lib.rs',
    'crates/ptr-core/src/lib.rs','evaluations/README.md','docs/components/STATUS.md',
    'scripts/update_component_docs.py'
]
for rel in required:
    if not (ROOT/rel).exists(): errors.append(f'missing {rel}')

for p in ROOT.rglob('*.toml'):
    try: tomllib.loads(p.read_text(encoding='utf-8'))
    except Exception as e: errors.append(f'TOML {p.relative_to(ROOT)}: {e}')
for p in ROOT.rglob('*.json'):
    try: json.loads(p.read_text(encoding='utf-8'))
    except Exception as e: errors.append(f'JSON {p.relative_to(ROOT)}: {e}')
for p in ROOT.rglob('*.jsonl'):
    for n,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try: json.loads(line)
        except Exception as e: errors.append(f'JSONL {p.relative_to(ROOT)}:{n}: {e}')

exp_raw=tomllib.loads((ROOT/'experiments/registry.toml').read_text(encoding='utf-8'))
eval_raw=tomllib.loads((ROOT/'evaluations/registry.toml').read_text(encoding='utf-8'))
exp_ids={x['id'] for x in exp_raw.get('experiment',[])}
eval_ids={x['id'] for x in eval_raw.get('component',[])}

crate_dirs=sorted(p for p in (ROOT/'crates').iterdir() if p.is_dir() and (p/'Cargo.toml').exists())
for crate in crate_dirs:
    name=crate.name
    for rel in ['README.md','component.toml','src/lib.rs']:
        if not (crate/rel).exists(): errors.append(f'{name}: missing {rel}')
    diagram=ROOT/'docs/diagrams/components'/f'{name}.mmd'
    if not diagram.exists(): errors.append(f'{name}: missing component diagram')
    meta_path=crate/'component.toml'
    if meta_path.exists():
        try:
            meta=tomllib.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('id') != name: errors.append(f'{name}: component.toml id mismatch')
            for key in ['maturity','last_reviewed','implemented','missing','next','experiments','evaluations','decisions','checks']:
                if key not in meta: errors.append(f'{name}: component.toml missing {key}')
            for eid in meta.get('experiments',[]):
                if eid not in exp_ids: errors.append(f'{name}: unknown experiment {eid}')
            for eid in meta.get('evaluations',[]):
                if eid not in eval_ids: errors.append(f'{name}: unknown evaluation {eid}')
            for adr in meta.get('decisions',[]):
                if not (ROOT/'research/decisions'/adr).exists(): errors.append(f'{name}: missing ADR {adr}')
        except Exception as e:
            errors.append(f'{name}: invalid component.toml: {e}')
    readme=crate/'README.md'
    if readme.exists():
        txt=readme.read_text(encoding='utf-8')
        if '<!-- PTR:STATUS:BEGIN -->' not in txt or '<!-- PTR:STATUS:END -->' not in txt:
            errors.append(f'{name}: README generated status block missing')

mods=list((ROOT/'model/modifications').glob('MOD-*.md'))
if len(mods) < 10: errors.append('expected >=10 model modification specs')
comps=list((ROOT/'evaluations/components').glob('*/candidates.toml'))
if len(comps) < 20: errors.append('expected >=20 component evaluations')
exps=list((ROOT/'experiments').glob('**/experiment.toml'))
if len(exps) < 15: errors.append('expected >=15 experiments')

if errors:
    print('\n'.join('ERROR: '+x for x in errors)); sys.exit(1)
print(f'OK: {len(crate_dirs)} documented crates, {len(comps)} component evaluations, {len(exps)} experiments, {len(mods)} model modifications')
