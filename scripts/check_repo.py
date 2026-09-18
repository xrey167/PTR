from __future__ import annotations
import json, sys, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
errors=[]
required=[
    'Cargo.toml','README.md','docs/DEFINITION_OF_DONE.md','experiments/registry.toml',
    'datasets/registry.toml','crates/ptr-types/src/lib.rs','crates/ptr-semdb/src/lib.rs',
    'crates/ptr-core/src/lib.rs','evaluations/README.md'
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
mods=list((ROOT/'model/modifications').glob('MOD-*.md'))
if len(mods) < 10: errors.append('expected >=10 model modification specs')
comps=list((ROOT/'evaluations/components').glob('*/candidates.toml'))
if len(comps) < 20: errors.append('expected >=20 component evaluations')
exps=list((ROOT/'experiments').glob('**/experiment.toml'))
if len(exps) < 15: errors.append('expected >=15 experiments')
if errors:
    print('\n'.join('ERROR: '+x for x in errors)); sys.exit(1)
print(f'OK: {len(comps)} component evaluations, {len(exps)} experiments, {len(mods)} model modifications')
