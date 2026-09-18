from pathlib import Path
root=Path(__file__).resolve().parents[1]
for p in sorted(root.rglob('*')):
    if '.git' in p.parts: continue
    depth=len(p.relative_to(root).parts)-1
    if depth <= 2: print('  '*depth + ('/' if p.is_dir() else '') + p.name)
