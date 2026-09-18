from pathlib import Path
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--hypothesis',required=True); a=ap.parse_args()
root=Path(__file__).resolve().parents[1]/'experiments'/a.path
root.mkdir(parents=True,exist_ok=False); (root/'results').mkdir()
(root/'README.md').write_text(f'# {root.name}\n\n## Hypothesis\n{a.hypothesis}\n',encoding='utf-8')
(root/'experiment.toml').write_text(f'id = "{root.name.split("-")[0]}"\nstatus = "planned"\nhypothesis = {a.hypothesis!r}\n',encoding='utf-8')
(root/'results'/'.gitkeep').write_text('')
print(root)
