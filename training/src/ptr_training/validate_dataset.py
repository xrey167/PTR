from __future__ import annotations
import argparse, json
from pathlib import Path

def validate(path: Path) -> tuple[int,int]:
    ok=bad=0
    with path.open('r', encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try:
                obj=json.loads(line)
                if not isinstance(obj,dict): raise ValueError('record must be object')
                ok += 1
            except Exception as exc:
                bad += 1
                print(f'{path}:{line_no}: {exc}')
    return ok,bad

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path', type=Path); args=ap.parse_args()
    ok,bad=validate(args.path); print(f'ok={ok} bad={bad}'); raise SystemExit(1 if bad else 0)

if __name__=='__main__': main()
