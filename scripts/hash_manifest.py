from __future__ import annotations
import argparse, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def rows():
    for p in sorted(x for x in ROOT.rglob('*') if x.is_file() and '.git' not in x.parts):
        if p.name in {'Cargo.lock'}:
            pass
        yield f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    args=ap.parse_args()
    text='\n'.join(rows())+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text,encoding='utf-8')
        print(args.out)
    else:
        print(text,end='')

if __name__=='__main__':
    main()
