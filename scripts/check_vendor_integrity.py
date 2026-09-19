"""Check checksum-pinned downstream dependencies without fetching or rewriting them."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> dict:
    vendor = root / 'vendor'
    origins_path = vendor / 'ORIGINS.json'
    origins = json.loads(origins_path.read_text(encoding='utf-8'))
    if not origins:
        raise ValueError('no recorded dependency origins')
    packages = []
    claimed = set()
    for origin in origins:
        relative = PurePosixPath(origin.get('directory', 'vendor/' + origin['name']))
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'vendor':
            raise ValueError(f'unsafe dependency directory: {relative}')
        directory = root / relative
        if directory.is_symlink() or not directory.resolve().is_relative_to(vendor.resolve()):
            raise ValueError(f'unsafe dependency path: {relative}')
        if relative.as_posix() in claimed:
            raise ValueError(f'duplicate dependency directory: {relative}')
        claimed.add(relative.as_posix())
        package = tomllib.loads((directory / 'Cargo.toml').read_text(encoding='utf-8'))['package']
        if (package['name'], package['version']) != (origin['name'], origin['version']):
            raise ValueError(f'package identity changed: {relative}')
        if len(origin['sha256']) != 64 or any(c not in '0123456789abcdef' for c in origin['sha256']):
            raise ValueError(f'invalid archive digest: {relative}')
        files = {}
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'symlink in dependency: {path}')
            if path.is_file():
                files[path.relative_to(directory).as_posix()] = digest(path.read_bytes())
        upstream = origin['upstream_files']
        for path, expected in upstream.items():
            if PurePosixPath(path).name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE', 'PATENTS')):
                if files.get(path) != expected:
                    raise ValueError(f'upstream legal notice changed or missing: {relative}/{path}')
        changed = {path: {'upstream_sha256': upstream.get(path), 'current_sha256': files.get(path)}
                   for path in sorted(files.keys() | upstream.keys()) if files.get(path) != upstream.get(path)}
        packages.append({'directory': relative.as_posix(), 'name': origin['name'],
                         'version': origin['version'], 'archive_sha256': origin['sha256'],
                         'changes': changed, 'files': files})
    actual_directories = {p.relative_to(root).as_posix() for p in vendor.iterdir() if p.is_dir()}
    if actual_directories != claimed:
        raise ValueError(f'unrecorded/missing dependency directories: {actual_directories ^ claimed}')
    return {'schema': 1, 'origins_sha256': digest(origins_path.read_bytes()), 'packages': packages}


def check(root: Path) -> None:
    expected = json.loads((root / 'vendor/CURRENT.json').read_text(encoding='utf-8'))
    actual = inventory(root)
    if actual != expected:
        raise ValueError('vendor integrity mismatch: review changes and explicitly regenerate CURRENT.json')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Explicit maintenance only; normal CI never uses this.')
    args = parser.parse_args()
    try:
        if args.write:
            (ROOT / 'vendor/CURRENT.json').write_text(json.dumps(inventory(ROOT), indent=2) + '\n', encoding='utf-8')
        else:
            check(ROOT)
        print('Vendor identities, original legal notices and complete file inventory verified.')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
