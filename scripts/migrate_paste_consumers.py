"""Checksum-verified, explicit compatibility migration; never invoked by normal CI.

Alias the maintained pastey package in actual parent dependencies instead of
renaming the obsolete paste package or suppressing RUSTSEC-2024-0436.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ['Cargo.toml', 'model/burn-a0/Cargo.toml']


def metadata(manifest: str) -> dict:
    return json.loads(subprocess.check_output(['cargo', '+stable', 'metadata', '--manifest-path', manifest,
                      '--format-version', '1', '--all-features'], cwd=ROOT, text=True))


def extract_pinned(name: str, version: str, digest: str, target: Path) -> dict:
    if target.exists():
        raise ValueError(f'destination already exists: {target}')
    url = f'https://static.crates.io/crates/{name}/{name}-{version}.crate'
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != digest:
        raise ValueError(f'archive checksum mismatch: {name}@{version}')
    prefix = f'{name}-{version}/'
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        for item in bundle:
            if item.isdir():
                continue
            if not item.isfile() or not item.name.startswith(prefix):
                raise ValueError(f'unsafe archive member: {item.name}')
            relative = Path(item.name[len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'unsafe archive path: {relative}')
            stream = bundle.extractfile(item)
            if stream is None:
                raise ValueError(f'missing archive content: {item.name}')
            data = stream.read()
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    return {'name': name, 'version': version, 'archive': url, 'sha256': digest,
            'directory': target.relative_to(ROOT).as_posix(), 'upstream_files': files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plans = {}
    for manifest in MANIFESTS:
        graph = metadata(manifest)
        packages = {p['id']: p for p in graph['packages']}
        paste = {p['id'] for p in graph['packages'] if p['name'] == 'paste'}
        parents = [packages[n['id']] for n in graph['resolve']['nodes'] if paste.intersection(n['dependencies'])]
        lock = tomllib.loads((ROOT / manifest).with_name('Cargo.lock').read_text())
        checksums = {(p['name'], p['version']): p.get('checksum') for p in lock['package']}
        plans[manifest] = [(p['name'], p['version'], checksums[(p['name'], p['version'])]) for p in parents]
    print(json.dumps(plans, indent=2))
    if not args.execute:
        return
    origins = json.loads((ROOT / 'vendor/ORIGINS.json').read_text())
    downloaded = {}
    for manifest, parents in plans.items():
        additions = []
        for name, version, digest in parents:
            if not digest:
                raise ValueError(f'no registry checksum: {name}@{version}')
            key = name, version
            directory = ROOT / 'vendor' / f'{name}-{version}'
            if key not in downloaded:
                origin = extract_pinned(name, version, digest, directory)
                cargo = directory / 'Cargo.toml'
                source = cargo.read_text()
                pattern = r'(\[[^\]\n]*dependencies\.paste\]\n)version = "[^"]+"'
                source, count = re.subn(pattern, r'\1version = "=0.2.3"\npackage = "pastey"', source)
                if count == 0:
                    raise ValueError(f'no patchable paste dependency: {cargo}')
                cargo.write_text(source)
                origins.append(origin)
                downloaded[key] = directory
            alias = name.replace('-', '_') + '_pastey_' + re.sub(r'\W', '_', version)
            relative = os.path.relpath(directory, (ROOT / manifest).parent).replace(os.sep, '/')
            additions.append(f'{alias} = {{ package = "{name}", path = "{relative}" }}')
        path = ROOT / manifest
        source = path.read_text()
        if additions:
            if '[patch.crates-io]' not in source:
                source += '\n[patch.crates-io]\n'
            # Both owned manifests keep patch.crates-io as their last table.
            if source.rfind('\n[') != source.rfind('\n[patch.crates-io]'):
                raise ValueError(f'patch table is not last: {path}')
            path.write_text(source + '\n' + '\n'.join(additions) + '\n')
    path = ROOT / 'Cargo.toml'
    source = path.read_text().replace('"vendor/raft", "vendor/raft-proto", "vendor/raft-engine"', '"vendor/*"')
    path.write_text(source)
    (ROOT / 'vendor/ORIGINS.json').write_text(json.dumps(origins, indent=2) + '\n')
    for manifest in MANIFESTS:
        graph = metadata(manifest)
        active = {n['id'] for n in graph['resolve']['nodes']}
        bad = [p['id'] for p in graph['packages'] if p['name'] == 'paste' and p['id'] in active]
        if bad:
            raise ValueError(f'active obsolete macro dependencies remain: {bad}')

if __name__ == '__main__':
    main()
