"""One-shot dependency migration for the security review branch, never main.

Archives are pinned to checksums from PTR's original lockfile. Original source
and licenses are retained; downstream modifications remain a PTR maintenance
responsibility, not a claim that upstream released a patched version.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'raft': ('0.7.0', 'f12688b23a649902762d4c11d854d73c49c9b93138f2de16403ef9f571ad5bae'),
    'raft-proto': ('0.7.0', 'fb6884896294f553e8d5cfbdb55080b9f5f2f43394afff59c9f077e0f4b46d6b'),
    'raft-engine': ('0.4.2', '1213c3a24e3fee8afcc74b2be08c4081adde96f092e0fc1c607abb3e16ae722e'),
}

def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text()
    if old not in source:
        raise ValueError(f'migration precondition does not match: {path}: {old!r}')
    target.write_text(source.replace(old, new))


def vendor() -> None:
    records = []
    for name, (version, digest) in PINS.items():
        destination = ROOT / 'vendor' / name
        if destination.exists():
            raise ValueError(f'vendor directory already exists: {destination}')
        url = f'https://static.crates.io/crates/{name}/{name}-{version}.crate'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ValueError(f'archive checksum mismatch for {name}: {actual}')
        prefix = f'{name}-{version}/'
        files = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for entry in archive.getmembers():
                if entry.isdir():
                    continue
                if not entry.isfile() or not entry.name.startswith(prefix):
                    raise ValueError(f'unsafe crate archive entry: {entry.name}')
                relative = Path(entry.name[len(prefix):])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'unsafe archive path: {relative}')
                stream = archive.extractfile(entry)
                assert stream is not None
                content = stream.read()
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files[relative.as_posix()] = hashlib.sha256(content).hexdigest()
        records.append({'name': name, 'version': version, 'archive': url,
                        'sha256': digest, 'upstream_files': files})
    (ROOT / 'vendor' / 'ORIGINS.json').write_text(json.dumps(records, indent=2) + '\n')


def main() -> None:
    vendor()
    path = ROOT / 'vendor/raft/Cargo.toml'
    text = path.read_text()
    text, count = re.subn(r'(\[dependencies.protobuf\]\nversion = "2[^\n]*\n)', r'\1optional = true\n', text)
    if count != 1:
        raise ValueError('expected raft protobuf dependency')
    text = text.replace('[dependencies.fxhash]\nversion = "0.2.1"', '[dependencies.rustc-hash]\nversion = "2.1"')
    text = text.replace('"raft-proto/protobuf-codec",', '"raft-proto/protobuf-codec",\n    "dep:protobuf",')
    path.write_text(text)
    for path in (ROOT / 'vendor/raft').rglob('*.rs'):
        text = path.read_text()
        path.write_text(text.replace('fxhash::', 'rustc_hash::'))
    path = ROOT / 'vendor/raft-proto/Cargo.toml'
    text = path.read_text()
    text, count = re.subn(r'(\[dependencies.protobuf\]\nversion = "2[^\n]*\n)', r'\1optional = true\n', text)
    if count != 1:
        raise ValueError('expected raft-proto protobuf dependency')
    text = re.sub(r'(\[build-dependencies.protobuf-build\]\nversion = )"[^"]+"', r'\1"0.15.1"', text)
    path.write_text(text)
    path = ROOT / 'vendor/raft-engine/Cargo.toml'
    text = path.read_text()
    text = re.sub(r'(\[dependencies.protobuf\]\nversion = )"[^"]+"', r'\1"3.7.2"', text)
    text = re.sub(r'(\[dependencies.prometheus\]\nversion = )"[^"]+"', r'\1"0.14"', text)
    path.write_text(text)
    for path in (ROOT / 'vendor/raft-engine/src').rglob('*.rs'):
        text = path.read_text()
        path.write_text(text.replace('protobuf::ProtobufError', 'protobuf::Error'))
    replace('Cargo.toml', 'exclude = ["fuzz", "model/burn-a0"]',
            'exclude = ["fuzz", "model/burn-a0", "vendor/raft", "vendor/raft-proto", "vendor/raft-engine"]')
    with (ROOT / 'Cargo.toml').open('a') as output:
        output.write('\n# Compatibility patches: original archives and changes are recorded in vendor/.\n[patch.crates-io]\nraft = { path = "vendor/raft" }\nraft-proto = { path = "vendor/raft-proto" }\nraft-engine = { path = "vendor/raft-engine" }\n')
    path = ROOT / 'crates/ptr-net/Cargo.toml'
    text = path.read_text()
    text = re.sub(r'iroh-backend = \[[^\n]+', 'iroh-backend = ["dep:iroh"]', text)
    text = re.sub(r'^iroh = .*$', 'iroh = { version = "=1.2.0", optional = true, default-features = false, features = ["tls-ring"] }', text, flags=re.M)
    text = re.sub(r'^(?:url|idna|idna_adapter|time) = .*\n', '', text, flags=re.M)
    path.write_text(text)
    replace('crates/ptr-net/src/lib.rs', 'Endpoint, NodeAddr, RelayMode,', 'Endpoint, EndpointAddr, RelayMode,')
    replace('crates/ptr-net/src/lib.rs', 'Endpoint::builder()', 'Endpoint::builder(iroh::endpoint::presets::Minimal)')
    replace('crates/ptr-net/src/lib.rs', '.bind_addr_v4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))', '.clear_ip_transports()\n                .bind_addr(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))\n                .map_err(|error| error.to_string())?')
    replace('crates/ptr-net/src/lib.rs', '.node_id()', '.id()')
    replace('crates/ptr-net/src/lib.rs', 'pub fn direct_addr(&self) -> NodeAddr {\n            NodeAddr::new(self.endpoint.id())\n                .with_direct_addresses([self.endpoint.bound_sockets().0])\n        }', 'pub fn direct_addr(&self) -> EndpointAddr {\n            self.endpoint.addr()\n        }')
    replace('crates/ptr-net/src/lib.rs', 'peer: NodeAddr,', 'peer: EndpointAddr,')
    replace('crates/ptr-net/src/lib.rs', 'peer.node_id', 'peer.id')
    path = ROOT / 'crates/ptr-net/src/lib.rs'
    text = path.read_text()
    text = re.sub(r'connection\s*\.remote_node_id\(\)\s*\.map_err\(\|error\| error.to_string\(\)\)\?', 'connection.remote_id()', text)
    path.write_text(text)
    replace('model/burn-a0/Cargo.toml', 'version = "=0.18.0"', 'version = "=0.22.0-pre.3"')
    replace('model/burn-a0/Cargo.toml', 'rust-version = "1.85"', 'rust-version = "1.95"')
    replace('model/burn-a0/Cargo.toml', 'bytemuck = "=1.23.1"', 'bytemuck = "1.23.1"')
    replace('crates/ptr-ledger/Cargo.toml', '[dependencies]\n', '[dependencies]\nfs4 = { version = "=1.1.0", default-features = false, features = ["sync"] }\n')
    (ROOT / 'vendor/SECURITY-PATCHES.md').write_text('''# Dependency compatibility patches — review candidate\n\nOriginal crate names/versions are retained; there are no forged version numbers\nor RustSec ignore entries. ORIGINS.json pins published archives and every original\nfile. This directory is an explicit PTR maintenance responsibility, not a claim\nthat upstream has released these patches.\n\n* raft 0.7.0: protobuf-codec dependency is optional, matching upstream; prost\n  remains selected. Replace abandoned fxhash with maintained rustc-hash.\n* raft-proto 0.7.0: optional protobuf dependency; protobuf-build 0.15.1 supports\n  selecting prost without dragging in protobuf 2.\n* raft-engine 0.4.2: migrate real protobuf dependency/API to 3.7.2 and the\n  corresponding Prometheus 0.14 API. No log-payload or protocol format waiver.\n\nCompilation, ledger round-trip, replay and corruption tests must pass on the\nactual selected features before integration. Further compiler-driven migrations\nare recorded in the PR review. All upstream sources and licenses are retained.\n''')

if __name__ == '__main__':
    main()
