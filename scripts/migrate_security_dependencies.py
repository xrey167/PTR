#!/usr/bin/env python3
"""One-time, fail-closed dependency migration. Run on the remediation branch only.

Downloads retain upstream identities, licenses and source checksums. Local patches
change real dependencies/APIs; they never rename an affected package to evade an
audit. No RustSec advisory is ignored. Inspect vendor/manifest.json and the diff.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
RAFT_REV = "ad13f3d90780f53aea2488c6a4b76c0d334bf136"
RECORDS: list[dict] = []
PATCHES: dict[str, dict[tuple[str, str], Path]] = {"Cargo.toml": {}, "model/burn-a0/Cargo.toml": {}}


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True)


def replace(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text()
    if text.count(old) != count:
        raise ValueError(f"{path}: expected {count} occurrences of {old!r}, found {text.count(old)}")
    path.write_text(text.replace(old, new))


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "PTR-security-migration/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def extract(data: bytes, destination: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(destination, filter="data")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("unexpected source archive layout")
    return roots[0]


def registry_source(name: str, version: str) -> Path:
    dest = VENDOR / f"{name}-{version}"
    if dest.exists():
        return dest
    info = json.loads(fetch(f"https://crates.io/api/v1/crates/{name}/{version}"))["version"]
    url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
    data = fetch(url)
    digest = hashlib.sha256(data).hexdigest()
    if digest != info["checksum"]:
        raise ValueError(f"registry checksum mismatch for {name} {version}")
    with tempfile.TemporaryDirectory() as tmp:
        source = extract(data, Path(tmp))
        shutil.copytree(source, dest)
    RECORDS.append({"name": name, "version": version, "source": url, "archive_sha256": digest,
                    "patches": [], "path": dest.relative_to(ROOT).as_posix()})
    return dest


def note(path: Path, text: str) -> None:
    record = next(item for item in RECORDS if item["path"] == path.relative_to(ROOT).as_posix())
    record["patches"].append(text)


def dependency(path: Path, alias: str, version: str, package: str | None = None) -> None:
    """Modify normalized Cargo dependency tables while retaining features/optionality."""
    text = path.read_text()
    pattern = re.compile(r"(?m)^(\[(?:[^\n]*\.)?(?:dependencies|dev-dependencies|build-dependencies)\." + re.escape(alias) + r"\]\n)([^\[]*)")
    found = 0
    def edit(match: re.Match[str]) -> str:
        nonlocal found
        body = match.group(2)
        body, changed = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', body, count=1)
        if changed != 1:
            raise ValueError(f"missing registry version for {alias} in {path}")
        if package is not None:
            body = re.sub(r'(?m)^package\s*=\s*"[^"]+"\n?', '', body)
            body = f'package = "{package}"\n' + body
        found += 1
        return match.group(1) + body
    text = pattern.sub(edit, text)
    if not found:
        raise ValueError(f"missing dependency {alias} in {path}")
    path.write_text(text)
    tomllib.loads(text)


def register(graph: str, name: str, version: str, source: Path) -> None:
    PATCHES[graph][(name, version)] = source


def write_patch_tables() -> None:
    for graph, patches in PATCHES.items():
        manifest = ROOT / graph
        text = manifest.read_text().split("\n# PTR SECURITY PATCHES\n", 1)[0].rstrip() + "\n"
        text += "\n# PTR SECURITY PATCHES\n[patch.crates-io]\n"
        for (name, version), path in sorted(patches.items()):
            import os
            relative = Path(os.path.relpath(path, manifest.parent)).as_posix()
            key = name + "-" + version.replace(".", "-")
            text += f'{key} = {{ package = "{name}", path = "{relative}" }}\n'
        manifest.write_text(text)


def prepare_raft() -> None:
    url = f"https://api.github.com/repos/tikv/raft-rs/tarball/{RAFT_REV}"
    data = fetch(url)
    with tempfile.TemporaryDirectory() as tmp:
        source = extract(data, Path(tmp))
        for name, relative in [("raft", "."), ("raft-proto", "proto")]:
            dest = VENDOR / f"{name}-0.7.0"
            shutil.copytree(source / relative, dest)
            if name == "raft":
                # The sibling proto is copied independently. Other workspace
                # packages are not part of the published raft dependency.
                for child in ["proto", "harness", "datadriven", ".github"]:
                    shutil.rmtree(dest / child, ignore_errors=True)
                (dest / "Cargo.lock").unlink(missing_ok=True)
                cargo = dest / "Cargo.toml"
                replace(cargo, '[workspace]\nmembers = ["proto", "harness", "datadriven"]\n', '')
                replace(cargo, 'path = "proto", version = "0.7.0"', 'path = "../raft-proto-0.7.0", version = "0.7.0"')
                replace(cargo, 'fxhash = "0.2.1"', 'fxhash = { package = "rustc-hash", version = "=2.1.3" }')
                replace(cargo, 'datadriven = { path = "datadriven", version = "0.1.0" }', 'datadriven = "0.1.0"')
            RECORDS.append({"name": name, "version": "0.7.0", "source": url,
                            "upstream_revision": RAFT_REV, "archive_sha256": hashlib.sha256(data).hexdigest(),
                            "path": dest.relative_to(ROOT).as_posix(),
                            "patches": ["Pinned upstream prost-codec removes mandatory protobuf 2; preserve upstream licenses."] +
                            (["Use maintained rustc-hash through the existing fxhash alias; remove uncopied workspace members and local dev paths."] if name == "raft" else [])})
            # A vendored dependency must not become a root workspace member:
            # --all-features must not enable its unused protobuf-codec backend.
            with (dest / "Cargo.toml").open("a") as f:
                f.write("\n[workspace]\n")
            register("Cargo.toml", name, "0.7.0", dest)
    engine = registry_source("raft-engine", "0.4.2")
    dependency(engine / "Cargo.toml", "protobuf", "=3.7.2")
    dependency(engine / "Cargo.toml", "prometheus", "=0.14.0")
    replace(engine / "src/errors.rs", "protobuf::ProtobufError", "protobuf::Error")
    source = engine / "src/engine.rs"
    replace(source, "use protobuf::{parse_from_bytes, Message};", "use protobuf::Message;")
    replace(source, "Some(parse_from_bytes(&value)?)", "Some(S::parse_from_bytes(&value)?)")
    replace(source, "if let Ok(v) = parse_from_bytes(raw_v)", "if let Ok(v) = S::parse_from_bytes(raw_v)")
    replace(source, "let e = parse_from_bytes(", "let e = <M::Entry as Message>::parse_from_bytes(")
    with (engine / "Cargo.toml").open("a") as f:
        f.write("\n[workspace]\n")
    note(engine, "Upgrade protobuf 2 to 3.7.2 and prometheus to 0.14.0; adapt Message parsing/error APIs without changing PTR record encoding.")
    register("Cargo.toml", "raft-engine", "0.4.2", engine)


def prepare_network() -> None:
    cargo = ROOT / "crates/ptr-net/Cargo.toml"
    text = cargo.read_text()
    start = text.index('iroh-backend = [')
    end = text.index('\n[dependencies]', start)
    text = text[:start] + 'iroh-backend = ["dep:iroh"]\n' + text[end:]
    start = text.index('iroh = {')
    end = text.index('[dev-dependencies]', start)
    text = text[:start] + '# Optional transport requires Rust 1.91; the default core retains Rust 1.85.\niroh = { version = "=1.2.0", optional = true, default-features = false, features = ["tls-ring"] }\n\n' + text[end:]
    cargo.write_text(text)
    source = ROOT / "crates/ptr-net/src/lib.rs"
    replace(source, 'endpoint::{Connection, SendStream}', 'endpoint::{presets, Connection, SendStream}')
    replace(source, 'Endpoint, NodeAddr, RelayMode,', 'Endpoint, EndpointAddr, RelayMode, TransportAddr,')
    replace(source, 'Endpoint::builder()', 'Endpoint::builder(presets::Minimal)')
    replace(source, '.bind_addr_v4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))', '.bind_addr(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))\n                .map_err(|error| error.to_string())?')
    replace(source, 'self.endpoint.node_id()', 'self.endpoint.id()', count=2)
    replace(source, 'pub fn direct_addr(&self) -> NodeAddr {\n            NodeAddr::new(self.endpoint.id())\n                .with_direct_addresses([self.endpoint.bound_sockets().0])\n        }', 'pub fn direct_addr(&self) -> EndpointAddr {\n            EndpointAddr::from_parts(\n                self.endpoint.id(),\n                self.endpoint.bound_sockets().into_iter().map(TransportAddr::Ip),\n            )\n        }')
    replace(source, 'peer: NodeAddr,', 'peer: EndpointAddr,')
    replace(source, 'peer.node_id', 'peer.id')
    replace(source, 'connection\n                .remote_node_id()\n                .map_err(|error| error.to_string())?', 'connection.remote_id()', count=2)


def replace_retired_dependencies() -> None:
    """Patch direct consumers, not advisory metadata or the retired packages."""
    for graph in PATCHES:
        metadata = json.loads(run('cargo', '+stable', 'metadata', '--manifest-path', graph, '--format-version', '1', '--all-features'))
        for package in metadata['packages']:
            if not (package.get('source') or '').startswith('registry+'):
                continue
            retired = {dep['name'] for dep in package['dependencies']} & {'paste', 'bincode'}
            if not retired:
                continue
            if 'bincode' in retired and not any(dep['name'] == 'bincode' and ('2.' in dep['req']) for dep in package['dependencies']):
                retired.remove('bincode')
            if not retired:
                continue
            dest = registry_source(package['name'], package['version'])
            manifest = dest / 'Cargo.toml'
            if 'paste' in retired:
                dependency(manifest, 'paste', '=0.2.1', 'pastey')
                note(dest, 'Replace the retired paste dependency with pinned pastey 0.2.1, retaining the paste API alias.')
            if 'bincode' in retired:
                dependency(manifest, 'bincode', '=2.0.4', 'bincode-next')
                note(dest, 'Replace retired bincode 2 with pinned bincode-next 2.0.4, retaining the bincode API alias; record compatibility requires executed tests.')
            if '[workspace]' not in manifest.read_text():
                with manifest.open('a') as f:
                    f.write('\n[workspace]\n')
            register(graph, package['name'], package['version'], dest)
        write_patch_tables()


def update_docs() -> None:
    for name, item in [('ptr-net', 'Iroh 1.2 transport API with current DNS/TLS dependency graph; optional feature MSRV 1.91, default core MSRV 1.85'), ('ptr-ledger', 'Reviewed pinned Raft prost-codec source and raft-engine protobuf 3 migration; upstream source and local patch provenance retained in vendor/manifest.json')]:
        file = ROOT / 'crates' / name / 'component.toml'
        text = file.read_text()
        text = re.sub(r'last_reviewed = "[^"]+"', 'last_reviewed = "2026-09-19"', text)
        text = text.replace('implemented = [', 'implemented = [\n  ' + json.dumps(item) + ',')
        file.write_text(text)
    run('cargo', '+stable', 'fmt', '--all')
    run('python3', 'scripts/update_component_docs.py', '--write')


def main() -> None:
    if run('git', 'branch', '--show-current').strip() != 'agent/security-remediation-20260919':
        raise SystemExit('Refuse to migrate outside the explicit remediation branch')
    if VENDOR.exists():
        raise SystemExit('Refuse to overwrite an existing vendor directory')
    VENDOR.mkdir()
    prepare_raft()
    prepare_network()
    write_patch_tables()
    replace_retired_dependencies()
    write_patch_tables()
    for manifest in ['Cargo.toml', 'model/burn-a0/Cargo.toml', 'fuzz/Cargo.toml', 'templates/rust-crate/Cargo.toml']:
        run('cargo', '+stable', 'update', '--manifest-path', manifest)
    for record in RECORDS:
        path = ROOT / record['path']
        record['files'] = {f.relative_to(path).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in sorted(path.rglob('*')) if f.is_file()}
    (VENDOR / 'manifest.json').write_text(json.dumps({'schema': 1, 'owner': 'PTR maintainers',
        'purpose': 'Real source/API dependency migration; no advisory ignores or falsified package versions.',
        'packages': RECORDS}, indent=2) + '\n')
    (VENDOR / 'README.md').write_text('# Reviewed dependency source patches\n\n'
        'The JSON manifest records immutable upstream inputs, archive SHA256, local changes and file hashes. '
        'Package identities are retained. These sources replace dependencies/APIs, not audit metadata. '
        'All upstream license and notice files are preserved. Review these forks when upgrading; '
        'root and model workspace patches are deliberately separate.\n')
    update_docs()
    print(json.dumps({'vendored_packages': len(RECORDS), 'graphs': list(PATCHES)}, indent=2))


if __name__ == '__main__':
    main()
