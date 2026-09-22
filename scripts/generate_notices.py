"""Assemble THIRD-PARTY-NOTICES.md from the owned workspaces' lockfiles and crate archives.

`docs/SECURITY_P0_REVIEW_20260919.md` states the obligation: "Release packaging
must carry third-party notices/source availability", and for the one MPL
exception, "preserve their notices and provide the corresponding MPL-covered
source and modifications". An SBOM is an inventory and does not discharge that;
this file carries the notices themselves.

Three properties make the artifact worth committing:

* **Coverage cannot silently shrink.** Workspaces come from
  `security_workspaces.workspaces`, which refuses to lose one, and every
  lockfile it finds is read. `colored 3.1.1`, the only MPL-covered package, is
  reachable only through the A0 lockfile — a notices file built from the root
  workspace alone would omit the package the obligation is about.
* **It is reproducible.** Texts come from the pinned `.crate` archives and the
  vendored trees, not from whatever a machine happens to have built, so the same
  lockfiles produce the same document. Run `cargo fetch --locked` in each
  workspace first; a missing archive is an error, never a silent omission.
* **A package without a license expression is a failure**, not a quiet gap. A
  notices file that drops what it cannot describe is worse than none.

Identical texts are emitted once and referenced, because 920 packages carry
hundreds of copies of the same licence. Two MIT texts with different copyright
lines are *not* identical and are both kept: the copyright line is the notice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from security_workspaces import workspaces  # noqa: E402

TEXT_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "UNLICENSE", "NOTICE")
MPL_PACKAGE = ("colored", "3.1.1")


def registry_cache() -> Path:
    home = Path(__file__).resolve()
    for candidate in Path.home().glob(".cargo/registry/cache/*"):
        return candidate
    raise SystemExit(f"no cargo registry cache found (looked under {home.home()})")


def is_text_name(name: str) -> bool:
    return name.upper().startswith(TEXT_PREFIXES)


def collect_packages(root: Path) -> dict[tuple[str, str], dict]:
    """Every package in every owned workspace, with the workspaces it came from."""
    found: dict[tuple[str, str], dict] = {}
    for manifest in workspaces(root):
        lock = root / manifest.parent / "Cargo.lock"
        if not lock.is_file():
            continue
        label = manifest.parent.as_posix() or "."
        data = tomllib.loads(lock.read_text(encoding="utf-8"))
        for package in data.get("package", []):
            key = (package["name"], package["version"])
            entry = found.setdefault(key, {"source": package.get("source"), "workspaces": set()})
            entry["workspaces"].add(label)
            if package.get("source"):
                entry["source"] = package["source"]
    return found


def license_of(meta: dict) -> str | None:
    if meta.get("license"):
        return meta["license"]
    if meta.get("license-file"):
        # A crate may ship a text instead of an SPDX expression; naming the file
        # is honest, and the text itself is included below.
        return f"see {meta['license-file']}"
    return None


def from_vendor(directory: Path) -> tuple[str | None, dict[str, bytes]]:
    meta = tomllib.loads((directory / "Cargo.toml").read_text(encoding="utf-8"))
    texts = {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file() and is_text_name(path.name)
    }
    return license_of(meta.get("package", {})), texts


def from_archive(archive: Path, name: str, version: str) -> tuple[str | None, dict[str, bytes]]:
    prefix = f"{name}-{version}/"
    expression: str | None = None
    texts: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if relative == "Cargo.toml":
                raw = tar.extractfile(member).read().decode("utf-8", "replace")
                expression = license_of(tomllib.loads(raw).get("package", {}))
            elif "/" not in relative and is_text_name(relative):
                texts[relative] = tar.extractfile(member).read()
    return expression, texts


def package_set_digest(keys) -> str:
    material = "\n".join(f"{name} {version}" for name, version in sorted(keys))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build(root: Path, cache: Path) -> tuple[str, list[str]]:
    origins = json.loads((root / "vendor/ORIGINS.json").read_text(encoding="utf-8"))
    vendored = {
        (origin["name"], origin["version"]): origin.get(
            "directory", "vendor/" + origin["name"]
        )
        for origin in origins
    }

    packages = collect_packages(root)
    third_party = {
        key: entry
        for key, entry in packages.items()
        if entry["source"] is not None or key in vendored
    }

    errors: list[str] = []
    texts: dict[str, bytes] = {}
    rows = []

    for (name, version), entry in sorted(third_party.items()):
        if (name, version) in vendored:
            directory = root / vendored[(name, version)]
            expression, files = from_vendor(directory)
            origin_note = f"retained in `{vendored[(name, version)]}`"
        else:
            archive = cache / f"{name}-{version}.crate"
            if not archive.is_file():
                errors.append(
                    f"{name} {version}: no cached archive; run `cargo fetch --locked` "
                    "in each owned workspace"
                )
                continue
            expression, files = from_archive(archive, name, version)
            origin_note = "crates.io"

        if not expression:
            errors.append(f"{name} {version}: no license expression")
            continue

        references = []
        for filename, data in sorted(files.items()):
            digest = hashlib.sha256(data).hexdigest()
            texts.setdefault(digest, data)
            references.append((filename, digest))

        rows.append(
            {
                "name": name,
                "version": version,
                "license": expression,
                "origin": origin_note,
                "workspaces": sorted(entry["workspaces"]),
                "texts": references,
            }
        )

    if errors:
        return "", errors

    order = {digest: index for index, digest in enumerate(sorted(texts), start=1)}
    digest = package_set_digest(row_key(row) for row in rows)
    return render(rows, texts, order, digest), []


def row_key(row: dict) -> tuple[str, str]:
    return (row["name"], row["version"])


def render(rows, texts, order, digest: str) -> str:
    out: list[str] = []
    out.append("# Third-party notices\n")
    out.append(
        "This file is generated by `scripts/generate_notices.py` and verified by\n"
        "`scripts/check_notices.py`, which fails when the dependency graph and this\n"
        "document disagree. Do not edit it by hand.\n"
    )
    out.append(
        "It lists every third-party package reachable from an owned Cargo workspace —\n"
        "the production workspace, the isolated A0 model workspace, the fuzz harness and\n"
        "the crate template — together with the licence text each package ships. Packages\n"
        "belonging to PTR itself are not listed.\n"
    )
    out.append(
        "**This is not legal advice and not a distribution clearance.** It assembles the\n"
        "notices the licences require to travel with redistributed software; deciding\n"
        "whether a particular distribution satisfies them is not something a generator\n"
        "can do.\n"
    )
    out.append(f"`Package-set-digest: sha256:{digest}`\n")
    out.append(f"**{len(rows)} third-party packages · {len(texts)} distinct licence texts**\n")

    out.append("## MPL-2.0 source availability\n")
    mpl = next((row for row in rows if row_key(row) == MPL_PACKAGE), None)
    if mpl is None:
        out.append(
            f"`{MPL_PACKAGE[0]} {MPL_PACKAGE[1]}` is no longer in any owned workspace, so no\n"
            "MPL-covered source accompanies this tree. `deny.toml` still carries the\n"
            "version-scoped exception; remove it once that is intended.\n"
        )
    else:
        out.append(
            f"`{mpl['name']} {mpl['version']}` is covered by MPL-2.0 and is the only\n"
            "MPL-covered package permitted, as a version-scoped exception in `deny.toml`\n"
            f"reached through {', '.join(mpl['workspaces'])}. MPL-2.0 section 3.2 requires that\n"
            "recipients of a binary be able to obtain the covered source.\n"
        )
        if mpl["origin"] == "crates.io":
            # Derived from where the generator actually read the package, so this
            # paragraph cannot keep claiming "unmodified" after someone vendors it.
            out.append(
                "The covered files are the crate as published and PTR has not modified them:\n"
                "the package is absent from `vendor/ORIGINS.json`, so it carries no retained\n"
                "PTR patch, and its version and checksum are pinned in the lockfile. The\n"
                "corresponding source is therefore the published archive\n"
                f"`https://static.crates.io/crates/{mpl['name']}/{mpl['name']}-{mpl['version']}.crate`.\n"
                "A distributor who modifies these files must make their modifications\n"
                "available under the same terms; PTR has made none.\n"
            )
        else:
            out.append(
                f"The covered files are retained in this repository ({mpl['origin']}) and carry a\n"
                "PTR patch, so the corresponding source *and PTR's modifications* must both\n"
                "accompany a redistributed binary. `vendor/CURRENT.json` enumerates exactly\n"
                "which files differ from the published archive.\n"
            )

    out.append("## Packages\n")
    out.append("| Package | Version | Licence | Source | Workspaces | Notices |")
    out.append("|---|---|---|---|---|---|")
    for row in rows:
        references = (
            ", ".join(f"[{name}](#text-{order[digest]:03d})" for name, digest in row["texts"])
            if row["texts"]
            else "_none shipped_"
        )
        out.append(
            f"| {row['name']} | {row['version']} | {row['license']} | {row['origin']} "
            f"| {', '.join(row['workspaces'])} | {references} |"
        )
    out.append("")

    out.append("## Licence texts\n")
    out.append(
        "Identical texts appear once. Two texts that differ only in a copyright line are\n"
        "not identical and both appear, because that line is the notice.\n"
    )
    carriers: dict[str, list[str]] = {}
    for row in rows:
        for _, digest in row["texts"]:
            carriers.setdefault(digest, []).append(f"{row['name']} {row['version']}")
    for digest, index in order.items():
        out.append(f'### <a id="text-{index:03d}"></a>Text {index:03d}\n')
        out.append(f"Carried by: {', '.join(sorted(carriers.get(digest, [])))}\n")
        out.append(f"`sha256:{digest}`\n")
        body = texts[digest].decode("utf-8", "replace").replace("\r\n", "\n").rstrip("\n")
        out.append("```text")
        out.append(body)
        out.append("```\n")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "THIRD-PARTY-NOTICES.md")
    args = parser.parse_args()

    document, errors = build(ROOT, registry_cache())
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    args.out.write_text(document, encoding="utf-8")
    print(f"{args.out}: {len(document) / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
