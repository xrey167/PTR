#!/usr/bin/env python3
"""Extract the structure of the PTR workspace into one JSON document.

Everything here is read from the source tree: Cargo manifests, `mod`
declarations, `use` paths, `#[test]` attributes and public items. Nothing is
estimated. The output feeds the atlas page.
"""
import glob
import json
import os
import re
import sys

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "atlas_data.json"

PUB_ITEM = re.compile(
    r"^\s*pub(?:\([a-z]+\))?\s+(?:async\s+|unsafe\s+|const\s+)*(struct|enum|trait|fn|type|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
MOD_DECL = re.compile(r"^\s*(pub(?:\([a-z]+\))?\s+)?mod\s+([a-z_][a-z0-9_]*)\s*[;{]")
TEST_ATTR = re.compile(r"#\[(?:tokio::)?test(?:\(|\])")
USE_CRATE = re.compile(r"\b(?:crate|super|self)::([a-z_][a-z0-9_]*)")


def read(path):
    with open(path, errors="ignore") as handle:
        return handle.read()


def non_blank(text):
    return sum(1 for line in text.splitlines() if line.strip())


def analyse_source(path, text, module_names):
    doc = []
    for line in text.splitlines():
        if line.startswith("//!"):
            doc.append(line[3:].strip())
        elif line.strip() and not line.startswith("//"):
            break
    items = []
    for line in text.splitlines():
        match = PUB_ITEM.match(line)
        if match and not line.lstrip().startswith("//"):
            items.append({"kind": match.group(1), "name": match.group(2)})
    uses = set()
    for match in USE_CRATE.finditer(text):
        name = match.group(1)
        if name in module_names:
            uses.add(name)
    return {
        "loc": non_blank(text),
        "doc": " ".join(doc).strip(),
        "tests": len(TEST_ATTR.findall(text)),
        "items": items,
        "uses": sorted(uses),
    }


def module_name_of(rel):
    base = os.path.basename(rel)
    if base in ("lib.rs", "main.rs"):
        return "(root)"
    if base == "mod.rs":
        return os.path.basename(os.path.dirname(rel))
    return base[:-3]


def package_info(directory, kind):
    manifest = tomllib.load(open(os.path.join(directory, "Cargo.toml"), "rb"))
    package = manifest.get("package", {})
    name = package.get("name", os.path.basename(directory))

    def ptr_deps(table):
        found = {}
        for key, value in manifest.get(table, {}).items():
            if key.startswith("ptr"):
                optional = isinstance(value, dict) and bool(value.get("optional"))
                found[key] = {"optional": optional}
        return found

    src_files = sorted(glob.glob(os.path.join(directory, "src", "**", "*.rs"), recursive=True))
    module_names = {module_name_of(os.path.relpath(f, os.path.join(directory, "src"))) for f in src_files}
    modules = []
    for path in src_files:
        rel = os.path.relpath(path, os.path.join(directory, "src"))
        text = read(path)
        info = analyse_source(path, text, module_names)
        info["path"] = rel
        info["name"] = module_name_of(rel)
        info["declared"] = [m.group(2) for m in map(MOD_DECL.match, text.splitlines()) if m]
        modules.append(info)

    tests = []
    for path in sorted(glob.glob(os.path.join(directory, "tests", "**", "*.rs"), recursive=True)):
        text = read(path)
        tests.append({
            "path": os.path.relpath(path, directory),
            "loc": non_blank(text),
            "tests": len(TEST_ATTR.findall(text)),
        })

    features = sorted(manifest.get("features", {}).keys())
    return {
        "name": name,
        "kind": kind,
        "path": os.path.relpath(directory, ROOT),
        "description": package.get("description", ""),
        "features": features,
        "deps": ptr_deps("dependencies"),
        "dev_deps": ptr_deps("dev-dependencies"),
        "modules": modules,
        "test_files": tests,
        "src_loc": sum(m["loc"] for m in modules),
        "src_tests": sum(m["tests"] for m in modules),
        "integration_tests": sum(t["tests"] for t in tests),
    }


def main():
    packages = []
    for directory in sorted(glob.glob(os.path.join(ROOT, "crates", "ptr-*"))):
        if os.path.exists(os.path.join(directory, "Cargo.toml")):
            packages.append(package_info(directory, "crate"))
    for directory in sorted(glob.glob(os.path.join(ROOT, "bins", "*"))):
        if os.path.exists(os.path.join(directory, "Cargo.toml")):
            packages.append(package_info(directory, "bin"))
    for directory, kind in (("model/burn-a0", "separate"), ("fuzz", "separate")):
        full = os.path.join(ROOT, directory)
        if os.path.exists(os.path.join(full, "Cargo.toml")):
            info = package_info(full, kind)
            if directory == "fuzz":
                for path in glob.glob(os.path.join(full, "fuzz_targets", "*.rs")):
                    text = read(path)
                    rel = os.path.relpath(path, full)
                    info["modules"].append({
                        "path": rel, "name": os.path.basename(path)[:-3], "loc": non_blank(text),
                        "doc": "", "tests": 0, "items": [], "uses": [], "declared": [],
                    })
                info["src_loc"] = sum(m["loc"] for m in info["modules"])
            packages.append(info)

    by_name = {p["name"]: p for p in packages}
    # Which declared dependencies does the source really name? A crate `ptr-x`
    # appears in Rust source as the path root `ptr_x`.
    for package in packages:
        source_text = ""
        for module in package["modules"]:
            full = os.path.join(ROOT, package["path"], "src" if package["kind"] != "separate" or package["name"] != "ptr-fuzz" else "", module["path"])
            if os.path.exists(full):
                source_text += read(full)
            else:
                alt = os.path.join(ROOT, package["path"], module["path"])
                if os.path.exists(alt):
                    source_text += read(alt)
        for test in package["test_files"]:
            source_text += read(os.path.join(ROOT, package["path"], test["path"]))
        for table in ("deps", "dev_deps"):
            for dep, meta in package[table].items():
                token = dep.replace("-", "_")
                meta["used"] = bool(re.search(r"\b" + re.escape(token) + r"\b", source_text))

    for package in packages:
        package["used_by"] = sorted(
            other["name"] for other in packages
            if package["name"] in other["deps"] and other["deps"][package["name"]].get("used")
        )
        package["declared_by"] = sorted(
            other["name"] for other in packages if package["name"] in other["deps"]
        )

    with open(OUT, "w") as handle:
        json.dump({"packages": packages}, handle, indent=1)
    print(f"{len(packages)} packages, {sum(len(p['modules']) for p in packages)} modules -> {OUT}")


if __name__ == "__main__":
    main()
