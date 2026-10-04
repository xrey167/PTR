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
USE_PATH = re.compile(r"\b((?:crate|super|self)(?:::[a-z_][a-z0-9_]*)+)")
USE_STATEMENT = re.compile(r"\buse\s+([^;]+);")
USE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|::|[{},*]")
RAW_STRING = re.compile(r'b?r(#*)"')
CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|.)|[^\\'\n])'")


def posix(path):
    """A relative path with `/` separators, so keys match the description files on any OS."""
    return path.replace(os.sep, "/")


def read(path):
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def non_blank(text):
    return sum(1 for line in text.splitlines() if line.strip())


def code_only(text):
    """The source with comments, string literals and char literals removed.

    A path in a comment (a Rustdoc link such as ``[x](crate::certify)``) or in a string
    names nothing the code uses, so it must not become an edge of the module map.
    Lifetimes (``'a``) are not char literals and stay. Nested block comments and raw
    strings are handled; everything removed is replaced by a space so tokens do not
    merge.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")
        elif c == "/" and text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif c in "rb" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) and RAW_STRING.match(text, i):
            hashes = RAW_STRING.match(text, i).group(1)
            end = text.find('"' + hashes, RAW_STRING.match(text, i).end())
            i = n if end == -1 else end + 1 + len(hashes)
            out.append(" ")
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append(" ")
        elif c == "'" and CHAR_LITERAL.match(text, i):
            i = CHAR_LITERAL.match(text, i).end()
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def expand_use(spec):
    """Expand one `use` tree into the list of full paths it names.

    ``crate::{a, b::{c, d as e}, f::*}`` gives ``crate::a``, ``crate::b::c``,
    ``crate::b::d`` and ``crate::f``. A rename keeps the original path, a glob names the
    module it globs, and ``self`` inside a group names the group's own prefix.
    """
    tokens = USE_TOKEN.findall(spec)
    position = 0

    def item(prefix):
        nonlocal position
        segments = list(prefix)
        while position < len(tokens) and tokens[position] not in ("{", "}", ",", "*"):
            if tokens[position] == "::":
                position += 1
            elif tokens[position] == "as":
                position += 2
            else:
                segments.append(tokens[position])
                position += 1
        found = []
        if position < len(tokens) and tokens[position] == "{":
            position += 1
            found += group(tuple(segments))
        else:
            if position < len(tokens) and tokens[position] == "*":
                position += 1
            if segments and segments[-1] == "self" and len(segments) > len(prefix):
                segments.pop()
            found.append(tuple(segments))
        return found

    def group(prefix):
        nonlocal position
        found = []
        while position < len(tokens) and tokens[position] != "}":
            found += item(prefix)
            if position < len(tokens) and tokens[position] == ",":
                position += 1
        position += 1  # the closing brace
        return found

    return ["::".join(path) for path in item(()) if path]


def module_path_of(rel):
    """The module path of a source file under `src/`, as a tuple of segments.

    `lib.rs` and `main.rs` are the crate root, `a/mod.rs` is `a`, `a/b.rs` is `a::b`.
    """
    parts = posix(rel)[:-3].split("/")
    if parts[-1] in ("lib", "main") and len(parts) == 1:
        return ()
    if parts[-1] == "mod":
        parts = parts[:-1]
    return tuple(parts)


def resolve_use(path, module_paths, current):
    """Resolve `crate::a::b::item`, `self::x` or `super::x` to the module it names.

    The path is walked from its base (the crate root, the current module, or the
    current module's parent for each `super`) and the longest prefix that is a known
    module of this package wins, so `crate::experiments::json::json_string` names the
    module `json` and not `experiments`. A path that reaches no module below its base
    names an item the base module defines or re-exports, so it names the base module
    (the crate root for `crate::Item`, the parent for `super::Item`), unless the base
    is the current module itself.
    """
    segments = path.split("::")
    base = {"crate": (), "self": current}.get(segments[0])
    rest = segments[1:]
    if base is None:  # one or more `super`
        base = current
        while segments and segments[0] == "super":
            base = base[:-1]
            segments = segments[1:]
        rest = segments
    for length in range(len(rest), 0, -1):
        candidate = base + tuple(rest[:length])
        if candidate in module_paths:
            return module_paths[candidate]
    # No module below the base matches, so the path names an item the base module
    # itself defines or re-exports (`use super::{PgSubstrate}`): a dependency on the base.
    if rest and base != current and base in module_paths:
        return module_paths[base]
    return None


def analyse_source(path, text, module_paths, current):
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
    code = code_only(text)
    paths = [match.group(1) for match in USE_PATH.finditer(code)]
    for statement in USE_STATEMENT.finditer(code):
        # Only paths written from `crate`, `self` or `super` are recorded; a relative
        # path such as `pub use child::Item` (a re-export of a child module) is not.
        paths += [path for path in expand_use(statement.group(1))
                  if path.split("::")[0] in ("crate", "self", "super")]
    for path in paths:
        target = resolve_use(path, module_paths, current)
        if target is not None:
            uses.add(target)
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
    module_paths = {}
    for f in src_files:
        rel_file = posix(os.path.relpath(f, os.path.join(directory, "src")))
        module_paths[module_path_of(rel_file)] = module_name_of(rel_file)
    modules = []
    for path in src_files:
        rel = posix(os.path.relpath(path, os.path.join(directory, "src")))
        text = read(path)
        info = analyse_source(path, text, module_paths, module_path_of(rel))
        info["path"] = rel
        info["name"] = module_name_of(rel)
        info["declared"] = [m.group(2) for m in map(MOD_DECL.match, text.splitlines()) if m]
        modules.append(info)

    tests = []
    for path in sorted(glob.glob(os.path.join(directory, "tests", "**", "*.rs"), recursive=True)):
        text = read(path)
        tests.append({
            "path": posix(os.path.relpath(path, directory)),
            "loc": non_blank(text),
            "tests": len(TEST_ATTR.findall(text)),
        })

    features = sorted(manifest.get("features", {}).keys())
    return {
        "name": name,
        "kind": kind,
        "path": posix(os.path.relpath(directory, ROOT)),
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
                    rel = posix(os.path.relpath(path, full))
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

    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"packages": packages}, handle, indent=1)
    print(f"{len(packages)} packages, {sum(len(p['modules']) for p in packages)} modules -> {OUT}")


if __name__ == "__main__":
    main()
