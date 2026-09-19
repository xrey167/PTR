from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DECL_RE = re.compile(
    r"^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?"
    r"(?:(?:async|const|unsafe|extern(?:\s+\"[^\"]+\")?)\s+)*"
    r"(?P<kind>fn|struct|enum|trait|type|const|static|mod)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
PUB_USE_RE = re.compile(r"^\s*(?P<vis>pub(?:\([^)]*\))?)\s+use\s+(?P<target>.+?);\s*$")
IMPL_START_RE = re.compile(r"^\s*impl(?:<[^>]*>)?\s+")
MACRO_RULES_RE = re.compile(r"^\s*macro_rules!\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class ApiItem:
    crate: str
    module: str
    file: str
    line: int
    visibility: str
    kind: str
    name: str
    signature: str


def visibility(raw: str | None) -> str:
    if raw is None:
        return "private"
    return raw.strip()


def module_name(src: Path, file: Path) -> str:
    rel = file.relative_to(src)
    if rel.name == "lib.rs":
        return "crate"
    if rel.name == "mod.rs":
        parts = rel.parent.parts
    else:
        parts = (*rel.parent.parts, rel.stem)
    return "::".join(parts) if parts else "crate"


def collect_signature(lines: list[str], start: int) -> str:
    parts: list[str] = []
    paren_depth = 0
    angle_depth = 0
    for index in range(start, min(len(lines), start + 30)):
        text = lines[index].strip()
        if not text:
            continue
        parts.append(text)
        paren_depth += text.count("(") - text.count(")")
        angle_depth += text.count("<") - text.count(">")
        if ("{" in text or text.endswith(";")) and paren_depth <= 0 and angle_depth <= 0:
            break
    return " ".join(parts)


def impl_name(signature: str) -> str:
    head = signature.split("{", 1)[0].strip()
    head = re.sub(r"^impl(?:<[^>]*>)?\s+", "", head)
    head = head.split(" where ", 1)[0].strip()
    return head


def source_path(crate: str, src: Path, file: Path) -> str:
    """Use repo-relative paths, or crate-relative paths for external fixtures.

    Resolving both roots makes relative and absolute callers consistent. Never
    expose the machine-specific temporary directory in an API inventory.
    """
    source = file.resolve()
    try:
        relative = source.relative_to(ROOT.resolve())
    except ValueError:
        relative = Path(crate) / source.relative_to(src.resolve().parent)
    return relative.as_posix()


def scan_file(crate: str, src: Path, file: Path) -> list[ApiItem]:
    lines = file.read_text(encoding="utf-8").splitlines()
    module = module_name(src, file)
    rel = source_path(crate, src, file)
    items: list[ApiItem] = []

    for number, line in enumerate(lines, 1):
        macro_match = MACRO_RULES_RE.match(line)
        if macro_match:
            previous = lines[number - 2].strip() if number >= 2 else ""
            exported = previous == "#[macro_export]"
            signature = collect_signature(lines, number - 1)
            items.append(
                ApiItem(
                    crate=crate,
                    module=module,
                    file=rel,
                    line=number,
                    visibility="pub" if exported else "private",
                    kind="macro",
                    name=macro_match.group("name"),
                    signature=signature,
                )
            )
            continue

        use_match = PUB_USE_RE.match(line)
        if use_match:
            target = use_match.group("target").strip()
            items.append(
                ApiItem(
                    crate=crate,
                    module=module,
                    file=rel,
                    line=number,
                    visibility=visibility(use_match.group("vis")),
                    kind="use",
                    name=target,
                    signature=line.strip(),
                )
            )
            continue

        if IMPL_START_RE.match(line):
            signature = collect_signature(lines, number - 1)
            items.append(
                ApiItem(
                    crate=crate,
                    module=module,
                    file=rel,
                    line=number,
                    visibility="implementation",
                    kind="impl",
                    name=impl_name(signature),
                    signature=signature,
                )
            )
            continue

        match = DECL_RE.match(line)
        if not match:
            continue
        items.append(
            ApiItem(
                crate=crate,
                module=module,
                file=rel,
                line=number,
                visibility=visibility(match.group("vis")),
                kind=match.group("kind"),
                name=match.group("name"),
                signature=collect_signature(lines, number - 1),
            )
        )

    return items


def scan_crate(crate_dir: Path) -> list[ApiItem]:
    src = crate_dir / "src"
    if not src.exists():
        return []
    items: list[ApiItem] = []
    for file in sorted(src.rglob("*.rs")):
        items.extend(scan_file(crate_dir.name, src, file))
    return items


def all_crates(selected: list[str]) -> list[Path]:
    available = sorted(
        path
        for path in (ROOT / "crates").glob("ptr-*")
        if (path / "Cargo.toml").exists()
    )
    if not selected:
        return available
    wanted = set(selected)
    return [path for path in available if path.name in wanted]


def markdown(items: list[ApiItem], show_signatures: bool = False) -> str:
    out = ["# PTR Rust API Inventory", ""]
    current_crate = None
    current_module = None
    for item in items:
        if item.crate != current_crate:
            current_crate = item.crate
            current_module = None
            out.extend([f"## {current_crate}", ""])
        if item.module != current_module:
            current_module = item.module
            out.extend([f"### {current_module}", ""])
        out.append(
            f"- {item.visibility} {item.kind} {item.name} "
            f"— {item.file}:{item.line}"
        )
        if show_signatures:
            out.append(f"  - signature: {item.signature}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Group Rust declarations by crate/module/visibility, including trait implementors "
            "and macro_rules declarations. Use --signatures to expose generics, bounds, "
            "lifetimes and macro signatures."
        )
    )
    parser.add_argument("crate", nargs="*", help="optional ptr-* crate names")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument(
        "--signatures",
        action="store_true",
        help="include full declarations so generics, bounds and lifetimes are visible",
    )
    args = parser.parse_args()

    items: list[ApiItem] = []
    for crate_dir in all_crates(args.crate):
        items.extend(scan_crate(crate_dir))
    items.sort(key=lambda item: (item.crate, item.module, item.file, item.line))

    if args.public_only:
        items = [
            item
            for item in items
            if item.visibility.startswith("pub") or item.kind == "impl"
        ]

    if args.format == "json":
        print(json.dumps([asdict(item) for item in items], indent=2))
    else:
        print(markdown(items, show_signatures=args.signatures), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
