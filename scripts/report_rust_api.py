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
    depth = 0
    for index in range(start, min(len(lines), start + 20)):
        text = lines[index].strip()
        if not text:
            continue
        parts.append(text)
        depth += text.count("(") + text.count("<") - text.count(")") - text.count(">")
        if ("{" in text or text.endswith(";")) and depth <= 0:
            break
    return " ".join(parts)


def scan_file(crate: str, src: Path, file: Path) -> list[ApiItem]:
    lines = file.read_text(encoding="utf-8").splitlines()
    module = module_name(src, file)
    rel = str(file.relative_to(ROOT)).replace("\\", "/")
    items: list[ApiItem] = []

    for number, line in enumerate(lines, 1):
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


def markdown(items: list[ApiItem]) -> str:
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
            out.extend([f"### `{current_module}`", ""])
        out.append(
            f"- `{item.visibility}` **{item.kind}** `{item.name}` "
            f"— `{item.file}:{item.line}`"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group current Rust declarations by crate/module and visibility."
    )
    parser.add_argument("crate", nargs="*", help="optional ptr-* crate names")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--public-only", action="store_true")
    args = parser.parse_args()

    items: list[ApiItem] = []
    for crate_dir in all_crates(args.crate):
        items.extend(scan_crate(crate_dir))
    items.sort(key=lambda item: (item.crate, item.module, item.file, item.line))

    if args.public_only:
        items = [item for item in items if item.visibility.startswith("pub")]

    if args.format == "json":
        print(json.dumps([asdict(item) for item in items], indent=2))
    else:
        print(markdown(items), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
