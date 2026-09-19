from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FALLIBLE_PREFIXES = ("check_", "validate_", "ensure_")
FN_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")


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


def rust_sources() -> list[Path]:
    paths: list[Path] = []
    for crate in sorted((ROOT / "crates").glob("ptr-*")):
        if (crate / "Cargo.toml").exists():
            paths.extend(sorted((crate / "src").rglob("*.rs")))
    template_src = ROOT / "templates" / "rust-crate" / "src"
    if template_src.exists():
        paths.extend(sorted(template_src.rglob("*.rs")))
    return paths


def check() -> list[str]:
    errors: list[str] = []

    for common_dir in sorted((ROOT / "crates").glob("ptr-*/tests/common")):
        for path in sorted(common_dir.rglob("*.rs")):
            text = path.read_text(encoding="utf-8")
            if "#[test]" in text or "#[tokio::test" in text:
                errors.append(
                    f"{path.relative_to(ROOT)}: common test helpers must not contain test annotations"
                )

    template_common = ROOT / "templates" / "rust-crate" / "tests" / "common"
    if template_common.exists():
        for path in sorted(template_common.rglob("*.rs")):
            text = path.read_text(encoding="utf-8")
            if "#[test]" in text or "#[tokio::test" in text:
                errors.append(
                    f"{path.relative_to(ROOT)}: common test helpers must not contain test annotations"
                )

    for path in rust_sources():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = FN_RE.search(line)
            if not match:
                continue
            name = match.group(1)
            if not name.startswith(FALLIBLE_PREFIXES):
                continue
            signature = collect_signature(lines, index)
            if "-> Result<" not in signature and "-> std::result::Result<" not in signature:
                errors.append(
                    f"{path.relative_to(ROOT)}:{index + 1}: {name} must return Result<..., ...>"
                )

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print("OK: Rust test/common and fallible check-function conventions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
