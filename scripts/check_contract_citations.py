"""A contract that names a test must name one that exists.

Three bullets in `docs/architecture/` asserted things the tree had stopped doing:
one said `PodRegistry::resolve` "still matches on capability and input type alone"
after it had taken a `ProjectId` for two commits, one said a race had no test in
the same pull request that added six, and `31-cluster-integrity.md` asserted and
denied the same fact about a deposed leader fourteen lines apart. Nothing checks a
sentence, which is how all three survived.

This does not check sentences either, and pretending otherwise would be the same
mistake one level up. What it checks is the part of a bullet that *is* mechanical:
**a cited test name resolves to a test that exists.** The corrections now cite
tests rather than merely asserting properties, so deleting or renaming one makes
this checker fail and point at the document that relies on it — the bullet and the
test fail together, which is the most a checker can offer here.

The heuristic is deliberately narrow. Only a backticked lower-snake identifier of
five or more segments is treated as a citation, because test names in this
repository are sentences (`a_pod_registered_for_one_project_is_invisible_to_another`)
and ordinary identifiers are not (`provenance_bucket_count`, `upstream_retire_when`).
A name that is a citation and does not look like one is missed; that is the cost of
not producing false failures, and it is the right way round.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "target", "vendor", ".venv", "node_modules", "__pycache__"}
# Documents that make claims about what the tree does. Deliberately not every
# document: `docs/RUST_API_STYLE.md:715` writes
# "Rust test functions themselves should name the behavior, e.g.
# `stale_generation_returns_expected_and_actual`" - an illustration of a naming
# convention, not an assertion that such a test exists. This checker found it on
# its first run, which is the right kind of finding and the wrong kind of failure.
# A style guide gives examples; a contract makes claims, and only claims are
# checkable.
DOCUMENTS = (
    "docs/architecture/*.md",
    "docs/PRIORITIES.md",
    "docs/OPEN_ITEMS_PLAN_*.md",
)
# Where a cited name could be defined.
SOURCE_AREAS = ("crates", "model", "bins", "scripts", "training")
# Five or more lower-snake segments: long enough that an ordinary identifier does
# not reach it, short enough that every test name in this repository does.
CITATION = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+){4,})`")
DEFINITION = "(?:fn|def)"


def owned(root: Path, paths):
    return (p for p in paths if not SKIP.intersection(p.relative_to(root).parts))


def definitions(root: Path) -> str:
    """Every source file that could define a cited name, concatenated."""
    chunks = []
    for area in SOURCE_AREAS:
        base = root / area
        if not base.is_dir():
            continue
        for pattern in ("*.rs", "*.py"):
            for path in owned(root, base.rglob(pattern)):
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def citations(root: Path) -> dict[str, set[str]]:
    """Every cited name, mapped to the documents citing it."""
    found: dict[str, set[str]] = {}
    seen: set[Path] = set()
    for pattern in DOCUMENTS:
        for document in sorted(root.glob(pattern)):
            if document in seen:
                continue
            seen.add(document)
            text = document.read_text(encoding="utf-8")
            for match in CITATION.finditer(text):
                found.setdefault(match.group(1), set()).add(
                    str(document.relative_to(root))
                )
    return found


def check(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    blob = definitions(root)
    cited = citations(root)
    for name in sorted(cited):
        if not re.search(rf"\b{DEFINITION} {re.escape(name)}\b", blob):
            where = ", ".join(sorted(cited[name]))
            errors.append(
                f"{name}: cited by {where} and defined nowhere - either the test "
                "was renamed or removed and the contract still relies on it, or "
                "the contract names a test that was never written"
            )
    return errors, len(cited)


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else ROOT
    errors, total = check(root)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: {total} test citations in contracts resolve to tests that exist")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
