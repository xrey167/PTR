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
# A Rust attribute that makes the function below it a test. `#[should_panic]` and
# `#[ignore]` sit beside one rather than replacing it, so matching "test" anywhere
# in the attribute covers `#[test]`, `#[tokio::test(...)]` and the rest.
RUST_TEST_ATTRIBUTE = re.compile(r"^\s*#\[[^]]*test")
RUST_FN = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)")
PYTHON_DEF = re.compile(r"^\s*def\s+([a-z_][a-z0-9_]*)")
# `unittest discover` collects `test*.py` only, so a definition in any other file
# is not a test however much it looks like one.
PYTHON_TEST_FILE = "test"


def owned(root: Path, paths):
    return (p for p in paths if not SKIP.intersection(p.relative_to(root).parts))


def rust_tests(text: str) -> set[str]:
    """Functions carrying a test attribute.

    Attributes accumulate until a non-attribute line, so `#[test]` followed by
    `#[should_panic(expected = "...")]` and then `fn name` is one test.
    """
    found: set[str] = set()
    attributed = False
    for line in text.splitlines():
        stripped = line.strip()
        if RUST_TEST_ATTRIBUTE.match(line):
            attributed = True
            continue
        match = RUST_FN.match(line)
        if match:
            if attributed:
                found.add(match.group(1))
            attributed = False
            continue
        # Blank lines, comments and further attributes do not break the run; any
        # other code does, so an attribute cannot reach past the function it sits on.
        if stripped and not stripped.startswith(("#[", "//", "///", "#!")):
            attributed = False
    return found


def python_tests(text: str) -> set[str]:
    """Every `def` in a file `unittest discover` collects."""
    return {m.group(1) for m in (PYTHON_DEF.match(line) for line in text.splitlines()) if m}


def test_definitions(root: Path) -> set[str]:
    """Names of tests that actually exist, rather than text that resembles one.

    This used to concatenate every source file and regex for `fn|def <name>`,
    which accepted three things that are not a test: a mention inside a comment
    or a string, an ordinary helper function, and a Python definition in a file
    `unittest discover` never collects. A citation could then survive the removal
    of the very test it cites, which is the one thing this checker exists to stop.
    """
    found: set[str] = set()
    for area in SOURCE_AREAS:
        base = root / area
        if not base.is_dir():
            continue
        for path in owned(root, base.rglob("*.rs")):
            found |= rust_tests(path.read_text(encoding="utf-8", errors="replace"))
        for path in owned(root, base.rglob("*.py")):
            if not path.name.startswith(PYTHON_TEST_FILE):
                continue
            found |= python_tests(path.read_text(encoding="utf-8", errors="replace"))
    return found


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
    defined = test_definitions(root)
    cited = citations(root)
    for name in sorted(cited):
        if name not in defined:
            where = ", ".join(sorted(cited[name]))
            errors.append(
                f"{name}: cited by {where} and is not a test - either it was "
                "renamed or removed and the contract still relies on it, or the "
                "contract names something that was never a test. A helper "
                "function, a mention in a comment and a Python definition in a "
                "file `unittest discover` does not collect all count as absent"
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
