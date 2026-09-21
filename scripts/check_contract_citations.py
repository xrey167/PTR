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

import ast
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
# A Rust attribute that makes the function below it a test: one whose path *is*
# `test`, possibly qualified - `#[test]`, `#[tokio::test]`, `#[tokio::test(...)]`.
# Matching "test" anywhere in the attribute also accepted `#[cfg(test)]`, which
# compiles a function for the test build without making it a test: `cargo test
# -- --list` names only the `#[test]` one. `#[should_panic]` and `#[ignore]` sit
# beside a test attribute rather than replacing it, and are handled by the
# accumulation rule below rather than by this pattern.
RUST_TEST_ATTRIBUTE = re.compile(r"^\s*#\[\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*test\s*[\])(]")
RUST_FN = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)")
# `unittest discover` collects `test*.py` only, so a definition in any other file
# is not a test however much it looks like one.
PYTHON_TEST_FILE = "test"
# ... and within such a file it collects `test*` methods of `TestCase` subclasses
# only. `TestLoader.testMethodPrefix` is `test` and CI leaves it there:
# `.github/workflows/ci.yml` runs plain `python -m unittest discover -s <dir>`
# for every Python test directory in the tree.
PYTHON_TEST_METHOD_PREFIX = "test"
# The module a case class has to come from, and the names it exports that are
# one. `IsolatedAsyncioTestCase` and `FunctionTestCase` are `TestCase` subclasses
# shipped by `unittest` itself, so a class deriving from either is collected
# exactly as one deriving from `TestCase`. The module matters as much as the
# name: a file defining its own `class TestCase` gets nothing collected, so
# matching on the trailing name alone would accept a citation to a test that
# never runs.
UNITTEST_MODULE = "unittest"
TEST_CASE_EXPORTS = frozenset({"TestCase", "IsolatedAsyncioTestCase", "FunctionTestCase"})


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


def case_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The names in this file that really refer to `unittest`'s case classes.

    Two ways in: a name bound to the `unittest` module itself, for a base
    written `unittest.TestCase`, and a name bound directly to one of its case
    classes by `from unittest import TestCase [as ...]`.

    Only imports written directly in the module body count. An import the
    module does not unconditionally execute binds nothing by the time the class
    statement runs - `if False:`, `if TYPE_CHECKING:` and an import inside a
    function all leave the base name undefined, so the module raises on import
    and `unittest` collects a failure in place of the test. Walking the whole
    tree accepted all three.

    A name that is also defined in the file - a class, a function or an
    assignment - is dropped from both sets rather than guessed at. Import order
    and conditional definitions decide which binding wins at runtime, and a
    checker that picks one is inventing an answer; dropping it makes the file's
    tests read as absent, which fails loudly instead of passing wrongly.

    Both rules cost the same thing in the same direction. A `try: import ... /
    except ImportError:` at module scope is rejected although the import may
    well succeed, because whether it does is not something a parse can settle.
    """
    modules: set[str] = set()
    direct: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    # `import unittest.mock as m` binds `m` to the submodule, so
                    # only a plain `unittest` alias names the module we want.
                    if alias.name == UNITTEST_MODULE:
                        modules.add(alias.asname)
                elif alias.name == UNITTEST_MODULE or alias.name.startswith(
                    f"{UNITTEST_MODULE}."
                ):
                    # `import unittest.mock` binds `unittest` as well.
                    modules.add(UNITTEST_MODULE)
        elif isinstance(node, ast.ImportFrom):
            from_unittest = node.level == 0 and node.module in (
                UNITTEST_MODULE,
                f"{UNITTEST_MODULE}.case",
                f"{UNITTEST_MODULE}.async_case",
            )
            if from_unittest:
                for alias in node.names:
                    if alias.name in TEST_CASE_EXPORTS:
                        direct.add(alias.asname or alias.name)

    shadowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    shadowed.add(target.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                shadowed.add(node.target.id)
    return modules - shadowed, direct - shadowed


def is_case_base(node: ast.expr, modules: set[str], direct: set[str]) -> bool:
    """Whether this base expression is one of `unittest`'s case classes."""
    if isinstance(node, ast.Name):
        return node.id in direct
    if isinstance(node, ast.Attribute):
        return (
            node.attr in TEST_CASE_EXPORTS
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
        )
    return False


def python_tests(text: str) -> set[str]:
    """Test methods `unittest discover` would actually collect from this file.

    Collection is narrower than "a `def` in a `test*.py` file" in three ways,
    and each one was a place this function let a dead citation through. The name
    must begin with `TestLoader.testMethodPrefix`, which CI leaves at `test`; it
    must be a method of a class, not a module-level function; and that class
    must derive from a case class **that came from `unittest`**. The third is
    not pedantry - a file holding its own `class TestCase` gets nothing
    collected at all, so a base matched by its trailing name is a citation to a
    test that never runs.

    The file is parsed rather than imported. Importing would answer the question
    exactly - it is what the runner does - and would also execute module-level
    code from every `test*.py` in the tree during an invariant check, which is a
    trade this checker is not entitled to make. Parsing costs one thing: a class
    whose base is only resolvable at import time (a base imported from another
    module, or built by a factory) is not recognised, and its tests read as
    absent. That is a false failure rather than a false pass, and it is the right
    way round for the same reason the citation pattern is narrow.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # A file that does not parse defines no collectable test either, and a
        # checker is not the place to report a syntax error.
        return set()

    modules, direct = case_bindings(tree)

    # Module scope only, for the same reason the imports are: `unittest` finds
    # cases by looking at the module's own attributes. A class nested inside
    # another class, defined inside a function, or written under a conditional
    # the module does not take is not one of them, and is not collected.
    classes: dict[str, list[ast.ClassDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.setdefault(node.name, []).append(node)

    # A subclass of a case in the same file is a case too, and how many links the
    # chain has is not something to assume, so this runs to a fixed point.
    cases: set[str] = set()
    growing = True
    while growing:
        growing = False
        for name, definitions in classes.items():
            if name in cases:
                continue
            for definition in definitions:
                inherited = any(
                    isinstance(base, ast.Name) and base.id in cases
                    for base in definition.bases
                )
                if inherited or any(
                    is_case_base(base, modules, direct) for base in definition.bases
                ):
                    cases.add(name)
                    growing = True
                    break

    found: set[str] = set()
    for name in cases:
        for definition in classes[name]:
            for item in definition.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith(PYTHON_TEST_METHOD_PREFIX):
                        found.add(item.name)
    return found


def test_definitions(root: Path) -> set[str]:
    """Names of tests that actually exist, rather than text that resembles one.

    This used to concatenate every source file and regex for `fn|def <name>`,
    which accepted three things that are not a test: a mention inside a comment
    or a string, an ordinary helper function, and a Python definition no runner
    collects. A citation could then survive the removal of the very test it
    cites, which is the one thing this checker exists to stop.

    The Python half was wrong twice, and the second time is the one worth
    recording: narrowing it to files named `test*.py` looked like the fix and was
    only half of one, because `unittest discover` collects `test*` methods of
    `TestCase` subclasses from such a file and nothing else. The fixture written
    to prove the narrowing worked was itself a top-level `def` that no runner
    would ever execute - the checker accepted it, and so did I.
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
                "function, a mention in a comment, and a Python definition "
                "`unittest discover` does not collect - one outside a "
                "`TestCase` subclass, one whose name does not start with "
                "`test`, one whose base only shares the name of a `unittest` "
                "class, or one in a file not named `test*.py` - all count as "
                "absent"
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
