"""A cited test that no longer exists must fail, and an illustration must not.

The bullets this checker was written for cited no tests at all, which is how they
went stale unnoticed. The corrections cite tests, so this is what makes the bullet
and the test fail together - and the checker's own first run found a naming example
in a style guide, which is why what counts as a citation is narrow.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_contract_citations", ROOT / "scripts/check_contract_citations.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Fixtures(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        (self.root / "docs/architecture").mkdir(parents=True)
        (self.root / "crates/ptr-x/tests").mkdir(parents=True)

    def contract(self, text):
        (self.root / "docs/architecture/99-thing.md").write_text(text, encoding="utf-8")

    def source(self, text):
        (self.root / "crates/ptr-x/tests/thing.rs").write_text(text, encoding="utf-8")

    def test_a_cited_test_that_exists_passes(self):
        # The control. Every refusal below has to be about the citation and not
        # about the scanner finding nothing.
        self.contract("A Pod is scoped: `a_pod_for_one_project_is_invisible_to_another`.")
        self.source("#[test]\nfn a_pod_for_one_project_is_invisible_to_another() {}\n")
        errors, total = mod.check(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(total, 1)

    def test_a_cited_test_that_was_renamed_fails_and_names_the_document(self):
        self.contract("A Pod is scoped: `a_pod_for_one_project_is_invisible_to_another`.")
        self.source("#[test]\nfn a_pod_for_one_project_is_hidden_from_another() {}\n")
        errors, _ = mod.check(self.root)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("a_pod_for_one_project_is_invisible_to_another", errors[0])
        self.assertIn("docs/architecture/99-thing.md", errors[0])

    def test_an_ordinary_identifier_is_not_a_citation(self):
        # Four segments or fewer. `provenance_bucket_count` (3),
        # `upstream_retire_when` (3) and `require_current_revision` (3) are fields
        # and settings, not tests, and a checker demanding a function for each
        # would be unusable. Five segments is where real test names start:
        # `two_projects_may_each_register` would be treated as a citation.
        self.contract(
            "The width `provenance_bucket_count`, the field `upstream_retire_when` "
            "and the setting `require_current_revision` are not citations."
        )
        self.source("")
        errors, total = mod.check(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(total, 0)

    def test_a_style_guide_illustration_is_not_scanned(self):
        # The checker's own first run flagged `docs/RUST_API_STYLE.md`, which
        # writes such a name after "e.g." to illustrate a naming convention. A
        # style guide gives examples; a contract makes claims.
        (self.root / "docs/RUST_API_STYLE.md").write_text(
            "Name the behavior, e.g. `stale_generation_returns_expected_and_actual`.",
            encoding="utf-8",
        )
        self.source("")
        self.assertEqual(mod.check(self.root), ([], 0))

    def test_a_name_cited_by_two_documents_names_both(self):
        self.contract("See `a_widely_cited_test_that_does_not_exist`.")
        (self.root / "docs/PRIORITIES.md").write_text(
            "Also `a_widely_cited_test_that_does_not_exist`.", encoding="utf-8"
        )
        self.source("")
        errors, _ = mod.check(self.root)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("99-thing.md", errors[0])
        self.assertIn("PRIORITIES.md", errors[0])


class WhatCountsAsATest(unittest.TestCase):
    """A citation must resolve to a test, not to text that resembles one.

    The first form of this checker concatenated every `.rs` and `.py` file and
    regexed for `fn|def <name>`. That accepted a mention inside a comment, an
    ordinary helper, and a Python definition no runner collects - so a citation
    could outlive the very test it cites, which is the one thing this exists to
    stop. Each case below was verified to pass before the rule was tightened.
    """

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        (self.root / "docs/architecture").mkdir(parents=True)
        (self.root / "crates/ptr-x").mkdir(parents=True)
        (self.root / "docs/architecture/99-thing.md").write_text(
            "Asserted by `a_property_this_repository_actually_holds`.", encoding="utf-8"
        )

    def source(self, text):
        (self.root / "crates/ptr-x/t.rs").write_text(text, encoding="utf-8")

    def errors(self):
        return mod.check(self.root)[0]

    NAME = "a_property_this_repository_actually_holds"

    def test_a_plain_test_attribute_counts(self):
        self.source(f"#[test]\nfn {self.NAME}() {{}}\n")
        self.assertEqual(self.errors(), [])

    def test_an_async_test_attribute_with_arguments_counts(self):
        self.source(
            f'#[tokio::test(flavor = "multi_thread", worker_threads = 4)]\n'
            f"async fn {self.NAME}() {{}}\n"
        )
        self.assertEqual(self.errors(), [])

    def test_a_should_panic_attribute_beside_test_counts(self):
        # Attributes accumulate; `#[should_panic]` sits beside `#[test]` rather
        # than replacing it, and A5's guard tests are written exactly this way.
        self.source(
            f'#[test]\n#[should_panic(expected = "another codebook version")]\n'
            f"fn {self.NAME}() {{}}\n"
        )
        self.assertEqual(self.errors(), [])

    def test_a_mention_inside_a_comment_is_not_a_test(self):
        self.source(f"// see fn {self.NAME} for details\n")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("is not a test", errors[0])

    def test_an_ordinary_helper_is_not_a_test(self):
        self.source(f"fn {self.NAME}() {{}}\n")
        self.assertEqual(len(self.errors()), 1)

    def test_an_attribute_that_is_not_a_test_does_not_count(self):
        self.source(f"#[derive(Debug)]\nfn {self.NAME}() {{}}\n")
        self.assertEqual(len(self.errors()), 1)

    def test_an_attribute_cannot_reach_past_the_function_it_sits_on(self):
        # The control for the accumulation rule: real code between the attribute
        # and the function breaks the run, so `#[test]` on one function does not
        # silently bless a later one.
        self.source(f"#[test]\nfn something_else_entirely_here() {{}}\nfn {self.NAME}() {{}}\n")
        self.assertEqual(len(self.errors()), 1)


class WhatUnittestCollects(unittest.TestCase):
    """A cited Python name must be one the configured runner actually runs.

    CI runs plain `python -m unittest discover -s <dir>` for every Python test
    directory, so collection means: a file matching `test*.py`, a class derived
    from `unittest.TestCase`, and a method whose name starts with `test`. Every
    case below was first put to `unittest.TestLoader().discover()` itself, which
    collected `test_a_method_of_a_subclass_of_a_local_base` from a file holding
    all five shapes and nothing else.

    The review that produced this class caught the checker accepting a top-level
    `def` in a `test*.py` file - written by an earlier fixture of mine, as proof
    that the Python rule worked.
    """

    NAME = "test_a_dependent_the_lockfile_shows_must_be_recorded"

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        (self.root / "docs/architecture").mkdir(parents=True)
        (self.root / "scripts/tests").mkdir(parents=True)
        self.contract(self.NAME)

    def contract(self, name):
        (self.root / "docs/architecture/99-thing.md").write_text(
            f"A dependent is re-derived offline: `{name}`.", encoding="utf-8"
        )

    def python(self, body, name="test_blockers.py"):
        (self.root / "scripts/tests" / name).write_text(body, encoding="utf-8")

    def errors(self):
        return mod.check(self.root)[0]

    def test_a_test_method_of_a_test_case_counts(self):
        # The control. Every refusal below has to be about what was written and
        # not about the scanner failing to read Python at all.
        self.python(
            f"import unittest\n\n\nclass Blockers(unittest.TestCase):\n"
            f"    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(self.errors(), [])

    def test_the_base_class_may_be_imported_by_name(self):
        # `from unittest import TestCase` and `unittest.TestCase` are the same
        # class, and the runner does not care which spelling reached it.
        self.python(
            f"from unittest import TestCase\n\n\nclass Blockers(TestCase):\n"
            f"    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(self.errors(), [])

    def test_a_method_of_a_subclass_of_a_local_case_counts(self):
        # Verified against `unittest` itself: a subclass of a local `TestCase`
        # subclass is collected, so refusing it would be a false failure.
        self.python(
            f"import unittest\n\n\nclass Base(unittest.TestCase):\n    pass\n\n\n"
            f"class Blockers(Base):\n    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(self.errors(), [])

    def test_a_top_level_definition_is_not_collected(self):
        # The defect this class exists for. `unittest` collects methods of cases;
        # a module-level function in `test_blockers.py` is never executed, so a
        # contract citing one cites nothing.
        self.python(f"def {self.NAME}(self):\n    pass\n")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("is not a test", errors[0])

    def test_a_method_whose_name_lacks_the_prefix_is_not_collected(self):
        # `TestLoader.testMethodPrefix` is `test` and CI leaves it there, so a
        # method named without it sits in a collected class and never runs.
        name = "a_dependent_the_lockfile_shows_must_be_recorded"
        self.contract(name)
        self.python(
            f"import unittest\n\n\nclass Blockers(unittest.TestCase):\n"
            f"    def {name}(self):\n        pass\n"
        )
        self.assertEqual(len(self.errors()), 1)

    def test_a_method_of_a_class_that_is_not_a_case_is_not_collected(self):
        self.python(
            f"class Blockers:\n    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(len(self.errors()), 1)

    def test_a_file_the_discovery_pattern_does_not_match_is_not_read(self):
        # `discover` takes `test*.py`. A properly written case in `blockers.py`
        # is still never loaded.
        self.python(
            f"import unittest\n\n\nclass Blockers(unittest.TestCase):\n"
            f"    def {self.NAME}(self):\n        pass\n",
            name="blockers.py",
        )
        self.assertEqual(len(self.errors()), 1)

    def test_a_local_class_named_test_case_does_not_make_a_case(self):
        # A base is matched by where it came from, not by what it is called. A
        # file defining its own `class TestCase` gets nothing collected at all -
        # confirmed against `unittest.TestLoader().discover()`, which returns an
        # empty suite for exactly this file.
        self.python(
            f"class TestCase:\n    pass\n\n\nclass Blockers(TestCase):\n"
            f"    def {self.NAME}(self):\n        pass\n"
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("is not a test", errors[0])

    def test_a_base_imported_from_another_module_is_not_a_case(self):
        # The same name from somewhere else. Recognising it would be a guess
        # about a module this checker never reads.
        self.python(
            f"from mypkg import TestCase\n\n\nclass Blockers(TestCase):\n"
            f"    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(len(self.errors()), 1)

    def test_a_base_with_no_import_at_all_is_not_a_case(self):
        self.python(
            f"class Blockers(TestCase):\n    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(len(self.errors()), 1)

    def test_the_module_may_be_aliased(self):
        self.python(
            f"import unittest as ut\n\n\nclass Blockers(ut.TestCase):\n"
            f"    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(self.errors(), [])

    def test_the_class_may_be_aliased(self):
        self.python(
            f"from unittest import TestCase as Case\n\n\nclass Blockers(Case):\n"
            f"    def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(self.errors(), [])

    def test_an_asynchronous_case_counts(self):
        # `IsolatedAsyncioTestCase` is a `TestCase` subclass shipped by
        # `unittest`, and discovery treats it identically.
        self.python(
            f"import unittest\n\n\nclass Blockers(unittest.IsolatedAsyncioTestCase):\n"
            f"    async def {self.NAME}(self):\n        pass\n"
        )
        self.assertEqual(self.errors(), [])

    def test_a_file_that_does_not_parse_defines_no_test(self):
        # Reporting a syntax error is not this checker's job, but crashing on one
        # would take down an invariant run over a file it only skims.
        self.python(f"class Blockers(unittest.TestCase:\n    def {self.NAME}(self)\n")
        self.assertEqual(len(self.errors()), 1)


class ThisRepository(unittest.TestCase):
    def test_every_test_a_contract_names_exists(self):
        errors, total = mod.check(ROOT)
        self.assertEqual(errors, [])
        self.assertGreater(total, 0, "no citations found; the scanner is not reading")
