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

    def test_a_python_test_counts_too(self):
        self.contract("Checked by `a_dependent_the_lockfile_shows_must_be_recorded`.")
        (self.root / "scripts/tests").mkdir(parents=True)
        (self.root / "scripts/tests/t.py").write_text(
            "def a_dependent_the_lockfile_shows_must_be_recorded(self):\n    pass\n",
            encoding="utf-8",
        )
        self.assertEqual(mod.check(self.root)[0], [])

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


class ThisRepository(unittest.TestCase):
    def test_every_test_a_contract_names_exists(self):
        errors, total = mod.check(ROOT)
        self.assertEqual(errors, [])
        self.assertGreater(total, 0, "no citations found; the scanner is not reading")
