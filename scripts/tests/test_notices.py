"""The notices check has to catch a drifted dependency graph, not merely run."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = load("generate_notices")
check = load("check_notices")

LOCK = """\
version = 3

[[package]]
name = "ptr-thing"
version = "0.1.0"

[[package]]
name = "widget"
version = "1.2.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
"""


class Tree:
    """A tree with one owned workspace, one third-party package and one of ours."""

    def __init__(self, directory: Path):
        self.root = directory
        self.lock = LOCK
        self.origins = []
        self.notices = None

    def write(self) -> Path:
        (self.root / "vendor").mkdir(parents=True, exist_ok=True)
        (self.root / "vendor/ORIGINS.json").write_text(
            json.dumps(self.origins), encoding="utf-8"
        )
        (self.root / "Cargo.toml").write_text(
            "[workspace]\nmembers = []\n", encoding="utf-8"
        )
        (self.root / "Cargo.lock").write_text(self.lock, encoding="utf-8")
        for required in ("model/burn-a0", "fuzz", "templates/rust-crate"):
            path = self.root / required
            path.mkdir(parents=True, exist_ok=True)
            (path / "Cargo.toml").write_text("[workspace]\nmembers = []\n", encoding="utf-8")
        if self.notices is not None:
            (self.root / "THIRD-PARTY-NOTICES.md").write_text(
                self.notices, encoding="utf-8"
            )
        return self.root


def notices_for(packages, digest=None):
    keys = sorted(packages)
    body = [
        "# Third-party notices",
        "",
        f"`Package-set-digest: sha256:{digest or generate.package_set_digest(keys)}`",
        "",
        "| Package | Version | Licence | Source | Workspaces | Notices |",
        "|---|---|---|---|---|---|",
    ]
    for name, version in keys:
        body.append(f"| {name} | {version} | MIT | crates.io | . | _none shipped_ |")
    body.append("")
    return "\n".join(body) + "\n"


def run(build):
    with tempfile.TemporaryDirectory() as directory:
        tree = Tree(Path(directory))
        build(tree)
        root = tree.write()
        return check.check(root, root / "THIRD-PARTY-NOTICES.md")


class CoverageTests(unittest.TestCase):
    def test_a_matching_document_passes_and_excludes_our_own_crates(self):
        def build(tree):
            tree.notices = notices_for([("widget", "1.2.3")])

        self.assertEqual(run(build), [])

    def test_a_package_added_to_the_graph_fails_the_check(self):
        # The staleness the check exists for: a dependency arrives and nobody
        # regenerates.
        def build(tree):
            tree.notices = notices_for([("widget", "1.2.3")])
            tree.lock += (
                '\n[[package]]\nname = "newcomer"\nversion = "0.1.0"\n'
                'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
            )

        errors = run(build)
        self.assertTrue(
            any("newcomer 0.1.0: in a lockfile but not in the notices" in e for e in errors),
            errors,
        )

    def test_a_package_removed_from_the_graph_fails_the_check(self):
        def build(tree):
            tree.notices = notices_for([("widget", "1.2.3"), ("ghost", "9.9.9")])

        errors = run(build)
        self.assertTrue(
            any("ghost 9.9.9: in the notices but in no lockfile" in e for e in errors),
            errors,
        )

    def test_a_version_bump_is_a_difference_in_both_directions(self):
        def build(tree):
            tree.notices = notices_for([("widget", "1.2.2")])

        errors = run(build)
        self.assertTrue(any("widget 1.2.3: in a lockfile" in e for e in errors), errors)
        self.assertTrue(any("widget 1.2.2: in the notices" in e for e in errors), errors)

    def test_a_vendored_package_counts_as_third_party(self):
        # A retained patch is redistributed, so its notices are required even
        # though the lockfile gives it no registry source.
        def build(tree):
            tree.lock += '\n[[package]]\nname = "raft"\nversion = "0.7.0"\n'
            tree.origins = [{"name": "raft", "version": "0.7.0", "directory": "vendor/raft"}]
            tree.notices = notices_for([("widget", "1.2.3")])

        errors = run(build)
        self.assertTrue(any("raft 0.7.0: in a lockfile" in e for e in errors), errors)

    def test_a_hand_written_document_without_a_digest_is_refused(self):
        def build(tree):
            tree.notices = (
                "# Third-party notices\n\n"
                "| Package | Version | Licence | Source | Workspaces | Notices |\n"
                "|---|---|---|---|---|---|\n"
                "| widget | 1.2.3 | MIT | crates.io | . | _none shipped_ |\n"
            )

        errors = run(build)
        self.assertTrue(any("was not generated" in e for e in errors), errors)

    def test_a_tampered_digest_is_refused_even_when_the_table_matches(self):
        def build(tree):
            tree.notices = notices_for([("widget", "1.2.3")], digest="0" * 64)

        errors = run(build)
        self.assertTrue(any("does not match" in e for e in errors), errors)

    def test_a_missing_document_is_refused(self):
        errors = run(lambda tree: None)
        self.assertTrue(any("is missing" in e for e in errors), errors)


class GeneratorTests(unittest.TestCase):
    def test_a_package_without_a_license_expression_fails_generation(self):
        # Required by #20: such a package must fail, never be dropped from the
        # notices it cannot describe.
        self.assertIsNone(generate.license_of({}))
        self.assertEqual(generate.license_of({"license": "MIT"}), "MIT")
        self.assertEqual(
            generate.license_of({"license-file": "LICENSE.custom"}), "see LICENSE.custom"
        )

    def test_the_package_set_digest_depends_only_on_the_set(self):
        first = generate.package_set_digest([("a", "1"), ("b", "2")])
        self.assertEqual(first, generate.package_set_digest([("b", "2"), ("a", "1")]))
        self.assertNotEqual(first, generate.package_set_digest([("a", "1"), ("b", "3")]))

    def test_licence_file_names_are_recognised_but_sources_are_not(self):
        for name in ("LICENSE", "LICENSE-MIT", "licence.md", "COPYING", "NOTICE", "UNLICENSE"):
            self.assertTrue(generate.is_text_name(name), name)
        for name in ("lib.rs", "Cargo.toml", "README.md"):
            self.assertFalse(generate.is_text_name(name), name)


class RealTreeTests(unittest.TestCase):
    def test_the_committed_document_matches_this_repository(self):
        self.assertEqual(check.check(ROOT, ROOT / "THIRD-PARTY-NOTICES.md"), [])


if __name__ == "__main__":
    unittest.main()
