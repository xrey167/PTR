"""The retirement checker has to catch what it claims to catch, not merely run."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_vendor_retirement", ROOT / "scripts/check_vendor_retirement.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PASTE_MANIFEST = '[dependencies.paste]\nversion = "=0.2.3"\npackage = "pastey"\n'


class Tree:
    """A minimal vendor tree: one root-patched package aliasing pastey."""

    def __init__(self, directory: Path):
        self.root = directory
        self.origins = [
            {
                "name": "widget",
                "version": "0.19.0",
                "directory": "vendor/widget-0.19.0",
                "archive": "https://example.invalid/widget",
                "sha256": "0" * 64,
            }
        ]
        self.current = {
            "schema": "1",
            "packages": [
                {
                    "directory": "vendor/widget-0.19.0",
                    "name": "widget",
                    "version": "0.19.0",
                    "changes": {"Cargo.toml": {}},
                }
            ],
        }
        self.retirement = {
            "schema": "1",
            "classes": {
                "paste-alias": {
                    "change": "Aliases paste to pastey.",
                    "touches": ["Cargo.lock", "Cargo.toml"],
                    "retire_when": "Upstream drops paste.",
                }
            },
            "packages": [
                {
                    "name": "widget",
                    "version": "0.19.0",
                    "directory": "vendor/widget-0.19.0",
                    "workspace": "root",
                    "class": "paste-alias",
                    "upstream_observed": None,
                    "observed_at": None,
                }
            ],
        }
        self.manifest = PASTE_MANIFEST

    def write(self) -> Path:
        vendor = self.root / "vendor"
        package = self.root / "vendor/widget-0.19.0"
        package.mkdir(parents=True, exist_ok=True)
        (package / "Cargo.toml").write_text(self.manifest, encoding="utf-8")
        (vendor / "ORIGINS.json").write_text(json.dumps(self.origins), encoding="utf-8")
        (vendor / "CURRENT.json").write_text(json.dumps(self.current), encoding="utf-8")
        (vendor / "RETIREMENT.json").write_text(
            json.dumps(self.retirement), encoding="utf-8"
        )
        (self.root / "Cargo.toml").write_text(
            '[patch.crates-io]\nwidget = { path = "vendor/widget-0.19.0" }\n\n'
            "[workspace]\nmembers = []\n",
            encoding="utf-8",
        )
        # The second workspace exists and patches nothing, so a record naming it
        # is wrong rather than unrecognised.
        a0 = self.root / "model/burn-a0"
        a0.mkdir(parents=True, exist_ok=True)
        (a0 / "Cargo.toml").write_text("[patch.crates-io]\n", encoding="utf-8")
        return self.root


def run(build, require_observations=False):
    with tempfile.TemporaryDirectory() as directory:
        tree = Tree(Path(directory))
        build(tree)
        return mod.check(tree.write(), require_observations)


class ReleaseOrderTests(unittest.TestCase):
    def test_a_prerelease_sorts_below_the_release_it_precedes(self):
        self.assertLess(mod.release_key("0.22.0-pre.3"), mod.release_key("0.22.0"))
        self.assertLess(mod.release_key("0.22.0-pre.3"), mod.release_key("0.22.0-pre.4"))
        # Numeric, not lexicographic: 0.10.0 is newer than 0.9.0.
        self.assertLess(mod.release_key("0.9.0"), mod.release_key("0.10.0"))
        self.assertEqual(mod.release_key("0.19.0"), mod.release_key("0.19.0"))


class RetirementReportTests(unittest.TestCase):
    def test_a_known_newer_upstream_is_reported_as_a_candidate(self):
        def newer(tree):
            tree.retirement["packages"][0]["upstream_observed"] = "0.20.0"
            tree.retirement["packages"][0]["observed_at"] = "2026-09-21"

        errors, notes = run(newer)
        self.assertEqual(errors, [])
        self.assertTrue(
            any("RETIREMENT CANDIDATE" in note and "0.20.0" in note for note in notes),
            notes,
        )
        # The condition travels with the report, so the reader is not sent
        # looking for what "candidate" would require.
        self.assertTrue(any("Upstream drops paste." in note for note in notes), notes)

    def test_an_equal_upstream_is_not_a_candidate(self):
        def same(tree):
            tree.retirement["packages"][0]["upstream_observed"] = "0.19.0"
            tree.retirement["packages"][0]["observed_at"] = "2026-09-21"

        errors, notes = run(same)
        self.assertEqual(errors, [])
        self.assertFalse(any("CANDIDATE" in note for note in notes), notes)

    def test_an_unobserved_package_is_reported_and_can_be_made_fatal(self):
        errors, notes = run(lambda tree: None)
        self.assertEqual(errors, [])
        self.assertTrue(any("unobserved" in note for note in notes), notes)

        errors, _ = run(lambda tree: None, require_observations=True)
        self.assertTrue(any("no upstream version has been observed" in e for e in errors), errors)


class RefusalTests(unittest.TestCase):
    def test_a_vendored_package_without_a_record_is_refused(self):
        errors, _ = run(lambda tree: tree.retirement["packages"].clear())
        self.assertTrue(any("no retirement record" in e for e in errors), errors)

    def test_a_record_for_a_package_that_is_not_vendored_is_refused(self):
        def stray(tree):
            tree.retirement["packages"].append(
                {
                    "name": "ghost",
                    "version": "1.0.0",
                    "directory": "vendor/ghost-1.0.0",
                    "workspace": "root",
                    "class": "paste-alias",
                    "upstream_observed": None,
                    "observed_at": None,
                }
            )

        errors, _ = run(stray)
        self.assertTrue(any("is not vendored" in e for e in errors), errors)

    def test_a_record_naming_the_wrong_workspace_is_refused(self):
        def moved(tree):
            tree.retirement["packages"][0]["workspace"] = "burn-a0"

        errors, _ = run(moved)
        self.assertTrue(any("patched by root" in e for e in errors), errors)

    def test_a_cargo_toml_only_class_cannot_quietly_grow_a_source_patch(self):
        # The whole point: check_vendor_integrity records that src/lib.rs changed
        # and is content; this refuses it, because the recorded reason says the
        # patch touches manifests only.
        def source_patch(tree):
            tree.current["packages"][0]["changes"]["src/lib.rs"] = {}

        errors, _ = run(source_patch)
        self.assertTrue(any("src/lib.rs" in e and "allows only" in e for e in errors), errors)

    def test_a_paste_alias_record_whose_manifest_lacks_the_alias_is_refused(self):
        def no_alias(tree):
            tree.manifest = '[dependencies.paste]\nversion = "1"\n'

        errors, _ = run(no_alias)
        self.assertTrue(any("does not alias pastey" in e for e in errors), errors)

    def test_a_missing_workspace_manifest_is_reported_as_such(self):
        # Deleting the manifest must not turn 17 accurate records into
        # "unknown workspace"; the manifest is what went missing.
        def build(tree):
            tree.retirement["packages"][0]["workspace"] = "burn-a0"

        with tempfile.TemporaryDirectory() as directory:
            tree = Tree(Path(directory))
            build(tree)
            root = tree.write()
            (root / "model/burn-a0/Cargo.toml").unlink()
            errors, _ = mod.check(root, False)
        self.assertTrue(any("model/burn-a0/Cargo.toml is missing" in e for e in errors), errors)

    def test_an_unknown_class_and_an_empty_condition_are_refused(self):
        errors, _ = run(lambda tree: tree.retirement["packages"][0].update({"class": "mystery"}))
        self.assertTrue(any("unknown class" in e for e in errors), errors)

        def blank(tree):
            tree.retirement["classes"]["paste-alias"]["retire_when"] = "   "

        errors, _ = run(blank)
        self.assertTrue(any("empty retire_when" in e for e in errors), errors)


class RealTreeTests(unittest.TestCase):
    def test_the_repository_itself_passes(self):
        errors, notes = mod.check(ROOT, False)
        self.assertEqual(errors, [])
        self.assertEqual(len(notes), 21, notes)


if __name__ == "__main__":
    unittest.main()
