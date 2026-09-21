"""The MSRV checker must catch the burn-a0 drift and must not fire on 1.85 vs 1.85.0.

Both halves matter. Without the first the check is decorative; without the second
it fails on this repository's own root workspace the day it is added, because the
root declares `rust-version = "1.85"` against `msrv = "1.85.0"` and clippy treats
those as the same version.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_msrv_alignment", ROOT / "scripts/check_msrv_alignment.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write(root: Path, relative: str, text: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Fixtures(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def errors(self):
        return mod.check(self.root)[0]

    def checked(self):
        return mod.check(self.root)[1]

    def test_the_burn_a0_drift_is_caught(self):
        # A nested workspace requiring more than the root's clippy.toml allows.
        write(self.root, "clippy.toml", 'msrv = "1.85.0"\n')
        write(
            self.root,
            "Cargo.toml",
            '[workspace]\nmembers = []\n\n[workspace.package]\nrust-version = "1.85"\n',
        )
        write(
            self.root,
            "model/burn-a0/Cargo.toml",
            '[package]\nname = "a0"\nrust-version = "1.95"\n\n[workspace]\n',
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("model/burn-a0/Cargo.toml", errors[0])
        self.assertIn("1.85.0", errors[0])
        self.assertIn("1.95", errors[0])

    def test_its_own_clippy_toml_settles_it(self):
        # The control: the same tree with the fix applied is clean, so the test
        # above is evidence about the drift and not about nested workspaces.
        self.test_the_burn_a0_drift_is_caught()
        write(self.root, "model/burn-a0/clippy.toml", 'msrv = "1.95"\n')
        self.assertEqual(self.errors(), [])

    def test_two_digit_and_three_digit_versions_are_the_same_version(self):
        # This repository's root. A string comparison reports it as a mismatch.
        write(self.root, "clippy.toml", 'msrv = "1.85.0"\n')
        write(
            self.root,
            "Cargo.toml",
            '[workspace]\nmembers = []\n\n[workspace.package]\nrust-version = "1.85"\n',
        )
        self.assertEqual(self.errors(), [])
        self.assertEqual(len(self.checked()), 1)

    def test_an_inherited_rust_version_is_resolved(self):
        # Every crate and both binaries write `rust-version.workspace = true`, so a
        # checker that read only `[package]` would examine almost nothing.
        write(self.root, "clippy.toml", 'msrv = "1.90"\n')
        write(
            self.root,
            "Cargo.toml",
            '[workspace]\nmembers = ["crates/a"]\n\n[workspace.package]\nrust-version = "1.85"\n',
        )
        write(
            self.root,
            "crates/a/Cargo.toml",
            '[package]\nname = "a"\nrust-version.workspace = true\n',
        )
        errors = self.errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any("crates/a/Cargo.toml" in e for e in errors), errors)
        self.assertTrue(
            any("Cargo.toml [workspace.package]" in e for e in errors), errors
        )

    def test_a_package_declaring_no_msrv_is_not_reported(self):
        # `fuzz` and `templates/rust-crate` declare none. No claim, no contradiction.
        write(self.root, "clippy.toml", 'msrv = "1.85.0"\n')
        write(self.root, "Cargo.toml", '[workspace]\nmembers = []\n')
        write(self.root, "fuzz/Cargo.toml", '[package]\nname = "f"\n')
        self.assertEqual(self.errors(), [])

    def test_a_clippy_toml_without_an_msrv_constrains_nothing(self):
        write(self.root, "clippy.toml", "too-many-arguments-threshold = 8\n")
        write(
            self.root,
            "Cargo.toml",
            '[workspace]\nmembers = []\n\n[workspace.package]\nrust-version = "1.85"\n',
        )
        self.assertEqual(self.errors(), [])

    def test_the_nearest_clippy_toml_wins(self):
        # Clippy walks upward and stops at the first one; a distant root must not be
        # the one reported once a nearer file exists.
        write(self.root, "clippy.toml", 'msrv = "1.85.0"\n')
        write(self.root, "Cargo.toml", '[workspace]\nmembers = []\n')
        write(self.root, "nested/clippy.toml", 'msrv = "1.90"\n')
        write(
            self.root,
            "nested/Cargo.toml",
            '[package]\nname = "n"\nrust-version = "1.95"\n\n[workspace]\n',
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("nested/clippy.toml", errors[0])


class Semantics(unittest.TestCase):
    def test_versions_are_padded_to_three_components(self):
        self.assertEqual(mod.parse_version("1.85"), mod.parse_version("1.85.0"))
        self.assertNotEqual(mod.parse_version("1.85"), mod.parse_version("1.85.1"))
        self.assertNotEqual(mod.parse_version("1.9"), mod.parse_version("1.90"))


class ThisRepository(unittest.TestCase):
    def test_every_package_is_linted_at_the_msrv_it_declares(self):
        errors, checked = mod.check(ROOT)
        self.assertEqual(errors, [])
        self.assertTrue(checked)

    def test_burn_a0_carries_its_own_clippy_toml(self):
        # The fix itself, asserted rather than assumed: without this file the tree
        # is governed by the root's 1.85.0.
        self.assertTrue((ROOT / "model/burn-a0/clippy.toml").exists())


if __name__ == "__main__":
    unittest.main()
