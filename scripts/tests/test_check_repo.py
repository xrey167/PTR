"""The workspace-membership rule must catch a package no glob names, not merely run.

`members = ["bins/ptr-*"]` omitted `bins/ptrctl` and `bins/ptrd` for as long as
they existed, and nothing anywhere said so: cargo reports no error for a directory
it was never told about, so the two binaries were built, tested, linted and
formatted by nothing. These tests drive the rule against fixture trees, including
the exact shape of the original defect.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_repo", ROOT / "scripts/check_repo.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def tree(manifest: str, packages=()):
    """A root holding a workspace manifest and some package directories."""
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    (root / "Cargo.toml").write_text(manifest, encoding="utf-8")
    for relative in packages:
        package = root / relative
        package.mkdir(parents=True)
        (package / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
    return directory, root


WORKSPACE = '[workspace]\nmembers = {members}\nexclude = {exclude}\n'


def run(members, exclude, packages):
    directory, root = tree(
        WORKSPACE.format(members=repr(list(members)), exclude=repr(list(exclude))),
        packages,
    )
    with directory:
        return mod.check_workspace_membership(root)


class WorkspaceMembership(unittest.TestCase):
    def test_the_original_defect_is_caught(self):
        # The tree exactly as it was: the glob needs a hyphen after `ptr`.
        errors = run(
            ["crates/ptr-*", "bins/ptr-*"],
            ["fuzz", "model/burn-a0", "vendor/*"],
            ["bins/ptr-bench", "bins/ptr-worker", "bins/ptrctl", "bins/ptrd"],
        )
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any("bins/ptrctl" in e for e in errors), errors)
        self.assertTrue(any("bins/ptrd" in e for e in errors), errors)

    def test_a_named_member_is_accounted_for(self):
        # The control: the same tree, with the two named outright, is clean. Without
        # this the test above would pass for a rule that rejects everything.
        self.assertEqual(
            run(
                ["crates/ptr-*", "bins/ptr-*", "bins/ptrctl", "bins/ptrd"],
                ["fuzz"],
                ["bins/ptr-bench", "bins/ptrctl", "bins/ptrd"],
            ),
            [],
        )

    def test_an_excluded_package_is_accounted_for(self):
        # Excluding is a decision; the rule objects to the absence of one, not to
        # a package being outside the workspace.
        self.assertEqual(run(["bins/ptr-*"], ["bins/scratch"], ["bins/scratch"]), [])

    def test_a_glob_star_does_not_cross_a_separator(self):
        # `crates/*` must not account for `bins/orphan`, or the rule would pass on
        # any tree carrying one broad glob.
        errors = run(["crates/*"], [], ["crates/ptr-types", "bins/orphan"])
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("bins/orphan", errors[0])

    def test_a_basename_pattern_does_not_account_for_a_nested_package(self):
        # Cargo resolves a member path from the workspace root: put to `cargo
        # metadata`, a workspace with `members = ["ptrctl"]` and the package at
        # `bins/ptrctl` fails with "failed to read <root>/ptrctl/Cargo.toml". It
        # never looks in `bins/`, so the package is registered nowhere - which is
        # exactly what this check exists to report.
        errors = run(["ptrctl"], [], ["bins/ptrctl"])
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("bins/ptrctl", errors[0])

    def test_a_basename_exclusion_does_not_account_for_a_nested_package(self):
        # The same rule on the other list. An `exclude` that Cargo reads as a
        # root-level directory excludes nothing here, so the package is still
        # unregistered.
        errors = run(["crates/*"], ["ptrctl"], ["bins/ptrctl"])
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("bins/ptrctl", errors[0])

    def test_a_double_star_spans_separators(self):
        # The control for the two above, and the reason the matcher is not simply
        # "same number of segments": `cargo metadata` does accept `**/ptrctl` for
        # a package at `bins/ptrctl`, so refusing it would be a false failure.
        self.assertEqual(run(["**/ptrctl"], [], ["bins/ptrctl"]), [])

    def test_a_directory_without_a_manifest_is_not_a_package(self):
        directory, root = tree(WORKSPACE.format(members="['bins/ptr-*']", exclude="[]"))
        with directory:
            (root / "bins/notes").mkdir(parents=True)
            self.assertEqual(mod.check_workspace_membership(root), [])

    def test_a_manifest_that_is_not_a_workspace_is_not_judged(self):
        directory, root = tree("[package]\nname = 'solo'\n", ["bins/orphan"])
        with directory:
            self.assertEqual(mod.check_workspace_membership(root), [])


class ThisRepository(unittest.TestCase):
    def test_the_real_tree_accounts_for_every_package(self):
        self.assertEqual(mod.check_workspace_membership(ROOT), [])


class RepositoryShape(unittest.TestCase):
    def test_malformed_registry_and_missing_crates_are_reported(self):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        with directory:
            (root / "experiments").mkdir()
            (root / "experiments/registry.toml").write_text("[experiment\n", encoding="utf-8")

            errors, summary = mod.check(root)

        self.assertIn("missing README.md", errors)
        self.assertIn("TOML experiments/registry.toml:", "\n".join(errors))
        self.assertIn("missing crates", errors)
        self.assertTrue(summary.startswith("OK: 0 documented crates"), summary)

    def test_evaluation_registry_and_directories_name_the_same_slots(self):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        with directory:
            for slot in ["listed-and-present", "present-only"]:
                (root / "evaluations/components" / slot).mkdir(parents=True)
                (root / "evaluations/components" / slot / "candidates.toml").write_text(
                    f'component = "{slot}"\n', encoding="utf-8"
                )
            (root / "evaluations/registry.toml").write_text(
                "".join(
                    f'[[component]]\nid = "{slot}"\n'
                    f'path = "components/{slot}/candidates.toml"\nstatus = "open"\n'
                    for slot in ["listed-and-present", "listed-only"]
                ),
                encoding="utf-8",
            )

            errors, _ = mod.check(root)

        self.assertIn(
            "evaluations/components/present-only: not listed in evaluations/registry.toml",
            errors,
        )
        self.assertIn(
            "evaluations/registry.toml: listed-only has no "
            "evaluations/components/listed-only/candidates.toml",
            errors,
        )
        self.assertFalse(any("listed-and-present" in error for error in errors), errors)

    def test_the_real_tree_registers_every_evaluation(self):
        errors, _ = mod.check(ROOT)
        self.assertEqual([e for e in errors if "evaluations/" in e], [])


if __name__ == "__main__":
    unittest.main()
