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


def features_tree(manifests):
    """A root holding the given `relative path -> Cargo.toml text` manifests."""
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    for relative, text in manifests.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    with directory:
        return mod.check_test_only_features(root)


class TestOnlyFeatures(unittest.TestCase):
    """`ptr-net/test-support` forges an authenticated peer; only tests may have it."""

    def test_a_dev_dependency_may_enable_it(self):
        # The control. Without it, a rule that refused every mention would pass the
        # tests below.
        self.assertEqual(
            features_tree({
                "crates/a/Cargo.toml": (
                    "[package]\nname = 'a'\n[dependencies]\nptr-net = { workspace = true }\n"
                    "[dev-dependencies]\nptr-net = { workspace = true, features = ['test-support'] }\n"
                ),
            }),
            [],
        )

    def test_a_normal_dependency_may_not(self):
        errors = features_tree({
            "crates/a/Cargo.toml": (
                "[package]\nname = 'a'\n[dependencies]\n"
                "ptr-net = { workspace = true, features = ['test-support'] }\n"
            ),
        })
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("[dependencies]", errors[0])

    def test_a_target_specific_or_renamed_dependency_may_not(self):
        errors = features_tree({
            "crates/a/Cargo.toml": (
                "[package]\nname = 'a'\n"
                "[target.'cfg(unix)'.dependencies]\n"
                "net = { package = 'ptr-net', path = '../net', features = ['test-support'] }\n"
            ),
        })
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("target.cfg(unix).dependencies", errors[0])

    def test_the_workspace_table_may_not(self):
        errors = features_tree({
            "Cargo.toml": (
                "[workspace]\n[workspace.dependencies]\n"
                "ptr-net = { path = 'crates/ptr-net', features = ['test-support'] }\n"
            ),
        })
        self.assertEqual(len(errors), 1, errors)

    def test_a_feature_may_not_forward_to_it(self):
        errors = features_tree({
            "crates/a/Cargo.toml": (
                "[package]\nname = 'a'\n[features]\nhandy = ['ptr-net?/test-support']\n"
            ),
        })
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("handy", errors[0])

    def test_the_defining_crate_may_declare_it(self):
        self.assertEqual(
            features_tree({
                "crates/ptr-net/Cargo.toml": (
                    "[package]\nname = 'ptr-net'\n[features]\ntest-support = []\n"
                ),
            }),
            [],
        )


class ThisRepositoryFeatures(unittest.TestCase):
    def test_the_real_tree_enables_no_test_only_feature_in_production(self):
        self.assertEqual(mod.check_test_only_features(ROOT), [])

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
        """Report evaluation entries missing from either the registry or the filesystem."""
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

    def test_a_crate_missing_from_either_component_map_is_reported(self):
        """Require a component-map table row rather than a prose link for each crate."""
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        with directory:
            for crate in ["ptr-listed", "ptr-unlisted"]:
                (root / "crates" / crate / "src").mkdir(parents=True)
                (root / "crates" / crate / "Cargo.toml").write_text(
                    "[package]\nname = 'x'\n", encoding="utf-8"
                )
            (root / "docs/components").mkdir(parents=True)
            (root / "README.md").write_text(
                "| [ptr-listed](crates/ptr-listed/README.md) | x |\n"
                "| [ptr-unlisted](crates/ptr-unlisted/README.md) | x |\n",
                encoding="utf-8",
            )
            # Mentioned in prose is not listed in the map.
            (root / "docs/components/README.md").write_text(
                "| [ptr-listed](../../crates/ptr-listed/README.md) | x | x |\n"
                "See [ptr-unlisted](../../crates/ptr-unlisted/README.md) for more.\n",
                encoding="utf-8",
            )

            errors, _ = mod.check(root)

        self.assertIn(
            "ptr-unlisted: not listed in the component map in docs/components/README.md",
            errors,
        )
        self.assertFalse(any("ptr-listed: not listed" in e for e in errors), errors)
        self.assertFalse(any("in README.md" in e for e in errors), errors)

    def test_the_real_tree_lists_every_crate_in_both_maps(self):
        """Check that both repository component maps include all actual crates."""
        errors, _ = mod.check(ROOT)
        self.assertEqual([e for e in errors if "component map" in e], [])

    def test_the_real_tree_registers_every_evaluation(self):
        """Check that actual evaluation directories and registry entries agree."""
        errors, _ = mod.check(ROOT)
        self.assertEqual([e for e in errors if "evaluations/" in e], [])


if __name__ == "__main__":
    unittest.main()
