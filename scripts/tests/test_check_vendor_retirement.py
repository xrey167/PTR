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
                    "retire_when_signal": "paste-free",
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
        # A lockfile is where the dependent set is re-derived from. By default
        # nothing depends on the patched package, so `blocked_by` is empty and a
        # satisfied upstream is a genuine candidate.
        self.lock = {"package": [{"name": "widget", "version": "0.19.0"}]}

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
        (self.root / "Cargo.lock").write_text(
            "\n".join(
                "[[package]]\nname = \"{}\"\nversion = \"{}\"\ndependencies = [{}]".format(
                    entry["name"],
                    entry["version"],
                    ", ".join(f'"{d}"' for d in entry.get("dependencies", [])),
                )
                for entry in self.lock["package"]
            )
            + "\n",
            encoding="utf-8",
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
            tree.retirement["packages"][0]["upstream_retire_when"] = "satisfied"
            tree.retirement["packages"][0]["blocked_by"] = []

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


class RetirementConditionTests(unittest.TestCase):
    """A newer version number is not a retirement.

    The report used to say CANDIDATE whenever the observed version outranked the
    pinned one, which asks about a number rather than about the condition in
    `retire_when`. `macerator 0.4.0` was reported that way while still depending
    on `paste`, and `netlink-packet-core 0.9.0` was reported the same way while
    four dependents refuse anything in `0.9`. These are the two real shapes.
    """

    @staticmethod
    def observed(tree, version="0.20.0", when="satisfied", blocked=()):
        package = tree.retirement["packages"][0]
        package["upstream_observed"] = version
        package["observed_at"] = "2026-09-21"
        package["upstream_retire_when"] = when
        package["blocked_by"] = list(blocked)

    def test_the_macerator_shape_a_newer_release_that_retires_nothing(self):
        # 0.4.0 exists and still depends on paste, so the alias has exactly as
        # much left to redirect as before.
        errors, notes = run(lambda tree: self.observed(tree, when="unsatisfied"))
        self.assertEqual(errors, [])
        self.assertTrue(any("NOT A CANDIDATE" in note for note in notes), notes)
        self.assertFalse(any("RETIREMENT CANDIDATE" in note for note in notes), notes)
        # The reason travels with the verdict.
        self.assertTrue(any("Upstream drops paste." in note for note in notes), notes)

    def test_the_netlink_shape_a_real_move_its_dependents_refuse(self):
        def tree_with_a_refusing_dependent(tree):
            tree.lock["package"].append(
                {"name": "consumer", "version": "1.0.0", "dependencies": ["widget 0.19.0"]}
            )
            self.observed(
                tree,
                blocked=[
                    {
                        "name": "consumer",
                        "version": "1.0.0",
                        "requirement": "^0.19",
                        "in_lockfile": True,
                    }
                ],
            )

        errors, notes = run(tree_with_a_refusing_dependent)
        self.assertEqual(errors, [])
        self.assertTrue(any("BLOCKED" in note for note in notes), notes)
        self.assertTrue(any("consumer 1.0.0 at ^0.19" in note for note in notes), notes)
        # The actionable is moving the dependent, not bumping the vendored copy -
        # a renamed-key patch entry silently goes unused when its version stops
        # satisfying the requirement.
        self.assertTrue(any("not bumping the vendored copy" in note for note in notes), notes)

    def test_a_dependent_that_admits_the_release_does_not_block_it(self):
        # The control for the test above: same tree, wider requirement, and the
        # verdict flips to CANDIDATE. Without it, "BLOCKED" could mean only that
        # a dependent exists.
        def tree_with_a_permissive_dependent(tree):
            tree.lock["package"].append(
                {"name": "consumer", "version": "1.0.0", "dependencies": ["widget 0.19.0"]}
            )
            self.observed(
                tree,
                blocked=[
                    {
                        "name": "consumer",
                        "version": "1.0.0",
                        "requirement": "^0.20",
                        "in_lockfile": True,
                    }
                ],
            )

        errors, notes = run(tree_with_a_permissive_dependent)
        self.assertEqual(errors, [])
        self.assertTrue(any("RETIREMENT CANDIDATE" in note for note in notes), notes)

    def test_a_class_with_no_mechanical_signal_is_reported_as_needing_reading(self):
        # `raft-protobuf`: "carries that migration" is something a person reads,
        # and a class that pretended otherwise would be deciding from a number
        # again.
        def unreadable(tree):
            tree.retirement["classes"]["paste-alias"]["retire_when_signal"] = None
            self.observed(tree, when="unknown")

        errors, notes = run(unreadable)
        self.assertEqual(errors, [])
        self.assertTrue(any("needs reading" in note for note in notes), notes)
        self.assertFalse(any("CANDIDATE" in note for note in notes), notes)

    def test_a_newer_upstream_without_a_verdict_is_refused(self):
        def silent(tree):
            package = tree.retirement["packages"][0]
            package["upstream_observed"] = "0.20.0"
            package["observed_at"] = "2026-09-21"

        errors, _ = run(silent)
        self.assertTrue(any("upstream_retire_when is None" in e for e in errors), errors)

    def test_a_class_without_a_declared_signal_is_refused(self):
        def missing(tree):
            del tree.retirement["classes"]["paste-alias"]["retire_when_signal"]

        errors, _ = run(missing)
        self.assertTrue(any("no retire_when_signal" in e for e in errors), errors)


class BlockerDerivationTests(unittest.TestCase):
    """Existence is re-derived from the lockfile; requirements are observed.

    The distinction is the whole design: a lockfile records resolved versions and
    never requirements, and these dependents are not vendored, so their
    requirement strings cannot be checked here at all. What can be checked is who
    depends on the package, and that is checked in both directions.
    """

    @staticmethod
    def with_dependent(tree, blocked, dependencies=("widget 0.19.0",)):
        tree.lock["package"].append(
            {"name": "consumer", "version": "1.0.0", "dependencies": list(dependencies)}
        )
        package = tree.retirement["packages"][0]
        package["upstream_observed"] = "0.20.0"
        package["observed_at"] = "2026-09-21"
        package["upstream_retire_when"] = "satisfied"
        package["blocked_by"] = list(blocked)

    def test_a_dependent_the_lockfile_shows_must_be_recorded(self):
        errors, _ = run(lambda tree: self.with_dependent(tree, blocked=[]))
        self.assertTrue(
            any("shows consumer 1.0.0 depending on it" in e for e in errors), errors
        )

    def test_an_optional_inactive_dependent_is_kept_and_marked(self):
        # `burn-flex` requires `macerator ^0.3.4` optionally and is not selected,
        # so no lockfile can show it while it still constrains a bump.
        def invisible(tree):
            package = tree.retirement["packages"][0]
            package["upstream_observed"] = "0.20.0"
            package["observed_at"] = "2026-09-21"
            package["upstream_retire_when"] = "satisfied"
            package["blocked_by"] = [
                {
                    "name": "optional-consumer",
                    "version": "2.0.0",
                    "requirement": "^0.19",
                    "in_lockfile": False,
                }
            ]

        errors, notes = run(invisible)
        self.assertEqual(errors, [])
        self.assertTrue(any("optional-consumer 2.0.0 at ^0.19" in n for n in notes), notes)

    def test_claiming_an_invisible_dependent_is_visible_is_refused(self):
        def lying(tree):
            self.with_dependent(
                tree,
                blocked=[
                    {
                        "name": "ghost",
                        "version": "3.0.0",
                        "requirement": "^0.19",
                        "in_lockfile": True,
                    },
                    {
                        "name": "consumer",
                        "version": "1.0.0",
                        "requirement": "^0.19",
                        "in_lockfile": True,
                    },
                ],
            )

        errors, _ = run(lying)
        self.assertTrue(
            any("ghost 3.0.0 records in_lockfile=True" in e for e in errors), errors
        )

    def test_an_unrecognised_requirement_is_reported_rather_than_guessed(self):
        # Guessing here produces exactly the false actionable this was changed to
        # stop producing, so an unread requirement blocks and says it is unread.
        def odd(tree):
            self.with_dependent(
                tree,
                blocked=[
                    {
                        "name": "consumer",
                        "version": "1.0.0",
                        "requirement": ">=0.19, <0.25",
                        "in_lockfile": True,
                    }
                ],
            )

        errors, notes = run(odd)
        self.assertEqual(errors, [])
        self.assertTrue(any("(unread)" in note for note in notes), notes)
        self.assertTrue(any("BLOCKED" in note for note in notes), notes)


class CaretTests(unittest.TestCase):
    def test_a_leading_zero_narrows_the_range(self):
        # The two real cases: ^0.8 excludes 0.9.0, ^0.3.4 excludes 0.4.0.
        self.assertFalse(mod.requirement_admits("^0.8", "0.9.0"))
        self.assertTrue(mod.requirement_admits("^0.8", "0.8.2"))
        self.assertFalse(mod.requirement_admits("^0.3.4", "0.4.0"))
        self.assertTrue(mod.requirement_admits("^0.3.4", "0.3.9"))
        # And the control: above zero, a caret reaches across the minor series.
        self.assertTrue(mod.requirement_admits("^1.2.3", "1.9.0"))
        self.assertFalse(mod.requirement_admits("^1.2.3", "2.0.0"))
        self.assertFalse(mod.requirement_admits("^1.2.3", "1.2.2"))

    def test_a_double_zero_narrows_to_the_patch(self):
        self.assertTrue(mod.requirement_admits("^0.0.3", "0.0.3"))
        self.assertFalse(mod.requirement_admits("^0.0.3", "0.0.4"))

    def test_what_it_cannot_read_it_says_it_cannot_read(self):
        self.assertIsNone(mod.requirement_admits(">=0.8", "0.9.0"))
        self.assertIsNone(mod.requirement_admits("~0.8", "0.9.0"))
        self.assertIsNone(mod.requirement_admits("*", "0.9.0"))
        # A caret without a pre-release never matches one, which is a rule this
        # does not implement, so it declines rather than deciding.
        self.assertIsNone(mod.requirement_admits("^0.22", "0.22.0-pre.3"))


class ThisTreeTests(unittest.TestCase):
    """The two real records, so the change describes this repository and not only fixtures."""

    def setUp(self):
        self.errors, self.notes = mod.check(ROOT, require_observations=True)
        self.assertEqual(self.errors, [])

    def test_macerator_is_not_a_candidate_because_0_4_0_still_uses_paste(self):
        note = next(n for n in self.notes if n.startswith("macerator"))
        self.assertIn("NOT A CANDIDATE", note)
        self.assertIn("0.4.0", note)

    def test_netlink_packet_core_is_a_real_move_four_dependents_refuse(self):
        note = next(n for n in self.notes if n.startswith("netlink-packet-core"))
        self.assertIn("BLOCKED", note)
        self.assertIn("satisfies retire_when", note)
        for dependent in ("netdev", "netlink-packet-route", "netlink-proto", "netwatch"):
            self.assertIn(dependent, note)

    def test_nothing_in_this_tree_is_reported_as_a_plain_candidate(self):
        self.assertFalse([n for n in self.notes if "RETIREMENT CANDIDATE" in n], self.notes)


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
    def test_the_repository_itself_passes_with_every_patch_observed(self):
        errors, notes = mod.check(ROOT, False)
        self.assertEqual(errors, [])
        # Asserted as a property rather than a note count, which changes every
        # time an observation is refreshed.
        self.assertTrue(any("observations recorded" in note for note in notes), notes)

        strict, _ = mod.check(ROOT, True)
        self.assertEqual(strict, [], "every retained patch must carry an observation")


if __name__ == "__main__":
    unittest.main()


spec_refresh = importlib.util.spec_from_file_location(
    "refresh_vendor_upstream", ROOT / "scripts/refresh_vendor_upstream.py"
)
refresh = importlib.util.module_from_spec(spec_refresh)
spec_refresh.loader.exec_module(refresh)


def index_body(entries):
    return "\n".join(json.dumps(entry) for entry in entries) + "\n"


class IndexPathTests(unittest.TestCase):
    def test_the_sparse_layout_matches_the_length_rule(self):
        self.assertEqual(refresh.index_path("a"), "1/a")
        self.assertEqual(refresh.index_path("ab"), "2/ab")
        self.assertEqual(refresh.index_path("abc"), "3/a/abc")
        self.assertEqual(refresh.index_path("raft"), "ra/ft/raft")
        self.assertEqual(refresh.index_path("macerator"), "ma/ce/macerator")
        # Names are lowercased for the path even when the crate is not.
        self.assertEqual(refresh.index_path("Inflector"), "in/fl/inflector")


class NewestVersionTests(unittest.TestCase):
    def test_publication_order_does_not_decide(self):
        # A backport published after a newer release ends the file; taking the
        # last line would record 0.3.5 as newest and hide 0.4.0.
        body = index_body(
            [
                {"vers": "0.3.4", "yanked": False},
                {"vers": "0.4.0", "yanked": False},
                {"vers": "0.3.5", "yanked": False},
            ]
        )
        self.assertEqual(refresh.newest(body), "0.4.0")

    def test_a_yanked_release_is_not_something_to_move_to(self):
        body = index_body(
            [
                {"vers": "0.3.4", "yanked": False},
                {"vers": "0.5.0", "yanked": True},
            ]
        )
        self.assertEqual(refresh.newest(body), "0.3.4")

    def test_a_prerelease_does_not_outrank_its_release(self):
        body = index_body(
            [
                {"vers": "0.22.0-pre.3", "yanked": False},
                {"vers": "0.21.0", "yanked": False},
            ]
        )
        # 0.22.0-pre.3 is still ahead of 0.21.0, and behind a real 0.22.0.
        self.assertEqual(refresh.newest(body), "0.22.0-pre.3")
        with_release = index_body(
            [
                {"vers": "0.22.0-pre.3", "yanked": False},
                {"vers": "0.22.0", "yanked": False},
            ]
        )
        self.assertEqual(refresh.newest(with_release), "0.22.0")

    def test_an_index_with_nothing_unyanked_yields_no_observation(self):
        # The caller must report this rather than record a guess.
        self.assertIsNone(refresh.newest(index_body([{"vers": "1.0.0", "yanked": True}])))
        self.assertIsNone(refresh.newest(""))


class ObservationAgeTests(unittest.TestCase):
    def test_the_oldest_observation_date_is_reported_and_never_fatal(self):
        def observed(tree):
            tree.retirement["packages"][0]["upstream_observed"] = "0.19.0"
            tree.retirement["packages"][0]["observed_at"] = "2020-01-01"

        errors, notes = run(observed, require_observations=True)
        self.assertEqual(errors, [])
        self.assertTrue(any("oldest 2020-01-01" in note for note in notes), notes)
