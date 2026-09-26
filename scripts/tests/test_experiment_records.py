import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
# Imported as the aggregators import it, so patching it here patches theirs.
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records as mod  # noqa: E402

AGGREGATORS = {
    "L003": ROOT / "experiments/lifecycle/L003-fastmem-revocation",
    "L004": ROOT / "experiments/lifecycle/L004-projection-equivalence",
}
MANIFEST = {
    "id": "L900",
    "status": "running",
    "seeds": [1, 2],
    "entrypoint": "run <seed>",
    "smoke": "smoke <seed>",
}


def record(sha: str, seed: int = 1, started_at: str = "20260101T000000Z", **changes) -> dict:
    """A run record as `scripts/run_experiment.py run` writes it."""
    base = {
        "experiment_id": "L900",
        "git_sha": sha,
        "manifest": dict(MANIFEST),
        "cargo_lock_sha256": "c" * 64,
        "parameters": {"iterations": "30"},
        "entrypoint": "entrypoint",
        "seed": seed,
        "started_at": started_at,
        "exit_code": 0,
    }
    base.update(changes)
    return base


def git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c", "user.name=PTR tests",
        "-c", "user.email=tests@example.invalid",
        "-c", "commit.gpgsign=false",
        "-c", "init.defaultBranch=main",
        *args,
    ]
    return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def commit(root: Path, files: dict[str, str], message: str) -> str:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "--no-verify", "-m", message)
    return git(root, "rev-parse", "HEAD")


class AgreementTests(unittest.TestCase):
    def test_records_of_one_configuration_agree_whatever_the_status(self):
        # The manifest's status changes when the experiment completes.
        records = {"a.json": record("a" * 40), "b.json": record("b" * 40, seed=2)}
        self.assertEqual(mod.agreement_errors("L900", {**MANIFEST, "status": "completed"}, records), [])

    def test_a_record_of_another_experiment_or_of_no_commit_is_refused(self):
        records = {
            "other.json": record("a" * 40, experiment_id="L901"),
            "unknown.json": record("unknown"),
            "option.json": record("--output=/tmp/x"),
            "none.json": record(None),
        }
        errors = mod.agreement_errors("L900", MANIFEST, records)
        self.assertEqual(len(errors), 4)
        self.assertIn("other.json is a record of 'L901', not 'L900'", errors)
        for name in ("unknown.json", "option.json", "none.json"):
            self.assertTrue(any(error.startswith(f"{name} names no commit") for error in errors), name)

    def test_records_of_different_configurations_are_refused(self):
        cases = {
            "parameters": {"parameters": {"iterations": "3"}},
            "Cargo.lock": {"cargo_lock_sha256": "d" * 64},
            "entrypoint": {"entrypoint": "smoke"},
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                records = {"a.json": record("a" * 40), "b.json": record("a" * 40, seed=2, **change)}
                errors = mod.agreement_errors("L900", MANIFEST, records)
                self.assertEqual(len(errors), 1)
                self.assertTrue(errors[0].startswith(f"the records disagree on {label}: "), errors[0])
                self.assertIn("b.json", errors[0])

    def test_a_record_of_an_entrypoint_the_manifest_does_not_declare_is_refused(self):
        records = {
            "a.json": record("a" * 40, entrypoint="nightly"),
            "b.json": record("a" * 40, seed=2, entrypoint="nightly"),
        }
        self.assertEqual(
            mod.agreement_errors("L900", MANIFEST, records),
            [
                "a.json ran 'nightly', which experiment.toml declares no command for",
                "b.json ran 'nightly', which experiment.toml declares no command for",
            ],
        )

    def test_a_record_run_under_another_manifest_is_refused(self):
        records = {
            "a.json": record("a" * 40),
            "b.json": record("a" * 40, seed=2, manifest={**MANIFEST, "seeds": [1, 2, 3]}),
        }
        self.assertEqual(
            mod.agreement_errors("L900", MANIFEST, records),
            ["b.json ran under an experiment.toml that differs from the current one"],
        )
        with self.assertRaisesRegex(mod.ProvenanceError, "b.json ran under"):
            mod.source_revision("L900", MANIFEST, records, ROOT, ROOT / "experiments/L900")
        with self.assertRaisesRegex(mod.ProvenanceError, "no run records"):
            mod.source_revision("L900", MANIFEST, {}, ROOT, ROOT / "experiments/L900")


class RevisionTests(unittest.TestCase):
    """`source_revision` against a scratch repository."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.experiment = self.root / "experiments/L900-x"
        git(self.root, "init", "-q")
        self.first = commit(
            self.root,
            {
                "Cargo.toml": "[workspace]\n",
                "src/lib.rs": "pub fn f() {}\n",
                "scripts/experiment_records.py": "# judge\n",
                "scripts/run_experiment.py": "# recorder\n",
                "scripts/mutation_check.py": "# mutator\n",
                "experiments/L900-x/aggregate.py": "HARD = ['a', 'b']\n",
                "experiments/L900-x/tests/mutations.toml": "[[mutation]]\n",
                "experiments/L900-x/results/a.json": "{}\n",
            },
            "code",
        )
        # Archiving a record commits only the record.
        self.archived = commit(self.root, {"experiments/L900-x/results/b.json": "{}\n"}, "archive")

    def revision(self, records: dict[str, dict]) -> str:
        return mod.source_revision("L900", MANIFEST, records, self.root, self.experiment)

    def tearDown(self):
        self.directory.cleanup()

    def records(self, *shas: str) -> dict[str, dict]:
        return {
            f"run-{index}.json": record(sha, seed=index, started_at=f"20260101T00000{index}Z")
            for index, sha in enumerate(shas)
        }

    def test_records_archived_one_commit_at_a_time_share_their_code(self):
        revision = self.revision(self.records(self.first, self.archived))
        self.assertEqual(revision, self.first)
        # The earliest record names the revision, whatever order they come in.
        records = self.records(self.archived, self.first)
        records["run-0.json"]["started_at"] = "20260102T000000Z"
        self.assertEqual(self.revision(records), self.first)

    def test_records_that_ran_different_code_are_refused(self):
        changed = commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "change")
        with self.assertRaisesRegex(mod.ProvenanceError, r"run-1\.json ran at .* differs in src/lib\.rs"):
            self.revision(self.records(self.first, changed))

    def test_a_checkout_whose_code_changed_since_the_records_is_refused(self):
        records = self.records(self.first, self.archived)
        for relative in ("src/lib.rs", "Cargo.lock", "migrations/0001.sql", "crates/x/Cargo.toml"):
            with self.subTest(changed=relative):
                (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
                (self.root / relative).write_text("-- changed\n", encoding="utf-8")
                git(self.root, "add", "-A")
                # Uncommitted, and then committed.
                with self.assertRaisesRegex(mod.ProvenanceError, "code has changed since"):
                    self.revision(records)
                git(self.root, "commit", "-q", "--no-verify", "-m", relative)
                with self.assertRaisesRegex(mod.ProvenanceError, f"code has changed since, in {relative}"):
                    self.revision(records)
                git(self.root, "reset", "-q", "--hard", self.archived)

    def test_a_change_outside_the_code_is_not_a_change_of_code(self):
        (self.root / "README.md").write_text("notes\n", encoding="utf-8")
        commit(
            self.root,
            {
                "experiments/L900-x/results/run.json": "{}\n",
                "experiments/L900-x/README.md": "result\n",
                "scripts/check_repo.py": "# unrelated\n",
            },
            "aggregate",
        )
        records = self.records(self.first, self.archived)
        self.assertEqual(self.revision(records), self.first)

    def test_a_commit_git_cannot_find_is_refused(self):
        with self.assertRaisesRegex(mod.ProvenanceError, "cannot compare"):
            self.revision(self.records("0" * 40))

    def test_records_judged_by_changed_aggregation_logic_are_refused(self):
        # Dropping a hard counter from HARD after seeing the runs changes what
        # the records' verdict means, as much as changing the harness does.
        records = self.records(self.first, self.archived)
        for relative in (
            "experiments/L900-x/aggregate.py",
            "scripts/experiment_records.py",
            "scripts/run_experiment.py",
        ):
            with self.subTest(changed=relative):
                commit(self.root, {relative: "HARD = ['a']\n"}, relative)
                with self.assertRaisesRegex(mod.ProvenanceError, f"code has changed since, in {relative}"):
                    self.revision(records)
                git(self.root, "reset", "-q", "--hard", self.archived)
        # The mutation checker and its plan decide mutation evidence, not seeds.
        commit(self.root, {"scripts/mutation_check.py": "# changed\n"}, "mutator")
        self.assertEqual(self.revision(records), self.first)

    def test_the_seed_and_mutation_provenance_cover_their_own_scripts(self):
        seeds = mod.seed_record_paths(self.experiment, self.root)
        mutations = mod.mutation_record_paths(self.experiment, self.root)
        for paths in (seeds, mutations):
            self.assertTrue(set(mod.CODE_PATHS) <= set(paths))
            self.assertIn("scripts/experiment_records.py", paths)
            self.assertIn("experiments/L900-x/aggregate.py", paths)
        self.assertIn("scripts/run_experiment.py", seeds)
        self.assertIn("scripts/mutation_check.py", mutations)
        self.assertIn("experiments/L900-x/tests/mutations.toml", mutations)
        self.assertNotIn("experiments/L900-x/tests/mutations.toml", seeds)


class SourceTreeTests(unittest.TestCase):
    """`uncommitted_files` finds what a run would execute but HEAD does not hold."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.experiment = self.root / "experiments/L900-x"
        git(self.root, "init", "-q")
        commit(
            self.root,
            {
                ".gitignore": "__pycache__/\ntarget/\n",
                "Cargo.toml": "[workspace]\n",
                "src/lib.rs": "pub fn f() {}\n",
                "scripts/run_experiment.py": "# recorder\n",
                "experiments/L900-x/experiment.toml": "id = 'L900'\n",
                "experiments/L900-x/results/.gitkeep": "",
            },
            "code",
        )
        self.pathspecs = mod.tree_pathspecs(
            self.experiment,
            self.experiment / "results",
            self.root,
            mod.seed_record_paths(self.experiment, self.root),
        )

    def tearDown(self):
        self.directory.cleanup()

    def write(self, relative: str, text: str = "changed\n") -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_a_clean_tree_has_no_uncommitted_sources(self):
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), [])

    def test_modified_staged_deleted_and_untracked_sources_are_reported(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("crates/new/src/lib.rs")
        self.write("scripts/run_experiment.py", "# edited recorder\n")
        git(self.root, "add", "scripts/run_experiment.py")
        (self.root / "Cargo.toml").unlink()
        self.write("experiments/L900-x/config.toml")
        self.assertEqual(
            sorted(mod.uncommitted_files(self.root, self.pathspecs)),
            [
                "Cargo.toml",
                "crates/new/src/lib.rs",
                "experiments/L900-x/config.toml",
                "scripts/run_experiment.py",
                "src/lib.rs",
            ],
        )

    def test_results_ignored_files_and_unrelated_files_are_not_sources(self):
        # Records accumulate in results/ between runs; build output is ignored.
        self.write("experiments/L900-x/results/run-1-seed-1.json", "{}\n")
        self.write("experiments/L900-x/__pycache__/aggregate.cpython-313.pyc")
        self.write("target/release/ptr-bench.rs")
        self.write("README.md")
        self.write("scripts/check_repo.py")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), [])

    def test_git_that_cannot_list_the_tree_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(mod.ProvenanceError, "cannot list"):
                mod.uncommitted_files(Path(directory), self.pathspecs)


class StalenessTests(unittest.TestCase):
    """`staleness_errors`: archived results must describe HEAD's code or say
    since when they no longer do."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.experiment = self.root / "experiments/L900-x"
        self.results = self.experiment / "results"
        git(self.root, "init", "-q")
        self.code = commit(
            self.root,
            {
                "Cargo.toml": "[workspace]\n",
                "src/lib.rs": "pub fn f() {}\n",
                "experiments/L900-x/aggregate.py": "HARD = ['a']\n",
                "experiments/L900-x/tests/mutations.toml": "[[mutation]]\n",
            },
            "code",
        )
        self.archived = commit(
            self.root,
            {
                "experiments/L900-x/results/run.json": json.dumps({"git_sha": self.code}),
                "experiments/L900-x/results/mutations.json": json.dumps({"git_sha": self.code}),
            },
            "archive",
        )

    def tearDown(self):
        self.directory.cleanup()

    def errors(self) -> list[str]:
        return mod.staleness_errors("L900", self.experiment, self.results, self.root)

    def mark(self, **fields) -> None:
        marker = {"results_git_sha": self.code, "reason": "rerun pending"}
        marker.update(fields)
        text = "".join(f"{key} = {json.dumps(value)}\n" for key, value in marker.items())
        commit(self.root, {"experiments/L900-x/results/STALE.toml": text}, "mark")

    def test_results_of_the_current_code_are_not_stale(self):
        commit(self.root, {"README.md": "notes\n", "experiments/L900-x/README.md": "x\n"}, "docs")
        self.assertEqual(self.errors(), [])

    def test_results_of_other_code_fail_without_a_marker(self):
        cases = {
            "src/lib.rs": ["run.json", "mutations.json"],
            "experiments/L900-x/aggregate.py": ["run.json", "mutations.json"],
            "experiments/L900-x/tests/mutations.toml": ["mutations.json"],
        }
        for relative, stale in cases.items():
            with self.subTest(changed=relative):
                commit(self.root, {relative: "changed\n"}, relative)
                errors = self.errors()
                self.assertEqual(len(errors), len(stale), errors)
                for name, error in zip(stale, errors):
                    self.assertIn(f"results/{name} ran at {self.code}", error)
                    self.assertIn(f"changed since, in {relative}", error)
                git(self.root, "reset", "-q", "--hard", self.archived)

    def test_an_honest_marker_names_the_results_and_the_first_change(self):
        first = commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "first change")
        commit(self.root, {"src/lib.rs": "pub fn f() { h() }\n"}, "second change")
        self.mark(stale_since=first)
        self.assertEqual(self.errors(), [])
        # Abbreviations resolve to the same commits.
        git(self.root, "rm", "-q", "experiments/L900-x/results/STALE.toml")
        self.mark(stale_since=first[:10], results_git_sha=self.code[:10])
        self.assertEqual(self.errors(), [])

    def test_a_dishonest_or_incomplete_marker_is_refused(self):
        first = commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "first change")
        second = commit(self.root, {"src/lib.rs": "pub fn f() { h() }\n"}, "second change")
        cases = {
            "later commit": ({"stale_since": second}, f"the first commit after {self.code}"),
            "other results": ({"stale_since": first, "results_git_sha": first}, "names results of"),
            "no reason": ({"stale_since": first, "reason": " "}, "no reason"),
            "unknown commit": ({"stale_since": "0" * 40}, "not a commit"),
            "missing field": ({"stale_since": None}, "stale_since"),
        }
        for label, (fields, expected) in cases.items():
            with self.subTest(case=label):
                marker = {key: value for key, value in fields.items() if value is not None}
                if "stale_since" not in marker:
                    commit(
                        self.root,
                        {
                            "experiments/L900-x/results/STALE.toml": (
                                f'results_git_sha = "{self.code}"\nreason = "x"\n'
                            )
                        },
                        "mark",
                    )
                else:
                    self.mark(**marker)
                errors = self.errors()
                self.assertEqual(len(errors), 1, errors)
                self.assertIn(expected, errors[0])
                git(self.root, "reset", "-q", "--hard", second)

    def test_a_marker_may_name_any_commit_where_every_stale_file_has_the_code_it_ran(self):
        # Seeds archived one commit at a time and a mutation check run later:
        # run.json names the earliest record's commit, mutations.json a later
        # one with the same code. A marker naming either used to be refused
        # for the other file, so no marker could cover them.
        later = commit(self.root, {"experiments/L900-x/results/run-seed-1.json": "{}"}, "archive seed 1")
        commit(
            self.root,
            {"experiments/L900-x/results/mutations.json": json.dumps({"git_sha": later})},
            "mutation check",
        )
        self.assertEqual(self.errors(), [])
        first = commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "first change")
        for marked in (self.code, later):
            with self.subTest(marked=marked):
                self.mark(results_git_sha=marked, stale_since=first)
                self.assertEqual(self.errors(), [])
                git(self.root, "reset", "-q", "--hard", first)

    def test_a_marker_refuses_a_commit_where_a_stale_file_had_other_code(self):
        # The mutation plan changed before the mutation check ran, so at the
        # records' commit mutations.json's provenance differs; the seed
        # records' provenance holds at the mutation check's commit, which the
        # marker may name instead.
        planned = commit(
            self.root, {"experiments/L900-x/tests/mutations.toml": "[[mutation]]\nname = 'b'\n"}, "plan"
        )
        commit(
            self.root,
            {"experiments/L900-x/results/mutations.json": json.dumps({"git_sha": planned})},
            "mutation check",
        )
        first = commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "first change")
        self.mark(results_git_sha=self.code, stale_since=first)
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(f"names results of {self.code}, but mutations.json ran at {planned}", errors[0])
        self.assertIn("whose provenance files differ there in experiments/L900-x/tests/mutations.toml", errors[0])
        git(self.root, "reset", "-q", "--hard", first)
        self.mark(results_git_sha=planned, stale_since=first)
        self.assertEqual(self.errors(), [])

    def test_a_marker_on_current_results_is_refused(self):
        # A rerun makes the results current; its marker must not outlive it.
        self.mark(stale_since=self.code)
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("match HEAD's code; remove it", errors[0])

    def test_results_that_name_no_commit_are_refused(self):
        commit(self.root, {"experiments/L900-x/results/run.json": json.dumps({"git_sha": "unknown"})}, "bad")
        self.assertEqual(
            self.errors(), ["L900: experiments/L900-x/results/run.json names no commit it ran at: 'unknown'"]
        )


class StaleMergeTests(unittest.TestCase):
    """`staleness_errors` on merges: the results were archived on one line of
    history, and another line (the base branch of a pull request) changed a
    provenance file after the two diverged. CI checks the pull request's head
    on push and a merge of it into the base branch on pull_request, and the
    base branch after the merge."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.experiment = self.root / "experiments/L900-x"
        self.results = self.experiment / "results"
        git(self.root, "init", "-q")
        commit(
            self.root,
            {
                "Cargo.toml": "[workspace]\n",
                "Cargo.lock": "v1\n",
                "src/lib.rs": "pub fn f() {}\n",
                "experiments/L900-x/aggregate.py": "HARD = ['a']\n",
            },
            "fork point",
        )
        git(self.root, "checkout", "-q", "-b", "pr")
        self.code = commit(self.root, {"src/lib.rs": "pub fn f() { a() }\n"}, "pr code")
        commit(
            self.root,
            {"experiments/L900-x/results/run.json": json.dumps({"git_sha": self.code})},
            "archive",
        )

    def tearDown(self):
        self.directory.cleanup()

    def errors(self) -> list[str]:
        return mod.staleness_errors("L900", self.experiment, self.results, self.root)

    def mark(self, stale_since: str) -> None:
        commit(
            self.root,
            {
                "experiments/L900-x/results/STALE.toml": (
                    f'results_git_sha = "{self.code}"\nstale_since = "{stale_since}"\nreason = "rerun pending"\n'
                )
            },
            "mark",
        )

    def bump_main(self) -> str:
        """A dependency bump on main, which never saw the results."""
        git(self.root, "checkout", "-q", "main")
        bump = commit(self.root, {"Cargo.lock": "v2\n"}, "bump a dependency")
        git(self.root, "checkout", "-q", "pr")
        return bump

    def merge(self, into: str, other: str) -> str:
        git(self.root, "checkout", "-q", into)
        git(self.root, "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", "-m", "merge", other)
        return git(self.root, "rev-parse", "HEAD")

    def test_a_base_branch_commit_merged_in_does_not_move_the_first_change(self):
        first = commit(self.root, {"src/lib.rs": "pub fn f() { b() }\n"}, "later code change")
        self.mark(stale_since=first)
        self.assertEqual(self.errors(), [])
        self.bump_main()
        # The pull_request merge ref and main after the merge: the bump sorts
        # before the change in topological order but never descended from the
        # results, so it used to be named as their first change.
        self.merge("main", "pr")
        self.assertEqual(self.errors(), [])
        # The pull request merging its base in keeps its marker too.
        self.merge("pr", "main")
        self.assertEqual(self.errors(), [])

    def test_a_change_reaching_the_results_only_through_a_merge_is_stale_since_that_merge(self):
        bump = self.bump_main()
        merged = self.merge("pr", "main")
        # The bump changed the lock file the results never ran, and it reached
        # their line of history at the merge.
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("changed since, in Cargo.lock", errors[0])
        self.mark(stale_since=bump)
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(f"says stale since {bump}, but the first commit after {self.code}", errors[0])
        self.assertIn(f"is {merged}", errors[0])
        git(self.root, "reset", "-q", "--hard", merged)
        self.mark(stale_since=merged)
        self.assertEqual(self.errors(), [])


    def test_each_line_of_history_leaving_the_results_has_its_own_first_change(self):
        # Two branches off the results each change the code and are merged:
        # either change is where the results first went stale on its line,
        # and the merge, which both precede, is not.
        git(self.root, "checkout", "-q", "-b", "other")
        theirs = commit(self.root, {"Cargo.lock": "v3\n"}, "other line's change")
        git(self.root, "checkout", "-q", "pr")
        ours = commit(self.root, {"src/lib.rs": "pub fn f() { b() }\n"}, "this line's change")
        merged = self.merge("pr", "other")
        for since, accepted in ((ours, True), (theirs, True), (merged, False)):
            with self.subTest(since=since):
                self.mark(stale_since=since)
                errors = self.errors()
                if accepted:
                    self.assertEqual(errors, [])
                else:
                    self.assertEqual(len(errors), 1, errors)
                    self.assertIn(f"says stale since {merged}", errors[0])
                git(self.root, "reset", "-q", "--hard", merged)

    def test_a_marker_naming_a_commit_off_heads_history_is_refused(self):
        # main takes the same code as the results in a commit of its own: its
        # provenance matches, but no change after it leads to HEAD.
        git(self.root, "checkout", "-q", "main")
        twin = commit(self.root, {"src/lib.rs": "pub fn f() { a() }\n"}, "same code on main")
        git(self.root, "checkout", "-q", "pr")
        first = commit(self.root, {"src/lib.rs": "pub fn f() { b() }\n"}, "later code change")
        commit(
            self.root,
            {
                "experiments/L900-x/results/STALE.toml": (
                    f'results_git_sha = "{twin}"\nstale_since = "{first}"\nreason = "rerun pending"\n'
                )
            },
            "mark",
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(f"names results of {twin}, which is not on HEAD's history", errors[0])


class AggregatorTests(unittest.TestCase):
    """The L003 and L004 aggregators bind their output to the records' commit,
    to the harness results the records carry and to current mutation
    evidence, and pass only a complete run."""

    RESULT_KEYS = (
        "writes_replayed",
        "revocations_with_removal",
        "full_refold_writes",
        "window_denials",
        "window_reads",
        "append_races_append_first",
        "append_races",
        "checkpoint_races_stored",
        "checkpoint_races",
        "revocation_crashes_committed",
        "revocation_crashes",
        "max_replayed",
        "max_journal_len",
    )
    BENCHMARKS = {"L003": "fastmem-revocation", "L004": "projection-equivalence"}

    def load(self, exp_id: str):
        here = AGGREGATORS[exp_id]
        spec = importlib.util.spec_from_file_location(f"aggregate_{exp_id}", here / "aggregate.py")
        agg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg)
        return agg

    def aggregate(
        self,
        exp_id: str,
        change=None,
        result_change=None,
        code_changes=(),
        mutations=None,
        stale_code=None,
        marker=False,
        entrypoint="entrypoint",
    ):
        """Run `exp_id`'s aggregator over synthetic records, one per declared
        seed at commits "1111111" and "2222222" with the same code, into a
        scratch results directory. Every seed's harness result passes and
        reaches every probe. `change` edits the last record and
        `result_change` its harness result; `code_changes` are the code files
        the checkout changed since; `mutations` is written as
        results/mutations.json, and `stale_code` maps a commit to the files
        whose code differs at it from the checkout; `marker` plants
        results/STALE.toml. Returns the run.json it wrote, or raises the
        SystemExit it refused with."""
        stale_code = stale_code or {}
        calls = []

        def changes(base, head, _root, paths=mod.CODE_PATHS):
            calls.append((base, head, tuple(paths)))
            if base in stale_code:
                return list(stale_code[base])
            return [] if head is not None else list(code_changes)

        here = AGGREGATORS[exp_id]
        agg = self.load(exp_id)
        benchmark = self.BENCHMARKS[exp_id]
        manifest = tomllib.loads((here / "experiment.toml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            inputs = []
            for index, seed in enumerate(manifest["seeds"]):
                result = {
                    "benchmark": benchmark,
                    "iterations": 30,
                    "seed": seed,
                    "server": "PostgreSQL 18",
                    "hard_failures": 0,
                }
                if exp_id == "L004":
                    result["turso_oracle"] = True
                result.update({key: 0 for key in (*self.RESULT_KEYS, *agg.HARD)})
                result.update({key: 1 for key in agg.COVERAGE})
                last = index == len(manifest["seeds"]) - 1
                if result_change and last:
                    result.update(result_change)
                run = {
                    **record("1111111" if index < 2 else "2222222", seed=seed),
                    "experiment_id": exp_id,
                    "manifest": {**manifest, "status": "running"},
                    "started_at": f"20260101T00000{index}Z",
                    "command": ["cargo", "run", "--", benchmark, "30", str(seed)],
                    "entrypoint": entrypoint,
                    "stdout": "Compiling\n" + json.dumps(result) + "\n",
                }
                if change and last:
                    run.update(change)
                path = results / f"run-20260101T00000{index}Z-seed-{seed}.json"
                path.write_text(json.dumps(run), encoding="utf-8")
                inputs.append(str(path))
            if mutations is not None:
                (results / "mutations.json").write_text(json.dumps(mutations), encoding="utf-8")
            if marker:
                (results / "STALE.toml").write_text('stale_since = "1111111"\n', encoding="utf-8")
            with (
                mock.patch.object(agg, "RESULTS", results),
                mock.patch.object(sys, "argv", ["aggregate.py", *inputs]),
                mock.patch.object(mod, "code_changes", side_effect=changes),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    agg.main()
                except SystemExit:
                    self.assertFalse((results / "run.json").exists())
                    self.assertFalse((results / "metrics.json").exists())
                    self.assertEqual((results / "STALE.toml").exists(), marker)
                    raise
            self.assertFalse((results / "STALE.toml").exists())
            run = json.loads((results / "run.json").read_text(encoding="utf-8"))
            run["_code_change_calls"] = calls
            return run

    def mutations(self, exp_id: str, **changes) -> dict:
        """A mutations.json as `scripts/mutation_check.py` writes it."""
        evidence = {
            "experiment_id": exp_id,
            "git_sha": "3333333",
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "subcommand": self.BENCHMARKS[exp_id],
            "killed": 2,
            "total": 2,
            "mutations": [{"name": "a", "result": "killed"}, {"name": "b", "result": "killed"}],
        }
        evidence.update(changes)
        return evidence

    def test_the_aggregate_names_the_commit_the_records_ran_at(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id)
                self.assertTrue(run["hard_pass"])
                self.assertEqual(run["verdict"], "hard-pass")
                self.assertEqual(run["git_sha"], "1111111")
                self.assertIn("aggregated_at_git_sha", run)
                self.assertEqual(
                    [seed["git_sha"] for seed in run["seeds"]],
                    ["1111111", "1111111", "2222222", "2222222", "2222222"],
                )
                # The records are bound to the aggregator that judges them.
                relative = AGGREGATORS[exp_id].relative_to(ROOT).as_posix()
                self.assertTrue(
                    all(f"{relative}/aggregate.py" in paths for _, _, paths in run["_code_change_calls"])
                )

    def test_records_of_other_code_or_configuration_are_not_aggregated(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id, refused="stale code"):
                with self.assertRaisesRegex(SystemExit, "refusing to aggregate: .* changed since"):
                    self.aggregate(exp_id, code_changes=["crates/ptr-pg/src/lib.rs"])
            with self.subTest(experiment=exp_id, refused="parameters"):
                with self.assertRaisesRegex(SystemExit, "disagree on parameters"):
                    self.aggregate(exp_id, change={"parameters": {"iterations": "3"}})
            with self.subTest(experiment=exp_id, refused="experiment"):
                with self.assertRaisesRegex(SystemExit, "is a record of 'L999'"):
                    self.aggregate(exp_id, change={"experiment_id": "L999"})

    def test_a_result_line_of_another_seed_or_benchmark_is_refused(self):
        # The wrapper selected the record by its seed; the result it carries
        # must be that seed's run of this experiment's benchmark.
        for exp_id in AGGREGATORS:
            other = self.BENCHMARKS["L004" if exp_id == "L003" else "L003"]
            cases = {
                "seed": ({"seed": 17}, None, "reports seed 17, not 101"),
                "benchmark": ({"benchmark": other}, None, f"reports benchmark '{other}'"),
                "iterations": ({"iterations": 3}, None, "reports 3 iterations, not 30"),
                "command": (None, {"command": ["cargo", "run", "--", other, "30", "101"]}, "did not run"),
                "hard counter": ({"read_failures": None}, None, "reports no count of read_failures"),
                "no result": (None, {"stdout": "Compiling\n"}, "has no result line"),
                "bad result": (None, {"stdout": "{not json\n"}, "result line is not a JSON object"),
            }
            for label, (result_change, change, expected) in cases.items():
                with self.subTest(experiment=exp_id, refused=label):
                    with self.assertRaisesRegex(SystemExit, f"refusing to aggregate: .*{expected}"):
                        self.aggregate(exp_id, change=change, result_change=result_change)

    def test_current_mutation_evidence_is_carried_into_the_run(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id, mutations=self.mutations(exp_id))
                self.assertEqual(run["mutation_checks"], {"killed": 2, "total": 2, "git_sha": "3333333"})
                relative = AGGREGATORS[exp_id].relative_to(ROOT).as_posix()
                evidence = [paths for base, _, paths in run["_code_change_calls"] if base == "3333333"]
                self.assertTrue(evidence)
                for paths in evidence:
                    self.assertIn("scripts/mutation_check.py", paths)
                    self.assertIn(f"{relative}/tests/mutations.toml", paths)
                self.assertIsNone(self.aggregate(exp_id)["mutation_checks"])

    def test_mutation_evidence_of_other_code_or_another_experiment_is_refused(self):
        for exp_id in AGGREGATORS:
            cases = {
                "stale code": (
                    {},
                    {"3333333": ["crates/ptr-pg/src/adapters/projection.rs"]},
                    "mutations.json ran at 3333333",
                ),
                "stale plan": (
                    {},
                    {"3333333": ["tests/mutations.toml"]},
                    "changed since, in tests/mutations.toml",
                ),
                "experiment": ({"experiment_id": "L999"}, {}, "is evidence of 'L999'"),
                "subcommand": ({"subcommand": "semdb"}, {}, "mutated 'semdb'"),
                "commit": ({"git_sha": "unknown"}, {}, "names no commit"),
                "counts": ({"killed": 3}, {}, "counts 3 of 2 killed"),
            }
            for label, (fields, stale, expected) in cases.items():
                with self.subTest(experiment=exp_id, refused=label):
                    with self.assertRaisesRegex(SystemExit, f"refusing to aggregate: .*{expected}"):
                        self.aggregate(exp_id, mutations=self.mutations(exp_id, **fields), stale_code=stale)

    def test_a_run_that_missed_a_probe_is_not_a_hard_pass(self):
        for exp_id in AGGREGATORS:
            probe = self.load(exp_id).COVERAGE[0]
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id, result_change={probe: 0})
                self.assertFalse(run["probe_coverage_ok"])
                self.assertFalse(run["hard_pass"])
                self.assertEqual(run["verdict"], "coverage-incomplete")

    def test_a_hard_failure_or_failed_exit_is_a_hard_fail_whatever_the_coverage(self):
        for exp_id in AGGREGATORS:
            hard = self.load(exp_id).HARD[0]
            probe = self.load(exp_id).COVERAGE[0]
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id, result_change={hard: 1, probe: 0})
                self.assertEqual((run["hard_pass"], run["verdict"]), (False, "hard-fail"))
                run = self.aggregate(exp_id, change={"exit_code": 1})
                self.assertEqual((run["hard_pass"], run["verdict"]), (False, "hard-fail"))
                # A hard counter the aggregator does not list still fails the run.
                run = self.aggregate(exp_id, result_change={"hard_failures": 1, "new_hard_counter": 1})
                self.assertEqual((run["hard_pass"], run["verdict"]), (False, "hard-fail"))

    def test_a_postgres_only_l004_run_is_never_a_hard_pass(self):
        # postgres_entrypoint runs the harness without Turso, the third
        # implementation the baseline names.
        manifest = tomllib.loads((AGGREGATORS["L004"] / "experiment.toml").read_text(encoding="utf-8"))
        run = self.aggregate("L004", result_change={"turso_oracle": False}, entrypoint="postgres_entrypoint")
        self.assertFalse(run["hard_pass"])
        self.assertFalse(run["turso_oracle"])
        self.assertEqual(run["verdict"], "no-turso-oracle")
        self.assertEqual(run["entrypoint"], manifest["postgres_entrypoint"])
        self.assertEqual(self.aggregate("L004")["entrypoint"], manifest["entrypoint"])
        run = self.aggregate("L004", result_change={"turso_oracle": "true"})
        self.assertEqual(run["verdict"], "no-turso-oracle")

    def test_fresh_results_clear_the_stale_marker(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id):
                self.assertTrue(self.aggregate(exp_id, marker=True)["hard_pass"])
                with self.assertRaisesRegex(SystemExit, "refusing to aggregate"):
                    self.aggregate(exp_id, marker=True, code_changes=["src/lib.rs"])


if __name__ == "__main__":
    unittest.main()
