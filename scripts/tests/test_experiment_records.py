import contextlib
import datetime
import errno
import hashlib
import importlib.util
import io
import json
import os
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


@contextlib.contextmanager
def disk_full_writing(name: str):
    """Fail every write to a file whose name contains `name` as a full disk
    does: the file is opened, and truncated when its mode says so, and its
    first write raises ENOSPC."""
    real_open = io.open

    def opener(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and name in Path(file).name and set(mode) & set("wxa+"):

            def full(*_args, **_kwargs):
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

            handle.write = full
        return handle

    with mock.patch("io.open", side_effect=opener), mock.patch("builtins.open", side_effect=opener):
        yield


def files(directory: Path) -> dict[str, str]:
    """Every file in `directory` by name, with its text."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(directory.iterdir())}

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
        # No automatic maintenance: a commit would start it detached, and it
        # can still be writing into .git/objects when the test deletes the
        # repository.
        "-c", "maintenance.auto=false",
        "-c", "gc.auto=0",
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

    def test_a_manifest_agrees_with_the_record_as_json_holds_it(self):
        # JSON has no date and the record holds a TOML date as its text, and
        # NaN is not equal to itself: the manifest is compared as a record
        # can hold it.
        manifest = {**MANIFEST, "created": datetime.date(2026, 1, 2), "loss_cap": float("nan")}
        held = json.loads(json.dumps(manifest, default=mod.toml_time))
        self.assertEqual(held["created"], "2026-01-02")
        records = {"a.json": record("a" * 40, manifest=held)}
        self.assertEqual(mod.agreement_errors("L900", manifest, records), [])
        changed = {"b.json": record("b" * 40, manifest={**held, "created": "2026-01-03"})}
        self.assertEqual(
            mod.agreement_errors("L900", manifest, changed),
            ["b.json ran under an experiment.toml that differs from the current one"],
        )
        with self.assertRaises(TypeError):
            mod.toml_time(object())

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

    def test_an_untracked_provenance_file_in_the_checkout_is_refused(self):
        # git diff omits untracked files, and the workspace takes every
        # crates/ptr-* directory as a member: an untracked crate changes what
        # Cargo evaluates though no tracked file differs from the records'.
        records = self.records(self.first, self.archived)
        for relative in ("crates/ptr-new/Cargo.toml", "crates/ptr-new/src/lib.rs", "migrations/0002.sql"):
            with self.subTest(untracked=relative):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("-- new\n", encoding="utf-8")
                self.assertEqual(mod.code_changes(self.first, None, self.root), [])
                with self.assertRaisesRegex(mod.ProvenanceError, f"HEAD does not hold {relative} in the checkout"):
                    self.revision(records)
                path.unlink()
        self.assertEqual(self.revision(records), self.first)

    def test_mutation_evidence_beside_an_untracked_provenance_file_is_refused(self):
        evidence = self.experiment / "results/mutations.json"
        evidence.write_text(
            json.dumps(
                {
                    "experiment_id": "L900",
                    "git_sha": self.first,
                    "subcommand": "bench",
                    "killed": 1,
                    "total": 1,
                    "mutations": [{"name": "a", "result": "killed"}],
                }
            ),
            encoding="utf-8",
        )
        summary = mod.mutation_evidence("L900", evidence, "bench", self.root, self.experiment)
        self.assertEqual(summary, {"killed": 1, "total": 1, "git_sha": self.first})
        (self.root / "crates/ptr-new/src").mkdir(parents=True)
        (self.root / "crates/ptr-new/src/lib.rs").write_text("pub fn g() {}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            mod.ProvenanceError, "mutations.json ran at .*HEAD does not hold crates/ptr-new/src/lib.rs"
        ):
            mod.mutation_evidence("L900", evidence, "bench", self.root, self.experiment)

    def test_an_untracked_crate_git_does_not_look_into_in_the_checkout_is_refused(self):
        # git lists a nested repository or a symlinked directory as one entry
        # no file pathspec matches; the workspace still takes the crate.
        crates = commit(
            self.root,
            {"crates/ptr-a/Cargo.toml": "[package]\n", "crates/ptr-a/src/lib.rs": "pub fn a() {}\n"},
            "crate",
        )
        records = self.records(crates)
        nested = self.root / "crates/ptr-nested"
        (nested / "src").mkdir(parents=True)
        (nested / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        (nested / "src/lib.rs").write_text("pub fn n() {}\n", encoding="utf-8")
        git(nested, "init", "-q")
        with self.assertRaisesRegex(mod.ProvenanceError, "HEAD does not hold crates/ptr-nested in the checkout"):
            self.revision(records)
        try:
            os.symlink(nested, self.root / "crates/ptr-extsym")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")
        with self.assertRaisesRegex(
            mod.ProvenanceError, "HEAD does not hold crates/ptr-extsym, crates/ptr-nested in the checkout"
        ):
            self.revision(records)

    def test_a_directory_symlink_is_provenance_though_its_path_matches_no_pathspec(self):
        # Linking a crate in, or pointing the link at another crate, changes
        # what Cargo builds; git diff shows only the link's own path.
        crates = commit(
            self.root,
            {
                "crates/ptr-a/Cargo.toml": "[package]\n",
                "crates/ptr-a/src/lib.rs": "pub fn a() {}\n",
                "crates/ptr-b/Cargo.toml": "[package]\n",
                "crates/ptr-b/src/lib.rs": "pub fn b() {}\n",
            },
            "crates",
        )
        try:
            os.symlink("ptr-a", self.root / "crates/ptr-link")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")
        linked = commit(self.root, {}, "link a crate")
        self.assertEqual(mod.code_changes(crates, linked, self.root), ["crates/ptr-link"])
        (self.root / "crates/ptr-link").unlink()
        os.symlink("ptr-b", self.root / "crates/ptr-link")
        retargeted = commit(self.root, {}, "point the link at another crate")
        self.assertEqual(mod.code_changes(linked, retargeted, self.root), ["crates/ptr-link"])
        self.assertEqual(mod.code_changes(linked, None, self.root), ["crates/ptr-link"])
        with self.assertRaisesRegex(mod.ProvenanceError, "code has changed since, in crates/ptr-link"):
            self.revision(self.records(crates))

    def test_what_the_build_reads_besides_rust_and_sql_is_provenance(self):
        # ptr-protocol's build script compiles proto/*.proto into the harness,
        # Cargo prefers .cargo/config to .cargo/config.toml, and rustup prefers
        # rust-toolchain to rust-toolchain.toml.
        records = self.records(self.first, self.archived)
        for relative in ("proto/events.proto", ".cargo/config", "rust-toolchain"):
            with self.subTest(changed=relative):
                commit(self.root, {relative: "changed\n"}, relative)
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


class GitTree:
    """A repository holding one experiment and the provenance pathspecs of
    its seed records, committed."""

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

    def link(self, target, relative: str) -> None:
        """A symlink at `relative` to `target`, or skip the test where the
        platform cannot make one."""
        (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(target, self.root / relative)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")

    def crate(self, relative: str) -> Path:
        """A crate written at `relative` below the root, returned."""
        self.write(f"{relative}/Cargo.toml", "[package]\n")
        self.write(f"{relative}/src/lib.rs", "pub fn g() {}\n")
        return self.root / relative

    def outside_crate(self) -> Path:
        """A crate outside the repository, removed with the test."""
        outside = Path(self.enterContext(tempfile.TemporaryDirectory())) / "crate"
        (outside / "src").mkdir(parents=True)
        (outside / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        (outside / "src/lib.rs").write_text("pub fn e() {}\n", encoding="utf-8")
        return outside


class SourceTreeTests(GitTree, unittest.TestCase):
    """`uncommitted_files` finds what a run would execute but HEAD does not hold."""

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

    def test_a_source_only_this_clones_own_ignore_rules_hide_is_reported(self):
        # .git/info/exclude and core.excludesFile belong to this clone, not to
        # HEAD: a crate they hide is still one Cargo takes as a member.
        (self.root / ".git/info").mkdir(parents=True, exist_ok=True)
        (self.root / ".git/info/exclude").write_text("crates/scratch/\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as home:
            excludes = Path(home) / "excludes"
            excludes.write_text("*.sql\n", encoding="utf-8")
            git(self.root, "config", "core.excludesFile", str(excludes))
            self.write("crates/scratch/Cargo.toml")
            self.write("crates/scratch/src/lib.rs")
            self.write("migrations/0002.sql")
            self.assertEqual(git(self.root, "status", "--porcelain"), "")
            self.assertEqual(
                mod.uncommitted_files(self.root, self.pathspecs),
                ["crates/scratch/Cargo.toml", "crates/scratch/src/lib.rs", "migrations/0002.sql"],
            )

    def test_an_ignore_rule_head_does_not_hold_is_reported(self):
        # An uncommitted rule hides a new source as well as a local one does.
        self.write(".gitignore", "__pycache__/\ntarget/\ncrates/scratch/\n")
        self.write("crates/scratch/src/lib.rs")
        self.write("crates/other/.gitignore", "*.rs\n")
        self.write("crates/other/src/lib.rs")
        self.assertEqual(
            mod.uncommitted_files(self.root, self.pathspecs), [".gitignore", "crates/other/.gitignore"]
        )

    def test_heads_ignore_rules_are_read_as_committed_whatever_replaces_them(self):
        # A replacement object for HEAD's .gitignore would make every
        # untracked source look ignored by HEAD's own rules.
        self.write("everything", "*\n")
        replacement = git(self.root, "hash-object", "-w", "everything")
        (self.root / "everything").unlink()
        git(self.root, "replace", git(self.root, "rev-parse", "HEAD:.gitignore"), replacement)
        self.assertEqual(mod.head_rules(self.root), {".gitignore": b"__pycache__/\ntarget/\n"})
        self.write("crates/other/.gitignore", "*.rs\n")
        self.write("crates/other/src/lib.rs")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), ["crates/other/.gitignore"])

    def test_a_source_git_is_told_to_take_as_heads_is_reported(self):
        # git status and git diff do not look at an assume-unchanged or
        # skip-worktree file, so an edit to one is invisible to both.
        git(self.root, "update-index", "--assume-unchanged", "src/lib.rs")
        git(self.root, "update-index", "--skip-worktree", "Cargo.toml")
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("Cargo.toml", "[workspace]\nmembers = ['crates/*']\n")
        self.assertEqual(git(self.root, "status", "--porcelain"), "")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), ["Cargo.toml", "src/lib.rs"])

    def test_a_symlinked_source_is_reported_though_head_holds_the_link(self):
        # Git holds a symlink's target path, not the content the build reads
        # through it, which can change with no change git would see.
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "shared.rs"
            target.write_text("pub fn s() {}\n", encoding="utf-8")
            try:
                os.symlink(target, self.root / "src/shared.rs")
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"cannot create a symlink: {error}")
            commit(self.root, {}, "link a shared source")
            target.write_text("pub fn s() { panic!() }\n", encoding="utf-8")
            self.assertEqual(git(self.root, "status", "--porcelain"), "")
            self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), ["src/shared.rs"])

    def test_an_untracked_crate_git_does_not_look_into_is_reported(self):
        # git lists a symlinked directory or a nested repository as one entry,
        # which no file pathspec matches, and does not look into it; the
        # workspace takes a crates/ptr-* crate either way as a member.
        self.crate("crates/ptr-a")
        commit(self.root, {}, "a crate")
        self.link(self.outside_crate(), "crates/ptr-extsym")
        git(self.crate("crates/ptr-nested"), "init", "-q")
        self.assertEqual(
            git(self.root, "status", "--porcelain"), "?? crates/ptr-extsym\n?? crates/ptr-nested/"
        )
        self.assertEqual(
            mod.uncommitted_files(self.root, self.pathspecs), ["crates/ptr-extsym", "crates/ptr-nested"]
        )
        # Beside sources too: a module directory linked in from elsewhere.
        self.link(self.outside_crate() / "src", "crates/ptr-a/src/extra")
        self.assertIn("crates/ptr-a/src/extra", mod.uncommitted_files(self.root, self.pathspecs))

    def test_a_committed_directory_symlink_is_reported_though_head_holds_the_link(self):
        # Git holds the link's target path, not the crate read through it,
        # which can change with no change git would see.
        self.crate("crates/ptr-a")
        outside = self.outside_crate()
        self.link(outside, "crates/ptr-link")
        commit(self.root, {}, "link a crate")
        (outside / "src/lib.rs").write_text("pub fn e() { panic!() }\n", encoding="utf-8")
        self.assertEqual(git(self.root, "status", "--porcelain"), "")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), ["crates/ptr-link"])

    def test_a_submodule_where_the_build_reads_is_reported(self):
        # A submodule's entry names a commit of another repository, whose
        # files no pathspec of this one reaches.
        self.crate("crates/ptr-a")
        before = commit(self.root, {}, "a crate")
        git(self.root, "update-index", "--add", "--cacheinfo", f"160000,{before},crates/ptr-sub")
        git(self.root, "commit", "-q", "--no-verify", "-m", "a submodule")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), ["crates/ptr-sub"])
        self.assertEqual(mod.code_changes(before, "HEAD", self.root), ["crates/ptr-sub"])

    def test_a_symlink_or_nested_repository_no_build_reads_is_not_reported(self):
        # Cargo reads a new top-level directory only once a manifest names it:
        # a build directory or virtualenv linked in at the root, or a worktree
        # of the repository kept in an untracked directory, holds no source
        # of this checkout.
        elsewhere = self.outside_crate()
        self.link(elsewhere, "target")
        self.link(elsewhere, ".venv")
        git(self.crate(".claude/worktrees/other"), "init", "-q")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), [])
        # A link at the root counts where a provenance file is named beneath
        # it: Cargo reads .cargo/config.toml.
        (elsewhere / "config.toml").write_text("[build]\n", encoding="utf-8")
        self.link(elsewhere, ".cargo")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), [".cargo"])

    def test_an_ignore_rule_that_hides_no_source_is_not_reported(self):
        # JetBrains IDEs write .idea/.gitignore, tools write rules into the
        # caches they own, and a clone may add a rule of its own to the root
        # .gitignore; none of them hides a source HEAD's rules would show.
        self.write(".idea/.gitignore", "/shelf/\n/workspace.xml\n")
        self.write(".idea/workspace.xml", "<project/>\n")
        self.write(".ruff_cache/.gitignore", "*\n")
        self.write(".ruff_cache/0.15/cache", "x\n")
        self.write(".gitignore", "__pycache__/\ntarget/\n*.local\n")
        self.write("target/debug/build/out/generated.rs", "// generated\n")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), [])

    def test_a_source_hidden_by_a_rule_that_ignores_itself_is_reported(self):
        # An untracked .gitignore of `*` hides itself and every file beside
        # it, such as a build script Cargo compiles into the crate.
        self.crate("crates/ptr-fastmem")
        commit(self.root, {}, "a crate")
        self.write("crates/ptr-fastmem/.gitignore", "*\n")
        self.write("crates/ptr-fastmem/build.rs", 'fn main() { println!("cargo:rustc-cfg=evil"); }\n')
        self.assertEqual(git(self.root, "status", "--porcelain", "--untracked-files=all"), "")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), ["crates/ptr-fastmem/.gitignore"])


class ProvenanceWatchTests(GitTree, unittest.TestCase):
    """`ProvenanceWatch` tells whether a run's tree stayed the commit its
    record names from before the run until the record is written."""

    def setUp(self):
        super().setUp()
        # A source last written long ago, so rewriting it with the same text
        # changes its stamp whatever the clock's resolution.
        os.utime(self.root / "src/lib.rs", ns=(10**18, 10**18))
        self.watch = mod.ProvenanceWatch(self.root, self.pathspecs)

    def test_a_tree_nothing_touched_has_no_changes(self):
        self.assertEqual(self.watch.head, git(self.root, "rev-parse", "HEAD"))
        self.assertEqual(self.watch.uncommitted, [])
        self.write("experiments/L900-x/results/run-1-seed-1.json", "{}\n")
        self.write("README.md")
        self.assertEqual(self.watch.changes(), [])

    def test_an_edit_undone_before_the_second_look_is_a_change(self):
        # The run may have compiled the edit though HEAD holds the tree again.
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("src/lib.rs", "pub fn f() {}\n")
        self.assertEqual(mod.uncommitted_files(self.root, self.pathspecs), [])
        self.assertEqual(self.watch.changes(), ["src/lib.rs changed on disk"])

    def test_an_added_or_removed_source_is_a_change(self):
        self.write("crates/new/src/lib.rs")
        (self.root / "Cargo.toml").unlink()
        self.assertEqual(
            self.watch.changes(),
            [
                "Cargo.toml, crates/new/src/lib.rs changed on disk",
                "HEAD does not hold Cargo.toml, crates/new/src/lib.rs",
            ],
        )

    def test_a_source_this_clones_own_ignore_rules_hide_is_a_change_when_added(self):
        (self.root / ".git/info/exclude").write_text("crates/scratch/\n", encoding="utf-8")
        self.write("crates/scratch/src/lib.rs")
        self.assertEqual(
            self.watch.changes(),
            [
                "crates/scratch/src/lib.rs changed on disk",
                "HEAD does not hold crates/scratch/src/lib.rs",
            ],
        )

    def test_a_crate_git_does_not_look_into_is_a_change_when_added(self):
        self.crate("crates/ptr-a")
        commit(self.root, {}, "a crate")
        watch = mod.ProvenanceWatch(self.root, self.pathspecs)
        self.assertEqual(watch.uncommitted, [])
        git(self.crate("crates/ptr-nested"), "init", "-q")
        self.assertEqual(watch.changes(), ["HEAD does not hold crates/ptr-nested"])

    def test_an_ignore_rule_that_hides_no_source_is_no_change_when_written(self):
        # An IDE opening the project while the run builds writes .idea/.gitignore.
        self.write(".idea/.gitignore", "/shelf/\n/workspace.xml\n")
        self.assertEqual(self.watch.changes(), [])
        # One that hides a source the run may have compiled is.
        self.write("src/extra/.gitignore", "*\n")
        self.write("src/extra/mod.rs")
        self.assertEqual(self.watch.changes(), ["HEAD does not hold src/extra/.gitignore"])

    def test_head_moving_is_a_change_though_the_tree_is_clean(self):
        before = self.watch.head
        git(self.root, "commit", "-q", "--no-verify", "--allow-empty", "-m", "moved")
        after = git(self.root, "rev-parse", "HEAD")
        self.assertEqual(self.watch.changes(), [f"HEAD moved from {before} to {after}"])

    def test_the_runs_own_rewrites_are_no_change_but_a_write_between_them_is(self):
        with self.watch.rewriting(["src/lib.rs", "README.md"]):
            self.write("src/lib.rs", "pub fn f() { g() }\n")
            self.write("README.md")
        with self.watch.rewriting(["src/lib.rs"]):
            self.write("src/lib.rs", "pub fn f() {}\n")
        self.assertEqual(self.watch.changes(), [])

        with self.watch.rewriting(["src/lib.rs"]):
            self.write("src/lib.rs", "pub fn f() { g() }\n")
        # Something else writes the planted file; the run's restore then
        # erases the edit its build may have compiled.
        self.write("src/lib.rs", "pub fn f() { h(); g() }\n")
        with self.watch.rewriting(["src/lib.rs"]):
            self.write("src/lib.rs", "pub fn f() {}\n")
        self.assertEqual(self.watch.changes(), ["src/lib.rs changed on disk"])

    def test_a_tree_git_cannot_name_a_commit_of_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(mod.ProvenanceError, "cannot name the commit HEAD is at"):
                mod.ProvenanceWatch(Path(directory), self.pathspecs)


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

    def test_results_are_stale_once_a_directory_symlink_changes(self):
        # A link's own path matches no pathspec, but what Cargo builds changed.
        code = commit(
            self.root, {"crates/ptr-a/Cargo.toml": "[package]\n", "crates/ptr-a/src/lib.rs": "x\n"}, "crate"
        )
        commit(
            self.root,
            {
                "experiments/L900-x/results/run.json": json.dumps({"git_sha": code}),
                "experiments/L900-x/results/mutations.json": json.dumps({"git_sha": code}),
            },
            "archive",
        )
        self.assertEqual(self.errors(), [])
        try:
            os.symlink("ptr-a", self.root / "crates/ptr-link")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")
        linked = commit(self.root, {}, "link a crate")
        commit(self.root, {"README.md": "notes\n"}, "docs")
        errors = self.errors()
        self.assertEqual(len(errors), 2, errors)
        for error in errors:
            self.assertIn("changed since, in crates/ptr-link", error)
        # The link is the first change after the results, and a marker says so.
        self.mark(results_git_sha=code, stale_since=linked)
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


class AggregateBindingTests(unittest.TestCase):
    """`publish_aggregate` writes run.json and metrics.json as one aggregate,
    and `aggregate_errors` (run by `scripts/check_research_gates.py`) refuses a
    run.json beside metrics or mutation evidence it was not aggregated with."""

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
                "experiments/L900-x/results/.gitkeep": "",
            },
            "code",
        )
        self.summary = {"killed": 2, "total": 2, "git_sha": self.code}
        self.metrics = {"experiment_id": "L900", "totals": {"cases": 30}, "mutation_checks": None}
        self.run = {"experiment_id": "L900", "git_sha": self.code, "mutation_checks": None}

    def tearDown(self):
        self.directory.cleanup()

    def errors(self) -> list[str]:
        return mod.aggregate_errors("L900", self.experiment, self.results, self.root)

    def write(self, name: str, value) -> None:
        (self.results / name).write_text(json.dumps(value) + "\n", encoding="utf-8")

    def evidence(self, killed: int = 2) -> dict:
        outcomes = [
            {"name": name, "result": "killed" if index < killed else "survived"} for index, name in enumerate("ab")
        ]
        return {"experiment_id": "L900", "git_sha": self.code, "killed": killed, "total": 2, "mutations": outcomes}

    def test_a_published_aggregate_binds_its_metrics_and_then_clears_the_stale_marker(self):
        (self.results / "STALE.toml").write_text('reason = "old"\n', encoding="utf-8")
        mod.publish_aggregate(self.results, self.metrics, self.run)
        metrics = (self.results / "metrics.json").read_bytes()
        run = json.loads((self.results / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(metrics), self.metrics)
        self.assertEqual(run, {**self.run, "metrics_sha256": hashlib.sha256(metrics).hexdigest()})
        # Nothing but the pair: no marker and no temporary file.
        self.assertEqual(sorted(files(self.results)), [".gitkeep", "metrics.json", "run.json"])
        self.assertEqual(self.errors(), [])

    def test_run_json_beside_other_metrics_is_refused(self):
        mod.publish_aggregate(self.results, self.metrics, self.run)
        # An aggregation interrupted between its two files, or a hand edit.
        self.write("metrics.json", {**self.metrics, "totals": {"cases": 31}})
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(
            "L900: experiments/L900-x/results/metrics.json is not the metrics run.json was aggregated with",
            errors[0],
        )
        (self.results / "metrics.json").unlink()
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("metrics.json, which run.json binds, cannot be read", errors[0])

    def test_a_current_run_json_that_binds_no_metrics_is_refused_until_it_is_stale(self):
        # Only aggregates written before run.json bound its metrics lack the
        # binding, and those ran at code HEAD has changed since.
        self.write("metrics.json", self.metrics)
        self.write("run.json", self.run)
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("run.json names no metrics_sha256", errors[0])
        commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "change")
        self.assertEqual(self.errors(), [])

    def test_an_unbound_stale_run_json_passes_only_beside_the_metrics_committed_with_it(self):
        # A run.json aggregated before run.json bound its metrics was archived
        # with them. New metrics beside it, whether left by an interrupted
        # aggregation, restored around or committed on their own, are the
        # mixed pair the binding refuses.
        self.write("metrics.json", self.metrics)
        self.write("run.json", self.run)
        commit(self.root, {}, "archive")
        commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "change")
        self.assertEqual(self.errors(), [])
        archived = (self.results / "metrics.json").read_text(encoding="utf-8")
        self.write("metrics.json", {**self.metrics, "totals": {"cases": 999}})
        for state in ("uncommitted", "committed on their own"):
            with self.subTest(metrics=state):
                if state != "uncommitted":
                    commit(self.root, {}, "new metrics")
                errors = self.errors()
                self.assertEqual(len(errors), 1, errors)
                self.assertIn(
                    "L900: experiments/L900-x/results/metrics.json is not the metrics.json committed with "
                    "run.json, which names no metrics_sha256",
                    errors[0],
                )
        (self.results / "metrics.json").write_text(archived, encoding="utf-8")
        self.assertEqual(self.errors(), [])
        # A run.json no commit holds binds nothing either.
        self.write("run.json", {**self.run, "verdict": "hard-pass"})
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("results/run.json names no metrics_sha256 and is not the run.json committed", errors[0])

    def test_mutation_evidence_run_json_does_not_carry_is_refused(self):
        self.write("mutations.json", self.evidence())
        carried = {"mutation_checks": self.summary}
        mod.publish_aggregate(self.results, {**self.metrics, **carried}, {**self.run, **carried})
        self.assertEqual(self.errors(), [])
        # A mutation check rerun after the aggregate, with another outcome.
        self.write("mutations.json", self.evidence(killed=1))
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("results/mutations.json is not the evidence run.json carries", errors[0])
        (self.results / "mutations.json").unlink()
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("mutations.json, whose checks run.json carries, cannot be read", errors[0])
        # An aggregate of no mutation evidence beside evidence that came later.
        mod.publish_aggregate(self.results, self.metrics, self.run)
        self.write("mutations.json", self.evidence())
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("run.json carries no mutation checks, but mutations.json is there", errors[0])

    def test_a_publish_that_finds_other_evidence_beside_it_keeps_the_stale_marker(self):
        (self.results / "STALE.toml").write_text('reason = "old"\n', encoding="utf-8")
        self.write("mutations.json", self.evidence())
        with self.assertRaisesRegex(mod.ProvenanceError, "run.json carries no mutation checks"):
            mod.publish_aggregate(self.results, self.metrics, self.run)
        self.assertTrue((self.results / "STALE.toml").exists())


class BindingAggregatorTests(unittest.TestCase):
    """The gate asks a current run.json to name the SHA-256 of its metrics and
    tells a completed experiment to rerun its aggregate.py to get one, so
    every aggregate.py must publish through `publish_aggregate`."""

    def test_every_aggregate_py_publishes_its_run_json_bound_to_its_metrics(self):
        aggregators = sorted((ROOT / "experiments").glob("*/*/aggregate.py"))
        self.assertTrue(aggregators)
        for path in aggregators:
            with self.subTest(aggregator=path.relative_to(ROOT).as_posix()):
                self.assertIn("experiment_records.publish_aggregate(", path.read_text(encoding="utf-8"))

    def test_the_l001_aggregate_is_one_the_gate_accepts_as_current(self):
        here = ROOT / "experiments/lifecycle/L001-revocation-crash"
        spec = importlib.util.spec_from_file_location("aggregate_L001", here / "aggregate.py")
        agg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiments/lifecycle/L001-x"
            git(root, "init", "-q")
            commit(
                root,
                {
                    "Cargo.toml": "[workspace]\n",
                    "src/lib.rs": "pub fn f() {}\n",
                    "experiments/lifecycle/L001-x/aggregate.py": "# aggregator\n",
                    "experiments/lifecycle/L001-x/results/.gitkeep": "",
                },
                "code",
            )
            results = experiment / "results"
            seed = results / "seed-17.json"
            seed.write_text(
                json.dumps({"iterations": 10, "false_accepts": 0, "recovery_errors": 0, "tail_trim_errors": 0}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(agg, "ROOT", root),
                mock.patch.object(agg, "RESULTS", results),
                mock.patch.object(sys, "argv", ["aggregate.py", str(seed)]),
            ):
                agg.main()
            run = json.loads((results / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["git_sha"], git(root, "rev-parse", "HEAD"))
            self.assertEqual(
                run["metrics_sha256"], hashlib.sha256((results / "metrics.json").read_bytes()).hexdigest()
            )
            self.assertEqual(mod.aggregate_errors("L001", experiment, results, root), [])


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
        uncommitted=(),
        results=None,
        previous=None,
        during=None,
        refused_untouched=True,
    ):
        """Run `exp_id`'s aggregator over synthetic records, one per declared
        seed at commits "1111111" and "2222222" with the same code, into a
        scratch results directory. Every seed's harness result passes and
        reaches every probe. `change` edits the last record and
        `result_change` its harness result; `code_changes` are the code files
        the checkout changed since, and `uncommitted` the provenance files
        HEAD does not hold; `mutations` is written as results/mutations.json,
        and `stale_code` maps a commit to the files whose code differs at it
        from the checkout; `marker` plants results/STALE.toml and `previous`
        (file name -> text) the files an earlier aggregation left. `results`
        is the results directory to use, a fresh one when None, and `during`
        a context the aggregator runs in. Returns the
        run.json it wrote, or raises what the aggregator raised; a SystemExit
        must have left run.json, metrics.json and the marker as they were,
        unless `refused_untouched` is false."""
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
        with contextlib.ExitStack() as stack:
            if results is None:
                results = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            for name, text in (previous or {}).items():
                (results / name).write_text(text, encoding="utf-8")
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
            aggregate = ("run.json", "metrics.json", "STALE.toml")
            before = {name: (previous or {}).get(name) for name in aggregate}
            if marker:
                before["STALE.toml"] = 'stale_since = "1111111"\n'
            with (
                mock.patch.object(agg, "RESULTS", results),
                mock.patch.object(sys, "argv", ["aggregate.py", *inputs]),
                mock.patch.object(mod, "code_changes", side_effect=changes),
                mock.patch.object(mod, "uncommitted_files", return_value=list(uncommitted)) as listed,
                during if during is not None else contextlib.nullcontext(),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    agg.main()
                except SystemExit:
                    if refused_untouched:
                        self.assertEqual(
                            {
                                name: (results / name).read_text(encoding="utf-8")
                                if (results / name).exists()
                                else None
                                for name in aggregate
                            },
                            before,
                        )
                    raise
            self.assertFalse((results / "STALE.toml").exists())
            run = json.loads((results / "run.json").read_text(encoding="utf-8"))
            run["_code_change_calls"] = calls
            run["_uncommitted_calls"] = [tuple(call.args[1]) for call in listed.call_args_list]
            run["_metrics_sha256"] = hashlib.sha256((results / "metrics.json").read_bytes()).hexdigest()
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

    def test_a_checkout_holding_provenance_files_head_does_not_is_not_aggregated(self):
        # git diff, which compares the records' commit with the checkout,
        # omits untracked files such as a new crates/ptr-* member.
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id)
                relative = AGGREGATORS[exp_id].relative_to(ROOT).as_posix()
                self.assertTrue(run["_uncommitted_calls"])
                for paths in run["_uncommitted_calls"]:
                    self.assertIn("*.rs", paths)
                    self.assertIn(f"{relative}/aggregate.py", paths)
                with self.assertRaisesRegex(
                    SystemExit, "refusing to aggregate: .*HEAD does not hold crates/ptr-new/src/lib.rs"
                ):
                    self.aggregate(exp_id, uncommitted=["crates/ptr-new/src/lib.rs"], marker=True)

    def test_the_aggregate_binds_the_metrics_written_with_it(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id)
                self.assertEqual(run["metrics_sha256"], run["_metrics_sha256"])

    PREVIOUS = {"metrics.json": '{"previous": "metrics"}\n', "run.json": '{"previous": "run"}\n'}
    MARKER = 'stale_since = "1111111"\n'

    def test_a_write_that_fails_leaves_the_previous_aggregate_and_its_marker(self):
        # A full disk while run.json was written used to leave the new
        # metrics.json beside the previous run.json, or an empty one.
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id), tempfile.TemporaryDirectory() as directory:
                results = Path(directory)
                full = disk_full_writing("run.json")
                with self.assertRaises(OSError):
                    self.aggregate(exp_id, results=results, previous=self.PREVIOUS, marker=True, during=full)
                self.assertEqual(
                    {name: text for name, text in files(results).items() if not name.startswith("run-")},
                    {**self.PREVIOUS, "STALE.toml": self.MARKER},
                )

    def test_an_interrupted_publish_leaves_no_run_json_beside_other_metrics(self):
        # Interrupted between moving the two files into place, the aggregate
        # is left without run.json, which check_research_gates.py fails as a
        # missing artifact, never with the previous run.json.
        real_replace = os.replace

        def replace(source, target):
            if Path(target).name == "run.json":
                raise KeyboardInterrupt
            return real_replace(source, target)

        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id), tempfile.TemporaryDirectory() as directory:
                results = Path(directory)
                interrupted = mock.patch.object(mod.os, "replace", side_effect=replace)
                with self.assertRaises(KeyboardInterrupt):
                    self.aggregate(
                        exp_id, results=results, previous=self.PREVIOUS, marker=True, during=interrupted
                    )
                left = {name: text for name, text in files(results).items() if not name.startswith("run-")}
                self.assertEqual(sorted(left), ["STALE.toml", "metrics.json"])
                self.assertEqual(json.loads(left["metrics.json"])["experiment_id"], exp_id)

    def test_the_marker_is_cleared_only_once_the_published_files_are_one_aggregate(self):
        # Another writer replaces metrics.json as this aggregate is published.
        real_replace = os.replace

        def replace(source, target):
            real_replace(source, target)
            if Path(target).name == "run.json":
                (Path(target).parent / "metrics.json").write_text('{"other": "metrics"}\n', encoding="utf-8")

        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id), tempfile.TemporaryDirectory() as directory:
                results = Path(directory)
                raced = mock.patch.object(mod.os, "replace", side_effect=replace)
                with self.assertRaisesRegex(SystemExit, f"{exp_id}: .*changed while they were published"):
                    self.aggregate(exp_id, results=results, marker=True, during=raced, refused_untouched=False)
                self.assertEqual((results / "STALE.toml").read_text(encoding="utf-8"), self.MARKER)


class PreregistrationTests(unittest.TestCase):
    TABLE = tomllib.loads(
        "schema = 1\n"
        'harness = "fixture"\n'
        "seeds = [17, 29]\n"
        'programs = ["rmw", "set_op"]\n'
        "see_intent = false\n"
        "low_cells = []\n"
        'note = "say \\"hi\\" \\\\ bye ~"\n'
    )

    def test_the_preregistration_canonical_text_is_sorted_compact_json_and_its_digest_is_sha256(self):
        text = (
            '{"harness":"fixture","low_cells":[],"note":"say \\"hi\\" \\\\ bye ~","programs":["rmw","set_op"],'
            '"schema":1,"see_intent":false,"seeds":[17,29]}'
        )
        self.assertEqual(mod.preregistration_canonical(self.TABLE), text)
        self.assertEqual(mod.preregistration_digest(self.TABLE), hashlib.sha256(text.encode("utf-8")).hexdigest())
        # The order the table was written in does not move the text.
        reordered = dict(reversed(list(self.TABLE.items())))
        self.assertEqual(mod.preregistration_canonical(reordered), text)
        # Every value moves the digest, a list's order included.
        for key, value in (("schema", 2), ("seeds", [29, 17]), ("see_intent", True), ("low_cells", ["L0N2"])):
            with self.subTest(key=key):
                changed = {**self.TABLE, key: value}
                self.assertNotEqual(mod.preregistration_digest(changed), mod.preregistration_digest(self.TABLE))

    def test_a_value_without_a_canonical_text_is_refused(self):
        refused = tomllib.loads(
            "rate = 0.1\n"
            "day = 2026-09-28\n"
            "nested = { a = 1 }\n"
            'mixed = [1, "a"]\n'
            "flags = [true, false]\n"
            "lists = [[1], [2]]\n"
            'accented = "Grüße"\n'
            'tab = "a\\tb"\n'
            'newline = "a\\nb"\n'
            'delete = "a\\u007fb"\n'
            'names = ["ok", "Grüße"]\n'
        )
        for key, value in refused.items():
            with self.subTest(key=key):
                self.assertIsNotNone(mod.canonical_value_problem(value))
                with self.assertRaisesRegex(ValueError, f"preregistration key {key} "):
                    mod.preregistration_canonical({**self.TABLE, key: value})
                with self.assertRaisesRegex(ValueError, f"preregistration key {key} "):
                    mod.preregistration_digest({**self.TABLE, key: value})
        for value in self.TABLE.values():
            self.assertIsNone(mod.canonical_value_problem(value))
        # A key outside printable ASCII is refused as well.
        for key in ("Größe", "a\tb"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "outside printable ASCII"):
                    mod.preregistration_canonical({**self.TABLE, key: 1})
        # Every printable ASCII character is accepted, and only the quote and
        # the backslash are escaped.
        printable = "".join(chr(code) for code in range(0x20, 0x7F))
        self.assertIsNone(mod.canonical_value_problem(printable))
        self.assertEqual(
            mod.preregistration_canonical({"all": printable}),
            '{"all":"' + printable.replace("\\", "\\\\").replace('"', '\\"') + '"}',
        )
        # Integers are held to what RFC 8785 writes exactly.
        bound = 2**53 - 1
        for value in (bound, -bound, 0, [bound, -bound]):
            self.assertIsNone(mod.canonical_value_problem(value))
        for value, problem in (
            (bound + 1, "is an integer beyond"),
            (-bound - 1, "is an integer beyond"),
            ([1, bound + 1], "has element 1 that is an integer beyond"),
        ):
            with self.subTest(value=value):
                self.assertTrue(mod.canonical_value_problem(value).startswith(problem))
                with self.assertRaisesRegex(ValueError, f"preregistration key big {problem}"):
                    mod.preregistration_canonical({"big": value})

    def test_the_canonical_text_is_spelled_exactly_and_is_rfc_8785_on_its_domain(self):
        # Every rule of the spelling, written out: keys in byte order ("B" <
        # "_" < "a"), no whitespace, only the quote and the backslash escaped
        # ("/" is not), booleans and integers as JSON writes them.
        table = {"a": 'p/q "r" \\ s', "_": [-1, 0, 9007199254740991], "B": ["x/y", ""], "b": True, "c": False}
        text = '{"B":["x/y",""],"_":[-1,0,9007199254740991],"a":"p/q \\"r\\" \\\\ s","b":true,"c":false}'
        self.assertEqual(mod.canonical_text(table), text)
        self.assertEqual(mod.preregistration_canonical(table), text)
        self.assertEqual(mod.canonical_digest(table), hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(mod.preregistration_digest(table), mod.canonical_digest(table))
        # It is what Python's json module writes with sorted keys and no
        # whitespace, for every table the domain allows.
        for sample in (table, self.TABLE, {"all": "".join(chr(code) for code in range(0x20, 0x7F))}, {}):
            with self.subTest(sample=sample):
                self.assertEqual(mod.canonical_text(sample), json.dumps(sample, sort_keys=True, separators=(",", ":")))
        # Nothing outside the domain has a text.
        with self.assertRaises(ValueError):
            mod.canonical_text(0.5)

    def test_a_preregistered_file_is_digested_with_crlf_read_as_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            unix = Path(directory) / "unix.md"
            windows = Path(directory) / "windows.md"
            unix.write_bytes(b"# Protocol\nline\n")
            windows.write_bytes(b"# Protocol\r\nline\r\n")
            expected = hashlib.sha256(b"# Protocol\nline\n").hexdigest()
            self.assertEqual(mod.preregistered_file_digest(unix), expected)
            self.assertEqual(mod.preregistered_file_digest(windows), expected)
            # A lone CR is content, not a line ending.
            windows.write_bytes(b"# Protocol\rline\n")
            self.assertNotEqual(mod.preregistered_file_digest(windows), expected)
            # A file that is not text is digested byte for byte: its carriage
            # returns are content, which a converting checkout leaves alone.
            for binary in (b"\x00\x01\r\n\x02", b"\xff\xfe\r\n"):
                with self.subTest(binary=binary):
                    self.assertEqual(mod.preregistered_bytes_digest(binary), hashlib.sha256(binary).hexdigest())
                    self.assertNotEqual(
                        mod.preregistered_bytes_digest(binary), mod.preregistered_bytes_digest(binary.replace(b"\r\n", b"\n"))
                    )

    def test_git_reads_the_repository_it_is_given_whatever_the_environment_says(self):
        with tempfile.TemporaryDirectory() as directory:
            ours, other = Path(directory) / "ours", Path(directory) / "other"
            heads = {}
            for root in (ours, other):
                root.mkdir()
                (root / "file.txt").write_text(root.name, encoding="utf-8")
                for args in (("init", "-q"), ("add", "-A"),
                             ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", root.name)):
                    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
                heads[root.name] = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
                ).stdout.strip()
            # Variables that would point git at the other repository, or make
            # it read pathspecs as globs, are not passed on.
            with mock.patch.dict(os.environ, {
                "GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other), "GIT_GLOB_PATHSPECS": "1",
            }):
                self.assertEqual(mod.git(ours, "rev-parse", "HEAD").stdout.strip(), heads["ours"])
                self.assertEqual(mod.git(ours, "ls-files", "--", "file.txt").stdout.strip(), "file.txt")


if __name__ == "__main__":
    unittest.main()
