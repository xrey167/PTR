import contextlib
import errno
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "mutation_check", ROOT / "scripts/mutation_check.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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


def experiments_with_mutations():
    for item in mod.load_toml(mod.REGISTRY).get("experiment", []):
        root = ROOT / "experiments" / item["path"]
        if (root / "tests/mutations.toml").exists():
            yield item["id"], root


class MutationCheckTests(unittest.TestCase):
    def test_every_listed_mutation_still_matches_its_source_once(self):
        found = list(experiments_with_mutations())
        self.assertTrue(found, "no experiment lists mutations")
        for exp_id, root in found:
            with self.subTest(experiment=exp_id):
                plan = mod.load_plan(root)
                self.assertTrue(plan.get("mutation"), f"{exp_id} lists no mutation")
                self.assertEqual(mod.anchor_errors(plan), [])

    def test_a_drifted_or_ambiguous_anchor_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.rs").write_text("let a = 1;\nlet b = 2;\nlet b = 2;\n", encoding="utf-8")
            plan = {
                "mutation": [
                    {"name": "fine", "file": "code.rs", "find": "let a = 1;\n", "replace": "let a = 2;\n"},
                    {"name": "twice", "file": "code.rs", "find": "let b = 2;\n", "replace": "let b = 3;\n"},
                    {"name": "gone", "file": "code.rs", "find": "let c = 3;\n", "replace": "let c = 4;\n"},
                    {"name": "same", "file": "code.rs", "find": "let a = 1;\n", "replace": "let a = 1;\n"},
                    {"name": "missing", "file": "none.rs", "find": "x", "replace": "y"},
                    {
                        "name": "second-edit-drifted",
                        "file": "code.rs",
                        "find": "let a = 1;\n",
                        "replace": "let a = 3;\n",
                        "also": [{"find": "let d = 4;\n", "replace": "let d = 5;\n"}],
                    },
                ]
            }
            errors = mod.anchor_errors(plan, root)
        self.assertEqual(len(errors), 5)
        self.assertTrue(any(error.startswith("second-edit-drifted: the anchor occurs 0") for error in errors))
        self.assertTrue(any(error.startswith("twice: the anchor occurs 2 times") for error in errors))
        self.assertTrue(any(error.startswith("gone: the anchor occurs 0 times") for error in errors))
        self.assertTrue(any(error.startswith("same: the replacement equals") for error in errors))
        self.assertTrue(any(error.startswith("missing:") for error in errors))

    def test_the_harness_is_built_locked_and_run_with_cases_and_seed(self):
        plan = {"package": "ptr-bench", "features": "postgres-experiments", "subcommand": "fastmem-revocation"}
        build, run = mod.binary_command(plan, 4, 17)
        self.assertEqual(
            build,
            ["cargo", "build", "--release", "--locked", "-p", "ptr-bench", "--features", "postgres-experiments"],
        )
        self.assertEqual(run[1:], ["fastmem-revocation", "4", "17"])
        self.assertTrue(run[0].endswith("target/release/ptr-bench"))

    def test_every_mutation_names_a_hard_counter_it_expects(self):
        for exp_id, root in experiments_with_mutations():
            with self.subTest(experiment=exp_id):
                plan = mod.load_plan(root)
                self.assertEqual(mod.expectation_errors(plan), [])
        plan = {
            "hard_counters": ["a", "b"],
            "mutation": [
                {"name": "none"},
                {"name": "empty", "expect": []},
                {"name": "unknown", "expect": ["a", "c"]},
                {"name": "fine", "expect": ["b"]},
            ],
        }
        errors = mod.expectation_errors(plan)
        self.assertEqual(len(errors), 3)
        self.assertTrue(any(error.startswith("none: no expected") for error in errors))
        self.assertTrue(any(error.startswith("empty: no expected") for error in errors))
        self.assertTrue(any(error.startswith("unknown: 'c' is not") for error in errors))

    def test_a_mutation_is_killed_only_by_a_counter_it_expects(self):
        plan = {"hard_counters": ["lost", "read_failures"]}
        mutation = {"expect": ["lost"]}
        self.assertEqual(
            mod.classify(1, {"hard_failures": 3, "lost": 3}, plan, mutation),
            ("killed", {"lost": 3}),
        )
        # A defect that broke a query first shows nothing about the defect.
        self.assertEqual(
            mod.classify(1, {"hard_failures": 2, "read_failures": 2}, plan, mutation),
            ("failed-elsewhere", {"read_failures": 2}),
        )
        self.assertEqual(mod.classify(0, {"hard_failures": 0}, plan, mutation)[0], "survived")
        self.assertEqual(mod.classify(101, {}, plan, mutation)[0], "crashed")
        self.assertEqual(mod.classify(1, {}, plan, mutation)[0], "crashed")

    def test_the_last_json_line_is_the_result(self):
        stdout = 'Compiling\n{"a":1}\nnoise\n{"hard_failures":2}\n'
        self.assertEqual(mod.last_json_line(stdout), {"hard_failures": 2})
        self.assertEqual(mod.last_json_line("nothing"), {})

    def test_only_must_name_mutations_the_plan_lists(self):
        plan = {"mutation": [{"name": "a"}, {"name": "b"}]}
        self.assertEqual(mod.select(plan, []), ([{"name": "a"}, {"name": "b"}], []))
        self.assertEqual(mod.select(plan, ["b"]), ([{"name": "b"}], []))
        selected, errors = mod.select(plan, ["b", "typo"])
        self.assertEqual(selected, [{"name": "b"}])
        self.assertEqual(errors, ["no mutation named 'typo'"])
        selected, errors = mod.select(plan, ["typo"])
        self.assertEqual(selected, [])
        self.assertEqual(errors, ["no mutation named 'typo'", "no mutation selected"])
        self.assertEqual(mod.select({}, []), ([], ["no mutation selected"]))

    def run_main(self, only, rebuild_exit, dirty=None):
        """`main` over a one-mutation plan whose mutation is killed, with the
        final rebuild exiting `rebuild_exit`; returns its exit status and the
        mutations it ran. `--only` keeps it from writing a record; without it
        (`only` None) the working tree reports the uncommitted files `dirty`."""
        plan = {
            "package": "ptr-bench",
            "features": "postgres-experiments",
            "subcommand": "fastmem-revocation",
            "cases": 1,
            "seed": 17,
            "hard_counters": ["lost"],
            "mutation": [{"name": "drop-row-lock", "expect": ["lost"]}],
        }
        ran = []
        real_run = subprocess.run

        def run_mutation(_plan, mutation, _timeout, _watch=None):
            ran.append(mutation["name"])
            return {"name": mutation["name"], "result": "killed", "counters": {"lost": 1}}

        def rebuild(command, **kwargs):
            if command[0] == "git":
                return real_run(command, **kwargs)
            return subprocess.CompletedProcess(command, rebuild_exit, "", "error: disk full")

        argv = ["mutation_check.py", "L003"] + ([] if only is None else ["--only", *only])
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                mod, "experiment_root", return_value=ROOT / "experiments/lifecycle/L003-fastmem-revocation"
            ),
            mock.patch.object(mod.experiment_records, "uncommitted_files", return_value=dirty) as listed,
            mock.patch.object(mod, "load_plan", return_value=plan),
            mock.patch.object(mod, "anchor_errors", return_value=[]),
            mock.patch.object(mod, "run_mutation", side_effect=run_mutation),
            mock.patch.object(mod.subprocess, "run", side_effect=rebuild),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = mod.main()
        # A run that records nothing has no need of a clean tree.
        self.assertEqual(listed.called, only is None)
        return status, ran

    def test_an_unknown_only_name_fails_without_running_anything(self):
        self.assertEqual(self.run_main(["drop-row-lock"], 0), (0, ["drop-row-lock"]))
        self.assertEqual(self.run_main(["drop-row-lok"], 0), (2, []))

    def test_a_failed_rebuild_of_the_unmutated_harness_fails_the_run(self):
        # Every mutation was killed, but the binary may still hold the last one.
        self.assertEqual(self.run_main(["drop-row-lock"], 101), (1, ["drop-row-lock"]))

    def test_a_recorded_run_from_a_dirty_source_tree_is_refused_before_it_mutates(self):
        # mutations.json names HEAD as the code it mutated.
        self.assertEqual(self.run_main(None, 0, dirty=["crates/ptr-pg/src/adapters/fastmem.rs"]), (2, []))

    def test_the_recorded_run_checks_the_mutation_provenance_of_its_experiment(self):
        experiment = "experiments/lifecycle/L003-fastmem-revocation"
        specs = mod.experiment_records.tree_pathspecs(
            ROOT / experiment,
            ROOT / experiment / "results",
            ROOT,
            mod.experiment_records.mutation_record_paths(ROOT / experiment, ROOT),
        )
        for spec in ("*.rs", "scripts/mutation_check.py", f"{experiment}/tests/mutations.toml", experiment):
            self.assertIn(spec, specs)
        self.assertIn(f":(exclude){experiment}/results", specs)


def files(directory: Path) -> dict[str, str]:
    """Every file in `directory` by name, with its text."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(directory.iterdir())}


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


class RecordedRunWatchTests(unittest.TestCase):
    """`mutations.json` names the commit the mutation run started from, so the
    run's sources must stay that commit's, apart from the defects the checker
    plants and removes itself, until the record is written."""

    PLANTED = "pub fn f() -> u8 { 1 }\n"
    OTHER = "pub fn g() {}\n"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.experiment = self.root / "experiments/x/L900-x"
        files = {
            ".gitignore": "__pycache__/\n",
            "Cargo.lock": "# lock\n",
            "README.md": "readme\n",
            "src/lib.rs": self.PLANTED,
            "src/other.rs": self.OTHER,
            "experiments/registry.toml": '[[experiment]]\nid = "L900"\npath = "x/L900-x"\nstatus = "running"\n',
            "experiments/x/L900-x/experiment.toml": 'id = "L900"\nstatus = "running"\n',
            "experiments/x/L900-x/results/.gitkeep": "",
            "experiments/x/L900-x/tests/mutations.toml": (
                'package = "ptr-bench"\nfeatures = "postgres-experiments"\nsubcommand = "bench"\n'
                'cases = 1\nseed = 17\nhard_counters = ["wrong"]\n\n'
                '[[mutation]]\nname = "return-two"\nfile = "src/lib.rs"\n'
                'find = "{ 1 }"\nreplace = "{ 2 }"\nexpect = ["wrong"]\n'
            ),
        }
        for relative, text in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "code")
        self.head = git(self.root, "rev-parse", "HEAD")
        # Sources last written long ago, so rewriting one with the same text
        # during a run changes its stamp whatever the clock's resolution.
        for relative in ("src/lib.rs", "src/other.rs"):
            os.utime(self.root / relative, ns=(10**18, 10**18))
        self.record = self.experiment / "results/mutations.json"

    def tearDown(self):
        self.directory.cleanup()

    def write(self, relative: str, text: str) -> None:
        (self.root / relative).write_text(text, encoding="utf-8")

    def run_check(self, during_build=None, rebuild_exit=0):
        """`main` over the one-mutation plan in the temporary repository, with
        cargo and the harness faked (the mutated harness is killed),
        `during_build` called while the mutated harness builds and the final
        rebuild of the unmutated harness exiting `rebuild_exit`; returns the
        exit status and what `main` printed."""
        real_run = subprocess.run
        builds = []

        def run(command, **kwargs):
            if command[0] == "git":
                return real_run(command, **kwargs)
            if command[0] == "cargo":
                builds.append((self.root / "src/lib.rs").read_text(encoding="utf-8"))
                if len(builds) == 1 and during_build is not None:
                    during_build()
                if len(builds) == 2:
                    return subprocess.CompletedProcess(command, rebuild_exit, "", "error: could not compile")
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 1, json.dumps({"hard_failures": 1, "wrong": 1}) + "\n", "")

        stdout = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["mutation_check.py", "L900"]),
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "anchor_errors", return_value=[]),
            mock.patch.object(mod.subprocess, "run", side_effect=run),
            contextlib.redirect_stdout(stdout),
        ):
            status = mod.main()
        # The mutated harness was built from the defect, and the checker put
        # the source back.
        self.assertEqual(builds[0], "pub fn f() -> u8 { 2 }\n")
        self.assertEqual((self.root / "src/lib.rs").read_text(encoding="utf-8"), self.PLANTED)
        return status, stdout.getvalue()

    def test_a_mutation_run_whose_tree_stays_at_head_is_recorded_at_the_commit_it_started_from(self):
        status, _ = self.run_check()
        self.assertEqual(status, 0)
        record = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertEqual((record["git_sha"], record["killed"], record["total"]), (self.head, 1, 1))

    PREVIOUS = '{"previous": "record"}\n'

    def test_a_mutation_run_whose_clean_rebuild_fails_writes_no_record(self):
        # Every mutation was killed, but the checker fails the run; nothing
        # downstream records or checks that, so the record must not exist.
        self.record.write_text(self.PREVIOUS, encoding="utf-8")
        status, stdout = self.run_check(rebuild_exit=101)
        self.assertEqual(status, 1)
        self.assertEqual(self.record.read_text(encoding="utf-8"), self.PREVIOUS)
        self.assertIn("rebuilding the unmutated harness failed", stdout)
        self.assertIn("no record is written", stdout)

    def test_a_record_whose_write_fails_leaves_the_previous_record_whole(self):
        self.record.write_text(self.PREVIOUS, encoding="utf-8")
        with disk_full_writing("mutations.json"), self.assertRaises(OSError):
            self.run_check()
        self.assertEqual(files(self.record.parent), {".gitkeep": "", "mutations.json": self.PREVIOUS})

    def test_a_planted_file_whose_write_fails_is_left_whole(self):
        # Planting the defect and removing it again replace the file in one
        # step; a write that failed half way used to leave it truncated.
        with disk_full_writing("lib.rs"), self.assertRaises(OSError):
            self.run_check()
        self.assertEqual(files(self.root / "src"), {"lib.rs": self.PLANTED, "other.rs": self.OTHER})
        self.assertFalse(self.record.exists())

    def test_a_planted_file_is_restored_though_the_disk_fills_after_the_plant(self):
        # The build fills the disk after the defect is planted; putting the
        # original back used to need room for a new copy of it, and failing
        # left the defect in the source tree.
        real_write_temporary = mod.experiment_records.write_temporary
        writes = []

        def fill_after_the_plant(path, data, mode=None):
            if path.name == "lib.rs":
                writes.append(data)
                if len(writes) > 1:
                    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            return real_write_temporary(path, data, mode)

        with mock.patch.object(mod.experiment_records, "write_temporary", side_effect=fill_after_the_plant):
            status, _ = self.run_check()
        self.assertEqual(status, 0)
        self.assertEqual(files(self.root / "src"), {"lib.rs": self.PLANTED, "other.rs": self.OTHER})

    def test_a_planted_file_whose_restore_fails_is_named_with_where_its_original_is(self):
        real_replace = os.replace
        onto = []

        def replace(source, target):
            if Path(target).name == "lib.rs":
                onto.append(source)
                if len(onto) == 2:
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
            return real_replace(source, target)

        plan = mod.load_toml(self.experiment / "tests/mutations.toml")

        def run(command, **_kwargs):
            if command[0] == "cargo":
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 1, json.dumps({"hard_failures": 1, "wrong": 1}) + "\n", "")

        stdout = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod.subprocess, "run", side_effect=run),
            mock.patch.object(mod.os, "replace", side_effect=replace),
            contextlib.redirect_stdout(stdout),
            self.assertRaises(OSError),
        ):
            mod.run_mutation(plan, plan["mutation"][0], 60)
        self.assertEqual((self.root / "src/lib.rs").read_text(encoding="utf-8"), "pub fn f() -> u8 { 2 }\n")
        sources = files(self.root / "src")
        kept = {name: text for name, text in sources.items() if name not in ("lib.rs", "other.rs")}
        self.assertEqual(list(kept.values()), [self.PLANTED])
        self.assertIn("src/lib.rs still holds the mutation return-two", stdout.getvalue())
        self.assertIn(f"src/{next(iter(kept))}", stdout.getvalue())

    def killed_run(self, planted_at=None, linked_at=None):
        """A fake `subprocess.run` for `run_mutation`: cargo succeeds, noting in
        `planted_at` the modification time the planted file had when it was
        built and, given `linked_at` (nanoseconds), linking the harness binary
        with that modification time; the harness fails on the mutation's
        expected counter."""

        def run(command, **_kwargs):
            if command[0] == "cargo":
                if planted_at is not None:
                    planted_at.append((self.root / "src/lib.rs").stat().st_mtime_ns)
                if linked_at is not None:
                    self.binary.parent.mkdir(parents=True, exist_ok=True)
                    self.binary.write_bytes(b"harness")
                    os.utime(self.binary, ns=(linked_at, linked_at))
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 1, json.dumps({"hard_failures": 1, "wrong": 1}) + "\n", "")

        return run

    @property
    def binary(self) -> Path:
        """Where `run_mutation` runs the harness of the plan in `setUp` from."""
        return self.root / "target/release/ptr-bench"

    def run_one(self, run, *patches):
        """`run_mutation` over the plan's one mutation, with `run` as
        `subprocess.run` and any further `patches` in force; returns its
        outcome and what it printed."""
        plan = mod.load_toml(self.experiment / "tests/mutations.toml")
        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(mod, "ROOT", self.root))
            stack.enter_context(mock.patch.object(mod.subprocess, "run", side_effect=run))
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            outcome = mod.run_mutation(plan, plan["mutation"][0], 60)
        return outcome, stdout.getvalue()

    def test_a_restored_source_is_stamped_newer_than_its_mutated_build(self):
        # Cargo rebuilds a crate only when one of its sources is newer than its
        # last build. Putting the original back by a rename brought back its
        # old modification time, so the crate stayed built with the defect: the
        # next mutation in another crate ran with both defects, and the
        # unmutated rebuild left the harness mutated for the next seed run.
        plan = mod.load_toml(self.experiment / "tests/mutations.toml")
        planted_at = []
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod.subprocess, "run", side_effect=self.killed_run(planted_at)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            outcome = mod.run_mutation(plan, plan["mutation"][0], 60)
        self.assertEqual(outcome["result"], "killed")
        restored = (self.root / "src/lib.rs").stat()
        self.assertEqual((self.root / "src/lib.rs").read_text(encoding="utf-8"), self.PLANTED)
        self.assertGreater(restored.st_mtime_ns, planted_at[0], "newer than the build of the defect")
        self.assertGreater(restored.st_mtime_ns, 10**18, "and not the original's old stamp")
        self.assertEqual((self.root / "src/other.rs").stat().st_mtime_ns, 10**18, "an unplanted file is untouched")

    def test_a_restored_source_is_stamped_newer_than_the_harness_its_build_linked(self):
        # Cargo stamps each crate when it builds it and links the harness after
        # all of them, so a source newer than the binary is newer than every
        # stamp of that build. Newer than the planted file alone is not enough:
        # a clock stepped back since the build, as here, left the source older
        # than the stamps and the crate built with the defect.
        linked_at = time.time_ns() + 30 * 10**9
        outcome, _ = self.run_one(self.killed_run(linked_at=linked_at))
        self.assertEqual(outcome["result"], "killed")
        self.assertEqual(self.binary.stat().st_mtime_ns, linked_at)
        self.assertGreater((self.root / "src/lib.rs").stat().st_mtime_ns, linked_at)

    def test_a_restored_source_is_stepped_past_a_filesystem_that_keeps_whole_seconds(self):
        # A nanosecond past the binary is stored as the binary's own second on
        # such a filesystem, which cargo takes as not newer; the checker reads
        # the stored time back and steps on until it is later.
        real_utime = os.utime

        def whole_seconds(path, *args, ns=None, **kwargs):
            if ns is not None:
                ns = tuple(value - value % 10**9 for value in ns)
            return real_utime(path, *args, ns=ns, **kwargs)

        linked_at = (time.time_ns() // 10**9 + 30) * 10**9
        outcome, _ = self.run_one(
            self.killed_run(linked_at=linked_at), mock.patch.object(mod.os, "utime", side_effect=whole_seconds)
        )
        self.assertEqual(outcome["result"], "killed")
        restored = (self.root / "src/lib.rs").stat().st_mtime_ns
        self.assertGreater(restored, linked_at)
        self.assertEqual(restored % 10**9, 0, "stored as the filesystem keeps it")

    def test_a_restored_source_that_stays_stored_as_old_fails_the_run(self):
        # A filesystem that ignores the new time leaves cargo linking the
        # defect; the run must stop rather than go on as though it were clean.
        with self.assertRaises(OSError):
            self.run_one(self.killed_run(), mock.patch.object(mod.os, "utime", side_effect=lambda *a, **k: None))
        self.assertEqual((self.root / "src/lib.rs").read_text(encoding="utf-8"), self.PLANTED)

    def test_a_restored_source_whose_stamp_cannot_be_moved_fails_the_run(self):
        # The content is back, but cargo may still link the defect, so the
        # run must not go on as though the tree were clean.
        plan = mod.load_toml(self.experiment / "tests/mutations.toml")

        def utime(*_args, **_kwargs):
            raise OSError(errno.EPERM, os.strerror(errno.EPERM))

        stdout = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod.subprocess, "run", side_effect=self.killed_run()),
            mock.patch.object(mod.os, "utime", side_effect=utime),
            contextlib.redirect_stdout(stdout),
            self.assertRaises(OSError),
        ):
            mod.run_mutation(plan, plan["mutation"][0], 60)
        self.assertEqual((self.root / "src/lib.rs").read_text(encoding="utf-8"), self.PLANTED)
        self.assertIn(
            "could not stamp them as newer than the build of the mutation return-two", stdout.getvalue()
        )
        self.assertIn("src/lib.rs", stdout.getvalue())

    def test_a_mutation_run_whose_plan_changes_as_it_is_read_is_not_recorded(self):
        # HEAD moves, a checkout in another terminal say, just after the plan
        # is read: the record would name a commit whose plan it did not run.
        real_load_plan = mod.load_plan

        def load_plan_then_move(exp_root):
            plan = real_load_plan(exp_root)
            path = self.experiment / "tests/mutations.toml"
            path.write_text(path.read_text(encoding="utf-8").replace("seed = 17", "seed = 29"), encoding="utf-8")
            git(self.root, "commit", "-q", "--no-verify", "-am", "another plan")
            return plan

        with mock.patch.object(mod, "load_plan", side_effect=load_plan_then_move):
            status, stdout = self.run_check()
        self.assertEqual(status, 2)
        self.assertFalse(self.record.exists())
        self.assertIn(f"HEAD moved from {self.head}", stdout)

    def test_a_mutation_run_whose_source_is_edited_and_restored_while_it_runs_is_not_recorded(self):
        def edit_and_restore():
            self.write("src/other.rs", "pub fn g() { panic!() }\n")
            self.write("src/other.rs", self.OTHER)

        status, stdout = self.run_check(edit_and_restore)
        self.assertEqual(status, 2)
        self.assertFalse(self.record.exists())
        self.assertIn("not recording", stdout)
        self.assertIn("src/other.rs", stdout)

    def test_an_edit_to_a_planted_file_during_its_run_is_seen_though_the_checker_undoes_it(self):
        # The checker restores the planted file from its own copy, which
        # erases the edit the mutated build may have compiled.
        status, stdout = self.run_check(lambda: self.write("src/lib.rs", "pub fn f() -> u8 { 3 + 4 }\n"))
        self.assertEqual(status, 2)
        self.assertFalse(self.record.exists())
        self.assertIn("src/lib.rs", stdout)

    def test_a_mutation_run_whose_sources_are_left_edited_is_not_recorded(self):
        status, stdout = self.run_check(lambda: self.write("src/other.rs", "pub fn g() { panic!() }\n"))
        self.assertEqual(status, 2)
        self.assertFalse(self.record.exists())
        self.assertIn("src/other.rs", stdout)

    def test_a_mutation_run_during_which_head_moves_is_not_recorded(self):
        status, stdout = self.run_check(
            lambda: git(self.root, "commit", "-q", "--no-verify", "--allow-empty", "-m", "moved")
        )
        self.assertEqual(status, 2)
        self.assertFalse(self.record.exists())
        self.assertIn(f"HEAD moved from {self.head}", stdout)


if __name__ == "__main__":
    unittest.main()
