import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_experiment", ROOT / "scripts/run_experiment.py"
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


class ExperimentRunnerTests(unittest.TestCase):
    def test_registry_resolves_known_experiment(self):
        item, root, data = mod.resolve("M001")
        self.assertEqual(data["id"], "M001")
        self.assertTrue(root.exists())
        self.assertEqual(item["status"], data["status"])

    def test_build_command_substitutes_seed_and_parameters(self):
        _, _, data = mod.resolve("L001")
        command = mod.build_command(
            data,
            entrypoint="entrypoint",
            seed=17,
            params={"iterations": "3"},
        )
        self.assertEqual(
            command,
            [
                "cargo",
                "run",
                "--release",
                "--locked",
                "-p",
                "ptr-bench",
                "--",
                "ledger-recovery",
                "3",
                "17",
            ],
        )

    def test_build_command_rejects_a_seed_given_as_a_parameter(self):
        # The record names the seed --seed gave, so no parameter replaces it.
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "the seed is given by --seed, not by a parameter"):
            mod.build_command(data, entrypoint="entrypoint", seed=17, params={"iterations": "3", "seed": "99"})

    def test_build_command_rejects_unlisted_seed(self):
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "not declared"):
            mod.build_command(
                data,
                entrypoint="entrypoint",
                seed=999,
                params={"iterations": "3"},
            )

    def test_build_command_takes_each_value_as_itself(self):
        # One pass: text in a value that reads as a placeholder is no
        # placeholder, so a value is never substituted into again.
        data = {"seeds": [17], "entrypoint": "bench <seed> --label=<label> <n>"}
        command = mod.build_command(data, entrypoint="entrypoint", seed=17, params={"label": "<n>/<seed>", "n": "3"})
        self.assertEqual(command, ["bench", "17", "--label=<n>/<seed>", "3"])
        # The seed is recorded whether or not the command takes it.
        data = {"seeds": [17], "entrypoint": "bench --fixed"}
        self.assertEqual(mod.build_command(data, entrypoint="entrypoint", seed=17), ["bench", "--fixed"])

    def test_build_command_refuses_a_value_no_placeholder_takes(self):
        # The record would name it as a parameter of a run it had no part in.
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "no placeholder of 'entrypoint' takes --set other, unused"):
            mod.build_command(data, entrypoint="entrypoint", seed=17, params={"iterations": "3", "unused": "1", "other": "2"})

    def test_build_command_rejects_unresolved_placeholder(self):
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "missing value"):
            mod.build_command(data, entrypoint="entrypoint", seed=17)

    def test_execute_command_captures_process_evidence(self):
        result = mod.execute_command(
            [sys.executable, "-c", "print('runner-ok')"]
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "runner-ok")
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(result["launch_error"])
        self.assertGreaterEqual(result["duration_ns"], 0)

    def test_python_in_the_command_reads_no_bytecode_cache_the_tree_holds(self):
        # A __pycache__ entry, which git ignores and HEAD does not hold, could
        # run in place of a tracked source: the command's Python keeps its
        # cache in a fresh directory, removed once the command has run.
        result = mod.execute_command([sys.executable, "-c", "import sys; print(sys.pycache_prefix)"])
        prefix = result["stdout"].strip()
        self.assertEqual(result["exit_code"], 0)
        self.assertNotIn(prefix, ("", "None"))
        self.assertFalse(Path(prefix).exists())
        self.assertNotIn("PYTHONPYCACHEPREFIX", os.environ)

    def test_a_seed_run_from_a_dirty_source_tree_is_refused_before_it_runs(self):
        # A record names HEAD as the code it ran; an edit reverted before
        # aggregation would otherwise be attributed to the clean revision.
        stderr = io.StringIO()
        with (
            mock.patch.object(
                mod.experiment_records, "uncommitted_files", return_value=["crates/ptr-pg/src/lib.rs"]
            ) as dirty,
            mock.patch.object(mod, "execute_command") as execute,
            mock.patch.object(mod, "write_json_exclusive") as write,
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L001", entrypoint="entrypoint", seed=17, params={"iterations": "3"})
        self.assertEqual(status, 2)
        execute.assert_not_called()
        write.assert_not_called()
        self.assertIn("refusing to run", stderr.getvalue())
        self.assertIn("crates/ptr-pg/src/lib.rs", stderr.getvalue())
        root, pathspecs = dirty.call_args.args
        self.assertEqual(root, mod.ROOT)
        experiment = "experiments/lifecycle/L001-revocation-crash"
        expected = ("*.rs", "Cargo.lock", "scripts/run_experiment.py", f"{experiment}/aggregate.py", experiment)
        for spec in expected:
            self.assertIn(spec, pathspecs)
        self.assertIn(f":(exclude){experiment}/results", pathspecs)

    def test_a_seed_run_from_a_clean_source_tree_runs_and_is_recorded(self):
        with (
            mock.patch.object(mod.experiment_records, "uncommitted_files", return_value=[]),
            mock.patch.object(
                mod, "execute_command", return_value={"exit_code": 0, "duration_ns": 1}
            ) as execute,
            mock.patch.object(mod, "write_json_exclusive") as write,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = mod.run_experiment("L001", entrypoint="entrypoint", seed=17, params={"iterations": "3"})
        self.assertEqual(status, 0)
        execute.assert_called_once()
        self.assertEqual(write.call_args.args[1]["seed"], 17)


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


class RunWatchTests(unittest.TestCase):
    """A seed record names the commit its run started from, so its sources must
    stay that commit's from before the run until the record is written: a
    `cargo run` entrypoint compiles whatever the tree holds while it runs."""

    ORIGINAL = "pub fn f() {}\n"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        files = {
            ".gitignore": "__pycache__/\n",
            "Cargo.lock": "# lock\n",
            "README.md": "readme\n",
            "src/lib.rs": self.ORIGINAL,
            "experiments/registry.toml": '[[experiment]]\nid = "L900"\npath = "x/L900-x"\nstatus = "running"\n',
            "experiments/x/L900-x/experiment.toml": (
                'id = "L900"\nstatus = "running"\nseeds = [17]\nentrypoint = "bench <seed>"\n'
            ),
            "experiments/x/L900-x/results/.gitkeep": "",
            # L900 is not listed, so it runs without a preregistration.
            "experiments/preregistration.toml": "version = 1\n",
        }
        for relative, text in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "code")
        self.head = git(self.root, "rev-parse", "HEAD")
        # A source last written long ago, so rewriting it with the same text
        # during a run changes its stamp whatever the clock's resolution.
        os.utime(self.root / "src/lib.rs", ns=(10**18, 10**18))
        self.results = self.root / "experiments/x/L900-x/results"

    def tearDown(self):
        self.directory.cleanup()

    def run_seed(self, during=None, launch_error: str | None = None):
        """Run seed 17 of L900 in the temporary repository, calling `during`
        while the command "runs", and failing to launch it with
        `launch_error` when given; returns the exit status, the records
        written and what was printed to stderr."""

        def execute(_command):
            if during is not None:
                during()
            if launch_error is not None:
                return {"exit_code": None, "stdout": "", "stderr": "", "launch_error": launch_error, "duration_ns": 1}
            return {"exit_code": 0, "stdout": "", "stderr": "", "launch_error": None, "duration_ns": 1}

        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "execute_command", side_effect=execute),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=17)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.results.glob("run-*.json"))]
        return status, records, stderr.getvalue()

    def write(self, relative: str, text: str) -> None:
        (self.root / relative).write_text(text, encoding="utf-8")

    def test_a_seed_run_whose_tree_stays_at_head_is_recorded_at_the_commit_it_started_from(self):
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual([record["git_sha"] for record in records], [self.head])

    def test_a_seed_run_whose_source_is_edited_and_restored_while_it_runs_is_not_recorded(self):
        # The build may have compiled the edit; the tree looks clean again
        # when the run ends.
        def edit_and_restore():
            self.write("src/lib.rs", "pub fn f() { panic!() }\n")
            self.write("src/lib.rs", self.ORIGINAL)

        status, records, stderr = self.run_seed(edit_and_restore)
        self.assertEqual((status, records), (2, []))
        self.assertIn("not recording", stderr)
        self.assertIn("src/lib.rs", stderr)
        self.assertEqual(git(self.root, "status", "--porcelain"), "")

    def test_a_seed_run_whose_sources_are_left_edited_or_added_to_is_not_recorded(self):
        for relative in ("src/lib.rs", "src/new.rs", "experiments/x/L900-x/config.toml"):
            with self.subTest(file=relative):
                try:
                    status, records, stderr = self.run_seed(lambda: self.write(relative, "changed\n"))
                    self.assertEqual((status, records), (2, []))
                    self.assertIn(relative, stderr)
                finally:
                    git(self.root, "checkout", "-q", "--", ".")
                    git(self.root, "clean", "-qfd")

    def test_a_seed_run_during_which_head_moves_is_not_recorded(self):
        status, records, stderr = self.run_seed(
            lambda: git(self.root, "commit", "-q", "--no-verify", "--allow-empty", "-m", "moved")
        )
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"HEAD moved from {self.head}", stderr)

    def test_a_record_whose_write_fails_leaves_no_partial_record(self):
        # A partial record is the newest of its seed, and the aggregator reads
        # every record it selects from.
        with disk_full_writing("-seed-17.json"), self.assertRaises(OSError):
            self.run_seed()
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])

    def test_a_record_never_replaces_another(self):
        path = self.results / "run-1-seed-17.json"
        path.write_text('{"first": true}\n', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            mod.write_json_exclusive(path, {"second": True})
        self.assertEqual(sorted(p.name for p in self.results.iterdir()), [".gitkeep", "run-1-seed-17.json"])
        self.assertEqual(path.read_text(encoding="utf-8"), '{"first": true}\n')
        other = self.results / "run-2-seed-17.json"
        mod.write_json_exclusive(other, {"second": True})
        self.assertEqual(json.loads(other.read_text(encoding="utf-8")), {"second": True})

    def preregister(
        self,
        status: str,
        table: str | None = "schema = 1\n",
        frozen: bool = True,
        required: str = "",
        entrypoint: str = "bench <seed>",
        manifest_lines: str = "",
    ) -> None:
        """List L900 as preregistering one integer, and the keys `required`
        adds, give it `status`, the `[preregistration]` table `table` (None
        for none), the command `entrypoint` and the further manifest lines
        `manifest_lines` and, when `frozen`, the manifest's digests of that
        table and of its list entry; commit."""
        listed = 'version = 1\n\n[experiment.L900.required]\nseeds = "int-list"\nschema = "int"\n' + required
        self.write("experiments/preregistration.toml", listed)
        if table is not None:
            table = "seeds = [17]\n" + table
        manifest = f'id = "L900"\nstatus = "{status}"\nseeds = [17]\nentrypoint = "{entrypoint}"\n' + manifest_lines
        self.write("experiments/x/L900-x/config.toml", "version = 1\n" + ("" if table is None else "\n[preregistration]\n" + table))
        if table is not None:
            if frozen:
                digest = mod.experiment_records.preregistration_digest(tomllib.loads(table))
                rules = mod.experiment_records.canonical_digest(tomllib.loads(listed)["experiment"]["L900"])
                manifest += f'preregistration_sha256 = "{digest}"\npreregistration_rules_sha256 = "{rules}"\n'
        self.write("experiments/x/L900-x/experiment.toml", manifest)
        self.write(
            "experiments/registry.toml",
            f'[[experiment]]\nid = "L900"\npath = "x/L900-x"\nstatus = "{status}"\n',
        )
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", f"preregister at {status}")
        self.head = git(self.root, "rev-parse", "HEAD")

    def test_a_listed_experiment_is_not_run_or_prepared_before_its_preregistration_is_frozen(self):
        cases = [
            ("planned", "schema = 1\n", True, "L900 preregisters (experiments/preregistration.toml) and is 'planned'"),
            ("running", None, True, "L900: config.toml has no [preregistration] table"),
            ("running", 'schema = "must-be-pinned-before-prepared"\n', True, "L900: preregistration key schema is a placeholder"),
            ("running", "schema = 1\n", False, "L900: experiment.toml names no preregistration_sha256"),
        ]
        for status, table, frozen, refusal in cases:
            with self.subTest(status=status, table=table, frozen=frozen):
                self.preregister(status, table, frozen)
                ran = []
                code, records, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((code, records, ran), (2, [], []))
                self.assertIn(refusal, stderr)
                prepared = io.StringIO()
                with (
                    mock.patch.object(mod, "ROOT", self.root),
                    mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                    contextlib.redirect_stderr(prepared),
                ):
                    self.assertEqual(mod.prepare("L900"), 2)
                self.assertIn(refusal, prepared.getvalue())
                self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])

    def test_a_listed_experiment_runs_once_frozen_and_its_record_is_bound_to_that_preregistration(self):
        self.preregister("running")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        manifest = records[0]["manifest"]
        self.assertEqual(
            manifest["preregistration_sha256"],
            mod.experiment_records.preregistration_digest({"seeds": [17], "schema": 1}),
        )
        self.assertEqual(records[0]["git_sha"], self.head)
        # Committed, the record passes the gate: the commit it names holds
        # the preregistration frozen now and the manifest it names.
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record")
        self.assertEqual(mod.check_research_gates.launch_errors(self.root, "L900"), [])

    def test_a_listed_experiment_runs_each_seed_once(self):
        # A seed run again after its outcome was seen could keep whichever
        # run came out best, so every run of a listed seed is its one run.
        self.preregister("running")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, len(records)), (0, "", 1))
        [first] = [path.relative_to(self.root).as_posix() for path in self.results.glob("run-*.json")]
        refusal = (
            f"ERROR: seed 17 of L900 already ran ({first}); a listed experiment runs each seed once, so no run of it "
            "is chosen by its outcome\n"
        )
        # Before its record is committed and after.
        for committed in (False, True):
            with self.subTest(committed=committed):
                if committed:
                    git(self.root, "add", "-A")
                    git(self.root, "commit", "-q", "--no-verify", "-m", "record")
                status, records, stderr = self.run_seed()
                self.assertEqual((status, stderr, len(records)), (2, refusal, 1))
        # A committed record deleted since is refused by the gate.
        (self.root / first).unlink()
        git(self.root, "commit", "-q", "--no-verify", "-am", "record deleted")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn("was committed and has since been deleted or renamed", stderr)
        # A command that failed to launch saw no outcome, so its seed runs.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        status, records, _ = self.run_seed(launch_error="no such file")
        self.assertEqual((status, [record["status"] for record in records]), (127, ["failed-to-launch"]))
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(sorted(record["status"] for record in records), ["completed", "failed-to-launch"])
        # A record the runner cannot read could be a run of the seed.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        self.write("experiments/x/L900-x/results/run-unreadable.json", "{")
        status, calls, stderr = self.launch_listed({})
        self.assertEqual((status, calls), (2, []))
        # The gate reads the results first; the runner's own look refuses it
        # all the same, as it does a record that is no JSON object.
        self.assertIn("L900: results/run-unreadable.json cannot be read", stderr)
        with mock.patch.object(mod, "ROOT", self.root):
            for text, refusal in (
                ("{", "experiments/x/L900-x/results/run-unreadable.json cannot be read, so whether seed 17 ran is unknown"),
                ("[17]", "experiments/x/L900-x/results/run-unreadable.json is not a JSON object, so whether seed 17 ran is unknown"),
            ):
                self.write("experiments/x/L900-x/results/run-unreadable.json", text)
                with self.assertRaises(ValueError) as caught:
                    mod.seed_runs(self.results, 17)
                self.assertIn(refusal, str(caught.exception))
            # A prepared record names no seed, and another seed is another.
            self.write("experiments/x/L900-x/results/run-unreadable.json", json.dumps({"status": "prepared"}))
            self.write("experiments/x/L900-x/results/run-29.json", json.dumps({"seed": 29, "status": "completed"}))
            self.assertEqual(mod.seed_runs(self.results, 17), [])
            self.write("experiments/x/L900-x/results/run-17.json", json.dumps({"seed": 17, "status": "failed"}))
            self.assertEqual(mod.seed_runs(self.results, 17), ["experiments/x/L900-x/results/run-17.json"])
        # An unlisted experiment runs a seed again, as L003 and L004 do.
        self.tearDown()
        self.setUp()
        for runs in (1, 2):
            status, records, stderr = self.run_seed()
            self.assertEqual((status, stderr, len(records)), (0, "", runs))

    def test_what_decides_a_launch_is_held_to_head_before_and_while_the_run(self):
        self.preregister("running")
        listed = (self.root / "experiments/preregistration.toml").read_text(encoding="utf-8")
        # Edited, and not committed: refused before it runs, even where the
        # edit leaves the experiment's entry as it was.
        self.write("experiments/preregistration.toml", listed + "# edited\n")
        ran = []
        status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("refusing to run", stderr)
        self.assertIn("experiments/preregistration.toml", stderr)
        # An entry taken out of the list does not open the runner either: an
        # experiment the list has named stays listed.
        self.write("experiments/preregistration.toml", "version = 1\n")
        status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("L900 was listed at", stderr)
        self.write("experiments/preregistration.toml", listed)
        # Edited while it runs, and put back: no record.
        def edit_and_restore():
            self.write("experiments/preregistration.toml", "version = 1\n")
            self.write("experiments/preregistration.toml", listed)

        status, records, stderr = self.run_seed(edit_and_restore)
        self.assertEqual((status, records), (2, []))
        self.assertIn("experiments/preregistration.toml", stderr)
        self.assertEqual(
            mod.check_research_gates.launch_inputs(self.root, "L900"),
            ["experiments/preregistration.toml", "experiments/registry.toml", "scripts/check_research_gates.py"],
        )

    def test_a_results_directory_below_the_experiment_and_no_link_holds_the_records(self):
        manifest = (self.root / "experiments/x/L900-x/experiment.toml").read_text(encoding="utf-8")
        # The experiment's own directory would take all of it out of the
        # HEAD check, and one outside it is not the experiment's.
        # Nor may git's own directory hold it, where no record is committed.
        for results_dir, refusal in (
            (".", "is not a directory below experiments/x/L900-x"),
            ("../elsewhere", "is not a directory below experiments/x/L900-x"),
            (".git", "passes through git's own directory, where no record can be committed"),
            ("results/.GIT", "passes through git's own directory, where no record can be committed"),
        ):
            with self.subTest(results_dir=results_dir):
                self.write("experiments/x/L900-x/experiment.toml", manifest + f'results_dir = "{results_dir}"\n')
                git(self.root, "commit", "-q", "--no-verify", "-am", f"results in {results_dir}")
                ran = []
                status, _, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, ran), (2, []))
                self.assertIn(f"results_dir {results_dir!r} {refusal}", stderr)
        # A file on the way would leave no directory to write the record
        # into once the run had run.
        self.write("experiments/x/L900-x/notes", "a file\n")
        for results_dir in ("notes", "notes/results"):
            with self.subTest(results_dir=results_dir):
                self.write("experiments/x/L900-x/experiment.toml", manifest + f'results_dir = "{results_dir}"\n')
                ran = []
                status, _, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, ran), (2, []))
                self.assertIn("results directory experiments/x/L900-x/notes is not a directory", stderr)
        (self.root / "experiments/x/L900-x/notes").unlink()
        self.write("experiments/x/L900-x/experiment.toml", manifest)
        git(self.root, "commit", "-q", "--no-verify", "-am", "results in results")
        # A link would carry the records, and what the check leaves out,
        # elsewhere.
        shutil.rmtree(self.results)
        try:
            os.symlink(self.root, self.results)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")
        ran = []
        status, _, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, ran), (2, []))
        self.assertIn("results directory experiments/x/L900-x/results is a symlink", stderr)

    def test_a_results_directory_the_command_replaces_holds_no_record(self):
        # The watch leaves the results directory out, so the command could
        # put a file or a link where it is: the record is then not written
        # through it, nor lost to a traceback.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))

        def with_file():
            shutil.rmtree(self.results)
            self.results.write_text("a file\n", encoding="utf-8")

        def with_link():
            shutil.rmtree(self.results)
            os.symlink(outside, self.results)

        for during, problem in ((with_file, "is not a directory"), (with_link, "is a symlink")):
            with self.subTest(problem=problem):
                status, records, stderr = self.run_seed(during)
                self.assertEqual((status, records), (2, []))
                self.assertIn(f"results directory experiments/x/L900-x/results {problem}", stderr)
                self.assertIn("its results directory changed while it ran", stderr)
                self.assertEqual(list(outside.iterdir()), [])
                self.results.unlink()
                git(self.root, "checkout", "-q", "--", "experiments/x/L900-x/results")

    def test_prepare_holds_what_decides_a_launch_to_head(self):
        self.preregister("prepared")
        listed = (self.root / "experiments/preregistration.toml").read_text(encoding="utf-8")

        def prepare() -> tuple[int, str]:
            stderr = io.StringIO()
            with (
                mock.patch.object(mod, "ROOT", self.root),
                mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                return mod.prepare("L900"), stderr.getvalue()

        # An uncommitted edit to the list, even one that leaves the entry as
        # it was, refuses the prepare record: it would name a HEAD that is not
        # the tree the decision read.
        self.write("experiments/preregistration.toml", listed + "# edited\n")
        status, stderr = prepare()
        self.assertEqual(status, 2)
        self.assertIn("refusing to run", stderr)
        self.assertIn("experiments/preregistration.toml", stderr)
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])
        self.write("experiments/preregistration.toml", listed)
        status, stderr = prepare()
        self.assertEqual((status, stderr), (0, ""))
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual((record["status"], record["git_sha"]), ("prepared", self.head))

    def test_a_listed_experiment_runs_only_from_a_commit_that_holds_what_froze_it(self):
        # The gate finds a freeze by what a commit holds. A protocol that
        # HEAD's .gitignore hides is in the tree, so the tree is frozen and
        # the watch sees nothing uncommitted, but HEAD does not hold it: a run
        # from there would leave a freeze the gate could not find.
        protocol = "experiments/x/L900-x/notes/protocol.md"
        text = "# Protocol\n"
        self.write(".gitignore", "__pycache__/\nnotes/\n")
        (self.root / protocol).parent.mkdir()
        self.write(protocol, text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.preregister(
            "running", f'schema = 1\nprotocol = "{protocol}"\nprotocol_sha256 = "{digest}"\n', required='protocol = "file"\n'
        )
        self.assertEqual(mod.check_research_gates.launch_errors(self.root, "L900"), [])
        refusal = f"L900: {self.head[:12]}, the commit its run would name, does not hold it frozen as the tree launches it"
        ran = []
        status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn(refusal, stderr)
        prepared = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            contextlib.redirect_stderr(prepared),
        ):
            self.assertEqual(mod.prepare("L900"), 2)
        self.assertIn(refusal, prepared.getvalue())
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])
        # Committed, the protocol is one HEAD holds, and the run is recorded.
        self.write(".gitignore", "__pycache__/\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "protocol committed")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, len(records)), (0, "", 1))

    def test_a_manifest_date_is_refused_before_anything_runs(self):
        # A record holds the manifest as JSON, which has no date: a date and
        # the string of its text would be one manifest to aggregation.
        manifest = self.root / "experiments/x/L900-x/experiment.toml"
        self.write(
            "experiments/x/L900-x/experiment.toml",
            manifest.read_text(encoding="utf-8") + "created = 2026-01-02\n[timing]\nsteps = [1, 07:32:00]\n",
        )
        git(self.root, "commit", "-q", "--no-verify", "-am", "dated")
        refusal = (
            "ERROR: L900: experiment.toml holds a TOML date or time at {}, which a run record, holding the manifest "
            "as JSON, cannot tell from a string; write it as a string\n"
        )
        expected = refusal.format("created") + refusal.format("timing.steps[1]")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records, stderr), (2, [], expected))
        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(mod.prepare("L900"), 2)
        self.assertEqual(stderr.getvalue(), expected)
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])
        # Aggregation refuses it too, whatever the records hold.
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        record = {"experiment_id": "L900", "git_sha": "a" * 40, "entrypoint": "entrypoint", "manifest": {
            **data, "created": "2026-01-02", "timing": {"steps": [1, "07:32:00"]},
        }}
        self.assertEqual(
            mod.experiment_records.agreement_errors("L900", data, {"record": record}),
            [line.removeprefix("ERROR: L900: ").rstrip("\n") for line in expected.splitlines(keepends=True)],
        )

    def test_validate_refuses_a_manifest_holding_a_date_or_time(self):
        self.write(
            "experiments/schema.toml",
            'version = 1\nrequired = ["id", "status", "seeds"]\nallowed_status = ["planned", "running"]\n',
        )
        (self.root / "experiments/x/L900-x/tests").mkdir()
        self.write("experiments/x/L900-x/config.toml", "version = 1\n")
        manifest = (self.root / "experiments/x/L900-x/experiment.toml").read_text(encoding="utf-8")

        def validate() -> tuple[int, str]:
            stdout = io.StringIO()
            with (
                mock.patch.object(mod, "ROOT", self.root),
                mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                contextlib.redirect_stdout(stdout),
            ):
                return mod.validate(), stdout.getvalue()

        self.assertEqual(validate(), (0, "OK: validated 1 experiments\n"))
        self.write("experiments/x/L900-x/experiment.toml", manifest + "created = 2026-01-02\n")
        self.assertEqual(validate(), (1, (
            "ERROR: L900: experiment.toml holds a TOML date or time at created, which a run record, holding the "
            "manifest as JSON, cannot tell from a string; write it as a string\n"
        )))

    def launch_listed(self, params: dict[str, str], entrypoint: str = "entrypoint") -> tuple[int, list, str]:
        """Runs seed 17 of L900 through `entrypoint` with `params`; returns
        the status, the commands the run executed, and what it wrote to
        stderr."""
        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "execute_command", return_value={
                "exit_code": 0, "stdout": "", "stderr": "", "launch_error": None, "duration_ns": 1,
            }) as execute,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint=entrypoint, seed=17, params=params)
        return status, execute.call_args_list, stderr.getvalue()

    def test_a_listed_experiments_command_takes_its_preregistered_values(self):
        # A value chosen at launch could be chosen after an outcome was seen:
        # each placeholder but <seed> takes the frozen [preregistration]
        # value, and --set may only repeat it.
        self.preregister(
            "running",
            'schema = 1\niterations = 30\nverbose = true\nharness = "fixture-1"\nlabel = "<iterations>/<seed>"\n',
            entrypoint="bench <seed> <iterations> <verbose> <harness> --label=<label>",
            manifest_lines='quick_entrypoint = "bench <seed> 1 <verbose>"\n',
        )
        run = self.launch_listed
        status, calls, stderr = run({})
        self.assertEqual((status, stderr), (0, ""))
        # A string is itself, placeholder text in it included.
        self.assertEqual(calls[0].args[0], ["bench", "17", "30", "true", "fixture-1", "--label=<iterations>/<seed>"])
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual(
            record["parameters"],
            {"iterations": "30", "verbose": "true", "harness": "fixture-1", "label": "<iterations>/<seed>"},
        )
        for params, refusal in (
            ({"iterations": "31"}, "--set iterations=31 is not the preregistered value 30"),
            ({"other": "1"}, "--set other names no placeholder the command takes from the frozen [preregistration] table"),
        ):
            with self.subTest(params=params):
                status, calls, stderr = run(params)
                self.assertEqual((status, calls), (2, []))
                self.assertIn(refusal, stderr)
        # Repeating the preregistered value is no other command, but seed 17
        # has run.
        status, calls, stderr = run({"iterations": "30"})
        self.assertEqual((status, calls), (2, []))
        self.assertIn("seed 17 of L900 already ran", stderr)
        # Another command the frozen manifest holds is chosen at launch too.
        status, calls, stderr = run({}, entrypoint="quick_entrypoint")
        self.assertEqual((status, calls), (2, []))
        self.assertIn("--entrypoint quick_entrypoint: a listed experiment runs only through its manifest's entrypoint", stderr)
        self.assertEqual(len(list(self.results.glob("run-*.json"))), 1)
        # A --set repeating the preregistered value runs, in a repository
        # where seed 17 has not.
        self.tearDown()
        self.setUp()
        self.preregister(
            "running", "schema = 1\niterations = 30\nverbose = true\n", entrypoint="bench <seed> <iterations> <verbose>"
        )
        status, calls, stderr = run({"iterations": "30"})
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(calls[0].args[0], ["bench", "17", "30", "true"])

    def test_an_unlisted_experiment_runs_through_any_entrypoint_with_its_set_values(self):
        # Only the list binds a command: L001 and L004 run their second
        # entrypoints with values given at launch.
        self.write(
            "experiments/x/L900-x/experiment.toml",
            'id = "L900"\nstatus = "running"\nseeds = [17]\nentrypoint = "bench <seed>"\n'
            'quick_entrypoint = "bench <seed> <n>"\n',
        )
        git(self.root, "commit", "-q", "--no-verify", "-am", "a second entrypoint")
        status, calls, stderr = self.launch_listed({"n": "5"}, entrypoint="quick_entrypoint")
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(calls[0].args[0], ["bench", "17", "5"])
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual((record["entrypoint"], record["parameters"]), ("quick_entrypoint", {"n": "5"}))
        # Its parameters are the values its placeholders took: a --set no
        # placeholder takes is refused.
        status, calls, stderr = self.launch_listed({"n": "5", "unused": "1"}, entrypoint="quick_entrypoint")
        self.assertEqual((status, calls), (2, []))
        self.assertIn("no placeholder of 'quick_entrypoint' takes --set unused", stderr)

    def test_a_placeholder_the_frozen_table_does_not_hold_or_holds_as_a_list_is_refused(self):
        # A placeholder with no frozen value would take one chosen at launch,
        # and a list has no single token to stand for it; neither runs.
        for table, entrypoint, gate_refusal, refusal in (
            ("schema = 1\n", "bench <seed> <missing>",
             "L900: entrypoint placeholder <missing> is no key of the [preregistration] table",
             "<missing> is no key of the frozen [preregistration] table"),
            ("schema = 1\ngrid = [1, 2]\n", "bench <seed> <grid>",
             "L900: entrypoint placeholder <grid> is preregistered as a list",
             "<grid> is preregistered as a list, which no command token takes"),
        ):
            with self.subTest(entrypoint=entrypoint):
                # A frozen table is never rewritten, so each case starts from
                # a repository of its own.
                self.tearDown()
                self.setUp()
                self.preregister("running", table, entrypoint=entrypoint)
                for params in ({}, {"missing": "1"}, {"grid": "1"}):
                    status, calls, stderr = self.launch_listed(params)
                    self.assertEqual((status, calls), (2, []))
                    self.assertEqual(list(self.results.glob("run-*.json")), [])
                # The gate refuses such a freeze first; the runner's own look
                # refuses it all the same.
                self.assertIn(gate_refusal, self.launch_listed({})[2])
                with mock.patch.object(mod, "ROOT", self.root):
                    data = tomllib.loads((self.root / "experiments/x/L900-x/experiment.toml").read_text(encoding="utf-8"))
                    with self.assertRaises(ValueError) as caught:
                        mod.command_parameters("L900", self.root / "experiments/x/L900-x", data, "entrypoint", {})
                self.assertIn(refusal, str(caught.exception))

    def test_a_manifest_changed_before_the_watch_looked_is_refused(self):
        # The command is built from the manifest read first; one committed
        # in its place before the watch looked would leave the watch holding
        # another than the one that ran.
        real = mod.experiment_records.ProvenanceWatch

        def replaced(*args, **kwargs):
            self.write(
                "experiments/x/L900-x/experiment.toml",
                'id = "L900"\nstatus = "running"\nseeds = [17]\nentrypoint = "other <seed>"\n',
            )
            git(self.root, "commit", "-q", "--no-verify", "-am", "replaced")
            return real(*args, **kwargs)

        ran = []
        with mock.patch.object(mod.experiment_records, "ProvenanceWatch", side_effect=replaced):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("experiment.toml changed while the launch was checked", stderr)

    def test_a_record_the_results_hold_is_no_change_of_the_sources(self):
        # Records accumulate in results/, which the run writes into itself.
        status, records, _ = self.run_seed(lambda: self.write("experiments/x/L900-x/results/other.json", "{}\n"))
        self.assertEqual(status, 0)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
