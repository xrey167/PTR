import contextlib
import datetime
import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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

    def test_a_record_holding_a_date_is_written_as_its_text(self):
        # The runner refuses a dated manifest before it runs; were a date to
        # reach the record all the same, it is written as its text rather
        # than lose the record of a run that ran.
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        mod.write_json_exclusive(directory / "run.json", {"created": datetime.date(2026, 1, 2)})
        self.assertEqual(json.loads((directory / "run.json").read_text(encoding="utf-8")), {"created": "2026-01-02"})
        mod.write_json_replacing(directory / "run.json", {"created": datetime.time(7, 32)})
        self.assertEqual(json.loads((directory / "run.json").read_text(encoding="utf-8")), {"created": "07:32:00"})

    def test_the_runner_and_the_gate_write_no_bytecode_cache_into_the_tree(self):
        # A `__pycache__` directory is one git ignores, and a listed
        # experiment runs only from a checkout that holds none: run as the
        # README shows, without -B, neither the runner nor the gate writes
        # one for the modules it imports.
        environment = {
            key: value for key, value in os.environ.items() if key not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")
        }
        for script, arguments in (("run_experiment.py", ["--help"]), ("check_research_gates.py", [])):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / "scripts"
                copy.mkdir()
                for name in ("run_experiment.py", "check_research_gates.py", "experiment_records.py"):
                    shutil.copy(ROOT / "scripts" / name, copy / name)
                subprocess.run(
                    [sys.executable, str(copy / script), *arguments], cwd=directory, env=environment, capture_output=True,
                    check=False,
                )
                self.assertEqual(sorted(path.name for path in copy.iterdir()), sorted(
                    ("run_experiment.py", "check_research_gates.py", "experiment_records.py")
                ))

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
        script = "import os, sys; print(sys.pycache_prefix); print(os.environ.get('CARGO_TARGET_DIR'))"
        result = mod.execute_command([sys.executable, "-c", script])
        prefix, target = result["stdout"].splitlines()
        # Without an environment of its own (an unlisted experiment's), Cargo
        # builds where it would.
        self.assertEqual(target, str(os.environ.get("CARGO_TARGET_DIR")))
        self.assertEqual(result["exit_code"], 0)
        self.assertNotIn(prefix, ("", "None"))
        self.assertFalse(Path(prefix).exists())
        self.assertNotIn("PYTHONPYCACHEPREFIX", os.environ)

    def test_a_command_that_cannot_start_has_its_launch_error_and_no_exit_status(self):
        # For want of a directory for its bytecode cache, or of its program,
        # the command never started: it saw no outcome.
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "no-such-directory"
            with mock.patch.object(mod.tempfile, "tempdir", str(missing)):
                result = mod.execute_command([sys.executable, "-c", "print('ran')"])
            self.assertEqual((result["exit_code"], result["stdout"], result["stderr"]), (None, "", ""))
            self.assertTrue(result["launch_error"].startswith("FileNotFoundError: "), result["launch_error"])
            self.assertIn(str(missing), result["launch_error"])
            self.assertGreaterEqual(result["duration_ns"], 0)
            result = mod.execute_command([str(missing / "program")])
            self.assertEqual((result["exit_code"], result["stdout"], result["stderr"]), (None, "", ""))
            self.assertTrue(result["launch_error"].startswith("FileNotFoundError: "), result["launch_error"])
            # Nor can a command take an argument holding a NUL character.
            result = mod.execute_command([sys.executable, "-c", "print('ran')", "a\0b"])
            self.assertEqual((result["exit_code"], result["stdout"]), (None, ""))
            self.assertEqual(result["launch_error"], "ValueError: embedded null byte")

    def test_a_command_given_its_scratch_directory_makes_it_fresh_and_removes_it(self):
        # A listed run builds and caches in a directory of a known name,
        # which it makes only if absent: an earlier run's build left there
        # is not taken for one of the commit's sources.
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "ptr-run-L900"
            script = "import os, sys; print(sys.pycache_prefix); print(os.environ['CARGO_TARGET_DIR'])"
            environment = {"PATH": os.environ.get("PATH", os.defpath)}
            result = mod.execute_command([sys.executable, "-c", script], environment, None, scratch)
            self.assertEqual((result["exit_code"], result["launch_error"]), (0, None))
            self.assertEqual(result["stdout"].splitlines(), [str(scratch), str(scratch / "cargo-target")])
            self.assertFalse(scratch.exists())
            scratch.mkdir()
            (scratch / "left").write_text("an earlier build\n", encoding="utf-8")
            result = mod.execute_command([sys.executable, "-c", "print('ran')"], environment, None, scratch)
            self.assertEqual((result["exit_code"], result["stdout"]), (None, ""))
            self.assertTrue(result["launch_error"].startswith("FileExistsError: "), result["launch_error"])
            self.assertTrue((scratch / "left").exists())
        # Named by the experiment's id as its lock is.
        self.assertEqual(
            mod.scratch_directory("team/trial", {"TMPDIR": "/scratch"}), Path("/scratch/ptr-run-team%2Ftrial")
        )

    def test_a_command_that_started_ran_whatever_happens_after(self):
        # A cache the command left that cannot be removed does not make the
        # run one that never started.
        script = (
            "import os, shutil, sys; prefix = sys.pycache_prefix; shutil.rmtree(prefix); "
            "open(prefix, 'w').close(); print(prefix)"
        )
        result = mod.execute_command([sys.executable, "-c", script])
        prefix = Path(result["stdout"].strip())
        self.addCleanup(prefix.unlink, missing_ok=True)
        self.assertEqual((result["exit_code"], result["launch_error"]), (0, None))
        self.assertTrue(prefix.is_file())
        # An exception raised in the runner while the command runs kills
        # the command, and is raised.
        started = []
        killed = []
        kill = subprocess.Popen.kill

        def interrupted(process, *args, **kwargs):
            started.append(process)
            raise KeyboardInterrupt

        def killing(process):
            killed.append(process)
            kill(process)

        with (
            mock.patch.object(subprocess.Popen, "communicate", autospec=True, side_effect=interrupted),
            mock.patch.object(subprocess.Popen, "kill", autospec=True, side_effect=killing),
        ):
            with self.assertRaises(KeyboardInterrupt):
                mod.execute_command([sys.executable, "-c", "import time; time.sleep(60)"])
        [process] = started
        self.assertEqual(killed, [process])
        self.assertEqual(process.wait(timeout=30), -signal.SIGKILL)

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
        # The results directory is left out only for the records the tools
        # write there; anything else there is held to HEAD.
        self.assertIn(experiment, pathspecs)
        for output in ("run-*.json", "run.json", "metrics.json", "mutations.json"):
            self.assertIn(f":(exclude,glob){experiment}/results/{output}", pathspecs)
        self.assertNotIn(f":(exclude){experiment}/results", pathspecs)

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
        # A listed run refuses Cargo configuration from outside the
        # repository: the machine's own is none of these runs'.
        cargo_home = Path(self.directory.name + "-cargo")
        cargo_home.mkdir(exist_ok=True)
        self.addCleanup(shutil.rmtree, cargo_home, ignore_errors=True)
        self.enterContext(mock.patch.dict(os.environ, {"CARGO_HOME": str(cargo_home)}))
        # The entrypoint's program, found first on the PATH: a listed run
        # starts only a program its launch found and named.
        tools = Path(self.directory.name + "-tools")
        tools.mkdir(exist_ok=True)
        self.addCleanup(shutil.rmtree, tools, ignore_errors=True)
        (tools / "bench").write_text("#!/bin/sh\n", encoding="utf-8")
        (tools / "bench").chmod(0o755)
        self.enterContext(mock.patch.dict(os.environ, {"PATH": os.pathsep.join((str(tools), os.environ.get("PATH", os.defpath)))}))
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

    def run_seed(self, during=None, launch_error: str | None = None, seed: int = 17):
        """Run `seed` of L900 in the temporary repository, calling `during`
        while the command "runs", and failing to launch it with
        `launch_error` when given; returns the exit status, the records
        written and what was printed to stderr."""

        def execute(_command, _environment=None, _program=None, _scratch=None):
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
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=seed)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.results.glob("run-*.json"))]
        return status, records, stderr.getvalue()

    def write(self, relative: str, text: str) -> None:
        (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
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
        seeds: str = "[17]",
    ) -> None:
        """List L900 as preregistering one integer, and the keys `required`
        adds, give it `status`, the `[preregistration]` table `table` (None
        for none), the command `entrypoint` and the further manifest lines
        `manifest_lines` and, when `frozen`, the manifest's digests of that
        table and of its list entry; commit."""
        listed = 'version = 1\n\n[experiment.L900.required]\nseeds = "int-list"\nschema = "int"\n' + required
        self.write("experiments/preregistration.toml", listed)
        if table is not None:
            table = f"seeds = {seeds}\n" + table
        manifest = f'id = "L900"\nstatus = "{status}"\nseeds = {seeds}\nentrypoint = "{entrypoint}"\n' + manifest_lines
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
        # Before its record is committed, no launch reads the results: every
        # earlier record is committed first. After, the seed has run.
        status, records, stderr = self.run_seed()
        self.assertEqual((status, len(records)), (2, 1))
        self.assertIn(f"commit or remove {first}", stderr)
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
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "failed to launch")
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
            attempts = mod.attempts_directory(mod.git_directory(), self.results)
            for text, refusal in (
                ("{", "experiments/x/L900-x/results/run-unreadable.json cannot be read, so whether seed 17 ran is unknown"),
                ("[17]", "experiments/x/L900-x/results/run-unreadable.json is not a JSON object, so whether seed 17 ran is unknown"),
            ):
                self.write("experiments/x/L900-x/results/run-unreadable.json", text)
                with self.assertRaises(ValueError) as caught:
                    mod.seed_runs(self.results, 17, attempts)
                self.assertIn(refusal, str(caught.exception))
            # A prepared record names no seed, and another seed is another.
            self.write("experiments/x/L900-x/results/run-unreadable.json", json.dumps({"status": "prepared"}))
            self.write("experiments/x/L900-x/results/run-29.json", json.dumps({"seed": 29, "status": "completed"}))
            self.assertEqual(mod.seed_runs(self.results, 17, attempts), [])
            self.write("experiments/x/L900-x/results/run-17.json", json.dumps({"seed": 17, "status": "failed"}))
            self.assertEqual(mod.seed_runs(self.results, 17, attempts), ["experiments/x/L900-x/results/run-17.json"])
            # The copy the runner keeps of a record counts as the record,
            # named by the record's path, whether or not the results hold it;
            # a copy of a run whose command failed to launch does not.
            attempts.mkdir(parents=True)
            (attempts / "run-17.json").write_text(json.dumps({"seed": 17, "status": "started"}), encoding="utf-8")
            (attempts / "run-kept.json").write_text(json.dumps({"seed": 17, "status": "completed"}), encoding="utf-8")
            (attempts / "run-unlaunched.json").write_text(
                json.dumps({"seed": 17, "status": "failed-to-launch"}), encoding="utf-8"
            )
            self.assertEqual(
                mod.seed_runs(self.results, 17, attempts),
                ["experiments/x/L900-x/results/run-17.json", "experiments/x/L900-x/results/run-kept.json"],
            )
            # The copy counts where the record of its name says otherwise.
            self.write("experiments/x/L900-x/results/run-other.json", json.dumps({"seed": 29, "status": "completed"}))
            (attempts / "run-other.json").write_text(json.dumps({"seed": 17, "status": "started"}), encoding="utf-8")
            self.write(
                "experiments/x/L900-x/results/run-unlaunched.json", json.dumps({"seed": 17, "status": "failed-to-launch"})
            )
            (attempts / "run-unlaunched.json").write_text(json.dumps({"seed": 17, "status": "started"}), encoding="utf-8")
            self.assertEqual(
                mod.seed_runs(self.results, 17, attempts),
                [
                    "experiments/x/L900-x/results/run-17.json",
                    "experiments/x/L900-x/results/run-kept.json",
                    "experiments/x/L900-x/results/run-other.json",
                    "experiments/x/L900-x/results/run-unlaunched.json",
                ],
            )
            for name in ("run-other.json", "run-unlaunched.json"):
                (attempts / name).unlink()
            (attempts / "run-kept.json").write_text("[17]", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                mod.seed_runs(self.results, 17, attempts)
            self.assertIn(f"{attempts / 'run-kept.json'} is not a JSON object, so whether seed 17 ran is unknown", str(caught.exception))
            (attempts / "run-kept.json").write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                mod.seed_runs(self.results, 17, attempts)
            self.assertIn(f"{attempts / 'run-kept.json'} cannot be read, so whether seed 17 ran is unknown", str(caught.exception))
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
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn("experiments/preregistration.toml", stderr)
        # The run went unrecorded, but its command ran: the reservation it
        # made before the command stays as the record that seed 17 ran.
        self.assertIn("stays as the record that seed 17 ran", stderr)
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
        # The command could put a file or a link where the results directory
        # is: the record is then not written through it, nor lost to a
        # traceback. The watch sees the tracked files there go; were it to
        # miss that, the results directory is checked again all the same.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))

        def with_file():
            shutil.rmtree(self.results)
            self.results.write_text("a file\n", encoding="utf-8")

        def with_link():
            shutil.rmtree(self.results)
            os.symlink(outside, self.results)

        for during, problem in ((with_file, "is not a directory"), (with_link, "is a symlink")):
            for watched in (True, False):
                with self.subTest(problem=problem, watched=watched):
                    with contextlib.ExitStack() as stack:
                        if not watched:
                            stack.enter_context(
                                mock.patch.object(mod.experiment_records.ProvenanceWatch, "changes", return_value=[])
                            )
                        status, records, stderr = self.run_seed(during)
                    self.assertEqual((status, records), (2, []))
                    if watched:
                        self.assertIn("experiments/x/L900-x/results/.gitkeep changed on disk", stderr)
                    else:
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
            manifest.read_text(encoding="utf-8")
            + "created = 2026-01-02\ncap = -nan\n[timing]\nsteps = [1, 07:32:00, nan]\n",
        )
        git(self.root, "commit", "-q", "--no-verify", "-am", "dated")
        refusal = (
            "ERROR: L900: experiment.toml holds a TOML date or time at {}, which a run record, holding the manifest "
            "as JSON, cannot tell from a string; write it as a string\n"
        )
        # Nor a NaN, which JSON writes as NaN whatever its sign.
        nan = (
            "ERROR: L900: experiment.toml holds a NaN at {}, which a run record, holding the manifest as JSON, "
            "cannot tell from a NaN of the other sign; write it as a string\n"
        )
        expected = (
            refusal.format("created") + nan.format("cap") + refusal.format("timing.steps[1]")
            + nan.format("timing.steps[2]")
        )
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
            **data, "created": "2026-01-02", "timing": {"steps": [1, "07:32:00", float("nan")]},
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
        self.write("experiments/x/L900-x/experiment.toml", manifest + "cap = +nan\n")
        self.assertEqual(validate(), (1, (
            "ERROR: L900: experiment.toml holds a NaN at cap, which a run record, holding the manifest as JSON, "
            "cannot tell from a NaN of the other sign; write it as a string\n"
        )))
        self.write("experiments/x/L900-x/experiment.toml", manifest + 'cap = "nan"\nceiling = -inf\n')
        self.assertEqual(validate(), (0, "OK: validated 1 experiments\n"))
        # An experiment is registered once: a second entry of its id would go
        # unchecked, and leave the runner no one place to launch it.
        registry = (self.root / "experiments/registry.toml").read_text(encoding="utf-8")
        self.write(
            "experiments/registry.toml", registry + '[[experiment]]\nid = "L900"\npath = "x/L900-x"\nstatus = "running"\n'
        )
        self.assertEqual(validate(), (1, "ERROR: L900: registered 2 times; an experiment is registered once\n"))

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

    def test_a_listed_experiments_run_holds_a_lock_and_leaves_a_reservation(self):
        # Two runs at once could both find seed 17 not yet run; and a run
        # stopped before its record is written left nothing. So a run holds
        # a lock, and reserves its record before the command starts.
        self.preregister("running")
        lock = Path(git(self.root, "rev-parse", "--absolute-git-dir")) / "ptr-run-L900.lock"
        lock.write_text("4242\n", encoding="utf-8")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"another run of L900 holds {lock}", stderr)
        lock.unlink()
        seen = []

        def during():
            # While the command runs, its record is reserved and the lock held.
            [reserved] = list(self.results.glob("run-*.json"))
            seen.append((json.loads(reserved.read_text(encoding="utf-8"))["status"], reserved.name, lock.exists()))

        status, records, stderr = self.run_seed(during)
        self.assertEqual((status, stderr), (0, ""))
        [(reserved_status, reserved_name, locked)] = seen
        [final] = list(self.results.glob("run-*.json"))
        self.assertEqual((reserved_status, locked, final.name), ("started", True, reserved_name))
        self.assertEqual(records[0]["status"], "completed")
        self.assertFalse(lock.exists())
        # A command that raises leaves the lock released and the reservation.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        lock = Path(git(self.root, "rev-parse", "--absolute-git-dir")) / "ptr-run-L900.lock"

        def dies():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_seed(dies)
        self.assertFalse(lock.exists())
        [reserved] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual((reserved["status"], reserved["seed"]), ("started", 17))
        # Committed, the reservation is a run of seed 17.
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "reservation")
        with mock.patch.object(mod, "ROOT", self.root):
            attempts = mod.attempts_directory(mod.git_directory(), self.results)
            self.assertEqual(len(mod.seed_runs(self.results, 17, attempts)), 1)
        status, records, stderr = self.run_seed()
        self.assertEqual(status, 2)
        self.assertIn("seed 17 of L900 already ran", stderr)

    def test_a_listed_experiments_prepare_holds_the_lock_on_its_runs(self):
        # A seed run watches every file of its results but its own record: a
        # prepare record written while its command ran would leave that run
        # unrecorded and its seed spent.
        self.preregister("prepared")
        lock = Path(git(self.root, "rev-parse", "--absolute-git-dir")) / "ptr-run-L900.lock"

        def prepare() -> tuple[int, str]:
            stderr = io.StringIO()
            with (
                mock.patch.object(mod, "ROOT", self.root),
                mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                return mod.prepare("L900"), stderr.getvalue()

        lock.write_text("4242\n", encoding="utf-8")
        status, stderr = prepare()
        self.assertEqual(status, 2)
        self.assertIn(f"another run of L900 holds {lock}", stderr)
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])
        lock.unlink()
        # Free, the lock is held while the record is written, and let go.
        held = []
        written = mod.write_json_exclusive

        def writing(path, record):
            held.append(lock.exists())
            return written(path, record)

        with mock.patch.object(mod, "write_json_exclusive", side_effect=writing):
            status, stderr = prepare()
        self.assertEqual((status, stderr, held, lock.exists()), (0, "", [True], False))

    def test_the_lock_on_an_experiments_runs_is_one_file_whatever_its_id(self):
        # No id format keeps a separator out of a listed experiment's id: the
        # lock's name encodes it, and a lock that cannot be made is refused.
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            lock = mod.run_lock("team/trial", directory)
            self.assertEqual(lock, directory / "ptr-run-team%2Ftrial.lock")
            self.assertEqual(lock.read_text(encoding="utf-8"), f"{os.getpid()}\n")
            self.assertIsNone(mod.run_lock("team/trial", directory))
            self.assertEqual(mod.run_lock("L900", directory), directory / "ptr-run-L900.lock")
            self.assertEqual(mod.run_lock("..", directory), directory / "ptr-run-...lock")
            self.assertIsNone(mod.run_lock("L901", directory / "missing"))
            (directory / "file").write_text("", encoding="utf-8")
            self.assertIsNone(mod.run_lock("L902", directory / "file"))
        self.assertIn(f"another run of team/trial holds {lock}", stderr.getvalue())
        self.assertIn(f"cannot take the lock on runs of L902 in {directory / 'file' / 'ptr-run-L902.lock'}", stderr.getvalue())
        self.assertIn(
            f"cannot take the lock on runs of L901 in {directory / 'missing' / 'ptr-run-L901.lock'}", stderr.getvalue()
        )

    def attempts(self) -> Path:
        """The directory in the temporary repository's git directory where
        the runner keeps its copies of L900's records."""
        with mock.patch.object(mod, "ROOT", self.root):
            return mod.attempts_directory(mod.git_directory(), self.results)

    def test_the_runs_of_a_clone_are_kept_in_git_s_own_directory_its_worktrees_share(self):
        # A worktree's runs lock and count as the clone's: a seed run in one
        # worktree has run in all of them.
        git_directory = Path(git(self.root, "rev-parse", "--absolute-git-dir")).resolve()
        self.assertEqual(self.attempts(), git_directory / "ptr-runs/experiments/x/L900-x/results")
        worktree = self.root / "worktree"
        git(self.root, "worktree", "add", "-q", "--detach", str(worktree), "HEAD")
        self.assertNotEqual(Path(git(worktree, "rev-parse", "--absolute-git-dir")).resolve(), git_directory)
        with mock.patch.object(mod, "ROOT", worktree):
            self.assertEqual(mod.git_directory(), git_directory)
            self.assertEqual(
                mod.attempts_directory(mod.git_directory(), worktree / "experiments/x/L900-x/results"),
                git_directory / "ptr-runs/experiments/x/L900-x/results",
            )
        # Where git cannot name its directory, a listed run runs nothing and
        # keeps no copy.
        refused = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal: not a git repository")
        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod.experiment_records, "git", return_value=refused),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertIsNone(mod.git_directory())
        self.assertEqual(stderr.getvalue(), "ERROR: cannot find git's directory: fatal: not a git repository\n")
        self.preregister("running")
        ran = []
        with mock.patch.object(mod, "git_directory", return_value=None):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertFalse((git_directory / "ptr-runs").exists())

    def test_a_listed_runs_record_outlasts_a_command_that_clears_its_results(self):
        # The command can write in its results: a reservation it removed or
        # rewrote while its runner was stopped would leave no trace that its
        # seed ran. A copy in git's own directory keeps one, and the next run
        # puts the record back before anything else.
        self.preregister("running")
        attempts = self.attempts()

        def clears_and_dies():
            shutil.rmtree(self.results)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_seed(clears_and_dies)
        self.assertFalse(self.results.exists())
        [kept] = list(attempts.glob("run-*.json"))
        self.assertEqual(json.loads(kept.read_text(encoding="utf-8"))["status"], "started")
        # The lock is released: the next run finds the copy.
        restored = f"experiments/x/L900-x/results/{kept.name}"
        status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertEqual(stderr, (
            f"ERROR: put back {restored} from the copies kept in {attempts}: records of runs at commits on HEAD's "
            "history, which the results directory held otherwise or not at all; commit them before the next run\n"
        ))
        self.assertEqual((self.root / restored).read_bytes(), kept.read_bytes())
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record put back")
        status, records, stderr = self.run_seed()
        self.assertEqual(status, 2)
        self.assertIn(f"seed 17 of L900 already ran ({restored})", stderr)

        # A reservation rewritten to say its command never started is put
        # back as the copy holds it.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        attempts = self.attempts()

        def rewrites_and_dies():
            [reserved] = list(self.results.glob("run-*.json"))
            record = json.loads(reserved.read_text(encoding="utf-8"))
            reserved.write_text(json.dumps({**record, "status": "failed-to-launch"}), encoding="utf-8")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_seed(rewrites_and_dies)
        status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn("ERROR: put back experiments/x/L900-x/results/run-", stderr)
        # Put back as it was, the record is left as it is.
        with mock.patch.object(mod, "ROOT", self.root):
            self.assertEqual(mod.restore_records(self.results, attempts), [])
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record put back")
        status, records, stderr = self.run_seed()
        self.assertEqual(status, 2)
        self.assertIn("seed 17 of L900 already ran", stderr)
        # Committed as the command rewrote it, before any run could put it
        # back, the reservation is the gate's; the copy alone says the seed
        # ran, and the seed stays run.
        for rewritten in ({"status": "failed-to-launch"}, {"seed": 29}):
            with self.subTest(rewritten=rewritten):
                self.tearDown()
                self.setUp()
                self.preregister("running")

                def rewrites_and_dies():
                    [reserved] = list(self.results.glob("run-*.json"))
                    record = json.loads(reserved.read_text(encoding="utf-8"))
                    reserved.write_text(json.dumps({**record, **rewritten}), encoding="utf-8")
                    raise KeyboardInterrupt

                with self.assertRaises(KeyboardInterrupt):
                    self.run_seed(rewrites_and_dies)
                git(self.root, "add", "-A")
                git(self.root, "commit", "-q", "--no-verify", "-m", "rewritten reservation")
                status, records, stderr = self.run_seed()
                self.assertEqual(status, 2)
                self.assertNotIn("put back", stderr)
                self.assertIn("seed 17 of L900 already ran", stderr)

        # A runner stopped between writing the whole record to the copy and
        # to the results leaves the whole record in the copy.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        attempts = self.attempts()
        replacing = mod.write_json_replacing

        def stops_before_the_results(path: Path, record: dict) -> None:
            if path.parent == self.results:
                raise KeyboardInterrupt
            replacing(path, record)

        with mock.patch.object(mod, "write_json_replacing", side_effect=stops_before_the_results):
            with self.assertRaises(KeyboardInterrupt):
                self.run_seed()
        [kept] = list(attempts.glob("run-*.json"))
        self.assertEqual(json.loads(kept.read_text(encoding="utf-8"))["status"], "completed")
        self.assertEqual(
            [json.loads(path.read_text(encoding="utf-8"))["status"] for path in self.results.glob("run-*.json")],
            ["started"],
        )
        # A record HEAD holds is the gate's to keep as committed, even one
        # that differs from its copy; its seed has run all the same.
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "reservation")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn("seed 17 of L900 already ran", stderr)
        self.assertNotIn("put back", stderr)
        # Uncommitted, the reservation gives way to the whole record.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        with mock.patch.object(mod, "write_json_replacing", side_effect=stops_before_the_results):
            with self.assertRaises(KeyboardInterrupt):
                self.run_seed()
        status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["completed"]))
        self.assertIn("ERROR: put back", stderr)

    def test_a_reservation_that_cannot_be_written_leaves_the_seed_unrun(self):
        self.preregister("running")
        attempts = self.attempts()
        exclusive = mod.write_json_exclusive

        def fails_in(directory: Path):
            def write(path: Path, record: dict) -> None:
                if path.parent == directory:
                    raise OSError(errno.ENOSPC, "No space left on device")
                exclusive(path, record)

            return write

        # Its copy cannot be written: nothing is.
        with mock.patch.object(mod, "write_json_exclusive", side_effect=fails_in(attempts)):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, records, list(attempts.glob("run-*.json"))), (2, [], []))
        self.assertIn("ERROR: cannot reserve experiments/x/L900-x/results/run-", stderr)
        self.assertIn(f" in {attempts}/run-", stderr)
        self.assertIn("No space left on device", stderr)
        # The reservation in the results cannot be written: its copy goes.
        with mock.patch.object(mod, "write_json_exclusive", side_effect=fails_in(self.results)):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, records, list(attempts.glob("run-*.json"))), (2, [], []))
        self.assertIn("No space left on device", stderr)
        self.assertNotIn("cannot be removed", stderr)
        # A copy that cannot be removed then is named, since it says the
        # seed ran.
        unlink = Path.unlink

        def stuck(path: Path, *args, **kwargs) -> None:
            if path.parent == attempts and path.name.startswith("run-"):
                raise PermissionError(errno.EACCES, "Permission denied")
            unlink(path, *args, **kwargs)

        with (
            mock.patch.object(mod, "write_json_exclusive", side_effect=fails_in(self.results)),
            mock.patch.object(Path, "unlink", autospec=True, side_effect=stuck),
        ):
            status, records, stderr = self.run_seed()
        [kept] = list(attempts.glob("run-*.json"))
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"and {kept}, which says seed 17 ran, cannot be removed: [Errno 13] Permission denied", stderr)
        kept.unlink()
        # With both written, the seed runs.
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
        [kept] = list(attempts.glob("run-*.json"))
        self.assertEqual(kept.read_bytes(), next(self.results.glob("run-*.json")).read_bytes())

    def test_copies_that_cannot_be_read_or_put_back_refuse_the_run(self):
        self.preregister("running")
        attempts = self.attempts()
        # A copy that is no file cannot be read.
        (attempts / "run-kept.json").mkdir(parents=True)
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"ERROR: cannot put back the records of L900's runs kept in {attempts}: [Errno 21]", stderr)
        (attempts / "run-kept.json").rmdir()
        # Nor can the runner put a record back where HEAD cannot be read.
        kept_text = json.dumps({"git_sha": self.head}) + "\n"
        (attempts / "run-kept.json").write_text(kept_text, encoding="utf-8")
        tree_entry = mod.check_research_gates.tree_entry

        def unreadable(root: Path, commit: str, relative: str):
            if relative == "experiments/x/L900-x/results/run-kept.json":
                raise mod.check_research_gates.HistoryUnreadable("cannot read HEAD")
            return tree_entry(root, commit, relative)

        with mock.patch.object(mod.check_research_gates, "tree_entry", side_effect=unreadable):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"ERROR: cannot put back the records of L900's runs kept in {attempts}: cannot read HEAD", stderr)
        # A directory or a link in the record's place the gate refuses
        # first; the runner would not write into or through either.
        (self.results / "run-kept.json").mkdir()
        status, calls, stderr = self.launch_listed({})
        self.assertEqual((status, calls), (2, []))
        self.assertIn("L900: results/run-kept.json is not a regular file reached through no symlink", stderr)
        with mock.patch.object(mod, "ROOT", self.root), self.assertRaises(IsADirectoryError):
            mod.restore_records(self.results, attempts)
        self.assertEqual(list((self.results / "run-kept.json").iterdir()), [])
        (self.results / "run-kept.json").rmdir()
        # A link is replaced, even one to a file holding the copy, and the
        # file it pointed to is left as it was.
        outside = self.root / "outside.json"
        (self.results / "run-kept.json").symlink_to(outside)
        status, calls, stderr = self.launch_listed({})
        self.assertEqual((status, calls), (2, []))
        self.assertIn("L900: results/run-kept.json is not a regular file reached through no symlink", stderr)
        for held in (None, kept_text, "[]\n"):
            with self.subTest(held=held):
                outside.unlink(missing_ok=True)
                if held is not None:
                    outside.write_text(held, encoding="utf-8")
                (self.results / "run-kept.json").unlink(missing_ok=True)
                (self.results / "run-kept.json").symlink_to(outside)
                with mock.patch.object(mod, "ROOT", self.root):
                    self.assertEqual(
                        mod.restore_records(self.results, attempts), ["experiments/x/L900-x/results/run-kept.json"]
                    )
                self.assertFalse((self.results / "run-kept.json").is_symlink())
                self.assertEqual((self.results / "run-kept.json").read_text(encoding="utf-8"), kept_text)
                self.assertEqual(outside.read_text(encoding="utf-8") if outside.exists() else None, held)

    def test_a_command_that_cannot_start_leaves_its_seed_unrun(self):
        # A command stopped before it started, for want of a directory for
        # its bytecode cache, saw no outcome.
        self.preregister("running")
        attempts = self.attempts()
        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod.tempfile, "tempdir", str(self.root / "no-such-directory")),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=17)
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        [kept] = [json.loads(path.read_text(encoding="utf-8")) for path in attempts.glob("run-*.json")]
        self.assertEqual((status, stderr.getvalue()), (127, ""))
        self.assertEqual((record["status"], record["exit_code"]), ("failed-to-launch", None))
        self.assertTrue(record["launch_error"].startswith("FileNotFoundError: "), record["launch_error"])
        self.assertEqual(kept, record)
        lock = Path(git(self.root, "rev-parse", "--absolute-git-dir")) / "ptr-run-L900.lock"
        self.assertFalse(lock.exists())
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "failed to launch")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(sorted(record["status"] for record in records), ["completed", "failed-to-launch"])

        # A command that could not start ran no code: its record is written
        # whatever the tree did meanwhile, listed or not.
        for listed in (True, False):
            with self.subTest(listed=listed):
                self.tearDown()
                self.setUp()
                if listed:
                    self.preregister("running")

                def edits():
                    self.write("src/lib.rs", "pub fn g() {}\n")

                status, records, stderr = self.run_seed(edits, launch_error="FileNotFoundError: bench")
                self.assertEqual((status, stderr), (127, ""))
                self.assertEqual([record["status"] for record in records], ["failed-to-launch"])
                self.assertEqual(
                    [json.loads(path.read_text(encoding="utf-8"))["status"] for path in self.attempts().glob("run-*.json")],
                    ["failed-to-launch"] if listed else [],
                )

        # With its results directory replaced meanwhile, the copy says the
        # command never started, and comes back once the directory does.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        attempts = self.attempts()

        def replaces_the_results():
            shutil.rmtree(self.results)
            self.results.write_text("", encoding="utf-8")

        status, _, stderr = self.run_seed(replaces_the_results, launch_error="FileNotFoundError: bench")
        self.assertEqual(status, 2)
        self.assertEqual(stderr, (
            "ERROR: results directory experiments/x/L900-x/results is not a directory\n"
            "ERROR: not recording the run (exit status None): its results directory changed while it ran\n"
        ))
        [kept] = list(attempts.glob("run-*.json"))
        self.assertEqual(json.loads(kept.read_text(encoding="utf-8"))["status"], "failed-to-launch")
        self.results.unlink()
        status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["failed-to-launch"]))
        self.assertIn("ERROR: put back", stderr)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record put back")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        # A command that started and whose results were replaced leaves its
        # reservation as the record that its seed ran, and its copy says so.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        status, _, stderr = self.run_seed(replaces_the_results)
        self.assertEqual(status, 2)
        self.assertIn("ERROR: not recording the run (exit status 0)", stderr)
        self.assertIn("stays as the record that seed 17 ran", stderr)
        [kept] = list(self.attempts().glob("run-*.json"))
        self.assertEqual(json.loads(kept.read_text(encoding="utf-8"))["status"], "started")
        # So it does where the watch saw nothing, the results directory
        # alone having changed.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        with mock.patch.object(mod.experiment_records.ProvenanceWatch, "changes", return_value=[]):
            status, _, stderr = self.run_seed(replaces_the_results)
        [kept] = list(self.attempts().glob("run-*.json"))
        self.assertEqual(status, 2)
        self.assertEqual(stderr, (
            "ERROR: results directory experiments/x/L900-x/results is not a directory\n"
            "ERROR: not recording the run (exit status 0): its results directory changed while it ran; "
            f"experiments/x/L900-x/results/{kept.name} stays as the record that seed 17 ran\n"
        ))
        self.assertEqual(json.loads(kept.read_text(encoding="utf-8"))["status"], "started")

    def run_bench(self) -> tuple[int, dict, dict]:
        """Run seed 17 of L900 through a real `bench` found on `PATH`, in a
        runner environment holding settings that load code and a credential;
        returns the status, the record and the environment `bench` saw."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        bench = Path(directory.name) / "bench"
        bench.write_text("#!/usr/bin/env python3\nimport json, os\nprint(json.dumps(dict(os.environ)))\n", encoding="utf-8")
        bench.chmod(0o755)
        self.bench = bench
        runner = {
            "PATH": f"{directory.name}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": "/elsewhere",
            "LD_LIBRARY_PATH": "/elsewhere",
            "RUSTC_WRAPPER": "/elsewhere/wrapper",
            "RUSTUP_TOOLCHAIN": "nightly",
            "GH_TOKEN": "credential",
            "TMPDIR": tempfile.gettempdir(),
        }
        with (
            mock.patch.dict(os.environ, runner),
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            allowed = mod.command_environment()
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=17)
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual(allowed["PATH"], runner["PATH"])
        self.allowed = allowed
        return status, record, json.loads(record["stdout"])

    def test_a_listed_command_runs_in_the_environment_the_runner_allows(self):
        # What the runner's environment sets could load code the commit does
        # not hold (PYTHONPATH, LD_LIBRARY_PATH, RUSTC_WRAPPER, a toolchain
        # other than the one it pins) or hand the command a credential: a
        # listed experiment's command sees only the allowed variables, and
        # its record names them and the program it started.
        self.preregister("running")
        status, record, seen = self.run_bench()
        self.assertEqual((status, record["status"]), (0, "completed"))
        self.assertLessEqual(
            set(seen) - {"PYTHONPYCACHEPREFIX", "CARGO_TARGET_DIR"}, {*mod.COMMAND_ENVIRONMENT, *mod.FIXED_ENVIRONMENT}
        )
        # Cargo builds into a fresh directory outside the repository, beside
        # Python's bytecode cache, both removed once the command has run: a
        # directory of the experiment's name in the command's temporary
        # directory, the same for every seed, which the record names.
        scratch = os.path.join(self.allowed["TMPDIR"], "ptr-run-L900")
        self.assertEqual(
            (seen["PYTHONPYCACHEPREFIX"], seen["CARGO_TARGET_DIR"]), (scratch, os.path.join(scratch, "cargo-target"))
        )
        self.assertFalse(Path(seen["CARGO_TARGET_DIR"]).is_relative_to(self.root))
        self.assertFalse(Path(scratch).exists())
        # Python reads no package from the user's own site directory.
        self.assertEqual(seen["PYTHONNOUSERSITE"], "1")
        for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "RUSTC_WRAPPER", "RUSTUP_TOOLCHAIN", "GH_TOKEN"):
            self.assertNotIn(name, seen)
        self.assertEqual(record["environment"], {
            **self.allowed, "PYTHONPYCACHEPREFIX": scratch, "CARGO_TARGET_DIR": os.path.join(scratch, "cargo-target"),
        })
        self.assertEqual({name: seen[name] for name in record["environment"]}, record["environment"])
        # A program found first on the PATH, wherever it lies, is named with
        # its content.
        self.assertEqual(record["executable"], {
            "path": str(self.bench.resolve()),
            "sha256": hashlib.sha256(self.bench.read_bytes()).hexdigest(),
        })
        # An unlisted experiment's command runs in the runner's environment,
        # as it did.
        self.tearDown()
        self.setUp()
        status, record, seen = self.run_bench()
        self.assertEqual((status, seen["PYTHONPATH"], seen["GH_TOKEN"]), (0, "/elsewhere", "credential"))
        self.assertEqual(seen.get("CARGO_TARGET_DIR"), os.environ.get("CARGO_TARGET_DIR"))
        self.assertNotIn("environment", record)
        self.assertNotIn("executable", record)

    def test_the_program_a_command_starts_is_found_as_the_process_finds_it(self):
        tools = self.root / "tools"
        tools.mkdir()
        script = tools / "run.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(0o755)
        (tools / "link.sh").symlink_to("run.sh")
        (tools / "plain.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        found = {"path": str(script.resolve()), "sha256": digest}
        missing = {"path": None, "sha256": None}
        with mock.patch.object(mod, "ROOT", self.root):
            # A name with a slash is read from the repository's root, through
            # any link, and must be an executable file.
            self.assertEqual(mod.resolved_executable(["tools/run.sh", "17"], {}), found)
            self.assertEqual(mod.resolved_executable(["tools/link.sh"], {}), found)
            self.assertEqual(mod.resolved_executable(["tools/plain.sh"], {}), missing)
            self.assertEqual(mod.resolved_executable(["tools/absent.sh"], {}), missing)
            # A name without one is looked up on the environment's PATH, and
            # on the default path when it sets none.
            self.assertEqual(mod.resolved_executable(["run.sh"], {"PATH": str(tools)}), found)
            self.assertEqual(mod.resolved_executable(["run.sh"], {"PATH": "/nowhere"}), missing)
            self.assertEqual(mod.resolved_executable(["plain.sh"], {"PATH": str(tools)}), missing)
            shell = shutil.which("sh", path=os.defpath)
            self.assertIsNotNone(shell)
            self.assertEqual(mod.resolved_executable(["sh"], {})["path"], os.path.realpath(shell))
        # A program that cannot be read is found, and named by nothing else.
        with mock.patch.object(mod, "ROOT", self.root), mock.patch.object(Path, "read_bytes", side_effect=PermissionError):
            self.assertEqual(mod.resolved_executable(["tools/run.sh"], {}), {"path": str(script.resolve()), "sha256": None})
        # So is one that changed while it was read, which holds no one
        # content; its stamp from before the read is kept, and a later
        # reading of the same file does not replace it.
        reading = Path.read_bytes

        def rewritten(path):
            content = reading(path)
            if path == script.resolve():
                path.write_text("#!/bin/sh\necho another\n", encoding="utf-8")
            return content

        before = mod.experiment_records.file_stamp(script.resolve())
        stamps = {}
        with mock.patch.object(mod, "ROOT", self.root), mock.patch.object(Path, "read_bytes", rewritten):
            self.assertEqual(
                mod.resolved_executable(["tools/run.sh"], {}, stamps), {"path": str(script.resolve()), "sha256": None}
            )
        self.assertEqual(stamps, {str(script.resolve()): before})
        with mock.patch.object(mod, "ROOT", self.root):
            self.assertEqual(mod.resolved_executable(["tools/run.sh"], {}, stamps)["path"], str(script.resolve()))
        self.assertEqual(stamps, {str(script.resolve()): before})
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        # A listed run refuses one, the program it starts or a tool it builds
        # with, before anything runs: named by its path alone, it could be
        # replaced between seeds while every record named the same. (The
        # tree holds no link a listed run would refuse first.)
        (tools / "link.sh").unlink()
        self.preregister("running")
        unread = {"path": str(script.resolve()), "sha256": None}
        for patched, value in (
            ("resolved_executable", unread),
            ("toolchain", {"rustc": unread, "cargo": {"path": None, "sha256": None}}),
        ):
            with self.subTest(patched=patched), mock.patch.object(mod, patched, return_value=value):
                ran = []
                status, records, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, records, ran, list(self.attempts().glob("run-*.json"))), (2, [], [], []))
                self.assertEqual(stderr, (
                    f"ERROR: refusing to run L900: {script.resolve()} cannot be read, or changed while it was read, so "
                    "its record could not name by its content a program the run starts or builds with; make it "
                    "readable, and leave it unchanged, for the run\n"
                ))
        # The allowed environment is what the runner's sets of the allowed
        # names, and nothing else.
        with mock.patch.dict(os.environ, {"PATH": "/bin", "HOME": "/home/runner", "GH_TOKEN": "x"}, clear=True):
            self.assertEqual(
                mod.command_environment(), {"PATH": "/bin", "HOME": "/home/runner", "PYTHONNOUSERSITE": "1"}
            )
        # A directory named by a relative path is one each program resolves
        # from wherever it starts: the runner from its own directory, the
        # command from the root. A listed run is refused while one is.
        self.assertEqual(mod.relative_directories({"PATH": "/bin", "HOME": "/home/runner"}), [])
        self.assertEqual(mod.relative_directories({"PATH": os.pathsep.join(("/bin", ""))}), ["PATH"])
        self.assertEqual(
            mod.relative_directories({
                "PATH": os.pathsep.join(("tools", "/bin")), "HOME": "home", "TMPDIR": "/tmp", "CARGO_HOME": "../cargo",
                "RUSTUP_HOME": "/rustup",
            }),
            ["PATH", "HOME", "CARGO_HOME"],
        )
        for name, value in (
            ("PATH", os.pathsep.join(("tools", os.environ.get("PATH", "/bin")))),
            ("PATH", os.environ.get("PATH", "/bin") + os.pathsep),
            ("TMPDIR", "tmp"),
            ("RUSTUP_HOME", "../rustup"),
        ):
            with self.subTest(name=name, value=value), mock.patch.dict(os.environ, {name: value}):
                ran = []
                status, records, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, records, ran, list(self.attempts().glob("run-*.json"))), (2, [], [], []))
                self.assertEqual(stderr, (
                    f"ERROR: refusing to run L900: {name} names a directory by a relative path, which each program "
                    "resolves from wherever it starts, so the record could not say which one the run used; set "
                    "absolute directories for the run\n"
                ))

    def test_a_listed_run_builds_in_a_directory_its_record_names_the_same_for_every_seed(self):
        # The command can read the variables naming where it builds and
        # caches, so they are the same for every seed and the record names
        # them; a directory left there by a run that died refuses the run
        # before anything is written.
        self.preregister("running", seeds="[17, 29]")
        temporary = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scratch = temporary / "ptr-run-L900"
        with mock.patch.dict(os.environ, {"TMPDIR": str(temporary)}):
            scratch.mkdir()
            status, records, stderr = self.run_seed()
            self.assertEqual((status, records, list(self.attempts().glob("run-*.json"))), (2, [], []))
            self.assertEqual(stderr, (
                f"ERROR: refusing to run L900: {scratch} exists, left by a run that died or held by one in progress; "
                "a listed run builds in a fresh directory there, so remove it once no run of L900 is in progress\n"
            ))
            scratch.rmdir()
            status, records, stderr = self.run_seed()
            self.assertEqual((status, stderr), (0, ""))
            git(self.root, "add", "-A")
            git(self.root, "commit", "-q", "--no-verify", "-m", "seed 17")
            status, records, stderr = self.run_seed(seed=29)
            self.assertEqual((status, stderr), (0, ""))
        first, second = (record["environment"] for record in records)
        self.assertEqual(first, second)
        self.assertEqual(
            (first["TMPDIR"], first["PYTHONPYCACHEPREFIX"], first["CARGO_TARGET_DIR"]),
            (str(temporary), str(scratch), str(scratch / "cargo-target")),
        )

    def test_a_program_the_record_names_changed_while_the_command_ran_leaves_it_unrecorded(self):
        # The record names the program the run starts and the tools it builds
        # with by the digests the launch read: one replaced for the run and
        # put back afterwards would leave every seed's record naming the same
        # while another program ran. Every write or replacement moves the
        # file's stamp, which ordinary tools cannot set back.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name in ("bench", "rustc"):
            (outside / name).write_text(f"#!/bin/sh\necho {name}\n", encoding="utf-8")
            (outside / name).chmod(0o755)
        self.preregister("running")

        def swapped_and_put_back():
            # Replaced by another file, and the original linked back.
            os.link(outside / "bench", outside / "kept")
            (outside / "other").write_text("#!/bin/sh\necho other\n", encoding="utf-8")
            (outside / "other").chmod(0o755)
            os.replace(outside / "other", outside / "bench")
            os.replace(outside / "kept", outside / "bench")

        def rewritten_and_put_back():
            # Rewritten in place with as many bytes, and its time set back.
            before = (outside / "rustc").stat()
            (outside / "rustc").write_text("#!/bin/sh\necho other\n", encoding="utf-8")
            (outside / "rustc").write_text("#!/bin/sh\necho rustc\n", encoding="utf-8")
            os.utime(outside / "rustc", ns=(before.st_atime_ns, before.st_mtime_ns))

        # Each change lands on a later tick of the clock than the file's last.
        time.sleep(0.05)
        search = os.pathsep.join((str(outside), os.environ.get("PATH", os.defpath)))
        for during, program in ((swapped_and_put_back, "bench"), (rewritten_and_put_back, "rustc")):
            with self.subTest(program=program), mock.patch.dict(os.environ, {"PATH": search}):
                content = (outside / program).read_bytes()
                status, records, stderr = self.run_seed(during)
                self.assertEqual((outside / program).read_bytes(), content)
                self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
                self.assertEqual(records[0]["executable"]["path"], str((outside / "bench").resolve()))
                self.assertEqual(records[0]["toolchain"]["rustc"]["path"], str((outside / "rustc").resolve()))
                self.assertIn(
                    f"{(outside / program).resolve()} changed since the launch read it, so the program that ran may "
                    "not be the one its digest in the record names",
                    stderr,
                )
                # The seed ran; the next case runs it anew.
                for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
                    record.unlink()
        # Left alone, the programs leave the run recorded.
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_a_listed_run_starts_the_program_its_record_names(self):
        # The process that starts a command looks its name up on the PATH
        # again: a program put earlier on it once the launch had read the
        # one the record names would run in its place.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))
        first, second = outside / "first", outside / "second"
        first.mkdir()
        second.mkdir()
        (second / "bench").write_text("#!/bin/sh\necho second\n", encoding="utf-8")
        (second / "bench").chmod(0o755)
        self.preregister("running")
        started = mod.execute_command

        def put_first_then_start(command, environment=None, program=None, scratch=None):
            (first / "bench").write_text("#!/bin/sh\necho first\n", encoding="utf-8")
            (first / "bench").chmod(0o755)
            return started(command, environment, program, scratch)

        stderr = io.StringIO()
        search = os.pathsep.join((str(first), str(second), os.environ.get("PATH", os.defpath)))
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "execute_command", side_effect=put_first_then_start),
            mock.patch.dict(os.environ, {"PATH": search}),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=17)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.results.glob("run-*.json"))]
        self.assertEqual((status, stderr.getvalue()), (0, ""))
        self.assertEqual(
            [(record["executable"]["path"], record["stdout"]) for record in records],
            [(str((second / "bench").resolve()), "second\n")],
        )

    def run_started(self, during=None, seed: int = 17) -> tuple[int, list[dict], str]:
        """Run `seed` of L900 through the real `execute_command`, calling
        `during` just before the command starts; returns the exit status, the
        records written and what was printed to stderr."""
        started = mod.execute_command

        def start(*arguments):
            if during is not None:
                during()
            return started(*arguments)

        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "execute_command", side_effect=start),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=seed)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.results.glob("run-*.json"))]
        return status, records, stderr.getvalue()

    def test_a_listed_run_starts_its_program_by_the_name_it_was_found_by(self):
        # A script reads the name it was started by ($0), and one reached
        # through a link may act on it, as xzfgrep, a link to xzgrep, does:
        # the command starts the name the launch found, and every link on
        # the way to the file its record names is stamped with it.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        (outside / "impl.sh").write_text('#!/bin/sh\ncase "$0" in *bench) echo "as bench $0";; *) echo "as other $0";; esac\n', encoding="utf-8")
        (outside / "impl.sh").chmod(0o755)
        (outside / "middle").symlink_to("impl.sh")
        (outside / "bench").symlink_to("middle")
        self.preregister("running")
        with mock.patch.dict(os.environ, {"PATH": os.pathsep.join((str(outside), os.environ["PATH"]))}):
            status, records, stderr = self.run_started()
            self.assertEqual((status, stderr), (0, ""))
            self.assertEqual(
                [(record["stdout"], record["executable"]["path"]) for record in records],
                [(f"as bench {outside / 'bench'}\n", str(outside / "impl.sh"))],
            )
            for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
                record.unlink()

            def turned_and_back():
                # A link on the way pointed elsewhere for the run, and back
                # after it.
                (outside / "turn").symlink_to("other.sh")
                os.replace(outside / "turn", outside / "middle")
                (outside / "turn").symlink_to("impl.sh")
                os.replace(outside / "turn", outside / "middle")

            status, records, stderr = self.run_started(turned_and_back)
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"{outside / 'middle'} changed since the launch read it", stderr)
        # A name with a slash is started as it stands, from the root.
        self.tearDown()
        self.setUp()
        self.write("tools/run.sh", '#!/bin/sh\necho "$0"\n')
        (self.root / "tools/run.sh").chmod(0o755)
        self.preregister("running", entrypoint="tools/run.sh <seed>")
        status, records, stderr = self.run_started()
        self.assertEqual((status, stderr, [record["stdout"] for record in records]), (0, "", ["tools/run.sh\n"]))

    def test_a_listed_run_whose_program_the_launch_did_not_find_starts_nothing(self):
        # The process that starts the command would look its name up again,
        # and could find a program put there meanwhile, which no record names:
        # the command fails to launch, and its seed may run once the program
        # is there.
        self.preregister("running", entrypoint="no-such-program <seed>")
        calls = []
        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "execute_command", side_effect=lambda *arguments: calls.append(arguments)),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=17)
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual((status, calls, stderr.getvalue()), (127, [], ""))
        self.assertEqual(
            (record["status"], record["exit_code"], record["launch_error"], record["executable"]),
            ("failed-to-launch", None, "FileNotFoundError: the launch found no program 'no-such-program' to start",
             {"path": None, "sha256": None}),
        )
        with mock.patch.object(mod, "ROOT", self.root):
            self.assertEqual(mod.seed_runs(self.results, 17, self.attempts()), [])

    def test_a_copy_from_another_line_of_history_is_not_put_back_but_its_seed_has_run(self):
        # A run on another branch, or another worktree's, belongs to that
        # history, whose record the gate here would refuse: its copy stays
        # in git's own directory and still counts its seed as run.
        self.preregister("running")
        base = git(self.root, "rev-parse", "--abbrev-ref", "HEAD")
        git(self.root, "switch", "-q", "-c", "elsewhere")
        git(self.root, "commit", "-q", "--no-verify", "--allow-empty", "-m", "elsewhere")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record elsewhere")
        git(self.root, "switch", "-q", base)
        self.assertEqual(list(self.results.glob("run-*.json")), [])
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertNotIn("put back", stderr)
        self.assertIn("ERROR: seed 17 of L900 already ran (experiments/x/L900-x/results/run-", stderr)
        self.assertEqual(mod.check_research_gates.launch_errors(self.root, "L900"), [])
        # Merged, the record is HEAD's, and the gate keeps it.
        git(self.root, "merge", "-q", "--no-edit", "elsewhere")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["completed"]))
        self.assertNotIn("put back", stderr)
        self.assertIn("ERROR: seed 17 of L900 already ran", stderr)

    def test_a_copy_git_s_own_directory_cannot_hold_leaves_the_record_in_the_results(self):
        # A full disk under .git: the copy cannot hold the whole record, so it
        # goes, and the record the command's outcome is in is written all
        # the same, to be committed before the next run.
        self.preregister("running")
        attempts = self.attempts()
        replacing = mod.write_json_replacing

        def full_in(directory: Path):
            def write(path: Path, record: dict) -> None:
                if path.parent == directory:
                    raise OSError(errno.ENOSPC, "No space left on device")
                replacing(path, record)

            return write

        with mock.patch.object(mod, "write_json_replacing", side_effect=full_in(attempts)):
            status, records, stderr = self.run_seed()
        [out] = [path.relative_to(self.root).as_posix() for path in self.results.glob("run-*.json")]
        self.assertEqual((status, [record["status"] for record in records]), (0, ["completed"]))
        self.assertEqual(list(attempts.glob("run-*.json")), [])
        self.assertIn(f"ERROR: cannot write the record's copy {attempts}/run-", stderr)
        self.assertIn(f"[Errno 28] No space left on device; commit {out} before the next run", stderr)
        # A copy that can be neither written nor removed is named: the next
        # run would put its reservation back over the record.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        attempts = self.attempts()
        unlink = Path.unlink

        def stuck(path: Path, *args, **kwargs) -> None:
            if path.parent == attempts and path.name.startswith("run-"):
                raise PermissionError(errno.EACCES, "Permission denied")
            unlink(path, *args, **kwargs)

        with (
            mock.patch.object(mod, "write_json_replacing", side_effect=full_in(attempts)),
            mock.patch.object(Path, "unlink", autospec=True, side_effect=stuck),
        ):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, [record["status"] for record in records]), (0, ["completed"]))
        self.assertIn("; nor remove it, so the next run would put its reservation back: [Errno 13]", stderr)
        # A command that could not start, whose record cannot be written,
        # ran nothing: its reservation and its copy go, and its seed runs.
        for stuck_too in (False, True):
            with self.subTest(stuck_too=stuck_too):
                self.tearDown()
                self.setUp()
                self.preregister("running")
                attempts = self.attempts()
                with (
                    mock.patch.object(mod, "write_json_replacing", side_effect=full_in(attempts)),
                    mock.patch.object(Path, "unlink", autospec=True, side_effect=stuck if stuck_too else unlink),
                ):
                    status, records, stderr = self.run_seed(launch_error="FileNotFoundError: bench")
                self.assertEqual(status, 2)
                self.assertIn(
                    "ERROR: cannot write the record of seed 17's launch, which failed: [Errno 28] No space left on device",
                    stderr,
                )
                if stuck_too:
                    self.assertIn("which says seed 17 ran, cannot be removed: [Errno 13] Permission denied", stderr)
                    continue
                self.assertEqual((records, list(attempts.glob("run-*.json"))), ([], []))
                status, records, stderr = self.run_seed()
                self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_a_listed_run_watches_the_whole_repository(self):
        # A listed command may run or read any file of the repository, a
        # script under scripts/ say: every one must be HEAD's before and
        # while it runs, but the record the run writes.
        self.preregister("running")
        self.write("scripts/harness.py", "print('unrecorded')\n")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn("commit or remove scripts/harness.py", stderr)
        stderr_prepare = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr_prepare),
        ):
            self.assertEqual(mod.prepare("L900"), 2)
        self.assertIn("commit or remove scripts/harness.py", stderr_prepare.getvalue())
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "harness")

        def edit_and_restore():
            self.write("scripts/harness.py", "print('changed')\n")
            self.write("scripts/harness.py", "print('unrecorded')\n")

        status, records, stderr = self.run_seed(edit_and_restore)
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn("scripts/harness.py changed on disk", stderr)
        # What git ignores is no file of the commit, and the watch does not
        # see it: a listed run starts only from a checkout that holds none,
        # a build's output (target/) or a virtual environment (.venv/) say.
        for ignored in ("scripts/__pycache__/harness.cpython-311.pyc", "target/release/harness", ".venv/bin/python"):
            with self.subTest(ignored=ignored):
                self.tearDown()
                self.setUp()
                self.write(".gitignore", "__pycache__/\ntarget/\n.venv/\n")
                self.preregister("running")
                self.write(ignored, "")
                shown = ignored.split("/")[0] + "/" if not ignored.startswith("scripts/") else "scripts/__pycache__/"
                status, records, stderr = self.run_seed()
                self.assertEqual((status, records), (2, []))
                self.assertTrue(stderr.startswith("ERROR: refusing to run L900 while the repository holds "), stderr)
                self.assertIn(shown, stderr)
                self.assertTrue(stderr.endswith(
                    ", which git ignores and its command could run or read unrecorded; a listed experiment runs from a "
                    "checkout that holds no such file (`git clean -ndX` lists them; a fresh worktree holds none)\n"
                ), stderr)
                shutil.rmtree(self.root / shown)
                status, records, stderr = self.run_seed()
                self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
        # One put there after the launch looked, and still there when the
        # command has ended, is one it could have run or read: the run is
        # not recorded, and its reservation stays.
        self.tearDown()
        self.setUp()
        self.write(".gitignore", "__pycache__/\ntarget/\n")
        self.preregister("running")
        status, records, stderr = self.run_seed(lambda: self.write("target/release/helper", ""))
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(
            "the repository holds target/, which git ignores and the command could have run or read unrecorded", stderr
        )
        # An unlisted experiment runs beside what git ignores, as it did.
        self.tearDown()
        self.setUp()
        self.write("scripts/__pycache__/harness.cpython-311.pyc", "")
        status, records, stderr = self.run_seed(lambda: self.write("target/release/helper", ""))
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
        # An unlisted experiment's watch is on its provenance files, as it was.
        self.tearDown()
        self.setUp()
        self.write("scripts/harness.py", "print('unrecorded')\n")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_cargo_configuration_outside_the_repository_refuses_a_listed_run(self):
        # Cargo reads its home's configuration and every .cargo directory
        # above the root, none of which the commit holds, and one could set
        # a rustc wrapper, flags or sources for the build.
        self.preregister("running")
        home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cargo_home = home / "cargo"
        cargo_home.mkdir()
        (cargo_home / "config.toml").write_text('[build]\nrustc-wrapper = "/elsewhere/wrapper"\n', encoding="utf-8")
        with mock.patch.dict(os.environ, {"CARGO_HOME": str(cargo_home)}):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, records, list(self.attempts().glob("run-*.json"))), (2, [], []))
        self.assertEqual(stderr, (
            f"ERROR: refusing to run L900: Cargo would read {cargo_home / 'config.toml'}, configuration outside the "
            "repository that could set a rustc wrapper, flags or sources the commit does not hold; move it aside for "
            "the run, or commit what it sets in the repository's .cargo/config.toml\n"
        ))
        # Without CARGO_HOME, Cargo's home is .cargo under HOME; the older
        # name counts as well.
        (home / ".cargo").mkdir()
        (home / ".cargo" / "config").write_text("[build]\n", encoding="utf-8")
        environment = {name: value for name, value in os.environ.items() if name != "CARGO_HOME"}
        with mock.patch.dict(os.environ, {**environment, "HOME": str(home)}, clear=True):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"Cargo would read {home / '.cargo' / 'config'}, configuration outside", stderr)
        # A .cargo directory above the root counts, the repository's own does
        # not.
        above = Path(self.enterContext(tempfile.TemporaryDirectory())) / "above"
        nested = above / "repository"
        (nested / ".cargo").mkdir(parents=True)
        (nested / ".cargo" / "config.toml").write_text("[build]\n", encoding="utf-8")
        (above / ".cargo").mkdir()
        (above / ".cargo" / "config.toml").write_text("[build]\n", encoding="utf-8")
        with mock.patch.object(mod, "ROOT", nested):
            self.assertEqual(
                mod.outside_cargo_configurations({"CARGO_HOME": str(cargo_home), "HOME": str(home)}),
                [str(above / ".cargo" / "config.toml"), str(cargo_home / "config.toml")],
            )
            (cargo_home / "config.toml").unlink()
            self.assertEqual(
                mod.outside_cargo_configurations({"HOME": str(home)}),
                [str(above / ".cargo" / "config.toml"), str(home / ".cargo" / "config")],
            )
            self.assertEqual(mod.outside_cargo_configurations({}), [str(above / ".cargo" / "config.toml")])
        # Cargo's home above the root is named once.
        with mock.patch.object(mod, "ROOT", home / "repository"):
            self.assertEqual(mod.outside_cargo_configurations({"HOME": str(home)}), [str(home / ".cargo" / "config")])
        # One put there while the command ran, after the launch looked, is
        # one its build could have read: the run is not recorded, and its
        # reservation stays.
        (home / ".cargo" / "config").unlink()
        (home / ".cargo").rmdir()
        late = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.dict(os.environ, {"CARGO_HOME": str(late)}):
            status, records, stderr = self.run_seed(
                lambda: (late / "config.toml").write_text('[build]\nrustc = "/elsewhere/rustc"\n', encoding="utf-8")
            )
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(
            f"Cargo would read {late / 'config.toml'}, configuration outside the repository its build could have read",
            stderr,
        )

    def test_the_toolchain_a_listed_run_builds_with_is_named_in_its_record(self):
        # rustup keeps toolchains outside the repository and may override the
        # one rust-toolchain.toml names: the record names rustc and cargo as
        # rustup resolves them from the root, with their content.
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name in ("pinned", "stable"):
            (tools / "toolchains" / name / "bin").mkdir(parents=True)
            for tool in ("rustc", "cargo"):
                (tools / "toolchains" / name / "bin" / tool).write_text(f"#!/bin/sh\necho {name} {tool}\n", encoding="utf-8")
                (tools / "toolchains" / name / "bin" / tool).chmod(0o755)
        toolchain = tools / "toolchains" / "pinned" / "bin"
        bin_directory = tools / "bin"
        bin_directory.mkdir()
        rustup = bin_directory / "rustup"
        rustup.write_text(
            '#!/bin/sh\n[ "$1" = which ] || exit 1\nshift\nname=pinned\n'
            'if [ "$1" = --toolchain ]; then name="$2"; shift 2; fi\n'
            f'path="{tools}/toolchains/$name/bin/$1"\n[ -e "$path" ] && echo "$path" && exit 0\nexit 1\n',
            encoding="utf-8",
        )
        rustup.chmod(0o755)
        # rustup installs its proxies beside itself, as links to it.
        for tool in ("rustc", "cargo"):
            (bin_directory / tool).symlink_to("rustup")

        def named(tool: str, name: str = "pinned") -> dict:
            path = tools / "toolchains" / name / "bin" / tool
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        with mock.patch.object(mod, "ROOT", self.root):
            environment = {"PATH": str(bin_directory)}
            self.assertEqual(mod.toolchain(environment, ["cargo", "run"]), {"rustc": named("rustc"), "cargo": named("cargo")})
            # Each tool rustup resolves is stamped as it is read, so a run
            # during which one changed goes unrecorded.
            stamps = {}
            mod.toolchain(environment, ["cargo", "run"], stamps)
            self.assertEqual(stamps, {
                named(tool)["path"]: mod.experiment_records.file_stamp(Path(named(tool)["path"]))
                for tool in ("rustc", "cargo")
            })
            # A proxy's first argument `+<toolchain>` names the toolchain that
            # builds, as rustup runs it.
            stable = {"rustc": named("rustc", "stable"), "cargo": named("cargo", "stable")}
            self.assertEqual(mod.toolchain(environment, ["cargo", "+stable", "run"]), stable)
            self.assertEqual(mod.toolchain(environment, ["/elsewhere/bin/cargo", "+stable", "run"]), stable)
            self.assertEqual(mod.toolchain(environment, ["rustc", "+stable", "lib.rs"]), stable)
            # So does `rustup run <toolchain>`, which rustup documents as the
            # same, past rustup's options and `+<toolchain>` and those of run.
            for command in (
                ["rustup", "run", "stable", "cargo", "build"],
                ["/elsewhere/bin/rustup", "-v", "run", "--install", "stable", "python3", "bench.py"],
                ["rustup", "+nightly", "run", "stable", "cargo"],
                # As Windows spells the programs, in any case and with .exe.
                ["rustup.exe", "run", "stable", "cargo", "build"],
                ["C:\\Rust\\RUSTUP.EXE", "run", "stable", "cargo"],
                ["cargo.exe", "+stable", "run"],
                ["C:\\Rust\\Cargo.Exe", "+stable", "run"],
            ):
                with self.subTest(command=command):
                    self.assertEqual(mod.toolchain(environment, command), stable)
            pinned = {"rustc": named("rustc"), "cargo": named("cargo")}
            for command in (["rustup", "which", "stable"], ["rustup", "run"], ["rustup", "--version"], ["rustup"]):
                with self.subTest(command=command):
                    self.assertEqual(mod.toolchain(environment, command), pinned)
            # Another program's `+` argument is its own.
            self.assertEqual(
                mod.toolchain(environment, ["python3", "+stable"]), {"rustc": named("rustc"), "cargo": named("cargo")}
            )
            self.assertEqual(mod.toolchain(environment, ["cargo"]), {"rustc": named("rustc"), "cargo": named("cargo")})
            # A standalone Cargo and rustc before rustup's proxies are what
            # runs, whatever rustup would resolve; a copy of rustup under a
            # proxy's name is its proxy all the same.
            standalone = os.pathsep.join((str(tools / "toolchains" / "stable" / "bin"), str(bin_directory)))
            self.assertEqual(mod.toolchain({"PATH": standalone}, ["cargo", "run"]), stable)
            # `rustup run` puts its toolchain first, whatever the PATH holds.
            self.assertEqual(
                mod.toolchain({"PATH": standalone}, ["rustup", "run", "pinned", "cargo"]),
                {"rustc": named("rustc"), "cargo": named("cargo")},
            )
            copies = tools / "copies"
            copies.mkdir()
            for tool in ("rustc", "cargo", "rustup"):
                shutil.copy2(rustup, copies / tool)
            self.assertEqual(
                mod.toolchain({"PATH": str(copies)}, ["cargo", "run"]), {"rustc": named("rustc"), "cargo": named("cargo")}
            )
            # What rustup prints when it fails names nothing, a path included.
            failing = tools / "failing"
            failing.mkdir()
            (failing / "rustup").write_text(f'#!/bin/sh\necho "{toolchain}/$2"\nexit 3\n', encoding="utf-8")
            (failing / "rustup").chmod(0o755)
            self.assertEqual(
                mod.toolchain({"PATH": str(failing)}, ["cargo", "run"]),
                {"rustc": {"path": None, "sha256": None}, "cargo": {"path": None, "sha256": None}},
            )
            # A tool rustup cannot resolve is named by nothing.
            (toolchain / "cargo").unlink()
            self.assertEqual(
                mod.toolchain(environment, ["cargo", "run"]),
                {"rustc": named("rustc"), "cargo": {"path": None, "sha256": None}},
            )
            self.assertEqual(
                mod.toolchain(environment, ["cargo", "+missing", "run"]),
                {"rustc": {"path": None, "sha256": None}, "cargo": {"path": None, "sha256": None}},
            )
            # Without rustup, the PATH's rustc and cargo are named.
            (toolchain / "cargo").write_text("#!/bin/sh\n", encoding="utf-8")
            (toolchain / "cargo").chmod(0o755)
            self.assertEqual(
                mod.toolchain({"PATH": str(toolchain)}, ["cargo", "+stable", "run"]),
                {"rustc": named("rustc"), "cargo": named("cargo")},
            )
            self.assertEqual(
                mod.toolchain({"PATH": str(tools / "nowhere")}, ["cargo", "run"]),
                {"rustc": {"path": None, "sha256": None}, "cargo": {"path": None, "sha256": None}},
            )
        # A listed run's record holds it.
        self.preregister("running")
        with mock.patch.object(mod, "toolchain", return_value={"rustc": named("rustc"), "cargo": named("cargo")}):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(records[0]["toolchain"], {"rustc": named("rustc"), "cargo": named("cargo")})

    def test_a_listed_experiment_reads_no_output_the_tools_left_uncommitted(self):
        # An output could be an input: what a listed experiment's command
        # could read in its results is held to HEAD like any other file, all
        # but the record the run itself writes.
        self.preregister("running")
        metrics = "experiments/x/L900-x/results/metrics.json"
        self.write(metrics, "{}\n")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"commit or remove {metrics}", stderr)
        stderr_prepare = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr_prepare),
        ):
            self.assertEqual(mod.prepare("L900"), 2)
        self.assertIn(f"commit or remove {metrics}", stderr_prepare.getvalue())
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "metrics")
        # Changed while the command runs and put back, the run goes
        # unrecorded, its reservation staying.
        def edit_and_restore():
            self.write(metrics, '{"adapted": true}\n')
            self.write(metrics, "{}\n")

        status, records, stderr = self.run_seed(edit_and_restore)
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"{metrics} changed on disk", stderr)

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
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record")
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

    def test_the_commit_a_run_names_decides_whether_it_is_listed(self):
        # The runner reads whether the list names the experiment from the
        # tree before the watch looks, and an unlisted run holds no lock,
        # reserves no seed and takes no frozen value: a commit dropping the
        # experiment from the list in between would let it run so, its
        # outcome be seen and its record discarded. The commit the record
        # would name decides.
        self.preregister("running")
        enrolled_at = self.head
        self.write("experiments/preregistration.toml", "version = 1\n")
        git(self.root, "commit", "-q", "--no-verify", "-am", "delisted")
        ran = []
        # As when the launch was checked before the commit dropped it.
        with mock.patch.object(mod, "launch_refused", return_value=False):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn(
            f"experiments/preregistration.toml: L900 was listed at {enrolled_at[:12]} and no longer is", stderr
        )
        # And a list read one way from the tree and held the other by HEAD.
        for listed, held in ((False, "names"), (True, "does not name")):
            with self.subTest(listed=listed):
                self.tearDown()
                self.setUp()
                if not listed:
                    self.preregister("running")
                refusal = (
                    f"experiments/preregistration.toml changed while the launch of L900 was checked: {self.head[:12]}, "
                    f"the commit its run would name, {held} it; rerun from a tree that holds HEAD"
                )
                ran = []
                with mock.patch.object(mod, "is_listed", return_value=listed):
                    status, records, stderr = self.run_seed(lambda: ran.append(True))
                    self.assertEqual((status, records, ran), (2, [], []))
                    self.assertIn(refusal, stderr)
                    prepared = io.StringIO()
                    with (
                        mock.patch.object(mod, "ROOT", self.root),
                        mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(prepared),
                    ):
                        self.assertEqual(mod.prepare("L900"), 2)
                    self.assertIn(refusal, prepared.getvalue())
                self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])

    def test_what_a_commit_made_before_the_watch_changed_decides_the_launch(self):
        # The launch is decided from the tree before the watch looks: a commit
        # made in between that changes what decides it, a setting outside the
        # frozen table say, is decided again once the watch holds the tree.
        self.preregister("running")
        real = mod.launch_refused
        config = self.root / "experiments/x/L900-x/config.toml"

        def refused_then_changed(exp_id):
            verdict = real(exp_id)
            config.write_text("learning_rate = 0.5\n" + config.read_text(encoding="utf-8"), encoding="utf-8")
            git(self.root, "commit", "-q", "--no-verify", "-am", "a setting changed")
            return verdict

        ran = []
        with mock.patch.object(mod, "launch_refused", side_effect=refused_then_changed):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("config.toml differs from the current one in learning_rate", stderr)
        # So is a registry that places the experiment elsewhere, where the
        # command's values would be read from another directory.
        self.tearDown()
        self.setUp()
        self.preregister("running")

        def refused_then_moved(exp_id):
            verdict = real(exp_id)
            git(self.root, "mv", "experiments/x", "experiments/y")
            registry = self.root / "experiments/registry.toml"
            registry.write_text(registry.read_text(encoding="utf-8").replace("x/L900-x", "y/L900-x"), encoding="utf-8")
            git(self.root, "commit", "-q", "--no-verify", "-am", "moved")
            return verdict

        ran = []
        with mock.patch.object(mod, "launch_refused", side_effect=refused_then_moved):
            status, _, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, ran), (2, []))
        self.assertIn("where the registry placed it in experiments/x/L900-x, not in experiments/y/L900-x", stderr)
        # Where the gate's own answer would not refuse it, the directory the
        # launch read does.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        ran = []
        with (
            mock.patch.object(mod, "launch_refused", side_effect=refused_then_moved),
            mock.patch.object(mod.check_research_gates, "launch_errors", return_value=[]),
        ):
            status, _, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, ran), (2, []))
        self.assertEqual(stderr, (
            "ERROR: experiments/registry.toml placed L900 elsewhere while its launch was checked; rerun from a tree "
            "that holds HEAD\n"
        ))

    def test_a_listed_runs_record_is_one_git_does_not_ignore(self):
        # A record git ignores could not be committed as written, and would
        # read as a file the command could have run once it ends.
        self.preregister("running")
        exclude = Path(git(self.root, "rev-parse", "--absolute-git-dir")) / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("results/\n", encoding="utf-8")
        ran = []
        status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran, list(self.attempts().glob("run-*.json"))), (2, [], [], []))
        self.assertTrue(stderr.startswith("ERROR: refusing to run L900: git would ignore its record "), stderr)
        self.assertTrue(stderr.endswith(
            ", which could then not be committed as written; no rule of the repository or the clone may ignore it\n"
        ), stderr)
        exclude.write_text("", encoding="utf-8")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_a_seed_that_could_not_join_the_earlier_ones_is_refused_before_it_runs(self):
        # The gate refuses a listed experiment's seeds that ran another tree
        # but for their seed records, or another program, toolchain or
        # environment, once the record exists; found only then, the seed
        # would be spent.
        self.preregister("running", seeds="[17, 29]")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "seed 17 recorded")
        seventeen = sorted(self.results.glob("run-*.json"))[0].name
        mutations = "experiments/x/L900-x/results/mutations.json"
        self.write(mutations, "{}\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "mutations between seeds")
        ran = []
        status, records, stderr = self.run_seed(lambda: ran.append(True), seed=29)
        self.assertEqual((status, ran, len(records)), (2, [], 1))
        self.assertIn(f"'s, where {seventeen} ran, in {mutations}; the seeds of a listed experiment run one tree", stderr)
        git(self.root, "rm", "-q", mutations)
        git(self.root, "commit", "-q", "--no-verify", "-m", "no mutations")
        with mock.patch.dict(os.environ, {"LANG": "xx_XX.UTF-8"}):
            status, records, stderr = self.run_seed(lambda: ran.append(True), seed=29)
        self.assertEqual((status, ran, len(records)), (2, [], 1))
        self.assertEqual(stderr, (
            f"ERROR: L900: {seventeen} ran another program, toolchain or environment than this run would; the seeds of "
            "a listed experiment run one of each\n"
        ))
        # Neither, it runs.
        environment = {name: value for name, value in os.environ.items() if name != "LANG"}
        recorded = json.loads((self.results / seventeen).read_text(encoding="utf-8"))["environment"]
        if "LANG" in recorded:
            environment["LANG"] = recorded["LANG"]
        with mock.patch.dict(os.environ, environment, clear=True):
            status, records, stderr = self.run_seed(seed=29)
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed", "completed"]))

    def test_a_record_the_results_hold_is_no_change_of_the_sources(self):
        # Records accumulate in results/, uncommitted while runs go on: what
        # the tools write there is no change of the sources.
        for name in ("run-concurrent-seed-29.json", "run.json", "metrics.json", "mutations.json"):
            self.write(f"experiments/x/L900-x/results/{name}", "{}\n")
        status, records, stderr = self.run_seed(lambda: self.write("experiments/x/L900-x/results/run-other.json", "{}\n"))
        self.assertEqual((status, stderr), (0, ""))

    def test_the_results_directory_holds_nothing_a_command_could_run_unrecorded(self):
        # Anything else there is held to HEAD like the rest of the
        # experiment: code or input kept there could change between runs
        # while every record named one commit.
        harness = "experiments/x/L900-x/results/harness.py"
        self.write(harness, "print('adapted')\n")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"commit or remove {harness}", stderr)
        # The repository ignores most of a results directory, and the watch
        # leaves out what git ignores, so an ignored file there is refused.
        self.write(".gitignore", "__pycache__/\n/experiments/**/results/*\n!/experiments/**/results/*.json\n")
        git(self.root, "commit", "-q", "--no-verify", "-am", "results ignored")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"results directory holds {harness}, which git ignores", stderr)
        (self.root / harness).unlink()
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, len(records)), (0, "", 1))
        # A file HEAD holds there is watched: changed while the command runs,
        # the run is not recorded.
        self.write("experiments/x/L900-x/results/notes.json", "{}\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "notes")
        status, records, stderr = self.run_seed(lambda: self.write("experiments/x/L900-x/results/notes.json", "[]\n"))
        self.assertEqual((status, len(records)), (2, 1))
        self.assertIn("experiments/x/L900-x/results/notes.json changed on disk", stderr)


if __name__ == "__main__":
    unittest.main()
