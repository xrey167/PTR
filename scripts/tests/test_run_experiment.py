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
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
import urllib.parse
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

    def test_output_no_encoding_reads_is_captured_and_leaves_no_seed_reserved(self):
        # A byte the encoding does not read is the command's output, not an
        # error of the runner: raised after the command ran, it would leave
        # the reservation the only record of a seed that saw its outcome.
        script = "import sys; sys.stdout.buffer.write(b'a\\xffb\\n'); sys.stderr.buffer.write(b'\\xfe\\xfdok')"
        result = mod.execute_command([sys.executable, "-c", script])
        self.assertEqual((result["exit_code"], result["launch_error"]), (0, None))
        self.assertEqual(result["stdout"], "a\\xffb\n")
        self.assertEqual(result["stderr"], "\\xfe\\xfdok")
        # Text it can read is read as it is, whatever the runner's locale.
        result = mod.execute_command([sys.executable, "-c", "import sys; sys.stdout.buffer.write('\u00e9\\n'.encode('utf-8'))"])
        self.assertEqual(result["stdout"], "\u00e9\n")

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
        # A listed run refuses to start while its scratch directory exists,
        # which is named by the experiment's id in the temporary directory:
        # each test has a temporary directory of its own, so a run of this
        # suite beside another one, or one an interrupted run left behind,
        # is never taken for this test's.
        temporary = Path(self.directory.name + "-tmp")
        temporary.mkdir(exist_ok=True)
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        self.enterContext(mock.patch.dict(os.environ, {"TMPDIR": str(temporary)}))
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
        # Nor a name Windows trims to a step: `.. ` and `...` would climb out
        # of the experiment's directory there, and `. ` stay in it as another
        # spelling of it.
        stepping = ("passes through a name Windows spells otherwise (trailing dots and spaces are trimmed, and a dot or two "
                    "with them is a step), which could climb out of the experiment's directory or name another directory")
        for results_dir, refusal in (
            (".", "is not a directory below experiments/x/L900-x"),
            ("../elsewhere", "is not a directory below experiments/x/L900-x"),
            (".git", "passes through git's own directory, where no record can be committed"),
            ("results/.GIT", "passes through git's own directory, where no record can be committed"),
            (".. /elsewhere", stepping),
            ("results/.../elsewhere", stepping),
            ("results/. /x", stepping),
            ("results/..:stream", stepping),
            ("results/ ", stepping),
            # And a name Windows trims to another: `archive.` is `archive`
            # there, and the record the history holds is under the first.
            ("results/archive.", stepping),
            ("a../results", stepping),
            ("results/.a.", stepping),
            ("archive./x", stepping),
            ("results/archive ", stepping),
            # Windows reads a backslash as a step and a drive letter with a
            # colon as another drive, and no path holds a NUL.
            ("..\\elsewhere", "is not a directory below experiments/x/L900-x"),
            ("results\\.git", "is not a directory below experiments/x/L900-x"),
            ("C:/out", "is not a directory below experiments/x/L900-x"),
            ("c:out", "is not a directory below experiments/x/L900-x"),
            ("results/a\0b", "is not a directory below experiments/x/L900-x"),
        ):
            with self.subTest(results_dir=results_dir):
                self.write("experiments/x/L900-x/experiment.toml", manifest + f"results_dir = {json.dumps(results_dir)}\n")
                git(self.root, "commit", "-q", "--no-verify", "-am", f"results in {results_dir!r}")
                ran = []
                status, _, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, ran), (2, []))
                self.assertIn(f"results_dir {results_dir!r} {refusal}", stderr)
        # A name that merely starts with dots, or holds a space inside, is a
        # name of its own on every platform, and is no step.
        experiment = self.root / "experiments/x/L900-x"
        with mock.patch.object(mod, "ROOT", self.root):
            for accepted in ("..x/results", "...x", ".hidden/results", "a b", "results/C:x", "ab:c/results"):
                with self.subTest(accepted=accepted):
                    self.assertEqual(mod.results_directory(experiment, {"results_dir": accepted}), experiment / accepted)
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
            # A long id, or one of characters that encode to several bytes,
            # names a file a file system can hold, one for each id.
            for long_id in ("x" * 300, "\u00e9" * 100):
                bounded = mod.run_lock(long_id, directory)
                self.assertLessEqual(len(bounded.name.encode("utf-8")), 255)
                self.assertTrue(bounded.exists())
                self.assertEqual(len(mod.id_file_name(long_id)), mod.ID_NAME_LIMIT)
            self.assertNotEqual(mod.id_file_name("x" * 300), mod.id_file_name("x" * 299 + "y"))
            # No id that fits is named as a long one: its name, read back as
            # an id, names another file.
            long_name = mod.id_file_name("sweep-" + "x" * 200)
            self.assertNotEqual(mod.id_file_name(urllib.parse.unquote(long_name)), long_name)
            self.assertEqual(mod.id_file_name("x" * mod.ID_NAME_LIMIT), "x" * mod.ID_NAME_LIMIT)
            self.assertEqual(
                mod.scratch_directory("x" * 300, {"TMPDIR": "/scratch"}).name, f"ptr-run-{mod.id_file_name('x' * 300)}"
            )
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
                # A record rewritten to another seed than the preregistered
                # ones is refused as that, before the seed's record is asked.
                self.assertIn(
                    "ran seed 29, which is not one of the preregistered seeds" if "seed" in rewritten
                    else "seed 17 of L900 already ran",
                    stderr,
                )

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
        # its bytecode cache, saw no outcome: here the runner's own
        # temporary directory, the one used when no TMPDIR is set, is gone.
        self.preregister("running")
        attempts = self.attempts()
        stderr = io.StringIO()
        unset = {name: value for name, value in os.environ.items() if name != "TMPDIR"}
        with (
            mock.patch.dict(os.environ, unset, clear=True),
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
            "TMPDIR": os.environ["TMPDIR"],
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
            set(seen) - {"PYTHONPYCACHEPREFIX", "CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"},
            {*mod.COMMAND_ENVIRONMENT, *mod.FIXED_ENVIRONMENT},
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
        # Python reads no package from the user's own site directory, and
        # hashes strings the same way for every seed, which the record names.
        self.assertEqual(seen["PYTHONNOUSERSITE"], "1")
        self.assertEqual((seen["PYTHONHASHSEED"], record["environment"]["PYTHONHASHSEED"]), ("0", "0"))
        for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "RUSTC_WRAPPER", "GH_TOKEN"):
            self.assertNotIn(name, seen)
        # The runner's RUSTUP_TOOLCHAIN does not reach the command; where
        # rustup resolves the toolchain, the one it resolved for the launch
        # is the command's, named in the record.
        self.assertNotEqual(seen.get("RUSTUP_TOOLCHAIN"), "nightly")
        pinned = {}
        if "RUSTUP_TOOLCHAIN" in record["environment"]:
            pinned["RUSTUP_TOOLCHAIN"] = record["environment"]["RUSTUP_TOOLCHAIN"]
            self.assertEqual(str(Path(record["toolchain"]["cargo"]["path"]).parent.parent), pinned["RUSTUP_TOOLCHAIN"])
        self.assertEqual(record["environment"], {
            **self.allowed, "PYTHONPYCACHEPREFIX": scratch, "CARGO_TARGET_DIR": os.path.join(scratch, "cargo-target"),
            **pinned,
        })
        self.assertEqual({name: seen[name] for name in record["environment"]}, record["environment"])
        # A program found first on the PATH, wherever it lies, is named with
        # its content, and so are the interpreters its first line names
        # (`#!/usr/bin/env python3`: `env`, and the `python3` it looks up on
        # the environment's PATH).
        self.assertEqual(record["executable"], {
            "path": str(self.bench.resolve()),
            "sha256": hashlib.sha256(self.bench.read_bytes()).hexdigest(),
            "interpreters": [
                mod.named_program("/usr/bin/env"),
                mod.named_program(shutil.which("python3", path=record["environment"]["PATH"])),
            ],
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
            ("named_script", unread),
            ("resolved_tool", unread),
            ("toolchain", {"rustc": unread, "cargo": {"path": None, "sha256": None}}),
            # An interpreter a tool runs through counts as well.
            ("toolchain", {"rustc": {"path": "/x/rustc", "sha256": "0" * 64, "interpreters": [unread]}}),
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
        # So does an interpreter the program runs through.
        with mock.patch.object(mod, "interpreters_of", return_value=[{"path": "/x/i", "sha256": None}]):
            ran = []
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran, list(self.attempts().glob("run-*.json"))), (2, [], [], []))
        self.assertEqual(stderr, (
            "ERROR: refusing to run L900: /x/i cannot be read, or changed while it was read, so "
            "its record could not name by its content a program the run starts or builds with; make it "
            "readable, and leave it unchanged, for the run\n"
        ))
        # The allowed environment is what the runner's sets of the allowed
        # names, and the fixed values, whatever the runner's sets for them.
        runner = {"PATH": "/bin", "HOME": "/home/runner", "GH_TOKEN": "x", "PYTHONHASHSEED": "random"}
        with mock.patch.dict(os.environ, runner, clear=True):
            self.assertEqual(
                mod.command_environment(),
                {"PATH": "/bin", "HOME": "/home/runner", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
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

    def test_a_listed_run_starts_from_and_leaves_a_checkout_holding_nothing_git_does_not_list(self):
        # An empty directory, whose presence a command can test, and an entry
        # named .git below the root, which git never looks into, are in no
        # commit and no watch sees them: a listed run is refused while the
        # repository holds one, and one there once its command has ended
        # leaves it unrecorded.
        self.preregister("running")
        for make, shown in (
            (lambda: (self.root / "empty/inner").mkdir(parents=True), "empty/inner"),
            (lambda: self.write("scripts/.git/helper.py", "x = 1\n"), "scripts/.git"),
            (lambda: self.write("docs/.git", "gitdir: /elsewhere\n"), "docs/.git"),
        ):
            with self.subTest(shown=shown):
                make()
                status, records, stderr = self.run_seed()
                self.assertEqual((status, records), (2, []))
                self.assertEqual(stderr, (
                    f"ERROR: refusing to run L900 while the repository holds {shown}, which git does not list (an "
                    "empty directory, an entry named .git below the root, or a directory that cannot be listed) and "
                    "its command could read unrecorded; a listed experiment runs from a checkout that holds none\n"
                ))
                shutil.rmtree(self.root / shown.split("/")[0])
        # However a platform names git's own directory: Windows reads `.git.`
        # and `git~1` as `.git`, at the root as below it.
        with mock.patch.object(mod, "ROOT", self.root):
            for make, shown in (
                (lambda: self.write("scripts/git~1/helper.py", "x = 1\n"), "scripts/git~1"),
                (lambda: self.write("docs/.GIT.", "gitdir: /elsewhere\n"), "docs/.GIT."),
                (lambda: self.write("GIT~1/helper.py", "x = 1\n"), "GIT~1"),
            ):
                with self.subTest(shown=shown):
                    make()
                    self.assertEqual(mod.unlisted_entries(), [shown])
                    shutil.rmtree(self.root / shown.split("/")[0])
            self.assertEqual(mod.unlisted_entries(), [])
            # A file at the root under such a name too: only a file named
            # `.git` itself is a linked worktree's, which git reads.
            for name in (".GIT", ".GIT.", "git~1"):
                with self.subTest(root_file=name):
                    self.write(name, "gitdir: /elsewhere\n")
                    self.assertEqual(mod.unlisted_entries(), [name])
                    (self.root / name).unlink()
            self.assertEqual(mod.unlisted_entries(), [])
            # A file that is neither a regular file nor a symlink, which git
            # skips: a FIFO or a socket, in a directory that holds nothing
            # else too, which a command can test for or read from.
            os.mkfifo(self.root / "flag")
            (self.root / "state").mkdir()
            os.mkfifo(self.root / "state" / "seen")
            listening = socket.socket(socket.AF_UNIX)
            self.addCleanup(listening.close)
            (self.root / "pg").mkdir()
            # Bound by a relative name, which no socket path length limits.
            with contextlib.chdir(self.root / "pg"):
                listening.bind(".s.PGSQL.5432")
            self.assertEqual(mod.unlisted_entries(), ["flag", "pg/.s.PGSQL.5432", "state/seen"])
            for path in ("flag", "state/seen", "pg/.s.PGSQL.5432"):
                (self.root / path).unlink()
            for directory in ("state", "pg"):
                (self.root / directory).rmdir()
            self.assertEqual(mod.unlisted_entries(), [])
        # One put there while the command ran.
        status, records, stderr = self.run_seed(lambda: (self.root / "made").mkdir())
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn("the repository holds made, which git does not list and the command could have read unrecorded", stderr)
        # An unlisted experiment runs beside them, as it did.
        self.tearDown()
        self.setUp()
        (self.root / "empty").mkdir()
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))

    def test_a_listed_run_refuses_a_directory_the_walk_cannot_list(self):
        # Git lists nothing of a directory without search permission, and
        # os.walk skips one silently, but a command can see it: its presence
        # or mode could differ between seeds while every record names one
        # commit. Run as root, where a mode changes nothing, the walk's own
        # listing is made to fail for the directory instead.
        self.preregister("running")
        (self.root / "nested/sealed").mkdir(parents=True)
        (self.root / "gone").mkdir()
        (self.root / "replaced").mkdir()
        scandir = os.scandir

        def failing(path=".", *arguments):
            name = Path(os.fspath(path)).name
            if name == "sealed":
                raise PermissionError(13, "Permission denied", os.fspath(path))
            if name == "gone":
                raise FileNotFoundError(2, "No such file or directory", os.fspath(path))
            if name == "replaced":
                raise NotADirectoryError(20, "Not a directory", os.fspath(path))
            return scandir(path, *arguments)

        with mock.patch.object(mod, "ROOT", self.root), mock.patch.object(os, "scandir", failing):
            self.assertEqual(mod.unlisted_entries(), ["nested/sealed"])
        # A directory removed, or replaced by a file, since the walk listed it
        # is one removed then.
        shutil.rmtree(self.root / "nested")
        with mock.patch.object(mod, "ROOT", self.root), mock.patch.object(os, "scandir", failing):
            self.assertEqual(mod.unlisted_entries(), [])
        # The launch refuses a checkout holding one.
        (self.root / "nested/sealed").mkdir(parents=True)
        (self.root / "gone").rmdir()
        (self.root / "replaced").rmdir()
        with mock.patch.object(os, "scandir", failing):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertEqual(stderr, (
            "ERROR: refusing to run L900 while the repository holds nested/sealed, which git does not list (an empty "
            "directory, an entry named .git below the root, or a directory that cannot be listed) and its command "
            "could read unrecorded; a listed experiment runs from a checkout that holds none\n"
        ))
        # One that cannot be listed once the command has ended leaves the run
        # unrecorded.
        shutil.rmtree(self.root / "nested")
        with mock.patch.object(os, "scandir", failing):
            status, records, stderr = self.run_seed(lambda: (self.root / "nested/sealed").mkdir(parents=True))
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(
            "the repository holds nested/sealed, which git does not list and the command could have read unrecorded",
            stderr,
        )

    def test_a_listed_run_from_a_checkout_holding_a_name_that_is_not_utf8_is_refused(self):
        # Git prints such a name's bytes as they are: the watch cannot read it
        # as a path, and refuses the run rather than failing.
        self.preregister("running")
        try:
            with open(os.path.join(os.fsencode(self.root), b"notes\xff.md"), "wb") as handle:
                handle.write(b"x\n")
        except OSError as error:
            self.skipTest(f"cannot name a file with bytes that are not UTF-8: {error}")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, records), (2, []))
        self.assertTrue(stderr.startswith("ERROR: "), stderr)
        self.assertIn("printed what is not UTF-8, such as a file name, which no check can read", stderr)
        self.assertNotIn("Traceback", stderr)

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
                # The directory it was put in goes with it: an empty one left
                # behind is one git does not list either.
                shutil.rmtree(self.root / ignored.split("/")[0])
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

    def test_a_listed_run_watches_the_directories_holding_the_files_it_watches(self):
        # A directory renamed aside and put back keeps every file in it, and
        # each file's stamp: the sources the command read were another
        # tree's, which the directory's own stamp and its parent's show.
        self.preregister("running")
        self.write("scripts/harness.py", "print('recorded')\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "harness")
        os.utime(self.root / "scripts/harness.py", ns=(10**18, 10**18))
        changed = (
            "changed on disk, a directory holding watched files that was renamed, or had an entry added or removed, "
            "since the run started"
        )

        def swap_and_restore():
            # Past the filesystem's timestamp resolution since the launch looked.
            time.sleep(0.05)
            (self.root / "scripts").rename(self.root / "scripts.aside")
            (self.root / "scripts.aside").rename(self.root / "scripts")

        status, records, stderr = self.run_seed(swap_and_restore)
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"; ., scripts {changed}; rerun from a working tree that stays at HEAD", stderr)

        # A file created in one of them and removed again is one the command
        # could have run.
        self.tearDown()
        self.setUp()
        self.preregister("running")

        def create_and_remove():
            time.sleep(0.05)
            self.write("src/generated.rs", "pub fn g() {}\n")
            (self.root / "src/generated.rs").unlink()

        status, records, stderr = self.run_seed(create_and_remove)
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"; src {changed}; rerun from a working tree that stays at HEAD", stderr)

        # The runner's own writes to the results before the command starts
        # are none of it: a run that changes nothing is recorded.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        status, records, stderr = self.run_seed(lambda: time.sleep(0.05))
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

        # An unlisted experiment's watch is on its provenance files, as it was.
        self.tearDown()
        self.setUp()

        def swap_sources_and_restore():
            time.sleep(0.05)
            (self.root / "src").rename(self.root / "src.aside")
            (self.root / "src.aside").rename(self.root / "src")

        status, records, stderr = self.run_seed(swap_sources_and_restore)
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_a_listed_run_watches_its_own_reservation_from_the_moment_its_command_starts(self):
        # The watch leaves the run's own record out, since the runner writes
        # it after the watch looked, but a command can read it: an in-place
        # edit that is put back, or a file replaced by a copy of itself,
        # leaves the bytes the runner wrote and must still be seen.
        self.preregister("running")
        changed = "changed on disk; rerun from a working tree that stays at HEAD"

        def reservation() -> Path:
            return next(self.results.glob("run-*.json"))

        def edit_and_restore():
            path = reservation()
            original = path.read_bytes()
            time.sleep(0.05)
            with path.open("r+b") as handle:
                handle.write(b"{" + b" " * (len(original) - 2) + b"}")
            with path.open("r+b") as handle:
                handle.write(original)

        status, records, stderr = self.run_seed(edit_and_restore)
        name = next(self.results.glob("run-*.json")).relative_to(self.root).as_posix()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"; {name} {changed}", stderr)

        # A file replaced by a copy of itself has another inode; the results
        # directory, which lost and regained an entry, is a change too.
        self.tearDown()
        self.setUp()
        self.preregister("running")

        def replace_by_a_copy():
            path = reservation()
            original = path.read_bytes()
            time.sleep(0.05)
            path.unlink()
            path.write_bytes(original)

        status, records, stderr = self.run_seed(replace_by_a_copy)
        name = next(self.results.glob("run-*.json")).relative_to(self.root).as_posix()
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"; {name} changed on disk; ", stderr)

        # A reservation the command removes is a change as well.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        status, records, stderr = self.run_seed(lambda: reservation().unlink())
        name = f"{self.results.relative_to(self.root).as_posix()}/{next(self.attempts().glob('run-*.json')).name}"
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"; {name} changed on disk; ", stderr)

        # A command that leaves it alone is recorded, as before.
        self.tearDown()
        self.setUp()
        self.preregister("running")
        status, records, stderr = self.run_seed(lambda: reservation().read_bytes())
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

        # An unlisted experiment writes no reservation and watches none.
        self.tearDown()
        self.setUp()
        status, records, stderr = self.run_seed(lambda: time.sleep(0.05))
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_a_reservation_edited_before_it_is_stamped_is_not_adopted(self):
        # The stamp binds the reservation as it is when it is taken. A local
        # actor who edits it between the write and the stamp would have the
        # edit adopted, and the command would read it: the bytes are checked
        # against the ones written, once stamped, and a run whose reservation
        # is not those refuses before its command starts.
        self.preregister("running")
        stamp = mod.experiment_records.ProvenanceWatch.stamp_reserved

        def edited_first(watch, names):
            for name in names:
                path = self.root / name
                path.write_bytes(path.read_bytes().replace(b'"started"', b'"attacker"'))
            stamp(watch, names)

        ran = []
        with mock.patch.object(mod.experiment_records.ProvenanceWatch, "stamp_reserved", edited_first):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        name = self.results.relative_to(self.root).as_posix()
        self.assertEqual((status, ran, records), (2, [], []))
        self.assertIn(f"ERROR: {name}/run-", stderr)
        self.assertIn(" was changed between being written and stamped; nothing ran, and it is removed so its seed may run", stderr)
        # Nothing ran: no reservation or copy stays, and the seed runs.
        self.assertEqual(list(self.attempts().glob("run-*.json")), [])
        status, records, stderr = self.run_seed(lambda: None)
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

        shell = Path(shutil.which("sh", path=os.defpath)).resolve()

        def named(tool: str, name: str = "pinned") -> dict:
            # The tools are shell scripts, so each names the shell it runs through.
            path = tools / "toolchains" / name / "bin" / tool
            return {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "interpreters": [{"path": str(shell), "sha256": hashlib.sha256(shell.read_bytes()).hexdigest()}],
            }

        with mock.patch.object(mod, "ROOT", self.root):
            environment = {"PATH": str(bin_directory)}
            self.assertEqual(mod.toolchain(environment, ["cargo", "run"]), {"rustc": named("rustc"), "cargo": named("cargo")})
            # Each tool rustup resolves is stamped as it is read, so a run
            # during which one changed goes unrecorded, and the toolchain it
            # resolved it from is named.
            stamps, selected = {}, {}
            mod.toolchain(environment, ["cargo", "run"], stamps, selected)
            pinned_directory = str(tools / "toolchains" / "pinned")
            self.assertEqual(selected, {"rustc": pinned_directory, "cargo": pinned_directory})
            selected = {}
            mod.toolchain(environment, ["cargo", "+stable", "run"], None, selected)
            stable_directory = str(tools / "toolchains" / "stable")
            self.assertEqual(selected, {"rustc": stable_directory, "cargo": stable_directory})
            # By its directory, which rustup takes as the toolchain itself,
            # whether it installed it under its home or a path names it (a
            # path toolchain, which has no name among those installed).
            self.assertEqual(
                mod.toolchain_directory("/rustup/toolchains/stable-x86_64/bin/cargo"), "/rustup/toolchains/stable-x86_64"
            )
            self.assertEqual(mod.toolchain_directory("/opt/toolchains/stable/bin/rustc"), "/opt/toolchains/stable")
            self.assertEqual(mod.toolchain_directory("/opt/rust-custom/bin/cargo"), "/opt/rust-custom")
            self.assertIsNone(mod.toolchain_directory("/opt/rust-custom/cargo"))
            # (each of them a shell script, so the shell it runs through is stamped too)
            self.assertEqual(
                {path: stamp for path, stamp in stamps.items() if path.startswith(str(tools))},
                {
                    named(tool)["path"]: mod.experiment_records.file_stamp(Path(named(tool)["path"]))
                    for tool in ("rustc", "cargo")
                },
            )
            self.assertIn(str(shell), stamps)
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
            # `rustup run` puts its toolchain first, whatever the PATH holds,
            # however a platform spells rustup.
            for rustup_run in ("rustup", "rustup.exe", "C:\\Rust\\RUSTUP.EXE"):
                with self.subTest(rustup_run=rustup_run):
                    self.assertEqual(
                        mod.toolchain({"PATH": standalone}, [rustup_run, "run", "pinned", "cargo"]),
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
        # A listed run's record holds it, and the command runs the toolchain
        # rustup resolved for the launch whatever override it would read by
        # then: its environment names it (RUSTUP_TOOLCHAIN).
        self.preregister("running")
        started = []

        def execute(command, environment=None, program=None, scratch=None):
            started.append(environment)
            return {"exit_code": 0, "stdout": "", "stderr": "", "launch_error": None, "duration_ns": 1}

        with mock.patch.dict(os.environ, {"PATH": os.pathsep.join((str(bin_directory), os.environ["PATH"]))}):
            status, records, stderr = self.run_seed()
            self.assertEqual((status, stderr), (0, ""))
            self.assertEqual(records[0]["toolchain"], {"rustc": named("rustc"), "cargo": named("cargo")})
            self.assertEqual(records[0]["environment"]["RUSTUP_TOOLCHAIN"], str(tools / "toolchains" / "pinned"))
            for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
                record.unlink()
            stderr = io.StringIO()
            with (
                mock.patch.object(mod, "ROOT", self.root),
                mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                mock.patch.object(mod, "execute_command", side_effect=execute),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(mod.run_experiment("L900", entrypoint="entrypoint", seed=17), 0)
        [environment] = started
        self.assertEqual(environment["RUSTUP_TOOLCHAIN"], str(tools / "toolchains" / "pinned"))
        # Tools rustup resolves from two toolchains pin neither.
        with mock.patch.object(mod, "toolchain", side_effect=lambda *arguments: (
            arguments[3].update({"rustc": "pinned", "cargo": "stable"}) or {"rustc": named("rustc"), "cargo": named("cargo")}
        )):
            for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
                record.unlink()
            status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertNotIn("RUSTUP_TOOLCHAIN", records[0]["environment"])

    def test_a_linked_toolchain_is_pinned_by_the_directory_it_leads_to(self):
        # rustup keeps a toolchain it was told to link as a link, which can
        # be pointed at another toolchain without changing a file of the one
        # it led to: the launch pins the directory, not the name of the link,
        # and refuses a command that selects the toolchain by that name
        # itself, which `RUSTUP_TOOLCHAIN` does not override.
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name in ("first", "second"):
            (tools / "toolchains" / name / "bin").mkdir(parents=True)
            for tool in ("rustc", "cargo"):
                (tools / "toolchains" / name / "bin" / tool).write_text(f"#!/bin/sh\necho {name} {tool}\n", encoding="utf-8")
                (tools / "toolchains" / name / "bin" / tool).chmod(0o755)
        try:
            (tools / "toolchains" / "pick").symlink_to(tools / "toolchains" / "first", target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")
        bin_directory = tools / "bin"
        bin_directory.mkdir()
        rustup = bin_directory / "rustup"
        rustup.write_text(
            '#!/bin/sh\n[ "$1" = which ] || exit 1\nshift\nname="${FAKE_TOOLCHAIN:-second}"\n'
            'if [ "$1" = --toolchain ]; then name="$2"; shift 2; fi\n'
            f'path="{tools}/toolchains/$name/bin/$1"\n[ -e "$path" ] && echo "$path" && exit 0\nexit 1\n',
            encoding="utf-8",
        )
        rustup.chmod(0o755)
        for tool in ("rustc", "cargo"):
            (bin_directory / tool).symlink_to("rustup")
        first = str((tools / "toolchains" / "first").resolve())
        second = str(tools / "toolchains" / "second")
        link = str(tools / "toolchains" / "pick")
        with mock.patch.object(mod, "ROOT", self.root):
            # The link resolved from the root, or named by the command, is
            # pinned as the directory it leads to now.
            for command, environment in (
                (["cargo", "run"], {"PATH": str(bin_directory), "FAKE_TOOLCHAIN": "pick"}),
                (["cargo", "+pick", "run"], {"PATH": str(bin_directory)}),
                (["rustup", "run", "pick", "cargo", "run"], {"PATH": str(bin_directory)}),
            ):
                with self.subTest(command=command):
                    selected, linked = {}, []
                    mod.toolchain(environment, command, None, selected, linked)
                    self.assertEqual(selected, {"rustc": first, "cargo": first})
                    self.assertEqual(linked, [link])
            # A toolchain rustup installed is no link: its directory is its name.
            selected, linked = {}, []
            mod.toolchain({"PATH": str(bin_directory)}, ["cargo", "+second", "run"], None, selected, linked)
            self.assertEqual((selected, linked), ({"rustc": second, "cargo": second}, []))
            self.assertFalse(mod.is_linked_toolchain(second))
            self.assertTrue(mod.is_linked_toolchain(link))
            self.assertFalse(mod.is_linked_toolchain(str(tools / "toolchains" / "absent")))
            # Given no `linked`, the tools are named as before.
            selected = {}
            mod.toolchain({"PATH": str(bin_directory), "FAKE_TOOLCHAIN": "pick"}, ["cargo", "run"], None, selected)
            self.assertEqual(selected, {"rustc": first, "cargo": first})

        # A listed run refuses a command that selects a toolchain by the name
        # of a link, before it reserves anything, and runs one that does not.
        self.preregister("running")
        unresolved = {"rustc": {"path": None, "sha256": None}, "cargo": {"path": None, "sha256": None}}

        def resolved(*arguments):
            arguments[4].append(link)
            arguments[3].update({"rustc": first, "cargo": first})
            return unresolved

        with mock.patch.object(mod, "toolchain", side_effect=resolved):
            with mock.patch.object(mod, "named_toolchain", return_value="pick"):
                status, records, stderr = self.run_seed()
            self.assertEqual((status, records), (2, []))
            self.assertIn(
                f"refusing to run L900: the command selects toolchain 'pick' itself, and rustup would resolve that "
                f"name again when it starts, from a link ({link}) that could be pointed at another toolchain "
                "meanwhile",
                stderr,
            )
            self.assertEqual(list(self.attempts().glob("run-*.json")), [])
            with mock.patch.object(mod, "named_toolchain", return_value=None):
                status, records, stderr = self.run_seed()
            self.assertEqual((status, stderr), (0, ""))
            self.assertEqual(records[0]["environment"]["RUSTUP_TOOLCHAIN"], first)
        # One that selects an installed toolchain, which is no link, runs.
        for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
            record.unlink()

        def installed(*arguments):
            arguments[3].update({"rustc": second, "cargo": second})
            return unresolved

        with mock.patch.object(mod, "toolchain", side_effect=installed):
            with mock.patch.object(mod, "named_toolchain", return_value="second"):
                status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(records[0]["environment"]["RUSTUP_TOOLCHAIN"], second)

    def fake_rust(self, tools: Path, resolves: bool = True) -> Path:
        """A directory of a fake rustup and its proxies for rustc, cargo and
        rustdoc, beside the toolchain `pinned` under `tools`, whose tools it
        resolves as `rustup which` does, or resolves none; returns it."""
        bin_directory = tools / "bin"
        bin_directory.mkdir()
        for tool in ("rustc", "cargo", "rustdoc"):
            (tools / "toolchains" / "pinned" / "bin").mkdir(parents=True, exist_ok=True)
            (tools / "toolchains" / "pinned" / "bin" / tool).write_bytes(f"{tool}\n".encode("utf-8"))
            (tools / "toolchains" / "pinned" / "bin" / tool).chmod(0o755)
        script = (
            '#!/bin/sh\n[ "$1" = which ] || exit 1\nshift\n'
            'if [ "$1" = --toolchain ]; then shift 2; fi\n'
            f'path="{tools}/toolchains/pinned/bin/$1"\n[ -e "$path" ] && echo "$path" && exit 0\nexit 1\n'
            if resolves
            else "#!/bin/sh\nexit 1\n"
        )
        (bin_directory / "rustup").write_text(script, encoding="utf-8")
        (bin_directory / "rustup").chmod(0o755)
        for tool in ("rustc", "cargo", "rustdoc"):
            (bin_directory / tool).symlink_to("rustup")
        return bin_directory

    def test_a_cargo_test_command_binds_rustdoc_beside_rustc_and_cargo(self):
        # `cargo test` runs rustdoc on a library's documentation tests, so a
        # rustdoc replaced between seeds would change what ran while every
        # record named the same rustc and cargo.
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools)

        def named(tool: str) -> dict:
            path = tools / "toolchains" / "pinned" / "bin" / tool
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        with mock.patch.object(mod, "ROOT", self.root):
            environment = {"PATH": str(bin_directory)}
            with_rustdoc = {"rustc": named("rustc"), "cargo": named("cargo"), "rustdoc": named("rustdoc")}
            without = {"rustc": named("rustc"), "cargo": named("cargo")}
            for command in (
                ["cargo", "test"],
                ["cargo", "test", "--", "--lib"],
                ["cargo", "test", "-p", "x", "--no-fail-fast"],
                ["cargo", "test", "--lib", "--doc"],
                ["cargo", "test", "--doc", "--bin", "x"],
                ["cargo", "+pinned", "test", "--doc"],
                ["/elsewhere/bin/cargo", "test", "--doc", "--", "filter"],
                ["rustup", "run", "pinned", "cargo", "test"],
                ["rustup", "-v", "run", "--install", "pinned", "cargo", "+pinned", "test"],
                ["rustup", "+pinned", "run", "pinned", "cargo", "test"],
                ["cargo.exe", "test"],
                ["C:\\Rust\\CARGO.EXE", "+pinned", "test"],
            ):
                with self.subTest(command=command):
                    self.assertEqual(mod.toolchain(environment, command), with_rustdoc)
            for command in (
                ["cargo", "test", "--all-targets"],
                ["cargo", "test", "--no-run"],
                ["cargo", "test", "--lib"],
                ["cargo", "test", "--bins"],
                ["cargo", "test", "--bin", "x"],
                ["cargo", "test", "--bin=x"],
                ["cargo", "test", "--tests"],
                ["cargo", "test", "--test", "t"],
                ["cargo", "test", "--test=t"],
                ["cargo", "test", "--examples"],
                ["cargo", "test", "--example", "e"],
                ["cargo", "test", "--benches"],
                ["cargo", "test", "--bench", "b"],
                ["cargo", "+pinned", "test", "-p", "x", "--lib", "--", "--doc"],
                ["cargo", "run"],
                ["cargo", "build", "--", "test"],
                ["cargo", "bench"],
                ["cargo", "check", "--tests"],
                ["cargo"],
                ["rustc", "lib.rs"],
                ["python3", "bench.py", "cargo", "test"],
                ["python3", "test"],
                ["rustc", "test"],
                ["python3", "run", "pinned", "cargo", "test"],
                ["rustup", "run", "pinned", "python3", "bench.py"],
                ["rustup", "show"],
                ["rustup", "toolchain", "install", "cargo", "test"],
                [],
            ):
                with self.subTest(command=command):
                    self.assertEqual(mod.toolchain(environment, command), without)
            # Stamped as it is read, so a run during which it changed goes
            # unrecorded.
            stamps: dict = {}
            mod.toolchain(environment, ["cargo", "test"], stamps)
            self.assertIn(named("rustdoc")["path"], stamps)

    def test_a_command_starts_rust_when_it_starts_rustup_or_one_of_its_proxies(self):
        # Rustup and its proxies resolve the toolchain's tools again when they
        # start, whatever spelling a platform gives them; another program, with
        # their names among its words, does not.
        for command in (
            ["cargo"],
            ["cargo", "run"],
            ["rustc", "lib.rs"],
            ["rustdoc", "lib.rs"],
            ["cargo-clippy"],
            ["cargo.exe", "test"],
            ["C:\\Rust\\CARGO.EXE", "test"],
            ["/opt/rust/bin/cargo", "test"],
            ["rustup"],
            ["rustup", "show"],
            ["rustup", "+pinned", "which", "cargo"],
            ["rustup", "run", "pinned", "cargo", "test"],
            ["rustup", "run", "pinned", "python3", "bench.py"],
            ["rustup.exe", "run", "pinned", "rustc"],
        ):
            with self.subTest(command=command):
                self.assertTrue(mod.starts_rust(command))
        for command in (
            [],
            ["python3", "bench.py", "cargo"],
            ["python3", "run", "pinned", "cargo"],
            ["sh", "-c", "cargo test"],
            ["cargo-foo", "test"],
            ["rustup-init"],
            ["./cargo/run"],
        ):
            with self.subTest(command=command):
                self.assertFalse(mod.starts_rust(command))

    def test_a_rust_tool_that_is_a_script_is_recorded_with_its_interpreters(self):
        # Cargo runs a `rustc` that is a script through the interpreter its
        # first line names, which a record must name by content as it does the
        # tool: one replaced between seeds would change the compiler that ran.
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools)
        interpreter = tools / "interp"
        interpreter.write_bytes(b"interpreter\n")
        interpreter.chmod(0o755)
        rustc = tools / "toolchains" / "pinned" / "bin" / "rustc"
        rustc.write_bytes(f"#!{interpreter}\n".encode("utf-8"))

        def named(path) -> dict:
            path = Path(path)
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        with mock.patch.object(mod, "ROOT", self.root):
            environment = {"PATH": str(bin_directory)}
            found = mod.toolchain(environment, ["cargo", "run"])
            self.assertEqual(found["rustc"], {**named(rustc), "interpreters": [named(interpreter)]})
            self.assertEqual(found["cargo"], named(tools / "toolchains" / "pinned" / "bin" / "cargo"))
            # Stamped as it is read, so a run during which it changed goes unrecorded.
            stamps: dict = {}
            mod.toolchain(environment, ["cargo", "run"], stamps)
            self.assertIn(str(interpreter.resolve()), stamps)
            # A standalone tool on the PATH is one as well.
            standalone = tools / "standalone"
            standalone.mkdir()
            script = standalone / "rustc"
            script.write_bytes(f"#!{interpreter}\n".encode("utf-8"))
            script.chmod(0o755)
            found = mod.toolchain({"PATH": str(standalone)}, ["rustc", "lib.rs"])
            self.assertEqual(found["rustc"], {**named(script), "interpreters": [named(interpreter)]})
            # One found nowhere refuses.
            rustc.write_bytes(b"#!/nowhere/interpreter\n")
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "the interpreter /nowhere/interpreter of a script"):
                mod.toolchain(environment, ["cargo", "run"])
        # A listed run records them, and refuses when one is found nowhere.
        rustc.write_bytes(f"#!{interpreter}\n".encode("utf-8"))
        search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
        self.preregister("running", entrypoint="cargo run -- <seed>")
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
        self.assertEqual(records[0]["toolchain"]["rustc"], {**named(rustc), "interpreters": [named(interpreter)]})
        self.tearDown()
        self.setUp()
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools)
        (tools / "toolchains" / "pinned" / "bin" / "rustc").write_bytes(b"#!/nowhere/interpreter\n")
        search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
        self.preregister("running", entrypoint="cargo run -- <seed>")
        ran = []
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("ERROR: refusing to run L900: the interpreter /nowhere/interpreter of a script is found nowhere", stderr)

    def test_a_rust_command_needs_the_tools_it_uses(self):
        # Cargo runs rustc (and rustdoc for documentation tests); rustc and
        # rustdoc run alone; rustup and its other proxies start whichever they
        # choose.
        tools = {"rustc": {}, "cargo": {}, "rustdoc": {}}
        everything = {"rustc", "cargo", "rustdoc"}
        for command, needed in (
            (["cargo", "run"], {"cargo", "rustc"}),
            (["cargo.exe", "build"], {"cargo", "rustc"}),
            (["cargo", "test"], everything),
            (["cargo", "test", "--doc"], everything),
            (["cargo", "test", "--lib"], {"cargo", "rustc"}),
            (["rustc", "lib.rs"], {"rustc"}),
            (["rustdoc", "lib.rs"], {"rustdoc"}),
            (["rustup", "run", "pinned", "rustc", "lib.rs"], {"rustc"}),
            (["rustup", "run", "pinned", "python3", "bench.py"], set()),
            (["rustup", "+pinned", "run", "--install", "pinned", "sh", "-c", "cargo test"], set()),
            (["python3", "bench.py"], set()),
            (["rustup", "run", "pinned", "cargo", "test"], everything),
            (["rustup", "show"], everything),
            (["rustfmt", "lib.rs"], everything),
            (["cargo-clippy"], everything),
        ):
            with self.subTest(command=command):
                self.assertEqual(mod.required_rust_tools(command, tools), needed)
        self.assertTrue(mod.invokes_rustdoc(["rustdoc", "lib.rs"]))
        self.assertTrue(mod.invokes_rustdoc(["rustup", "run", "pinned", "rustdoc", "lib.rs"]))

    def test_a_listed_rust_command_refuses_a_launch_when_a_tool_it_resolves_again_is_missing(self):
        # A proxy resolves its tools again when it starts: one rustup could
        # not name at the launch, and that appears later, would run code the
        # record never named, and one that stays absent would archive the
        # seed as a failed run.
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools, resolves=False)
        search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
        self.preregister("running", entrypoint="cargo run -- <seed>")
        ran = []
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, ran, records), (2, [], []))
        self.assertIn(
            "refusing to run L900: rustup or the PATH resolves no rustc, cargo, which the Rust command would resolve "
            "again when it starts",
            stderr,
        )
        self.assertEqual(list(self.attempts().glob("run-*.json")), [])
        # Cargo's test also needs rustdoc.
        self.tearDown()
        self.setUp()
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools)
        (tools / "toolchains" / "pinned" / "bin" / "rustdoc").unlink()
        search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
        self.preregister("running", entrypoint="cargo test -- <seed>")
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, ran, records), (2, [], []))
        self.assertIn("rustup or the PATH resolves no rustdoc, which the Rust command would resolve again", stderr)
        # Not where no documentation test runs: it needs no rustdoc then.
        for entrypoint in ("cargo test --all-targets -- <seed>", "cargo test --lib -- <seed>", "cargo run -- <seed>"):
            with self.subTest(entrypoint=entrypoint):
                self.tearDown()
                self.setUp()
                tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
                bin_directory = self.fake_rust(tools)
                (tools / "toolchains" / "pinned" / "bin" / "rustdoc").unlink()
                search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
                self.preregister("running", entrypoint=entrypoint)
                with mock.patch.dict(os.environ, {"PATH": search}):
                    status, records, stderr = self.run_seed()
                self.assertEqual(
                    (status, stderr, [record["status"] for record in records], sorted(records[0]["toolchain"])),
                    (0, "", ["completed"], ["cargo", "rustc"]),
                )
        # Only the tools the command uses are needed: a toolchain without Cargo
        # runs `rustc`, and one without rustc runs `rustdoc` alone.
        for entrypoint, removed, needed in (
            ("rustc -- <seed>", "cargo", ["cargo", "rustc"]),
            ("rustdoc -- <seed>", "cargo", ["cargo", "rustc", "rustdoc"]),
            ("rustdoc -- <seed>", "rustc", ["cargo", "rustc", "rustdoc"]),
        ):
            with self.subTest(entrypoint=entrypoint, removed=removed):
                self.tearDown()
                self.setUp()
                tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
                bin_directory = self.fake_rust(tools)
                (tools / "toolchains" / "pinned" / "bin" / removed).unlink()
                search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
                self.preregister("running", entrypoint=entrypoint)
                with mock.patch.dict(os.environ, {"PATH": search}):
                    status, records, stderr = self.run_seed()
                self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
                self.assertEqual(sorted(records[0]["toolchain"]), needed)
        # And the tool it uses is needed: Cargo without rustc, rustdoc without itself.
        for entrypoint, removed, message in (
            ("cargo run -- <seed>", "rustc", "resolves no rustc,"),
            ("cargo run -- <seed>", "cargo", "resolves no cargo,"),
            ("rustdoc -- <seed>", "rustdoc", "resolves no rustdoc,"),
        ):
            with self.subTest(entrypoint=entrypoint, removed=removed):
                self.tearDown()
                self.setUp()
                tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
                bin_directory = self.fake_rust(tools)
                (tools / "toolchains" / "pinned" / "bin" / removed).unlink()
                search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
                self.preregister("running", entrypoint=entrypoint)
                ran = []
                with mock.patch.dict(os.environ, {"PATH": search}):
                    status, records, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, records, ran), (2, [], []))
                self.assertIn(message, stderr)
        # A Rust command whose tools all resolve runs, whose record names them.
        self.tearDown()
        self.setUp()
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools)
        search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
        self.preregister("running", entrypoint="cargo run -- <seed>")
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))
        self.assertEqual(sorted(records[0]["toolchain"]), ["cargo", "rustc"])
        self.assertTrue(all(program["path"] is not None for program in records[0]["toolchain"].values()))
        # One that is found but cannot be read is refused as unread, not as unresolved.
        unread = {"path": str(tools / "toolchains" / "pinned" / "bin" / "cargo"), "sha256": None}
        found = {"path": str(tools / "toolchains" / "pinned" / "bin" / "rustc"), "sha256": "0" * 64}
        self.tearDown()
        self.setUp()
        self.preregister("running", entrypoint="cargo run -- <seed>")
        with (
            mock.patch.dict(os.environ, {"PATH": search}),
            mock.patch.object(mod, "toolchain", return_value={"rustc": found, "cargo": unread}),
        ):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn(f"refusing to run L900: {unread['path']} cannot be read, or changed while it was read", stderr)
        # A command that is no Rust command does not need them.
        self.tearDown()
        self.setUp()
        tools = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_directory = self.fake_rust(tools, resolves=False)
        search = os.pathsep.join((str(bin_directory), os.environ.get("PATH", os.defpath)))
        self.preregister("running")
        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr, [record["status"] for record in records]), (0, "", ["completed"]))

    def test_a_script_the_command_starts_binds_the_interpreter_its_shebang_names(self):
        # The kernel runs the interpreter a script's first line names, which
        # the record must name by content as it does the script: one replaced
        # between seeds would change what ran under records that agree.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))
        interpreter = outside / "interp"
        interpreter.write_bytes(b"interpreter\n")
        interpreter.chmod(0o755)
        fake = outside / "fake"
        fake.write_bytes(b"fake\n")
        fake.chmod(0o755)
        env = shutil.which("env")
        self.assertIsNotNone(env)

        def named(path) -> dict:
            path = Path(path)
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        def program(first_line: str) -> Path:
            (outside / "bench").write_text(first_line + "\necho bench\n", encoding="utf-8")
            (outside / "bench").chmod(0o755)
            return outside / "bench"

        def clear():
            for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
                record.unlink()

        self.preregister("running")
        search = os.pathsep.join((str(outside), os.environ.get("PATH", os.defpath)))
        # A named interpreter, and `env` with the program it looks up.
        for first_line, bound in (
            (f"#!{interpreter}", [named(interpreter)]),
            (f"#!{interpreter} -x", [named(interpreter)]),
            (f"#!{env} fake", [named(env), named(fake)]),
        ):
            with self.subTest(first_line=first_line), mock.patch.dict(os.environ, {"PATH": search}):
                script = program(first_line)
                status, records, stderr = self.run_seed()
                self.assertEqual((status, stderr), (0, ""))
                self.assertEqual(records[0]["executable"], {**named(script), "interpreters": bound})
                clear()
        # A file that is no script names none, nor does one whose first line
        # is a comment, or a `#!` naming no word.
        for content in (b"plain\n", b"# #!/no/interpreter\n", b"#!\n", b"#! \t\nrest\n", b"\n#!/no/interpreter\n"):
            with self.subTest(content=content), mock.patch.dict(os.environ, {"PATH": search}):
                (outside / "bench").write_bytes(content)
                (outside / "bench").chmod(0o755)
                status, records, stderr = self.run_seed()
                self.assertEqual((status, stderr, records[0]["executable"]), (0, "", named(outside / "bench")))
                clear()
        # An `env` given more than a program's name refuses the run: what it
        # would run is read by rules no record could bind.
        program(f"#!{env} -S PATH=relative fake")
        with mock.patch.dict(os.environ, {"PATH": search}):
            ran = []
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn(
            "ERROR: refusing to run L900: a script's first line gives env more than a program's name (an option, an "
            "assignment or several words), so its record could not name by its content what the script runs through",
            stderr,
        )
        # An interpreter found nowhere refuses it as well: it could appear before
        # the command starts and run with no record naming it.
        program(f"#!{outside}/nowhere")
        with mock.patch.dict(os.environ, {"PATH": search}):
            ran = []
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn(
            f"ERROR: refusing to run L900: the interpreter {outside}/nowhere of a script is found nowhere, and would "
            "be looked up again when the command starts, so its record could not name by its content what the "
            "script runs through",
            stderr,
        )
        # An interpreter replaced for the run and put back leaves it unrecorded.
        program(f"#!{interpreter}")
        time.sleep(0.05)

        def swapped_and_put_back():
            os.link(interpreter, outside / "kept")
            (outside / "other").write_bytes(b"other\n")
            (outside / "other").chmod(0o755)
            os.replace(outside / "other", interpreter)
            os.replace(outside / "kept", interpreter)

        with mock.patch.dict(os.environ, {"PATH": search}):
            status, records, stderr = self.run_seed(swapped_and_put_back)
        self.assertEqual((status, [record["status"] for record in records]), (2, ["started"]))
        self.assertIn(f"{interpreter.resolve()} changed since the launch read it", stderr)

    def test_a_scripts_interpreters_are_followed_through_scripts_to_a_bounded_depth(self):
        # An interpreter that is a script runs through its own, and so on as
        # far as the kernel follows them, so each is named by content; the
        # depth is bounded so that a script naming itself ends.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))

        def write(name: str, content: bytes) -> Path:
            (outside / name).write_bytes(content)
            (outside / name).chmod(0o755)
            return outside / name

        def named(path) -> dict:
            path = Path(path)
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        binary = write("binary", b"binary\n")
        chain = [binary]
        for index in range(10):
            chain.append(write(f"s{index}", f"#!{chain[-1]}\n".encode("utf-8")))
        environment = {"PATH": str(outside)}
        with mock.patch.object(mod, "ROOT", self.root):
            self.assertEqual(mod.interpreters_of(str(binary), environment, None), [])
            self.assertEqual(mod.interpreters_of(str(chain[1]), environment, None), [named(binary)])
            self.assertEqual(mod.interpreters_of(str(chain[2]), environment, None), [named(chain[1]), named(binary)])
            # As far as `SHEBANG_DEPTH` levels, above what any kernel follows;
            # a script found at the bound refuses, so no interpreter behind a
            # chain of scripts is left unbound.
            self.assertEqual(mod.SHEBANG_DEPTH, 8)
            self.assertEqual(
                mod.interpreters_of(str(chain[8]), environment, None),
                [named(path) for path in reversed(chain[:8])],
            )
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "more than 8 deep"):
                mod.interpreters_of(str(chain[9]), environment, None)
            # The program `env` runs is followed too, and a script `env` starts
            # is a new `exec`, whose interpreters the kernel follows anew.
            env = shutil.which("env")
            self.assertIsNotNone(env)
            fake = write("fake", f"#!{chain[1]}\n".encode("utf-8"))
            script = write("bench", f"#!{env} fake\n".encode("utf-8"))
            self.assertEqual(
                mod.interpreters_of(str(script), environment, None),
                [named(env), named(fake), named(chain[1]), named(binary)],
            )
            # A link of another name to `env` runs it as well, so it is read as
            # `env` is: the program it runs is followed, anything else refuses.
            alias = outside / "myenv"
            alias.symlink_to(env)
            script = write("bench", f"#!{alias} fake\n".encode("utf-8"))
            self.assertEqual(
                mod.interpreters_of(str(script), environment, None),
                [named(alias), named(fake), named(chain[1]), named(binary)],
            )
            script = write("bench", f"#!{alias} -i fake\n".encode("utf-8"))
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "gives env more than a program's name"):
                mod.interpreters_of(str(script), environment, None)
            # A program spelled `env` that is a link to a multi-call binary is
            # `env` as well, as the binary reads the name it is started by.
            multicall = write("multicall", b"multi-call binary\n")
            spelled = outside / "spelled"
            spelled.mkdir()
            (spelled / "env").symlink_to(multicall)
            script = write("bench", f"#!{spelled / 'env'} fake\n".encode("utf-8"))
            self.assertEqual(
                mod.interpreters_of(str(script), environment, None),
                [named(spelled / "env"), named(fake), named(chain[1]), named(binary)],
            )
            # `env` given no program is given the script itself by the kernel,
            # which starts itself again: refused.
            script = write("bench", f"#!{env}\nfake\n".encode("utf-8"))
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "gives env no program"):
                mod.interpreters_of(str(script), environment, None)
            # The kernel's levels start again for what `env` starts: a script
            # and seven behind it are followed to the end, which counted from
            # the top would pass the bound.
            top = write("top", f"#!{env} chain\n".encode("utf-8"))
            write("chain", f"#!{chain[7]}\n".encode("utf-8"))
            self.assertEqual(
                mod.interpreters_of(str(top), environment, None),
                [named(env), named(outside / "chain"), *[named(path) for path in reversed(chain[:8])]],
            )
            # Four such starts are followed, a fifth is not.
            for index in range(1, 6):
                write(f"e{index}", f"#!{env} e{index + 1}\n".encode("utf-8"))
            write("e5", f"#!{chain[1]}\n".encode("utf-8"))
            self.assertEqual(
                mod.interpreters_of(str(outside / "e1"), environment, None),
                [named(env), named(outside / "e2"), named(env), named(outside / "e3"), named(env),
                 named(outside / "e4"), named(env), named(outside / "e5"), named(chain[1]), named(binary)],
            )
            write("e5", f"#!{env} e6\n".encode("utf-8"))
            write("e6", f"#!{chain[1]}\n".encode("utf-8"))
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "through env more than 4 deep"):
                mod.interpreters_of(str(outside / "e1"), environment, None)
            # Scripts starting one another through `env` end at a bound.
            write("ping", f"#!{env} pong\n".encode("utf-8"))
            write("pong", f"#!{env} ping\n".encode("utf-8"))
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "through env more than 4 deep"):
                mod.interpreters_of(str(outside / "ping"), environment, None)
            # Any other form refuses: an option, an assignment, several words.
            for words in (
                "-i fake",
                "-S fake",
                "-S 'foo bar'",
                "--split-string=fake",
                "-u PYTHONPATH fake",
                "-C /tmp fake",
                "-P /bin fake",
                "PATH=/bin fake",
                "A=1",
                "fake extra",
                "-- fake",
                "-",
            ):
                with self.subTest(words=words):
                    script = write("bench", f"#!{env} {words}\n".encode("utf-8"))
                    with self.assertRaisesRegex(mod.ScriptInterpreterError, "gives env more than a program's name"):
                        mod.interpreters_of(str(script), environment, None)
            # A script naming itself ends at the bound, refused.
            looping = write("looping", f"#!{outside / 'looping'}\n".encode("utf-8"))
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "more than 8 deep"):
                mod.interpreters_of(str(looping), environment, None)
            # An interpreter found nowhere refuses, as the kernel or `env` would
            # look it up again when the command starts, and so does one whose
            # first line no encoding reads (it names a file no one has).
            missing = write("missing", b"#!/nowhere/interpreter\n")
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "the interpreter /nowhere/interpreter of a script"):
                mod.interpreters_of(str(missing), environment, None)
            undecodable = write("undecodable", b"#!/nowhere/\xff\xfe interpreter\n")
            with self.assertRaises(mod.ScriptInterpreterError):
                mod.interpreters_of(str(undecodable), environment, None)
            # The kernel takes a name without a slash as a path from the
            # directory the command starts in, the root, and looks up no PATH.
            (self.root / "bare").write_bytes(b"in the root\n")
            (self.root / "bare").chmod(0o755)
            write("bare", b"on the path\n")
            bare = write("bench", b"#!bare\n")
            self.assertEqual(mod.interpreters_of(str(bare), environment, None), [named(self.root / "bare")])
            (self.root / "bare").unlink()
            with self.assertRaisesRegex(mod.ScriptInterpreterError, "the interpreter bare of a script"):
                mod.interpreters_of(str(bare), environment, None)
            # Each is stamped as it is read.
            stamps: dict = {}
            mod.interpreters_of(str(chain[3]), environment, stamps)
            for interpreter in (chain[2], chain[1], binary):
                self.assertIn(str(interpreter.resolve()), stamps)

    def test_a_shebang_is_split_as_the_kernel_splits_it(self):
        # The kernel splits a first line on a space or a tab only, so a
        # carriage return, another whitespace byte or a byte no encoding reads
        # stays part of a name: a CRLF file's `python3\r` is not `python3`.
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

        def words(content: bytes):
            path = directory / "script"
            path.write_bytes(content)
            return mod.shebang_words(str(path))

        self.assertEqual(words(b"#!/usr/bin/env python3\n"), ["/usr/bin/env", "python3"])
        self.assertEqual(words(b"#!/usr/bin/env python3\r\n"), ["/usr/bin/env", "python3\r"])
        self.assertEqual(words(b"#!/x\r\n"), ["/x\r"])
        self.assertEqual(words(b"#!\t /a \t b \t\nrest\n"), ["/a", "b"])
        self.assertEqual(words(b"#!/a\x0bb\x0cc\n"), ["/a\x0bb\x0cc"])
        self.assertEqual(words("#!/a\u00a0b c\n".encode("utf-8")), ["/a\u00a0b", "c"])
        self.assertEqual(words(b"#!/x\xff\xfe y\n"), ["/x\udcff\udcfe", "y"])
        self.assertEqual(os.fsencode(words(b"#!/x\xff\xfe y\n")[0]), b"/x\xff\xfe")
        # Decoded as the file system encodes names, whatever that is: where it
        # is not UTF-8 the bytes of a name must come back the same.
        with mock.patch.object(mod.os, "fsdecode", side_effect=lambda name: name.decode("latin-1")):
            self.assertEqual(words(b"#!/x\xc3\xa9 y\n"), ["/x\xc3\xa9", "y"])
        for none in (b"", b"#", b"#!", b"#!\n", b"#! \t\nrest\n", b"# /x\n", b"\n#!/x\n", b"plain\n"):
            with self.subTest(content=none):
                self.assertIsNone(words(none))
        # A carriage return in a name found nowhere refuses the run.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))
        program = outside / "python3"
        program.write_bytes(b"python\n")
        program.chmod(0o755)
        env = shutil.which("env")
        script = outside / "bench"
        script.write_bytes(f"#!{env} python3\r\n".encode("utf-8"))
        script.chmod(0o755)
        with mock.patch.object(mod, "ROOT", self.root), self.assertRaisesRegex(
            mod.ScriptInterpreterError, "the interpreter python3\r of a script is found nowhere"
        ):
            mod.interpreters_of(str(script), {"PATH": str(outside)}, None)

    def test_env_in_a_shebang_looks_its_program_up_on_the_runners_path(self):
        # `#!/usr/bin/env prog` runs the `prog` of the PATH `env` starts with,
        # the runner's, by a name with a slash from the root; one absent
        # refuses the run, as `env` would look it up again.
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))
        first, second = outside / "first", outside / "second"
        first.mkdir()
        second.mkdir()

        def write(path: Path, content: bytes) -> Path:
            path.write_bytes(content)
            path.chmod(0o755)
            return path

        def named(path) -> dict:
            path = Path(path)
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        env = shutil.which("env")
        self.assertIsNotNone(env)
        for directory in (first, second):
            write(directory / "prog", f"{directory.name}\n".encode("utf-8"))

        def interpreters(words: str, path: str) -> list[dict]:
            script = write(outside / "bench", f"#!{env} {words}\n".encode("utf-8"))
            return mod.interpreters_of(str(script), {"PATH": path}, None)

        with mock.patch.object(mod, "ROOT", self.root):
            self.assertEqual(interpreters("prog", str(first)), [named(env), named(first / "prog")])
            self.assertEqual(interpreters("prog", str(second)), [named(env), named(second / "prog")])
            self.assertEqual(
                interpreters("prog", f"{second}{os.pathsep}{first}"), [named(env), named(second / "prog")]
            )
            (self.root / "rel").mkdir()
            write(self.root / "rel" / "tool", b"tool\n")
            self.assertEqual(interpreters("rel/tool", str(first)), [named(env), named(self.root / "rel" / "tool")])
            self.assertEqual(interpreters(f"{second}/prog", str(first)), [named(env), named(second / "prog")])
            for words, path in (("absent", str(first)), ("prog", "/nowhere"), ("./tool", str(first))):
                with self.subTest(words=words, path=path), self.assertRaisesRegex(
                    mod.ScriptInterpreterError, "of a script is found nowhere"
                ):
                    interpreters(words, path)

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

    def test_the_frozen_values_of_a_launch_are_read_from_the_commit_the_watch_holds(self):
        # A directory replaced while the launch reads it, and put back before
        # the watch stamps it, would supply a table no commit holds: the
        # values come from the commit's own blob, whatever the tree holds
        # when they are read.
        self.preregister("running", "schema = 1\nprogram = \"bench\"\n", entrypoint="<program> <seed>")
        directory = self.root / "experiments/x/L900-x"
        data = tomllib.loads((directory / "experiment.toml").read_text(encoding="utf-8"))
        self.write(
            "experiments/x/L900-x/config.toml",
            'version = 1\n\n[preregistration]\nseeds = [17]\nschema = 1\nprogram = "/tmp/attacker"\n',
        )
        with mock.patch.object(mod, "ROOT", self.root):
            self.assertEqual(mod.command_parameters("L900", directory, data, "entrypoint", {}), {"program": "/tmp/attacker"})
            self.assertEqual(
                mod.command_parameters("L900", directory, data, "entrypoint", {}, self.head), {"program": "bench"}
            )
            # A repeated --set value is compared with the commit's, not the tree's.
            with self.assertRaises(ValueError) as caught:
                mod.command_parameters("L900", directory, data, "entrypoint", {"program": "/tmp/attacker"}, self.head)
            self.assertIn("--set program=/tmp/attacker is not the preregistered value bench", str(caught.exception))
            # A commit that holds no such file freezes no values.
            with self.assertRaises(ValueError) as caught:
                mod.command_parameters("L900", directory / "elsewhere", data, "entrypoint", {}, self.head)
            self.assertIn("is not a TOML file that commit holds, so it freezes no values", str(caught.exception))
        # The launch passes the commit its watch holds.
        git(self.root, "checkout", "-q", "--", "experiments/x/L900-x/config.toml")
        seen = []
        real = mod.command_parameters

        def recording(*arguments):
            seen.append(arguments[5:])
            return real(*arguments)

        with mock.patch.object(mod, "command_parameters", side_effect=recording):
            status, records, stderr = self.run_seed()
        self.assertEqual((status, seen), (0, [(self.head,)]))
        self.assertEqual(records[0]["command"], ["bench", "17"])
        # The manifest the command was built from is compared with the one
        # the commit holds, not with what the tree holds when it is read.
        for record in (*self.results.glob("run-*.json"), *self.attempts().glob("run-*.json")):
            record.unlink()
        real_toml_at = mod.check_research_gates.toml_at
        real_launch_errors = mod.check_research_gates.launch_errors
        decided = []

        def after_decision(*arguments):
            problems = real_launch_errors(*arguments)
            decided.append(True)
            return problems

        def replaced(root, commit, relative):
            held = real_toml_at(root, commit, relative)
            # Once the launch has been decided twice (before the watch and
            # by it), what the watch reads last is replaced.
            if len(decided) >= 2 and relative.endswith("experiment.toml") and held is not None:
                return {**held, "entrypoint": "other <seed>"}
            return held

        ran = []
        with (
            mock.patch.object(mod.check_research_gates, "launch_errors", side_effect=after_decision),
            mock.patch.object(mod.check_research_gates, "toml_at", side_effect=replaced),
        ):
            status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("experiment.toml changed while the launch was checked", stderr)

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
