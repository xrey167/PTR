import contextlib
import errno
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
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

    def test_build_command_rejects_unlisted_seed(self):
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "not declared"):
            mod.build_command(
                data,
                entrypoint="entrypoint",
                seed=999,
                params={"iterations": "3"},
            )

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

    def run_seed(self, during=None):
        """Run seed 17 of L900 in the temporary repository, calling `during`
        while the command "runs"; returns the exit status, the records written
        and what was printed to stderr."""

        def execute(_command):
            if during is not None:
                during()
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

    def test_a_record_the_results_hold_is_no_change_of_the_sources(self):
        # Records accumulate in results/, which the run writes into itself.
        status, records, _ = self.run_seed(lambda: self.write("experiments/x/L900-x/results/other.json", "{}\n"))
        self.assertEqual(status, 0)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
