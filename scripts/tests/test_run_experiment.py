import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_experiment", ROOT / "scripts/run_experiment.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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


if __name__ == "__main__":
    unittest.main()
