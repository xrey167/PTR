import contextlib
import hashlib
import importlib.util
import io
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

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


    def test_records_measure_the_host_and_bind_the_declared_profile(self):
        _, root, data = mod.resolve("M001")
        record = mod.base_record("M001", data, root)
        self.assertEqual(record["schema_version"], 2)
        self.assertIn(record["git_dirty"], (True, False))
        self.assertEqual(len(record["git_tracked_diff_sha256"]), 64)
        host = record["host"]
        self.assertGreaterEqual(host["logical_cpus"], 1)
        self.assertTrue(host["system"])
        profile = record["hardware_profile_record"]
        self.assertEqual(profile["path"], data["hardware_profile"])
        self.assertEqual(
            profile["sha256"],
            hashlib.sha256((ROOT / data["hardware_profile"]).read_bytes()).hexdigest(),
        )
        # hardware/default.toml declares nothing, and the record says so.
        if data["hardware_profile"] == "hardware/default.toml":
            self.assertIn("cpu", profile["unspecified_fields"])

    def test_toolchain_asks_rustc_for_the_toolchain_the_command_names(self):
        calls = []
        real = mod.subprocess.check_output

        def fake(argv, **kwargs):
            calls.append(argv)
            return "rustc 9.99.0 (fake)\n"

        mod.subprocess.check_output = fake
        try:
            self.assertEqual(
                mod.toolchain(["cargo", "+1.95.0", "run", "--release"]), "rustc 9.99.0 (fake)"
            )
            self.assertEqual(mod.toolchain(["cargo", "run"]), "rustc 9.99.0 (fake)")
            self.assertIsNone(mod.toolchain([sys.executable, "-c", "pass"]))
        finally:
            mod.subprocess.check_output = real
        self.assertEqual(calls, [["rustc", "+1.95.0", "--version"], ["rustc", "--version"]])


SEEDS = [1, 2, 3]


class Aggregation(unittest.TestCase):
    """`aggregate` against fixture trees: what it computes, and what it refuses."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.experiment = self.root / "experiments/model/X001-fixture"
        self.results = self.experiment / "results"
        self.results.mkdir(parents=True)
        (self.root / "experiments/registry.toml").write_text(
            '[[experiment]]\nid = "X001"\npath = "model/X001-fixture"\nstatus = "planned"\n',
            encoding="utf-8",
        )
        (self.experiment / "experiment.toml").write_text(
            f'id = "X001"\nseeds = {SEEDS}\nresults_dir = "results"\n', encoding="utf-8"
        )
        saved = mod.ROOT, mod.REGISTRY
        mod.ROOT, mod.REGISTRY = self.root, self.root / "experiments/registry.toml"
        self.addCleanup(lambda: setattr(mod, "ROOT", saved[0]))
        self.addCleanup(lambda: setattr(mod, "REGISTRY", saved[1]))
        self.count = 0

    def write_run(self, seed, rows, *, status="completed", sha="a" * 40, dirty=False,
                  entrypoint="ablation", stdout=None):
        self.count += 1
        record = {
            "schema_version": 2,
            "experiment_id": "X001",
            "git_sha": sha,
            "git_dirty": dirty,
            "entrypoint": entrypoint,
            "seed": seed,
            "status": status,
            "exit_code": 0 if status == "completed" else 1,
            "stdout": stdout if stdout is not None else "".join(
                "compiling...\n" + json.dumps(row) + "\n" for row in rows
            ),
        }
        path = self.results / f"run-2026{self.count:04d}-seed-{seed}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def run_aggregate(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            code = mod.aggregate("X001", entrypoint="ablation", **kwargs)
        (out,) = sorted(self.results.glob("aggregate-*.json"))
        return code, json.loads(out.read_text(encoding="utf-8"))

    def test_rows_are_grouped_by_their_labels_and_summarized_across_seeds(self):
        values = {1: (0.50, 0.30), 2: (0.70, 0.20), 3: (0.60, 0.40)}
        paths = []
        for seed, (typed, ablated) in values.items():
            paths.append(self.write_run(seed, [
                {"arm": "typed", "split": "ood", "accuracy": typed, "seed": seed, "passed": True},
                {"arm": "ablated", "split": "ood", "accuracy": ablated, "seed": seed},
            ]))
        # A run of a different entrypoint is not this aggregate's input.
        self.write_run(1, [{"arm": "typed", "accuracy": 9.0}], entrypoint="other")

        code, summary = self.run_aggregate()

        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_seeds"], SEEDS)
        self.assertEqual(summary["git_sha"], "a" * 40)
        by_arm = {group["key"]["arm"]: group for group in summary["groups"]}
        self.assertEqual(set(by_arm), {"typed", "ablated"})
        typed = by_arm["typed"]["metrics"]
        self.assertEqual(set(typed), {"accuracy"})  # seed is a label, a bool is no metric
        self.assertEqual(typed["accuracy"]["n"], 3)
        self.assertAlmostEqual(typed["accuracy"]["mean"], 0.6)
        self.assertAlmostEqual(typed["accuracy"]["std"], statistics.stdev([0.5, 0.7, 0.6]))
        self.assertEqual(typed["accuracy"]["by_seed"], {"1": 0.5, "2": 0.7, "3": 0.6})
        self.assertEqual(
            summary["source_records"],
            [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(paths)],
        )

    def test_failed_and_missing_seeds_are_reported_not_dropped(self):
        self.write_run(1, [{"arm": "typed", "accuracy": 0.5}])
        failed = self.write_run(2, [], status="failed")

        code, summary = self.run_aggregate()

        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["missing_seeds"], [2, 3])
        self.assertEqual(
            summary["failed_runs"],
            [{"seed": 2, "status": "failed", "exit_code": 1, "record": failed.name}],
        )
        self.assertIsNone(summary["groups"][0]["metrics"]["accuracy"]["std"])

    def test_what_it_refuses_writes_nothing(self):
        cases = [
            ("span 2 commits", lambda: (self.write_run(1, [{"a": 1}]),
                                        self.write_run(2, [{"a": 1}], sha="b" * 40))),
            ("dirty or unrecorded", lambda: self.write_run(1, [{"a": 1}], dirty=None)),
            ("dirty or unrecorded", lambda: self.write_run(1, [{"a": 1}], dirty=True)),
            ("two completed records", lambda: (self.write_run(1, [{"a": 1}]),
                                               self.write_run(1, [{"a": 2}]))),
            ("printed more than once", lambda: self.write_run(1, [{"arm": "x", "a": 1},
                                                                  {"arm": "x", "a": 2}])),
            ("not valid JSON", lambda: self.write_run(1, [], stdout='{"a": 1\n')),
            ("no JSON metric rows", lambda: self.write_run(1, [], stdout="done\n")),
            ("no run records", lambda: None),
        ]
        for message, arrange in cases:
            with self.subTest(message=message):
                for path in self.results.glob("*.json"):
                    path.unlink()
                arrange()
                with self.assertRaisesRegex(ValueError, message):
                    mod.aggregate("X001", entrypoint="ablation")
                self.assertEqual(list(self.results.glob("aggregate-*.json")), [])

    def test_an_explicit_commit_and_allow_dirty_are_honoured_and_recorded(self):
        self.write_run(1, [{"a": 1.0}], sha="a" * 40)
        self.write_run(1, [{"a": 5.0}], sha="b" * 40, dirty=True)

        code, summary = self.run_aggregate(git_sha_filter="b" * 40, allow_dirty=True)

        self.assertEqual(code, 1)  # seeds 2 and 3 never ran
        self.assertEqual(summary["git_sha"], "b" * 40)
        self.assertTrue(summary["allow_dirty"])
        self.assertEqual(summary["groups"][0]["metrics"]["a"]["mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
