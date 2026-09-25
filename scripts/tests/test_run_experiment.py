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
        """Check run provenance includes worktree evidence, host facts, and the hardware-profile hash."""
        _, root, data = mod.resolve("M001")
        record = mod.base_record("M001", data, root)
        self.assertEqual(record["schema_version"], 2)
        if (ROOT / ".git").exists():
            self.assertIn(record["git_dirty"], (True, False))
            self.assertEqual(len(record["git_tracked_diff_sha256"]), 64)
        else:
            # An export without history cannot say whether it matches a commit,
            # and says so rather than claiming clean.
            self.assertIsNone(record["git_dirty"])
            self.assertIsNone(record["git_tracked_diff_sha256"])
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
        """Query the selected Cargo toolchain and avoid Rust queries for non-Cargo commands."""
        calls = []
        real = mod.subprocess.check_output

        def fake(argv, **kwargs):
            """Record the requested rustc command and return a synthetic version string."""
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
        """Create an isolated experiment registry and redirect runner paths for aggregation tests."""
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
                  entrypoint="ablation", stdout=None, extra=None, raw=None):
        """Write a configurable synthetic run record or raw malformed input and return its path."""
        self.count += 1
        path = self.results / f"run-2026{self.count:04d}-seed-{seed}.json"
        if raw is not None:
            path.write_text(raw, encoding="utf-8")
            return path
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
        record.update(extra or {})
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def run_aggregate(self, **kwargs):
        """Aggregate the fixture's ablation runs and return the exit code and JSON summary."""
        with contextlib.redirect_stdout(io.StringIO()):
            code = mod.aggregate("X001", entrypoint="ablation", **kwargs)
        (out,) = sorted(self.results.glob("aggregate-*.json"))
        return code, json.loads(out.read_text(encoding="utf-8"))

    def test_rows_are_grouped_by_their_labels_and_summarized_across_seeds(self):
        """Group metrics by arm and split, summarize seeds, and bind selected input hashes."""
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
        """Mark incomplete runs explicitly, preserving failed-run details and missing seeds."""
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
        """Reject incompatible, ambiguous, or malformed inputs without writing an aggregate."""
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

    def test_gaps_in_what_completed_runs_reported_make_it_incomplete(self):
        """Mark aggregates incomplete for non-finite values, missing metrics, and missing rows."""
        # Seed 1 diverged (NaN), seed 2 printed an extra row, seed 3 left a metric out.
        self.write_run(1, [], stdout='{"arm": "a", "loss": NaN, "acc": 0.5}\n')
        self.write_run(2, [{"arm": "a", "loss": 0.2, "acc": 0.7}, {"arm": "b", "loss": 0.1}])
        self.write_run(3, [{"arm": "a", "loss": 0.4, "acc": None}])

        code, summary = self.run_aggregate()

        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "incomplete")
        by_arm = {group["key"]["arm"]: group for group in summary["groups"]}
        loss = by_arm["a"]["metrics"]["loss"]
        self.assertEqual((loss["n"], loss["non_finite"]), (2, {"1": "nan"}))
        self.assertAlmostEqual(loss["mean"], 0.3)
        self.assertEqual(by_arm["a"]["metrics"]["acc"]["missing_seeds"], [3])
        self.assertEqual(by_arm["b"]["missing_seeds"], [1, 3])
        self.assertEqual(by_arm["a"]["missing_seeds"], [])

    def test_infinity_is_listed_not_averaged(self):
        """Exclude infinity from summary statistics while recording its seed as non-finite."""
        for seed in SEEDS:
            self.write_run(seed, [], stdout=f'{{"acc": {1e400 if seed == 2 else 0.5}}}\n'.replace("inf", "Infinity"))
        code, summary = self.run_aggregate()
        acc = summary["groups"][0]["metrics"]["acc"]
        self.assertEqual((code, acc["n"], acc["mean"], acc["non_finite"]), (1, 2, 0.5, {"2": "inf"}))

    def test_records_that_do_not_measure_the_same_thing_are_refused(self):
        """Reject runs with differing parameters, toolchains, hosts, or manifest hashes."""
        cases = [
            ("parameters", {"parameters": {"iterations": "100"}}, {"parameters": {"iterations": "100000"}}),
            ("rustc", {"rustc": "rustc 1.95.0"}, {"rustc": "rustc 1.98.1"}),
            ("host", {"host": {"cpu_model": "x"}}, {"host": {"cpu_model": "y"}}),
            ("manifest_sha256", {"manifest_sha256": "1" * 64}, {"manifest_sha256": "2" * 64}),
        ]
        for key, first, second in cases:
            with self.subTest(key=key):
                for path in self.results.glob("*.json"):
                    path.unlink()
                self.write_run(1, [{"a": 1}], extra=first)
                self.write_run(2, [{"a": 1}], extra=second)
                with self.assertRaisesRegex(ValueError, f"records differ in {key}"):
                    mod.aggregate("X001", entrypoint="ablation")
                self.assertEqual(list(self.results.glob("aggregate-*.json")), [])

    def test_different_tracked_edits_are_refused_even_with_allow_dirty(self):
        for seed in (1, 2):
            self.write_run(seed, [{"a": 1}], dirty=True,
                           extra={"git_tracked_diff_sha256": str(seed) * 64})
        with self.assertRaisesRegex(ValueError, "records differ in git_tracked_diff_sha256"):
            mod.aggregate("X001", entrypoint="ablation", allow_dirty=True)
        self.assertEqual(list(self.results.glob("aggregate-*.json")), [])

    def test_matching_tracked_diff_hashes_remain_compatible(self):
        for dirty, digest in ((False, hashlib.sha256(b"").hexdigest()), (True, "1" * 64)):
            with self.subTest(dirty=dirty):
                for path in self.results.glob("*.json"):
                    path.unlink()
                for seed in SEEDS:
                    self.write_run(seed, [{"a": 1}], dirty=dirty,
                                   extra={"git_tracked_diff_sha256": digest})
                code, summary = self.run_aggregate(allow_dirty=dirty)
                self.assertEqual((code, summary["status"]), (0, "complete"))

    def test_malformed_records_are_errors_not_tracebacks(self):
        """Return validation errors for invalid seeds, stdout, numeric values, and record JSON."""
        cases = [
            ("not one of the manifest's seeds", lambda: self.write_run(99, [{"a": 1}])),
            ("seed must be an integer", lambda: self.write_run("1", [{"a": 1}])),
            ("seed must be an integer", lambda: self.write_run(True, [{"a": 1}])),
            ("seed must be an integer", lambda: self.write_run(None, [{"a": 1}])),
            ("stdout must be text", lambda: self.write_run(1, [], extra={"stdout": None})),
            ("does not fit a float", lambda: self.write_run(1, [], stdout='{"a": 1' + "0" * 400 + '}\n')),
            ("not a readable run record", lambda: self.write_run(1, [], raw="{broken")),
            ("not a JSON object", lambda: self.write_run(1, [], raw="[1, 2]")),
        ]
        for message, arrange in cases:
            with self.subTest(message=message):
                for path in self.results.glob("*.json"):
                    path.unlink()
                arrange()
                with self.assertRaisesRegex(ValueError, message):
                    mod.aggregate("X001", entrypoint="ablation")
                self.assertEqual(list(self.results.glob("aggregate-*.json")), [])

    def test_the_cli_reports_a_refusal_as_an_error_line(self):
        """Translate aggregation refusal into an ERROR message and CLI exit status two."""
        self.write_run(99, [{"a": 1}])
        argv = sys.argv
        sys.argv = ["run_experiment.py", "aggregate", "X001", "--entrypoint", "ablation"]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as raised:
                mod.main()
        finally:
            sys.argv = argv
        self.assertEqual(raised.exception.code, 2)
        self.assertTrue(err.getvalue().startswith("ERROR: "), err.getvalue())

    def test_an_explicit_commit_and_allow_dirty_are_honoured_and_recorded(self):
        """Honor explicit commit and dirty-worktree overrides and record them in the summary."""
        self.write_run(1, [{"a": 1.0}], sha="a" * 40)
        self.write_run(1, [{"a": 5.0}], sha="b" * 40, dirty=True)

        code, summary = self.run_aggregate(git_sha_filter="b" * 40, allow_dirty=True)

        self.assertEqual(code, 1)  # seeds 2 and 3 never ran
        self.assertEqual(summary["git_sha"], "b" * 40)
        self.assertTrue(summary["allow_dirty"])
        self.assertEqual(summary["groups"][0]["metrics"]["a"]["mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
