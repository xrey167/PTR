import contextlib
import hashlib
import importlib.util
import io
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "aggregate_experiment", ROOT / "scripts/aggregate_experiment.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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

    def test_all_failed_runs_produce_an_incomplete_record_without_metrics(self):
        """Keep failure evidence even when no seed produced usable output."""
        paths = [self.write_run(seed, [], status="failed", stdout="{truncated")
                 for seed in SEEDS]
        code, summary = self.run_aggregate()
        self.assertEqual((code, summary["status"]), (1, "incomplete"))
        self.assertEqual(summary["completed_seeds"], [])
        self.assertEqual(summary["missing_seeds"], SEEDS)
        self.assertEqual(summary["groups"], [])
        self.assertEqual([r["record"] for r in summary["failed_runs"]],
                         [p.name for p in paths])
        self.assertEqual(len(summary["source_records"]), len(SEEDS))

    def test_successful_retry_keeps_the_previous_failure_visible(self):
        """A completed retry supplies metrics without erasing a failed attempt."""
        failed = self.write_run(1, [{"accuracy": 99}], status="failed")
        for seed in SEEDS:
            self.write_run(seed, [{"accuracy": 0.75}])
        code, summary = self.run_aggregate()
        self.assertEqual((code, summary["status"]), (1, "incomplete"))
        self.assertEqual(summary["missing_seeds"], [])
        self.assertEqual(summary["failed_runs"][0]["record"], failed.name)
        metric = summary["groups"][0]["metrics"]["accuracy"]
        self.assertEqual((metric["n"], metric["mean"], metric["std"]), (3, 0.75, 0.0))

    def test_all_non_finite_values_have_no_numeric_summary(self):
        """Distinguish reported divergence from missing observations."""
        for seed, value in zip(SEEDS, [float("nan"), float("inf"), -float("inf")]):
            self.write_run(seed, [{"arm": "full", "loss": value}])
        code, summary = self.run_aggregate()
        metric = summary["groups"][0]["metrics"]["loss"]
        self.assertEqual((code, metric["n"]), (1, 0))
        self.assertEqual(metric["by_seed"], {})
        self.assertEqual(metric["missing_seeds"], [])
        self.assertEqual(metric["non_finite"], {"1": "nan", "2": "inf", "3": "-inf"})
        for field in ("mean", "std", "min", "max"):
            self.assertIsNone(metric[field], field)

    def test_equivalent_provenance_is_independent_of_dictionary_order(self):
        """Object ordering must not split otherwise identical measurements."""
        for seed in SEEDS:
            extra = {"parameters": {"steps": "10", "lr": "0.01"},
                     "host": {"system": "Linux", "logical_cpus": 4}}
            if seed == 2:
                extra = {k: dict(reversed(list(v.items()))) for k, v in extra.items()}
            self.write_run(seed, [{"accuracy": 0.5}], extra=extra)
        code, summary = self.run_aggregate()
        self.assertEqual((code, summary["status"]), (0, "complete"))

    def test_timestamp_collision_cannot_overwrite_an_aggregate(self):
        """Preserve immutable evidence when two aggregations get the same timestamp."""
        for seed in SEEDS:
            self.write_run(seed, [{"accuracy": 0.5}])
        with patch.object(mod, "utc_stamp", return_value="fixed"):
            self.assertEqual(self.run_aggregate()[0], 0)
            path = self.results / "aggregate-fixed-ablation.json"
            before = path.read_bytes()
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                mod.aggregate("X001", entrypoint="ablation")
            self.assertEqual(path.read_bytes(), before)

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
            ("python", {"python": "3.11"}, {"python": "3.13"}),
            ("executable", {"executable": {"sha256": "a" * 64}}, {"executable": {"sha256": "b" * 64}}),
            ("toolchain", {"toolchain": {"rustc": "1.85"}}, {"toolchain": {"rustc": "1.95"}}),
            ("environment", {"environment": {"RUSTFLAGS": ""}}, {"environment": {"RUSTFLAGS": "-C opt-level=3"}}),
            ("toolchain", {}, {"toolchain": {"rustc": "1.95"}}),
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

    def test_dirty_records_require_valid_diff_hashes_even_with_override(self):
        for digest in (None, "", "not-a-hash", "a" * 63, 123, True):
            with self.subTest(digest=digest):
                for path in self.results.glob("*.json"):
                    path.unlink()
                for seed in SEEDS:
                    self.write_run(seed, [{"a": 1}], dirty=True,
                                   extra={"git_tracked_diff_sha256": digest})
                with self.assertRaisesRegex(ValueError, "requires a valid git_tracked_diff_sha256"):
                    mod.aggregate("X001", entrypoint="ablation", allow_dirty=True)
                self.assertEqual(list(self.results.glob("aggregate-*.json")), [])

    def test_aggregate_can_select_one_contingency_configuration(self):
        for seed in SEEDS:
            self.write_run(seed, [{"arm": "a", "accuracy": 0.8}], extra={"parameters": {"arms": "a"}})
            self.write_run(seed, [{"arm": "b", "accuracy": 0.2}], extra={"parameters": {"arms": "b"}})
        code, summary = self.run_aggregate(parameters_filter={"arms": "a"})
        self.assertEqual(code, 0)
        self.assertEqual(summary["parameters"], {"arms": "a"})
        self.assertAlmostEqual(summary["groups"][0]["metrics"]["accuracy"]["mean"], 0.8)
        self.assertEqual(len(summary["source_records"]), len(SEEDS))

    def test_foreign_or_missing_experiment_identity_is_rejected(self):
        for identity in ("M001", None, ""):
            with self.subTest(identity=identity):
                for path in self.results.glob("*.json"):
                    path.unlink()
                self.write_run(1, [{"a": 1}], extra={"experiment_id": identity})
                with self.assertRaisesRegex(ValueError, "experiment_id must be"):
                    mod.aggregate("X001", entrypoint="ablation")
                self.assertEqual(list(self.results.glob("aggregate-*.json")), [])

    def test_legacy_unknown_worktree_remains_explicitly_opt_in(self):
        for seed in SEEDS:
            self.write_run(seed, [{"a": 1}], dirty=None)
        code, summary = self.run_aggregate(allow_dirty=True)
        self.assertEqual((code, summary["status"]), (0, "complete"))

    def test_different_dependency_locks_are_rejected(self):
        for seed in SEEDS:
            self.write_run(seed, [{"a": 1}], extra={"cargo_lock_sha256": str(seed) * 64})
        with self.assertRaisesRegex(ValueError, "records differ in cargo_lock_sha256"):
            mod.aggregate("X001", entrypoint="ablation")

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
        self.write_run(1, [{"a": 5.0}], sha="b" * 40, dirty=True, extra={"git_tracked_diff_sha256": "1" * 64})

        code, summary = self.run_aggregate(git_sha_filter="b" * 40, allow_dirty=True)

        self.assertEqual(code, 1)  # seeds 2 and 3 never ran
        self.assertEqual(summary["git_sha"], "b" * 40)
        self.assertTrue(summary["allow_dirty"])
        self.assertEqual(summary["groups"][0]["metrics"]["a"]["mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
