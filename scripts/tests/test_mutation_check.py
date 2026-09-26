import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "mutation_check", ROOT / "scripts/mutation_check.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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

        def run_mutation(_plan, mutation, _timeout):
            ran.append(mutation["name"])
            return {"name": mutation["name"], "result": "killed", "counters": {"lost": 1}}

        def rebuild(command, **_kwargs):
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


if __name__ == "__main__":
    unittest.main()
