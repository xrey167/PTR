import importlib.util
import tempfile
import unittest
from pathlib import Path

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

    def test_the_last_json_line_is_the_result(self):
        stdout = 'Compiling\n{"a":1}\nnoise\n{"hard_failures":2}\n'
        self.assertEqual(mod.last_json_line(stdout), {"hard_failures": 2})
        self.assertEqual(mod.last_json_line("nothing"), {})


if __name__ == "__main__":
    unittest.main()
