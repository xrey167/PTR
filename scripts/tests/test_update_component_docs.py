"""The generated component views: the status of a listed experiment and the test counts.

The status tests check that a listed experiment past `planned` is shown as
frozen, so that completing it changes no generated file.

The counting tests check that the generated test counts see async tests and
ignore quoted attributes. The generator counted the substring `#[test]`, so every
`#[tokio::test]` was invisible: ptr-server's README said 1 test where it had 4,
and ptr-cluster's integration tests did not show up at all.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import update_component_docs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "update_component_docs_by_path", ROOT / "scripts/update_component_docs.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)



class ShownStatusTests(unittest.TestCase):
    def test_status_projection_does_not_mutate_registry_records_or_evaluations(self):
        experiment = {"id": "S003", "path": "semdb/S003", "status": "completed", "title": "Branches"}
        evaluation = {"id": "S003", "status": "completed"}
        # Each file answers for itself, so that a harmless reordering of the
        # three reads in registries() does not break the test.
        tables = {
            "experiments/registry.toml": {"experiment": [experiment]},
            "evaluations/registry.toml": {"component": [evaluation]},
            "experiments/preregistration.toml": {"experiment": {"S003": {}}},
        }

        def load_toml(path):
            return tables[Path(path).relative_to(update_component_docs.ROOT).as_posix()]

        with mock.patch.object(update_component_docs, "load_toml", side_effect=load_toml):
            experiments, evaluations = update_component_docs.registries()
        self.assertEqual(experiments["S003"], {**experiment, "status": "frozen"})
        self.assertEqual(experiment["status"], "completed")
        self.assertEqual(evaluations, {"S003": evaluation})
        self.assertEqual(evaluation["status"], "completed")

    def test_unlisted_experiments_keep_each_lifecycle_status(self):
        for status in ("planned", "prepared", "running", "completed", "failed", "superseded"):
            with self.subTest(status=status):
                self.assertEqual(update_component_docs.shown_status({"id": "L004", "status": status}, {"S003"}), status)

    def test_empty_registries_and_preregistration_produce_empty_views(self):
        with mock.patch.object(update_component_docs, "load_toml", return_value={}):
            self.assertEqual(update_component_docs.registries(), ({}, {}))

    def shown(self, statuses: dict[str, str], listed: list[str]) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "evaluations").mkdir()
            entries = "".join(
                f'[[experiment]]\nid = "{name}"\npath = "x/{name}"\nstatus = "{status}"\n\n'
                for name, status in statuses.items()
            )
            (root / "experiments" / "registry.toml").write_text("version = 1\n\n" + entries, encoding="utf-8")
            tables = "".join(f"[experiment.{name}.required]\nschema = \"int\"\n\n" for name in listed)
            (root / "experiments" / "preregistration.toml").write_text(tables, encoding="utf-8")
            (root / "evaluations" / "registry.toml").write_text("version = 1\n", encoding="utf-8")
            saved = update_component_docs.ROOT
            update_component_docs.ROOT = root
            try:
                exps, _ = update_component_docs.registries()
            finally:
                update_component_docs.ROOT = saved
        return {name: exp["status"] for name, exp in exps.items()}

    def test_a_listed_experiment_past_planned_is_shown_as_frozen_in_every_status(self):
        for status in ("prepared", "running", "completed", "failed"):
            with self.subTest(status=status):
                self.assertEqual(self.shown({"S003": status}, ["S003"]), {"S003": "frozen"})

    def test_the_views_do_not_change_when_a_listed_experiment_completes(self):
        before = self.shown({"S003": "running"}, ["S003"])
        after = self.shown({"S003": "completed"}, ["S003"])
        self.assertEqual(before, after)

    def test_a_listed_experiment_that_is_planned_or_superseded_shows_its_status(self):
        self.assertEqual(
            self.shown({"S003": "planned", "F003": "superseded"}, ["S003", "F003"]),
            {"S003": "planned", "F003": "superseded"},
        )

    def test_an_experiment_the_list_does_not_name_shows_its_status(self):
        self.assertEqual(
            self.shown({"L004": "completed", "S003": "running"}, ["S003"]),
            {"L004": "completed", "S003": "frozen"},
        )


class TestCounting(unittest.TestCase):
    def test_nested_sync_and_async_tests_count_once_with_crlf_and_tabs(self):
        """Traverse nested Rust files while ignoring fixtures and unrelated directories."""
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            files = {
                "src/nested/unit.rs": b'\t#[tokio::test]\r\nasync fn unit() {}\r\n',
                "tests/nested/integration.rs": b'\t#[test]\t// regression\r\nfn integration() {}\r\n',
                "tests/data.txt": b'#[test]\n',
                "examples/demo.rs": b'#[test]\n',
            }
            for name, contents in files.items():
                path = crate / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            self.assertEqual(mod.code_metrics(crate), {"files": 1, "loc": 2, "test_files": 1, "tests": 2})

    def test_sync_and_async_attributes_count_and_quoted_ones_do_not(self):
        """Count real sync and async test attributes while excluding quoted text and other macros."""
        directory = tempfile.TemporaryDirectory()
        crate = Path(directory.name)
        with directory:
            (crate / "src").mkdir()
            (crate / "tests").mkdir()
            (crate / "src/lib.rs").write_text(
                "pub fn f() {}\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    #[test]\n"
                "    fn a() {}\n"
                "    // A `#[test]` quoted in a comment is not a test.\n"
                '    const S: &str = "#[test]";\n'
                "}\n",
                encoding="utf-8",
            )
            (crate / "tests/it.rs").write_text(
                "#[tokio::test]\n"
                "async fn b() {}\n"
                '#[tokio::test(flavor = "multi_thread", worker_threads = 4)]\n'
                "async fn c() {}\n"
                "#[test] // trailing comment\n"
                "fn d() {}\n"
                "#[test_case(1)]\n"
                "fn not_counted(_: u8) {}\n"
                "/// Doc text naming #[test] and #[test] is not a test either.\n"
                "fn helper() {}\n",
                encoding="utf-8",
            )
            metrics = mod.code_metrics(crate)
        # Four attribute lines. The old substring count saw six here (two sync
        # attributes plus four quotations of the attribute) and none of the async
        # ones, so it cannot pass this by accident.
        self.assertEqual(metrics["tests"], 4)
        self.assertEqual(metrics["files"], 1)
        self.assertEqual(metrics["test_files"], 1)

    def test_async_crates_are_no_longer_undercounted(self):
        """Require the real server crate's metrics to include its async tests."""
        server = mod.code_metrics(ROOT / "crates/ptr-server")["tests"]
        self.assertGreaterEqual(server, 4)


if __name__ == "__main__":
    unittest.main()
