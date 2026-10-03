"""The generated component views and the status of a listed experiment."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import update_component_docs  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
