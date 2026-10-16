import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ptr_training.dragonfly.train import _spec_from_json, main

QUESTIONS = (
    "How long is the delivery time of {name}?",
    "Is {name} late with the X12 shipment?",
    "What did {name} quote for the lead time?",
    "Delay of the procurement order at {name}",
    "Which delivery date did {name} confirm?",
)


def pod_json(project: str | None = "demo", namespace: str | None = "supply") -> dict:
    data = {
        "pod_id": "pod-mueller",
        "descriptor": {
            "schema_version": 1,
            "domain": "supply",
            "kind": "Retrieval",
            "predicate": "lead_time",
            "semantic_role": "supplier-mueller",
            "aliases": ["Mueller GmbH", "Muller Werke"],
        },
        "generation": 1,
        "positives": [q.format(name="Mueller GmbH") for q in QUESTIONS],
        "negatives": [q.format(name="Schmidt AG") for q in QUESTIONS],
        "negative_aliases": ["Schmidt AG", "Schmitt Logistik"],
    }
    if project is not None:
        data["project"] = project
    if namespace is not None:
        data["namespace"] = namespace
    return data


class SpecFromJsonTest(unittest.TestCase):
    def test_the_project_is_read_from_the_spec(self) -> None:
        self.assertEqual(_spec_from_json(pod_json("acme")).project, "acme")

    def test_a_spec_without_a_project_is_refused(self) -> None:
        for data in (pod_json(None), pod_json("")):
            with self.assertRaises(ValueError):
                _spec_from_json(data)

    def test_the_namespace_is_read_from_the_spec(self) -> None:
        self.assertEqual(_spec_from_json(pod_json("acme", "ns-a")).namespace, "ns-a")

    def test_a_spec_without_a_namespace_is_refused(self) -> None:
        for data in (pod_json("acme", None), pod_json("acme", "")):
            with self.assertRaises(ValueError):
                _spec_from_json(data)


class CliTest(unittest.TestCase):
    def test_the_cli_trains_and_exports_a_pod_from_a_json_spec_with_its_project(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = Path(raw) / "spec.json"
            out = Path(raw) / "out"
            spec.write_text(json.dumps({"pods": [pod_json("acme")]}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(spec), str(out), "--dims", "32", "--steps", "60"])
            manifests = sorted(out.glob("dragonfly-*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["project"], "acme")
            self.assertEqual(manifest["namespace"], "supply")
            self.assertEqual(manifest["pod_id"], "pod-mueller")

    def test_the_cli_refuses_a_spec_without_a_project(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = Path(raw) / "spec.json"
            spec.write_text(json.dumps({"pods": [pod_json(None)]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                main([str(spec), str(Path(raw) / "out"), "--dims", "32", "--steps", "10"])


if __name__ == "__main__":
    unittest.main()
