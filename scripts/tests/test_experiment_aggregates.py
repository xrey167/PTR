import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGGREGATORS = {
    "L003": ROOT / "experiments/lifecycle/L003-fastmem-revocation/aggregate.py",
    "L004": ROOT / "experiments/lifecycle/L004-projection-equivalence/aggregate.py",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"aggregate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NewestPerSeedTests(unittest.TestCase):
    """`newest_per_seed` picks the record each aggregator reads for a seed."""

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def record(self, name: str, seed: int, status: str) -> Path:
        path = self.directory / name
        body = {"seed": seed, "status": status}
        # A run that finished names its outcome, as the runner writes it.
        if status in ("completed", "failed"):
            body.update(
                {"exit_code": 0 if status == "completed" else 1, "finished_at": "2026-01-01T00:00:00+00:00",
                 "stdout": "", "stderr": ""}
            )
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_a_launch_that_failed_never_stands_in_for_a_run_of_its_seed(self):
        # A clock set back between a failed launch and its retry names the
        # retry earlier; the failed launch ran nothing, so the retry is the
        # seed's run whatever the names say.
        retry = self.record("run-20260101T000000Z-seed-1.json", 1, "completed")
        failed = self.record("run-20270101T000000Z-seed-1.json", 1, "failed-to-launch")
        only_failed = self.record("run-20260101T000000Z-seed-2.json", 2, "failed-to-launch")
        later_failed = self.record("run-20260102T000000Z-seed-2.json", 2, "failed-to-launch")
        earlier = self.record("run-20260101T000000Z-seed-3.json", 3, "completed")
        later = self.record("run-20260102T000000Z-seed-3.json", 3, "failed")
        for name, path in AGGREGATORS.items():
            with self.subTest(aggregator=name):
                module = load(name, path)
                paths = [failed, retry, only_failed, later_failed, earlier, later]
                self.assertEqual(module.newest_per_seed(paths), {1: retry, 2: later_failed, 3: later})
                # Whatever order they come in.
                self.assertEqual(module.newest_per_seed(list(reversed(paths))), {1: retry, 2: later_failed, 3: later})

    def test_a_reservation_nothing_finished_never_stands_in_for_a_finished_run(self):
        # A run the runner died in leaves its reservation, `started`, with no
        # outcome: named later than a finished run of its seed, it is no run
        # to aggregate, and taken for one it would fail the aggregation.
        finished = self.record("run-20260101T000000Z-seed-1.json", 1, "completed")
        self.record("run-20260102T000000Z-seed-1.json", 1, "started")
        failed = self.record("run-20260101T000000Z-seed-2.json", 2, "failed")
        self.record("run-20260102T000000Z-seed-2.json", 2, "started")
        only = self.record("run-20260101T000000Z-seed-3.json", 3, "started")
        for name, path in AGGREGATORS.items():
            with self.subTest(aggregator=name):
                module = load(name, path)
                paths = sorted(self.directory.glob("run-*.json"))
                self.assertEqual(module.newest_per_seed(paths), {1: finished, 2: failed, 3: only})
                self.assertEqual(module.newest_per_seed(list(reversed(paths))), {1: finished, 2: failed, 3: only})


if __name__ == "__main__":
    unittest.main()
