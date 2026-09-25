"""The A0 study's report writer, run on a copy of the committed study outputs:
every verdict appears in RESULTS.md, a FALSIFIED note exists for exactly the
FALSIFIES and HARMFUL verdicts, and a note left from an earlier aggregate is
removed.
"""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = "research/falsification/A0-ablations-v1"
spec = importlib.util.spec_from_file_location("report_a0_ablation", ROOT / "scripts/report_a0_ablation.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@unittest.skipUnless((ROOT / STUDY / "results.json").exists(), "the study has not been aggregated")
class Report(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        study = self.tmp / STUDY
        study.mkdir(parents=True)
        for name in ("results.json", "references.json", "budget.json", "criteria.toml"):
            shutil.copy(ROOT / STUDY / name, study / name)
        for path in report.EXPERIMENTS.values():
            metrics = self.tmp / "experiments" / path / "results"
            metrics.mkdir(parents=True)
            shutil.copy(ROOT / "experiments" / path / "results/a0_internal_metrics.json", metrics)
        (study / "FALSIFIED-stale.md").write_text("from an earlier aggregate\n", encoding="utf-8")
        self.study = study
        self.saved = (report.ROOT, report.STUDY)
        report.ROOT, report.STUDY = self.tmp, study
        self.addCleanup(self.restore)

    def restore(self):
        report.ROOT, report.STUDY = self.saved

    def test_every_verdict_is_reported_and_nulls_get_a_note(self):
        self.assertEqual(report.main(), 0)
        text = (self.study / "RESULTS.md").read_text(encoding="utf-8")
        verdicts = json.loads((self.study / "results.json").read_text(encoding="utf-8"))["verdicts"]
        for cid, entry in verdicts.items():
            self.assertIn(f"| {cid} | **{entry['verdict']}** |", text)
        nulls = {cid for cid, entry in verdicts.items() if entry["verdict"] in ("FALSIFIES", "HARMFUL")}
        notes = {p.name[len("FALSIFIED-"):-len(".md")] for p in self.study.glob("FALSIFIED-*.md")}
        self.assertEqual(notes, nulls)


if __name__ == "__main__":
    unittest.main()
