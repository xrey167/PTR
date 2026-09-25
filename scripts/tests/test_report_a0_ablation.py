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
        """Copy study outputs into an isolated tree and redirect the report writer there."""
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
        """Restore the report writer's repository and study paths after the fixture run."""
        report.ROOT, report.STUDY = self.saved

    def test_every_verdict_is_reported_and_nulls_get_a_note(self):
        """Render every verdict and keep falsification notes only for FALSIFIES or HARMFUL outcomes."""
        self.assertEqual(report.main(), 0)
        text = (self.study / "RESULTS.md").read_text(encoding="utf-8")
        verdicts = json.loads((self.study / "results.json").read_text(encoding="utf-8"))["verdicts"]
        for cid, entry in verdicts.items():
            self.assertIn(f"| {cid} | **{entry['verdict']}** |", text)
        nulls = {cid for cid, entry in verdicts.items() if entry["verdict"] in ("FALSIFIES", "HARMFUL")}
        notes = {p.name[len("FALSIFIED-"):-len(".md")] for p in self.study.glob("FALSIFIED-*.md")}
        self.assertEqual(notes, nulls)

    def test_the_interpretation_follows_the_verdicts_it_describes(self):
        """Change verdict inputs and require the interpretation prose to follow their new outcomes."""
        # Other outcomes must not inherit this run's prose: turn M002-necessity into
        # SUPPORTS, M001-primary into FALSIFIES and the negative control into
        # EQUIVALENT, and every sentence about them has to change with them.
        path = self.study / "results.json"
        results = json.loads(path.read_text(encoding="utf-8"))
        results["verdicts"]["M002-necessity"]["verdict"] = "SUPPORTS"
        results["verdicts"]["M001-primary"]["verdict"] = "FALSIFIES"
        results["verdicts"]["M004-negative-control"]["verdict"] = "EQUIVALENT"
        path.write_text(json.dumps(results), encoding="utf-8")
        self.assertEqual(report.main(), 0)
        text = (self.study / "RESULTS.md").read_text(encoding="utf-8")
        interpretation = text.split("## Interpretation (not a rule output)")[1].split("## Reading these results")[0]
        self.assertNotIn("Typed slot content helped", interpretation)
        self.assertIn("M001-primary is FALSIFIES", interpretation)
        self.assertNotIn("typed attention bias", interpretation)
        self.assertIn("a second tied refinement step", interpretation)
        self.assertIn("equivalent to the learned one within the preregistered band, as predicted", interpretation)
        self.assertNotIn("could not be shown equivalent", interpretation)

    def test_harmful_note_is_replaced_when_the_verdict_changes(self):
        """Regeneration creates a harm note, is repeatable, and removes obsolete notes."""
        path = self.study / "results.json"
        results = json.loads(path.read_text(encoding="utf-8"))
        entry = results["verdicts"]["M002-necessity"]
        entry["verdict"] = "HARMFUL"
        entry["reason"] = "CI upper < 0: the ablated arm is better"
        entry["stats"] = {"deltas": [-0.02] * 5, "mean": -0.02, "sd": 0.0,
                          "ci": [-0.02, -0.02], "positive_seeds": 0}
        path.write_text(json.dumps(results), encoding="utf-8")
        self.assertEqual(report.main(), 0)
        note = self.study / "FALSIFIED-M002-necessity.md"
        text = note.read_text(encoding="utf-8")
        self.assertIn("**Verdict:** HARMFUL", text)
        self.assertIn("removing it made A0 better here", text)
        self.assertIn("mean -0.0200", text)
        before = {p.name: p.read_bytes() for p in self.study.glob("*.md")}
        self.assertEqual(report.main(), 0)
        self.assertEqual({p.name: p.read_bytes() for p in self.study.glob("*.md")}, before)
        entry["verdict"] = "INCONCLUSIVE"
        entry["reason"] = "G4 failed"
        path.write_text(json.dumps(results), encoding="utf-8")
        self.assertEqual(report.main(), 0)
        self.assertFalse(note.exists())


if __name__ == "__main__":
    unittest.main()
