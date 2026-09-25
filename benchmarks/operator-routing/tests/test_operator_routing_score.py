"""score.py: PRED parsing from run records, the three metrics and the subsets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _support import gen, prefix, score, score_items_from_tsv

N = 120


def run_record(stdout: str, seed: int = 17) -> dict:
    # The fields scripts/run_experiment.py writes that score.py reads.
    return {"schema_version": 2, "experiment_id": "M001", "seed": seed, "status": "completed", "stdout": stdout}


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name)
        # A miniature data directory: the first N examples of two test splits.
        for split in ("test_iid", "ood_compose_regime"):
            _jsonl, tsv, _labels = prefix(split, N)
            (cls.data / f"{split}.tsv").write_bytes(tsv)
        cls.items = {s: score_items_from_tsv(s, prefix(s, N)[1]) for s in ("test_iid", "ood_compose_regime")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write(self, name: str, record: dict) -> Path:
        path = self.data / name
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_perfect_predictions(self):
        items = self.items["test_iid"]
        report = score.score_predictions(items, [i.label for i in items])
        self.assertEqual(report["correct"], N)
        self.assertEqual(report["route_accuracy"], 1.0)
        self.assertEqual(report["cost_adjusted_regret"], 0.0)
        self.assertEqual(report["task_success"], 1.0)
        for tag in ("validity", "regime", "budget", "confidence", "epistemic", "clear_margin"):
            subset = report["subsets"][tag]
            self.assertEqual(subset["correct"], subset["n"])

    def test_constant_prediction_by_hand(self):
        items = self.items["test_iid"]
        semantic = score.OPERATORS["semantic"]
        report = score.score_predictions(items, [semantic] * N)
        correct = sum(i.label == semantic for i in items)
        regrets = [i.z[i.label] - i.z[semantic] for i in items]
        self.assertEqual(report["correct"], correct)
        self.assertAlmostEqual(report["cost_adjusted_regret"], sum(regrets) / N, places=12)
        self.assertEqual(report["task_success"], sum(r <= 0.25 + 1e-9 for r in regrets) / N)
        margin = [i for i in items if i.margin >= 0.25 - 1e-9]
        self.assertEqual(report["subsets"]["clear_margin"]["n"], len(margin))
        self.assertEqual(report["subsets"]["clear_margin"]["correct"], sum(i.label == semantic for i in margin))
        self.assertNotIn("heldout", report["subsets"])
        self.assertNotIn("transfer", report["subsets"])

    def test_transfer_subset_on_compose_regime(self):
        items = self.items["ood_compose_regime"]
        causal = score.OPERATORS["causal"]
        report = score.score_predictions(items, [causal] * N)
        transfer = [i for i in items if "transfer" in i.tags]
        self.assertEqual(report["subsets"]["transfer"]["n"], len(transfer))
        self.assertEqual(report["subsets"]["transfer"]["correct"], sum(i.label == causal for i in transfer))

    def test_pred_lines_in_a_run_record(self):
        labels = "".join(format(i.label, "x") for i in self.items["test_iid"])
        wrong = "".join(format((i.label + 1) % 11, "X") for i in self.items["test_iid"])  # upper case is accepted
        final = {"row": "final", "arm": "full", "split": "test_iid", "n": N, "correct": N,
                 "accuracy": 1.0, "nll": 0.1, "ece15": 0.01}
        stdout = "\n".join([
            json.dumps({"row": "data", "data_fnv1a64": "0"}),
            json.dumps(final),
            f"PRED full test_iid {labels}",
            f"PRED raw-blind test_iid {wrong}",
        ]) + "\n"
        path = self.write("run-a.json", run_record(stdout))
        document, problems = score.score_records([path], self.data)
        self.assertEqual(problems, [])
        by_arm = {r["arm"]: r for r in document["results"]}
        self.assertEqual(by_arm["full"]["route_accuracy"], 1.0)
        self.assertTrue(by_arm["full"]["rust_count_agrees"])
        self.assertEqual(by_arm["full"]["rust"]["nll"], 0.1)
        self.assertEqual(by_arm["raw-blind"]["correct"], 0)
        self.assertIsNone(by_arm["raw-blind"]["rust_count_agrees"])
        self.assertEqual(by_arm["full"]["seed"], 17)

    def test_rust_count_disagreement_is_reported(self):
        labels = "".join(format(i.label, "x") for i in self.items["test_iid"])
        final = {"row": "final", "arm": "full", "split": "test_iid", "n": N, "correct": N - 1}
        stdout = json.dumps(final) + "\n" + f"PRED full test_iid {labels}\n"
        path = self.write("run-b.json", run_record(stdout))
        _document, problems = score.score_records([path], self.data)
        self.assertTrue(any("G5" in p for p in problems))

    def test_wrong_length_and_bad_code_are_errors(self):
        short = "0" * (N - 1)
        bad = "b" * N  # 11 is not an operator code
        path = self.write("run-c.json", run_record(f"PRED full test_iid {short}\nPRED x test_iid {bad}\n"))
        _document, problems = score.score_records([path], self.data)
        self.assertEqual(len(problems), 2)

    def test_malformed_pred_line_raises(self):
        with self.assertRaises(ValueError):
            score.pred_lines("PRED full test_iid 01z\n")


if __name__ == "__main__":
    unittest.main()
