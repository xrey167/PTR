"""score.py: PRED parsing from run records, the three metrics and the subsets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _support import gen, prefix, score, score_items_from_tsv

N = 120


def run_record(stdout: str, seed: int = 17) -> dict:
    """Build the minimal completed experiment record consumed by the scorer."""
    # The fields scripts/run_experiment.py writes that score.py reads.
    return {"schema_version": 2, "experiment_id": "M001", "seed": seed, "status": "completed", "stdout": stdout}


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create two miniature TSV splits and independently scored expected items."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name)
        # A miniature data directory: the first N examples of two test splits.
        for split in ("test_iid", "ood_compose_regime"):
            _jsonl, tsv, _labels = prefix(split, N)
            (cls.data / f"{split}.tsv").write_bytes(tsv)
        cls.items = {s: score_items_from_tsv(s, prefix(s, N)[1]) for s in ("test_iid", "ood_compose_regime")}

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary split files and run records."""
        cls.tmp.cleanup()

    def write(self, name: str, record: dict) -> Path:
        """Write a JSON run record into the test data directory and return its path."""
        path = self.data / name
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_perfect_predictions(self):
        """Require perfect labels to produce full accuracy, zero regret, and successful subsets."""
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
        """Compare constant-predictor metrics and subset counts with direct calculations."""
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
        """Check transfer-subset membership and accuracy under a constant causal prediction."""
        items = self.items["ood_compose_regime"]
        causal = score.OPERATORS["causal"]
        report = score.score_predictions(items, [causal] * N)
        transfer = [i for i in items if "transfer" in i.tags]
        self.assertEqual(report["subsets"]["transfer"]["n"], len(transfer))
        self.assertEqual(report["subsets"]["transfer"]["correct"], sum(i.label == causal for i in transfer))

    def test_pred_lines_in_a_run_record(self):
        """Parse multiple arms and hex cases, preserving seed and Rust metric comparisons."""
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
        """Report gate G5 disagreement when Rust counts differ from rescored predictions."""
        labels = "".join(format(i.label, "x") for i in self.items["test_iid"])
        final = {"row": "final", "arm": "full", "split": "test_iid", "n": N, "correct": N - 1}
        stdout = json.dumps(final) + "\n" + f"PRED full test_iid {labels}\n"
        path = self.write("run-b.json", run_record(stdout))
        _document, problems = score.score_records([path], self.data)
        self.assertTrue(any("G5" in p for p in problems))

    def test_wrong_length_and_bad_code_are_errors(self):
        """Reject prediction strings with the wrong length or unknown operator codes."""
        short = "0" * (N - 1)
        bad = "b" * N  # 11 is not an operator code
        path = self.write("run-c.json", run_record(f"PRED full test_iid {short}\nPRED x test_iid {bad}\n"))
        _document, problems = score.score_records([path], self.data)
        self.assertEqual(len(problems), 2)

    def test_malformed_pred_line_raises(self):
        """Reject malformed PRED syntax before scoring its contents."""
        with self.assertRaises(ValueError):
            score.pred_lines("PRED full test_iid 01z\n")

    def test_bad_records_do_not_hide_valid_predictions(self):
        """Report missing stdout and unknown splits while retaining valid results."""
        missing = self.write("missing-stdout.json", {"seed": 17, "stdout": None})
        labels = "".join(format(i.label, "x") for i in self.items["test_iid"])
        mixed = self.write("mixed-splits.json", run_record(
            f"PRED full unknown_split 0\nPRED full test_iid {labels}\n"
        ))
        document, problems = score.score_records([missing, mixed], self.data)
        self.assertEqual(len(problems), 2)
        self.assertIn("no stdout text", problems[0])
        self.assertIn("unknown split 'unknown_split'", problems[1])
        self.assertEqual(len(document["results"]), 1)
        self.assertEqual(document["results"][0]["route_accuracy"], 1.0)

    def test_boolean_correct_is_not_a_rust_count(self):
        """Boolean status fields cannot masquerade as numeric metric evidence."""
        rows = [{"arm": "full", "split": "test_iid", "correct": True},
                {"arm": "other", "split": "test_iid", "correct": 120},
                {"arm": "full", "split": "val", "correct": 120}]
        self.assertIsNone(score.rust_final_row(rows, "full", "test_iid"))
        expected = {"arm": "full", "split": "test_iid", "correct": 0}
        self.assertEqual(score.rust_final_row(rows + [expected], "full", "test_iid"), expected)


class MetricBoundaries(unittest.TestCase):
    """Hand-built utilities isolate scoring from generator and rule behavior."""

    def item(self, regret=0.25, margin=0.25, split="test_iid", tags=()):
        """Make a two-candidate example with explicitly known regret and margin."""
        item = score.Item()
        item.id, item.split = "boundary", split
        item.label = 0
        item.z = [regret, 0.0] + [-1.0] * 9
        item.margin, item.tags = margin, list(tags)
        return item

    def test_success_tolerance_does_not_change_exact_route_accuracy(self):
        """A near-optimal wrong route can succeed without being counted correct."""
        for regret, success in [(0.25, 1.0), (0.25 + 0.5e-9, 1.0), (0.25 + 2e-9, 0.0)]:
            with self.subTest(regret=regret):
                report = score.score_predictions([self.item(regret=regret)], [1])
                self.assertEqual(report["route_accuracy"], 0.0)
                self.assertEqual(report["task_success"], success)
                self.assertAlmostEqual(report["cost_adjusted_regret"], regret, places=12)

    def test_clear_margin_tolerance_and_empty_subset_denominators(self):
        """Keep tolerance-edge examples and represent empty subsets as unavailable."""
        for margin, n in [(0.25, 1), (0.25 - 0.5e-9, 1), (0.25 - 2e-9, 0)]:
            with self.subTest(margin=margin):
                report = score.score_predictions([self.item(margin=margin)], [0])
                self.assertEqual(report["subsets"]["clear_margin"], {
                    "n": n, "correct": n, "accuracy": 1.0 if n else None,
                })
                self.assertEqual(report["subsets"]["validity"], {
                    "n": 0, "correct": 0, "accuracy": None,
                })

    def test_heldout_subset_is_reported_only_for_its_shift(self):
        """Held-out accuracy uses its own denominator and only the relevant split."""
        items = [self.item(split="ood_compose_epi", tags=["heldout"]),
                 self.item(split="ood_compose_epi")]
        report = score.score_predictions(items, [0, 1])
        self.assertEqual(report["route_accuracy"], 0.5)
        self.assertEqual(report["subsets"]["heldout"], {"n": 1, "correct": 1, "accuracy": 1.0})
        self.assertNotIn("transfer", report["subsets"])
        for item in items:
            item.split = "test_iid"
        self.assertNotIn("heldout", score.score_predictions(items, [0, 1])["subsets"])

    def test_prediction_count_and_operator_bounds_are_checked(self):
        """Reject both ends of the code range and both short and long predictions."""
        item = self.item()
        for predictions in ([], [0, 0]):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(ValueError, "predictions for 1 examples"):
                score.score_predictions([item], predictions)
        for code in (-1, 11):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "not an operator code"):
                score.score_predictions([item], [code])
        self.assertEqual(score.score_predictions([item], [10])["n"], 1)


if __name__ == "__main__":
    unittest.main()
