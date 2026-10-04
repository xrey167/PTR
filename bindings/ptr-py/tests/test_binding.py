"""Tests for the ptr_py bindings. Expected values are hand-derived or taken from
the Rust unit tests in crates/ptr-lineage (aimer.rs, subspace.rs, merge.rs,
forgetting.rs)."""

import math
import unittest

import ptr_py


class AimerTests(unittest.TestCase):
    def test_hand_value(self):
        self.assertAlmostEqual(ptr_py.aimer_score([3.0, 4.0]), 7 / (math.sqrt(2) * 5), places=12)

    def test_uniform_scores_one(self):
        self.assertAlmostEqual(ptr_py.aimer_score([2.0, -2.0, 2.0, 2.0]), 1.0, places=12)

    def test_one_hot_scores_inverse_root_n(self):
        self.assertAlmostEqual(ptr_py.aimer_score([0.0, 0.0, 5.0, 0.0]), 1 / math.sqrt(4), places=12)

    def test_ranking_highest_score_first_ties_stable(self):
        experts = [
            ("sparse", [0.0, 0.0, 1.0, 0.0]),
            ("uniform", [1.0, 1.0, 1.0, 1.0]),
            ("mid", [3.0, 4.0, 0.0, 0.0]),
            ("uniform2", [2.0, 2.0, 2.0, 2.0]),
        ]
        self.assertEqual(
            ptr_py.rank_experts_for_pruning(experts), ["uniform", "uniform2", "mid", "sparse"]
        )

    def test_errors_map_to_value_error_with_code(self):
        cases = [
            ([], "PTR_LINEAGE_EMPTY"),
            ([1.0, float("nan")], "PTR_LINEAGE_NON_FINITE"),
            ([0.0, 0.0], "PTR_LINEAGE_ZERO_NORM"),
        ]
        for weights, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as raised:
                    ptr_py.aimer_score(weights)
                self.assertIsInstance(raised.exception, ptr_py.PtrLineageError)
                self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))

    def test_ranking_errors(self):
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.rank_experts_for_pruning([("a", [1.0, 2.0]), ("b", [1.0])])
        self.assertIn("PTR_LINEAGE_SHAPE_MISMATCH", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.rank_experts_for_pruning([("a", [1.0, 2.0]), ("bad", [0.0, 0.0])])
        self.assertIn("PTR_LINEAGE_EXPERT_REJECTED", str(raised.exception))
        self.assertIn("bad", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError):
            ptr_py.rank_experts_for_pruning([])


class SubspaceTests(unittest.TestCase):
    def test_identical_and_orthogonal(self):
        e1 = [[1.0], [0.0], [0.0]]
        scaled = [[2.0], [0.0], [0.0]]
        e2 = [[0.0], [5.0], [0.0]]
        self.assertAlmostEqual(ptr_py.subspace_overlap(e1, scaled), 1.0, places=12)
        self.assertAlmostEqual(ptr_py.subspace_overlap(e1, e2), 0.0, places=12)

    def test_45_degree_line_overlaps_one_half(self):
        # cosine 1/sqrt(2), overlap is the squared cosine.
        self.assertAlmostEqual(
            ptr_py.subspace_overlap([[1.0], [0.0]], [[1.0], [1.0]]), 0.5, places=12
        )

    def test_plane_contains_its_line(self):
        plane = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        line = [[1.0], [1.0], [0.0]]
        self.assertAlmostEqual(ptr_py.subspace_overlap(plane, line), 1.0, places=12)

    def test_chance_overlap_is_larger_rank_over_dim(self):
        line = [[1.0], [0.0], [0.0], [0.0]]
        plane = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(ptr_py.chance_overlap(line, plane), 0.5)

    def test_errors(self):
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.subspace_overlap([[1.0], [0.0]], [[1.0], [0.0], [0.0]])
        self.assertIn("PTR_LINEAGE_SHAPE_MISMATCH", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.chance_overlap([[1.0], [0.0]], [[1.0], [0.0], [0.0]])
        self.assertIn("PTR_LINEAGE_SHAPE_MISMATCH", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.subspace_overlap([], [[1.0]])
        self.assertIn("PTR_LINEAGE_EMPTY", str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            ptr_py.subspace_overlap([[1.0, 2.0], [3.0]], [[1.0], [0.0]])
        self.assertIn("ragged", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.subspace_overlap([[float("inf")]], [[1.0]])
        self.assertIn("PTR_LINEAGE_NON_FINITE", str(raised.exception))


class InterferenceTests(unittest.TestCase):
    @staticmethod
    def update(layer, b, a):
        return (layer, b, a)

    def test_orthogonal_and_identical_updates(self):
        # delta_W = B A; B is d_out x r, A is r x d_in.
        earlier = [
            ("old", [self.update("q", [[1.0], [0.0]], [[1.0, 0.0]])]),
        ]
        same = [self.update("q", [[2.0], [0.0]], [[3.0, 0.0]])]
        other = [self.update("q", [[0.0], [1.0]], [[0.0, 1.0]])]
        report = ptr_py.measure_interference("new", same, earlier)
        self.assertEqual(report["candidate"], "new")
        layer = report["layers"][0]
        self.assertEqual(layer["layer"], "q")
        self.assertAlmostEqual(layer["output_overlap"], 1.0, places=12)
        self.assertAlmostEqual(layer["input_overlap"], 1.0, places=12)
        self.assertEqual(layer["output_chance"], 0.5)
        self.assertEqual(layer["worst"], "old")
        self.assertAlmostEqual(report["max_overlap"], 1.0, places=12)
        report = ptr_py.measure_interference("new", other, earlier)
        layer = report["layers"][0]
        self.assertAlmostEqual(layer["output_overlap"], 0.0, places=12)
        self.assertIsNone(layer["worst"])

    def test_no_earlier_update_on_layer_is_zero(self):
        report = ptr_py.measure_interference(
            "new", [self.update("q", [[1.0], [0.0]], [[1.0, 0.0]])], []
        )
        layer = report["layers"][0]
        self.assertEqual(layer["output_overlap"], 0.0)
        self.assertEqual(layer["output_chance"], 0.0)
        self.assertIsNone(layer["worst"])

    def test_errors(self):
        good = self.update("q", [[1.0], [0.0]], [[1.0, 0.0]])
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.measure_interference("new", [good, good], [])
        self.assertIn("PTR_LINEAGE_DUPLICATE_LAYER", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.measure_interference(
                "new", [self.update("q", [[1.0], [0.0]], [[1.0, 0.0], [0.0, 1.0]])], []
            )
        self.assertIn("PTR_LINEAGE_SHAPE_MISMATCH", str(raised.exception))


class TiesMergeTests(unittest.TestCase):
    def test_disagreeing_signs_do_not_cancel_the_majority(self):
        merged = ptr_py.ties_merge([[2.0, 0.0], [4.0, 1.0], [-1.0, 1.0]], 1.0)
        self.assertEqual(merged, [3.0, 1.0])

    def test_trimming_before_election(self):
        self.assertEqual(ptr_py.ties_merge([[5.0, -0.1], [0.2, 3.0]], 0.5), [5.0, 3.0])

    def test_errors(self):
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.ties_merge([[1.0], [1.0, 2.0]], 1.0)
        self.assertIn("PTR_LINEAGE_SHAPE_MISMATCH", str(raised.exception))
        for density in (0.0, 1.5, float("nan")):
            with self.subTest(density=density):
                with self.assertRaises(ptr_py.PtrLineageError) as raised:
                    ptr_py.ties_merge([[1.0]], density)
                self.assertIn("PTR_LINEAGE_INVALID_PARAMETER", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError):
            ptr_py.ties_merge([], 1.0)


class GateTests(unittest.TestCase):
    ACCURACY = [[0.9, 0.1, 0.1], [0.8, 0.85, 0.1], [0.7, 0.8, 0.9]]

    def evaluate(self, **overrides):
        thresholds = dict(
            max_average_forgetting=1.0,
            max_task_forgetting=1.0,
            min_backward_transfer=-1.0,
            max_public_regression=1.0,
        )
        thresholds.update(overrides)
        return ptr_py.forgetting_gate_evaluate("cand", self.ACCURACY, 0.5, 0.45, **thresholds)

    def test_pass(self):
        report = self.evaluate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["adapter"], "cand")
        self.assertAlmostEqual(report["average_accuracy"], 0.8, places=12)
        self.assertAlmostEqual(report["backward_transfer"], -0.125, places=12)
        self.assertAlmostEqual(report["task_forgetting"][0], 0.2, places=12)
        self.assertAlmostEqual(report["task_forgetting"][1], 0.05, places=12)

    def test_fail_reports_every_violated_threshold(self):
        report = self.evaluate(
            max_average_forgetting=0.1,
            max_task_forgetting=0.15,
            min_backward_transfer=-0.1,
            max_public_regression=0.01,
        )
        self.assertFalse(report["passed"])
        kinds = [v["kind"] for v in report["violations"]]
        self.assertEqual(len(kinds), 4)
        self.assertEqual(
            set(kinds),
            {"average_forgetting", "task_forgetting", "backward_transfer", "public_regression"},
        )
        task = next(v for v in report["violations"] if v["kind"] == "task_forgetting")
        self.assertEqual(task["task"], 0)
        self.assertAlmostEqual(task["observed"], 0.2, places=12)
        self.assertEqual(task["limit"], 0.15)

    def test_errors(self):
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            self.evaluate(max_task_forgetting=float("nan"))
        self.assertIn("PTR_LINEAGE_INVALID_PARAMETER", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.forgetting_gate_evaluate(
                "c", [[0.5, 0.5]], 0.5, 0.5,
                max_average_forgetting=1.0, max_task_forgetting=1.0,
                min_backward_transfer=-1.0, max_public_regression=1.0,
            )
        self.assertIn("PTR_LINEAGE_SHAPE_MISMATCH", str(raised.exception))
        with self.assertRaises(ptr_py.PtrLineageError) as raised:
            ptr_py.forgetting_gate_evaluate(
                "c", self.ACCURACY, float("nan"), 0.5,
                max_average_forgetting=1.0, max_task_forgetting=1.0,
                min_backward_transfer=-1.0, max_public_regression=1.0,
            )
        self.assertIn("PTR_LINEAGE_NON_FINITE", str(raised.exception))

    def test_thresholds_are_keyword_only(self):
        with self.assertRaises(TypeError):
            ptr_py.forgetting_gate_evaluate("c", self.ACCURACY, 0.5, 0.5, 1.0, 1.0, -1.0, 1.0)


class ModuleTests(unittest.TestCase):
    def test_docstrings(self):
        self.assertIn("governance", ptr_py.__doc__)
        for name in (
            "aimer_score", "rank_experts_for_pruning", "subspace_overlap", "chance_overlap",
            "measure_interference", "ties_merge", "forgetting_gate_evaluate",
        ):
            with self.subTest(name=name):
                self.assertIn("Governance check", getattr(ptr_py, name).__doc__)


if __name__ == "__main__":
    unittest.main()
