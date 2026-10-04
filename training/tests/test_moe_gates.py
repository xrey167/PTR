import sys
import types
import unittest
from unittest import mock

from ptr_training.moe import gates


def _e(flat, down):
    return {"flat": flat, "down": down}


def _weights():
    # [layer][expert]; 1 layer, 3 experts, down_proj is 2x1
    return [[_e([1.0, 2.0], [[1.0], [2.0]]), _e([3.0, 4.0], [[3.0], [4.0]]),
             _e([5.0, 6.0], [[5.0], [6.0]])]]


def _after():
    return [[_e([2.0, 2.0], [[2.0], [2.0]]), _e([3.0, 1.0], [[3.0], [1.0]]),
             _e([5.0, 6.0], [[5.0], [6.0]])]]


class MoeGatesTests(unittest.TestCase):
    def test_skipped_without_binding(self):
        with mock.patch.dict(sys.modules, {"ptr_py": None}):
            report = gates.run_gates(_weights(), _weights())
        self.assertEqual(report["status"], "skipped")
        self.assertIn("binding not installed", report["reason"])

    def test_skipped_when_binding_lacks_functions(self):
        stub = types.ModuleType("ptr_py")
        with mock.patch.dict(sys.modules, {"ptr_py": stub}):
            report = gates.run_gates(_weights(), _weights())
        self.assertEqual(report["status"], "skipped")
        self.assertIn("lacks", report["reason"])

    def test_deltas(self):
        d = gates.expert_deltas(_weights(), _after())
        self.assertEqual([e["flat"] for e in d[0]], [[1.0, 0.0], [0.0, -3.0], [0.0, 0.0]])
        self.assertEqual(d[0][1]["down"], [[0.0], [-3.0]])

    def test_plumbing_with_test_double_binding(self):
        """Test double only checks call wiring; its numbers are never reported as results."""
        calls = []
        stub = types.ModuleType("ptr_py")
        stub.aimer_score = lambda w: calls.append(("aimer", len(w))) or 0.0
        stub.rank_experts_for_pruning = lambda xs: [i for i, _ in xs]
        stub.subspace_overlap = lambda a, b: calls.append(("overlap", len(a), len(b))) or 0.0
        stub.chance_overlap = lambda a, b: calls.append(("chance", len(a), len(b))) or 0.0
        with mock.patch.dict(sys.modules, {"ptr_py": stub}):
            report = gates.run_gates(_weights(), _after(), max_pairs_per_layer=2)
        self.assertEqual(report["status"], "ok")
        layer = report["layers"][0]
        self.assertEqual(len(layer["aimer_score"]), 3)
        self.assertEqual(len(layer["delta_pairs"]), 2)
        self.assertEqual(sum(1 for c in calls if c[0] == "aimer"), 3)
        self.assertEqual(report["layers"][0]["prune_ranking"]["value"], ["0", "1", "2"])
        self.assertIn(("overlap", 2, 2), calls)

    def _stub(self):
        stub = types.ModuleType("ptr_py")
        stub.aimer_score = lambda w: 0.0
        stub.rank_experts_for_pruning = lambda xs: []
        stub.subspace_overlap = lambda a, b: 0.0
        stub.chance_overlap = lambda a, b: 0.0
        return stub

    def test_all_layers_skipped_is_not_ok(self):
        with mock.patch.dict(sys.modules, {"ptr_py": self._stub()}), mock.patch.object(
            gates, "MAX_FLAT_ELEMENTS", 1
        ):
            report = gates.run_gates(_weights(), _after())
        self.assertEqual(report["status"], "skipped")
        self.assertIn("all 1 layers skipped", report["reason"])

    def test_some_layers_skipped_is_partial_with_counts(self):
        w = _weights() + [[_e([1.0] * 5, [[1.0]]), _e([1.0] * 5, [[1.0]])]]
        a = _after() + [[_e([2.0] * 5, [[2.0]]), _e([2.0] * 5, [[2.0]])]]
        with mock.patch.dict(sys.modules, {"ptr_py": self._stub()}), mock.patch.object(
            gates, "MAX_FLAT_ELEMENTS", 4
        ):
            report = gates.run_gates(w, a)
        self.assertEqual(report["status"], "partial")
        self.assertEqual((report["layers_run"], report["layers_skipped"]), (1, 1))

    def test_pipeline_gate_status_keeps_error_precedence(self):
        from ptr_training.moe.pipeline import _gate_status

        err = {"status": "partial", "layers": [{"layer": 0, "x": {"status": "error"}},
                                               {"layer": 1, "status": "skipped"}]}
        out = _gate_status(err)
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["failed_calls"], 1)
        partial = _gate_status({"status": "partial", "layers": [{"status": "skipped"}]})
        self.assertEqual(partial["status"], "partial")
        skipped = {"status": "skipped", "reason": "r", "layers": []}
        self.assertEqual(_gate_status(skipped), skipped)

    def test_binding_errors_are_recorded_not_hidden(self):
        stub = types.ModuleType("ptr_py")
        stub.aimer_score = lambda w: 0.0
        stub.rank_experts_for_pruning = lambda xs: []

        def boom(*args):
            raise TypeError("unexpected signature")

        stub.subspace_overlap = boom
        stub.chance_overlap = lambda a, b: 0.0
        with mock.patch.dict(sys.modules, {"ptr_py": stub}):
            report = gates.run_gates(_weights(), _weights(), max_pairs_per_layer=1)
        pair = report["layers"][0]["delta_pairs"][0]
        self.assertEqual(pair["subspace_overlap"]["status"], "error")
        self.assertIn("unexpected signature", pair["subspace_overlap"]["error"])


if __name__ == "__main__":
    unittest.main()
