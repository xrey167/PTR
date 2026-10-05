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

    def test_absent_binding_is_skipped_not_error(self):
        for exc in (ModuleNotFoundError("x", name="ptr_py"),
                    ModuleNotFoundError("x", name="ptr_py._native")):
            with mock.patch.object(gates.importlib, "import_module", side_effect=exc):
                self.assertEqual(gates.binding_state()[0], "absent")
                self.assertEqual(gates.binding_problem()["status"], "skipped")
                self.assertEqual(gates.run_gates(_weights(), _weights())["status"], "skipped")

    def test_broken_binding_is_error_with_exception_text(self):
        broken = (
            ModuleNotFoundError("No module named 'numpy'", name="numpy"),
            ImportError("libptr.so: cannot open shared object file"),
            OSError("bad ELF header"),
        )
        for exc in broken:
            with self.subTest(exc=repr(exc)), mock.patch.object(
                gates.importlib, "import_module", side_effect=exc
            ):
                self.assertEqual(gates.binding_state()[0], "broken")
                report = gates.run_gates(_weights(), _weights())
                self.assertEqual(report["status"], "error")
                self.assertIn(str(exc), report["error"])
                self.assertIn("ptr_py import failed", report["error"])
                self.assertIsNone(gates.load_binding())

    def test_binding_problem_helper_with_stubs(self):
        with mock.patch.dict(sys.modules, {"ptr_py": types.ModuleType("ptr_py")}):
            self.assertIn("lacks", gates.binding_problem()["reason"])
        with mock.patch.dict(sys.modules, {"ptr_py": self._stub()}):
            self.assertIsNone(gates.binding_problem())

    def _overlap_stub(self):
        """Test double with the binding's chance formula max(rank)/dim (rank = #columns here)."""
        stub = self._stub()
        stub.subspace_overlap = lambda a, b: 0.5
        stub.chance_overlap = lambda a, b: max(len(a[0]), len(b[0])) / len(a)
        return stub

    def _delta_weights(self, dim, k, seed):
        import random

        rng = random.Random(seed)
        rows = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(dim)]
        return [_e([0.0, 0.0], rows)]

    def test_top_left_subspace_recovers_dominant_directions(self):
        m = [[5.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.05]]
        basis = gates.top_left_subspace(m, 2)  # dim x r row-major
        self.assertEqual((len(basis), len(basis[0])), (4, 2))
        for col in range(2):  # unit columns, supported on the first two coordinates
            self.assertAlmostEqual(sum(row[col] ** 2 for row in basis), 1.0, places=6)
            self.assertLess(sum(basis[i][col] ** 2 for i in (2, 3)), 1e-6)
        self.assertEqual(gates.top_left_subspace([[0.0, 0.0], [0.0, 0.0]], 1), [])

    def test_informative_flag_full_rank_vs_truncated(self):
        # two experts per layer, 8x8 deltas (full rank in the 8-dim ambient space)
        w0 = [[self._delta_weights(8, 8, 1)[0], self._delta_weights(8, 8, 2)[0]]]
        w1 = [[_e([0.0, 0.0], [[0.0] * 8 for _ in range(8)]),
               _e([0.0, 0.0], [[0.0] * 8 for _ in range(8)])]]
        with mock.patch.dict(sys.modules, {"ptr_py": self._overlap_stub()}):
            # full-rank comparison: chance = 8/8 = 1.0 -> not informative
            full = gates.run_gates(w1, w0, max_pairs_per_layer=1)
            pair = full["layers"][0]["delta_pairs"][0]
            self.assertFalse(pair["informative"])
            self.assertFalse(full["interference_informative"])
            # truncated to r=2 of 8: chance = 0.25 -> informative
            cut = gates.run_gates(w1, w0, max_pairs_per_layer=1, subspace_rank=2)
            pair = cut["layers"][0]["delta_pairs"][0]
            self.assertTrue(pair["informative"])
            self.assertEqual((pair["subspace_rank"], pair["ambient_dim"]), (2, 8))
            self.assertTrue(cut["interference_informative"])
            # r >= ambient dim is never informative, whatever the binding says
            wide = gates.run_gates(w1, w0, max_pairs_per_layer=1, subspace_rank=8)
            self.assertFalse(wide["layers"][0]["delta_pairs"][0]["informative"])

    def test_low_rank_inputs_are_informative_without_truncation(self):
        w0 = [[self._delta_weights(8, 1, 1)[0], self._delta_weights(8, 1, 2)[0]]]
        w1 = [[_e([0.0, 0.0], [[0.0] for _ in range(8)]) for _ in range(2)]]
        with mock.patch.dict(sys.modules, {"ptr_py": self._overlap_stub()}):
            report = gates.run_gates(w1, w0, max_pairs_per_layer=1)
        self.assertTrue(report["layers"][0]["delta_pairs"][0]["informative"])

    def test_zero_delta_with_truncation_is_not_informative(self):
        z = [[_e([0.0, 0.0], [[0.0] * 4 for _ in range(4)]) for _ in range(2)]]
        with mock.patch.dict(sys.modules, {"ptr_py": self._overlap_stub()}):
            report = gates.run_gates(z, z, max_pairs_per_layer=1, subspace_rank=2)
        pair = report["layers"][0]["delta_pairs"][0]
        self.assertFalse(pair["informative"])
        self.assertEqual(pair["subspace_overlap"]["status"], "skipped")

    def test_binding_version_from_distribution_metadata(self):
        stub = types.ModuleType("ptr_py")
        with mock.patch.object(gates.metadata, "version", return_value="0.1.0"):
            self.assertEqual(gates.binding_version(stub), "0.1.0")
        stub.__version__ = "9.9"
        err = gates.metadata.PackageNotFoundError("ptr_py")
        with mock.patch.object(gates.metadata, "version", side_effect=err):
            self.assertEqual(gates.binding_version(stub), "9.9")
            self.assertEqual(gates.binding_version(types.ModuleType("ptr_py")), "unknown")
        with mock.patch.dict(sys.modules, {"ptr_py": self._stub()}), mock.patch.object(
            gates.metadata, "version", return_value="0.1.0"
        ):
            report = gates.run_gates(_weights(), _after())
        self.assertEqual(report["binding_version"], "0.1.0")

    def test_pair_selection_is_strided_and_deterministic(self):
        pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)]  # 28
        self.assertEqual(gates.select_pairs(pairs, 100), pairs)
        chosen = gates.select_pairs(pairs, 6)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(chosen, gates.select_pairs(pairs, 6))
        self.assertEqual(len(set(chosen)), 6)
        firsts = {a for a, _ in chosen}
        self.assertGreater(len(firsts), 2)  # spread over experts, not just expert 0

    def test_report_records_pairs_evaluated_and_total(self):
        w = [[_e([1.0, 2.0], [[float(i)], [1.0]]) for i in range(5)]]
        after = [[_e([2.0, 2.0], [[float(i) + 1], [2.0]]) for i in range(5)]]
        with mock.patch.dict(sys.modules, {"ptr_py": self._stub()}):
            report = gates.run_gates(w, after, max_pairs_per_layer=4)
        layer = report["layers"][0]
        self.assertEqual((layer["pairs_total"], layer["pairs_evaluated"]), (10, 4))
        self.assertEqual(len(layer["delta_pairs"]), 4)

    def test_run_gates_upgrades_to_error_when_a_call_failed(self):
        stub = self._stub()

        def boom(*args):
            raise TypeError("bad signature")

        stub.aimer_score = boom
        with mock.patch.dict(sys.modules, {"ptr_py": stub}):
            report = gates.run_gates(_weights(), _after())  # called directly, not via pipeline
        self.assertEqual(report["status"], "error")
        self.assertGreaterEqual(report["failed_calls"], 1)

    def test_error_outranks_partial(self):
        stub = self._stub()
        stub.aimer_score = lambda w: (_ for _ in ()).throw(ValueError("nope"))
        w = _weights() + [[_e([1.0] * 5, [[1.0]]), _e([1.0] * 5, [[1.0]])]]
        a = _after() + [[_e([2.0] * 5, [[2.0]]), _e([2.0] * 5, [[2.0]])]]
        with mock.patch.dict(sys.modules, {"ptr_py": stub}), mock.patch.object(
            gates, "MAX_FLAT_ELEMENTS", 4
        ):
            report = gates.run_gates(w, a)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["layers_skipped"], 1)

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
