import math
import unittest

from ptr_training.moe import routing_metrics as rm


class MoeRoutingMetricsTests(unittest.TestCase):
    def test_top_k_set_and_tie_break(self):
        self.assertEqual(rm.top_k_set([0.1, 0.9, 0.5, 0.7], 2), frozenset({1, 3}))
        # ties resolve toward the lower expert index
        self.assertEqual(rm.top_k_set([1.0, 1.0, 1.0], 2), frozenset({0, 1}))
        with self.assertRaises(ValueError):
            rm.top_k_set([1.0, 2.0], 3)

    def test_churn_hand_computed(self):
        # k=2 over 4 experts; top-2 sets by hand:
        #   before: {0,1} {2,3} {2,3} {0,1}
        #   after:  {0,1} {0,1} {2,3} {1,2}
        # token2 keeps the same set with a different order (not churn), so 2 of 4 changed.
        before = [[3, 2, 1, 0], [0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1]]
        after = [[3, 2, 1, 0], [3, 2, 1, 0], [0, 1, 3, 2], [0, 3, 2, 1]]
        self.assertEqual(
            rm.top_k_sets(before, 2),
            [frozenset({0, 1}), frozenset({2, 3}), frozenset({2, 3}), frozenset({0, 1})],
        )
        self.assertEqual(
            rm.top_k_sets(after, 2),
            [frozenset({0, 1}), frozenset({0, 1}), frozenset({2, 3}), frozenset({1, 2})],
        )
        self.assertEqual(rm.expert_churn(before, after, 2), 0.5)

    def test_churn_extremes_and_errors(self):
        a = [[3, 2, 1, 0], [0, 1, 2, 3]]
        self.assertEqual(rm.expert_churn(a, a, 2), 0.0)
        b = [[0, 1, 2, 3], [3, 2, 1, 0]]
        self.assertEqual(rm.expert_churn(a, b, 2), 1.0)
        with self.assertRaises(ValueError):
            rm.expert_churn(a, a[:1], 2)
        with self.assertRaises(ValueError):
            rm.expert_churn([], [], 2)

    def test_entropy_uniform_and_peaked(self):
        self.assertAlmostEqual(rm.routing_entropy([[0.0, 0.0, 0.0, 0.0]]), math.log(4))
        self.assertAlmostEqual(rm.routing_entropy([[0.0, 0.0]] * 3), math.log(2))
        self.assertLess(rm.routing_entropy([[100.0, 0.0, 0.0, 0.0]]), 1e-9)
        # two-expert logits [0, ln 3] give p = (1/4, 3/4)
        h = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        self.assertAlmostEqual(rm.routing_entropy([[0.0, math.log(3)]]), h)

    def test_load_cv_dead_experts(self):
        # 4 tokens, k=1, 4 experts: tokens pick experts 0,0,0,1 -> load [3,1,0,0]
        layer = [[9, 0, 0, 0], [9, 0, 0, 0], [9, 0, 0, 0], [0, 9, 0, 0]]
        self.assertEqual(rm.expert_load(layer, 1), [3, 1, 0, 0])
        self.assertEqual(rm.dead_experts(layer, 1), 2)
        # mean 1, population variance (4+0+1+1)/4 = 1.5
        self.assertAlmostEqual(rm.load_cv(layer, 1), math.sqrt(1.5))

    def test_balanced_load_has_zero_cv(self):
        layer = [[9, 0], [0, 9], [9, 0], [0, 9]]
        self.assertEqual(rm.load_cv(layer, 1), 0.0)
        self.assertEqual(rm.dead_experts(layer, 1), 0)

    def test_layer_and_snapshot_metrics(self):
        layer = [[9, 0, 0, 0], [0, 9, 0, 0]]
        m = rm.layer_metrics(layer, 1)
        self.assertEqual(set(m), {"entropy", "load_cv", "dead_experts", "load"})
        self.assertEqual(m["dead_experts"], 2)
        snap = [layer, layer]
        self.assertEqual(len(rm.snapshot_metrics(snap, 1)), 2)
        self.assertEqual(rm.snapshot_churn(snap, snap, 1), [0.0, 0.0])
        with self.assertRaises(ValueError):
            rm.snapshot_churn(snap, snap[:1], 1)

    def test_adapter_with_stub_tensors(self):
        class Stub:
            def __init__(self, data):
                self.data = data

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def reshape(self, *_):
                return self

            def tolist(self):
                return self.data

        logits = (Stub([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]),)
        mask = Stub([1, 0, 1])
        out = rm.router_logits_to_lists(logits, mask)
        self.assertEqual(out, [[[1.0, 0.0], [5.0, 5.0]]])
        with self.assertRaises(ValueError):
            rm.router_logits_to_lists(logits, Stub([1, 1]))
        self.assertEqual(len(rm.router_logits_to_lists(logits)[0]), 3)


if __name__ == "__main__":
    unittest.main()
