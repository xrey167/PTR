import unittest

from ptr_training.vcw import budget as b
from ptr_training.vcw import plan


class BudgetTests(unittest.TestCase):
    def test_model_arithmetic(self):
        self.assertAlmostEqual(b.weights_gib(), 12.89, places=1)
        self.assertEqual(b.EXPERT_PARAMS + b.NON_EXPERT_PARAMS, b.TOTAL_PARAMS)
        self.assertGreater(b.NON_EXPERT_PARAMS, 0)

    def test_perft_r_parameter_arithmetic_matches_m011_target(self):
        n, r, h, layers = 8, 8, b.HIDDEN, b.LAYERS
        self.assertEqual(layers * (n * 2 * h * r + h * n), plan.PERFT_PARAMS)
        self.assertEqual(plan.PERFT_PARAMS, 4_456_448)
        self.assertEqual(8 * 2 * (b.LAYERS * 4 * 2 * h), plan.REPAIR_PARAMS)  # 8 classes x rank 2

    def test_checkpointing_saves_activation_memory(self):
        self.assertLess(b.train_activation_gib(4096, True), b.train_activation_gib(4096, False) / 4)

    def test_memory_grows_with_tokens(self):
        self.assertLess(b.train_peak_gib(1024, True, 1), b.train_peak_gib(4096, True, 1))

    def test_fit_rule_keeps_headroom(self):
        self.assertTrue(b.fits(22.5, "rtx4090-reference"))
        self.assertFalse(b.fits(22.6, "rtx4090-reference"))
        self.assertTrue(b.fits(30.5, "rtx5090-reference"))

    def test_decode_step_is_host_bound_not_bandwidth_bound(self):
        for pid in b.GPUS:
            low, mid, high = b.decode_step_seconds(32, pid)
            self.assertEqual((low, mid, high), b.DECODE_STEP_OVERHEAD_S)

    def test_ranges_are_ordered_slow_to_fast(self):
        for t in (b.train_seconds(1e6, True, "rtx4090-reference"), b.generate_seconds(64, 100, 32, "rtx5090-reference")):
            self.assertGreater(t[0], t[1])
            self.assertGreater(t[1], t[2])

    def test_pinned_m010_protocol_does_not_fit_either_card_with_this_backend(self):
        w = b.RlWorkload(2000, 4, 8, 4, 256, 512, plan.PERFT_PARAMS)
        w_ckpt = b.RlWorkload(2000, 4, 8, 4, 256, 512, plan.PERFT_PARAMS, True)
        for pid in b.GPUS:
            self.assertFalse(b.fits(b.rl_peak_gib(w), pid))
            self.assertFalse(b.fits(b.rl_peak_gib(w_ckpt), pid))

    def test_profiles_fit_their_own_protocols(self):
        for pid, p in plan.PROTOCOL.items():
            s, r, q = p["sft"], p["rl"], p["repair"]
            self.assertTrue(b.fits(b.sft_peak_gib(b.SftWorkload(s["steps"], s["micro_batch"], s["accumulation"], s["seq_tokens"], True, plan.PERFT_PARAMS)), pid))
            self.assertTrue(b.fits(b.rl_peak_gib(b.RlWorkload(r["steps"], r["prompts"], r["generations"], r["mini_steps"], r["prompt_tokens"], r["completion_tokens"], plan.PERFT_PARAMS)), pid))
            self.assertTrue(b.fits(b.sft_peak_gib(b.SftWorkload(q["steps"], q["micro_batch"], q["accumulation"], q["seq_tokens"], True, plan.REPAIR_PARAMS)), pid))

    def test_ledger_rows_are_well_formed(self):
        for pid in plan.PROTOCOL:
            rows = plan.ledger(pid)
            self.assertGreaterEqual(len(rows), 10)
            for exp, item, hrs, n, note in rows:
                self.assertTrue(hrs[0] > hrs[1] > hrs[2] > 0, (exp, item))
                self.assertGreaterEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
