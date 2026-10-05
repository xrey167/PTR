import unittest

from ptr_training.moe import rl


class RlSeriesTests(unittest.TestCase):
    HISTORY = [
        # step 1: reward/kl/clip logged in one entry
        {"step": 1, "reward": 0.5, "kl": 0.0, "clip_ratio/region_mean": 0.0, "loss": 0.1,
         "entropy": 4.0, "grad_norm": 0.3},
        # step 2: no reward (not a generation step); clip_ratio and kl live here only
        {"step": 2, "kl": 0.2, "clip_ratio/region_mean": 0.125, "loss": 0.2, "entropy": 3.9},
        # step 3: reward and clip logged in SEPARATE entries for the same step
        {"step": 3, "reward": 0.25, "loss": 0.3, "grad_norm": 0.2},
        {"step": 3, "clip_ratio/region_mean": 0.5, "kl": 0.3, "entropy": 3.8},
        {"train_runtime": 1.0, "train_loss": 0.2},  # summary: no step, not per-step data
    ]

    def _out(self):
        merged = rl.merge_log_history(self.HISTORY)
        out = {}
        rl._series(out, merged, "clip_ratio", "clip_ratio/region_mean", "none")
        rl._series(out, merged, "reward", "reward", "none")
        rl._series(out, merged, "kl", "kl", "none")
        rl._series(out, merged, "entropy", "entropy", "none")
        rl._series(out, merged, "grad_norm", "grad_norm", "none")
        rl._series(out, merged, "loss", "loss", "none")
        return out

    def test_merge_groups_entries_by_step(self):
        merged = rl.merge_log_history(self.HISTORY)
        self.assertEqual(list(merged), [1, 2, 3])
        self.assertEqual(merged[3]["reward"], 0.25)
        self.assertEqual(merged[3]["clip_ratio/region_mean"], 0.5)
        self.assertNotIn("train_runtime", {k for h in merged.values() for k in h})

    def test_clip_not_lost_when_logged_apart_from_reward(self):
        out = self._out()
        self.assertEqual(out["clip_ratio_per_step"], [0.0, 0.125, 0.5])
        self.assertEqual(out["series_steps"]["clip_ratio"], [1, 2, 3])
        self.assertEqual(out["reward_per_step"], [0.5, 0.25])
        self.assertEqual(out["series_steps"]["reward"], [1, 3])
        self.assertEqual(out["kl_per_step"], [0.0, 0.2, 0.3])
        self.assertEqual(out["series_steps"]["kl"], [1, 2, 3])
        self.assertEqual(out["entropy_per_step"], [4.0, 3.9, 3.8])
        self.assertEqual(out["grad_norm_per_step"], [0.3, 0.2])
        self.assertEqual(out["loss_per_step"], [0.1, 0.2, 0.3])

    def test_missing_metric_is_null_with_reason(self):
        out = {}
        rl._series(out, rl.merge_log_history(self.HISTORY), "nope", "nope", "why")
        self.assertIsNone(out["nope_per_step"])
        self.assertEqual(out["nope_unavailable_reason"], "why")
        self.assertEqual(out["series_steps"]["nope"], [])


if __name__ == "__main__":
    unittest.main()
