"""Tests of the task and the models. They need torch and skip without it (the repository's CI python has none)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/consciousness/gw"))
try:
    import torch
    import harness
    import model as gwmodel
    import task

    harness._load_torch()  # the harness binds torch and its modules lazily, after its environment check
except ModuleNotFoundError:  # no torch here
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TaskTests(unittest.TestCase):
    def setUp(self):
        self.cfg = harness.load_config()

    def batch(self, n_modules, n=512, seed=1, noise=0.0):
        """Noise-free by default, so that the one-hot features can be read back exactly."""
        g = torch.Generator().manual_seed(seed)
        return harness.make_batch(g, n, n_modules, {**self.cfg, "noise": noise})

    def test_exactly_two_modules_have_the_queried_type_at_any_module_count(self):
        d = self.cfg["n_types"]
        for n_modules in (8, 16, 24):
            b = self.batch(n_modules, n=64)
            types = b["x"][..., :d].round().argmax(-1)
            for row in range(64):
                cue = int(b["cue"][row])
                same = [i for i in range(n_modules) if i != cue and int(types[row, i]) == int(b["q"][row])]
                self.assertEqual(sorted(same), sorted([int(b["pi"][row]), int(b["pj"][row])]))
                self.assertNotIn(cue, same)

    def test_answer_is_the_sum_of_the_two_values(self):
        d, t = self.cfg["n_types"], self.cfg["n_values"]
        b = self.batch(8)
        vals = b["x"][..., d : d + t].argmax(-1)
        rows = torch.arange(vals.shape[0])
        self.assertTrue(torch.equal(b["y"], vals[rows, b["pi"]] + vals[rows, b["pj"]]))

    def test_a_single_module_does_not_determine_the_answer(self):
        d, t = self.cfg["n_types"], self.cfg["n_values"]
        b = self.batch(8, n=4096)
        vals = b["x"][..., d : d + t].argmax(-1)
        first = vals[torch.arange(vals.shape[0]), b["pi"]]
        for v in range(t):
            self.assertGreater(len(set(b["y"][first == v].tolist())), 1)

    def test_the_task_is_a_pure_function_of_the_generator(self):
        a, b = self.batch(8, seed=3), self.batch(8, seed=3)
        self.assertTrue(torch.equal(a["x"], b["x"]) and torch.equal(a["y"], b["y"]))
        self.assertFalse(torch.equal(a["x"], self.batch(8, seed=4)["x"]))

    def test_row_hashes_detect_identical_examples_only(self):
        b = self.batch(8, n=16, noise=0.5)
        hashes = set(harness.row_hashes(b["x"]))
        self.assertEqual(harness.overlap_count(hashes, b["x"][:4]), 4)
        self.assertEqual(harness.overlap_count(hashes, self.batch(8, n=16, seed=9, noise=0.5)["x"]), 0)

    def test_the_n_leak_check_counts_observed_ood_counts(self):
        self.assertEqual(harness.n_leak_count({8}, (16, 24)), 0)
        self.assertEqual(harness.n_leak_count({8, 16}, (16, 24)), 1)


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = harness.load_config()
        torch.manual_seed(0)

    def x(self, n_modules, batch=3):
        return torch.randn(batch, n_modules, self.cfg["d_in"])

    def test_every_arm_runs_at_any_module_count(self):
        for name in gwmodel.ARM_SPECS:
            net = gwmodel.build(name, self.cfg)
            for n_modules in (8, 16, 24):
                self.assertEqual(tuple(net(self.x(n_modules)).shape), (3, 2 * self.cfg["n_values"] - 1), (name, n_modules))

    def test_width_matching_reaches_the_tolerance_for_every_arm(self):
        target = gwmodel.count_params(gwmodel.build("workspace", self.cfg))
        for name in gwmodel.ARM_SPECS:
            if name == "workspace":
                continue
            ffn = gwmodel.match_width(name, self.cfg, target)
            n = gwmodel.count_params(gwmodel.build(name, self.cfg, ffn=ffn))
            self.assertLessEqual(1000 * abs(n - target), self.cfg["param_tolerance_permille"] * target, name)

    def test_arm_settings_make_the_ablations_what_they_say(self):
        s = {n: gwmodel.arm_settings(n, self.cfg) for n in gwmodel.ARM_SPECS}
        self.assertGreaterEqual(s["competition_off"]["top_k"], 24)
        self.assertEqual(s["capacity_off"]["slots"], self.cfg["n_train_modules"])
        self.assertEqual(s["crippled_all_to_all"]["rounds"], 1)
        self.assertEqual(s["all_to_all"]["rounds"], self.cfg["rounds"])
        self.assertLess(s["bw_all_to_all"]["d_value"], self.cfg["d_model"])
        self.assertIsNone(s["all_to_all"]["d_value"])

    def test_communication_cost_is_linear_in_modules_for_the_workspace_and_quadratic_for_pairwise(self):
        ws = [gwmodel.comm_ops("workspace", self.cfg, n) for n in (8, 16, 24)]
        a2a = [gwmodel.comm_ops("all_to_all", self.cfg, n) for n in (8, 16, 24)]
        self.assertEqual(ws[1] - ws[0], ws[2] - ws[1])
        self.assertGreater(a2a[2] - a2a[1], a2a[1] - a2a[0])
        self.assertEqual(gwmodel.comm_ops("no_comm", self.cfg, 8), 0)

    def test_no_comm_reporter_ignores_inputs_and_no_broadcast_isolates_sensory_modules(self):
        x = self.x(6, batch=4)
        self.assertEqual(harness.leak_count(gwmodel.build("no_comm", self.cfg), x, 5.0, "reporter"), 0)
        self.assertEqual(harness.leak_count(gwmodel.build("no_broadcast", self.cfg), x, 5.0, "sensory"), 0)
        self.assertGreater(harness.leak_count(gwmodel.build("workspace", self.cfg), x, 5.0, "sensory"), 0)
        self.assertGreater(harness.leak_count(gwmodel.build("all_to_all", self.cfg), x, 5.0, "sensory"), 0)

    def test_write_attention_is_top_k_sparse_and_competition_off_is_not(self):
        x = self.x(8, batch=5)
        for name in ("workspace", "no_broadcast", "capacity_off"):
            net = gwmodel.build(name, self.cfg)
            with torch.no_grad():
                net(x)
            self.assertEqual(harness.bottleneck_violations(net.trace, net.top_k), 0, name)
            self.assertGreater(harness.bottleneck_violations(net.trace, net.top_k - 1), 0, name)
        net = gwmodel.build("competition_off", self.cfg)
        with torch.no_grad():
            net(x)
        self.assertGreater(harness.bottleneck_violations(net.trace, self.cfg["top_k"]), 0)

    def test_dropped_module_input_has_no_effect(self):
        x = self.x(6, batch=4)
        for name in ("workspace", "all_to_all", "no_broadcast"):
            self.assertEqual(harness.dropped_leaks(gwmodel.build(name, self.cfg), x, 5.0), 0, name)
        self.assertTrue(harness.dropout_probe_is_sensitive(gwmodel.build("all_to_all", self.cfg), x, 5.0))


if __name__ == "__main__":
    unittest.main()
