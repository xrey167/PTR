"""M011 entrypoint: tiny end-to-end run per arm on CPU (2 steps), tuning, hyperparameter reuse."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from ptr_training.moe.pipeline import stack_available

HAVE = stack_available(("torch", "transformers", "peft"))

if HAVE:
    from ptr_training.moe import m011

ARMS = ("perft_r_emergent", "perft_r_prescribed", "plain_lora", "mixlora")
SMOKE = ["--model", "tiny", "--device", "cpu", "--max-steps", "2", "--eval-limit", "2",
         "--batch-size", "4"]


def finite_numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [obj]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in finite_numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in finite_numbers(v)]
    return []


@unittest.skipUnless(HAVE, "torch/transformers/peft not installed (moe extra)")
class TinyEndToEndTests(unittest.TestCase):
    def run_cli(self, arm, out, *extra, seed=3):
        rc = m011.main(["--arm", arm, "--seed", str(seed), "--out", str(out), *SMOKE, *extra])
        self.assertEqual(rc, 0)
        return out

    def test_each_arm_writes_finite_run_and_metrics(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                out = self.run_cli(arm, Path(tmp))
                run = json.loads((out / "run.json").read_text())
                metrics = json.loads((out / "metrics.json").read_text())
                self.assertEqual((run["arm"], run["seed"], run["device"]), (arm, 3, "cpu"))
                self.assertEqual(run["mode"], "final")
                self.assertEqual(run["training_steps"], {"pinned": 2000, "run": 2, "smoke_override": True})
                self.assertEqual(run["model"]["revision"] if "revision" in run["model"] else None, None)
                self.assertEqual(run["pins"]["model_checkpoint_revision"], "9b0c1aa87e34a20052389dce1f0cf01da783f654")
                self.assertTrue(run["expert_domain_assignment"]["matches_pinned_digest"])
                self.assertEqual(run["expert_domain_assignment"]["applied"], arm == "perft_r_prescribed")
                self.assertEqual(set(run["corpora"]), {"gsm8k", "humaneval", "bfcl_simple"})
                for report in run["corpora"].values():
                    self.assertEqual(report["validation_overlap_count"], 0)
                    self.assertEqual(len(report["selection_digest"]), 3)
                self.assertEqual(run["tuning_trials_used"], 0)
                for key in ("torch", "transformers", "peft"):
                    self.assertTrue(run["versions"][key])
                self.assertEqual(run["trainable_params"], metrics["trainable_params"])
                self.assertGreater(metrics["trainable_params"], 0)
                self.assertEqual(set(metrics["per_domain_accuracy"]), {"code", "math", "tool"})
                self.assertTrue(0.0 <= metrics["mean_task_accuracy"] <= 1.0)
                nums = finite_numbers(metrics)
                self.assertTrue(nums and all(math.isfinite(x) for x in nums))
                self.assertGreater(metrics["load_cv"], -1e-12)
                self.assertGreater(metrics["routing_entropy"], 0.0)
                self.assertEqual(
                    metrics["routing_source"], "base_router" if arm == "plain_lora" else "adapter_router"
                )
                self.assertEqual(metrics["training"]["steps"], 2)

    def test_prescribed_adapter_router_load_stays_in_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_cli("perft_r_prescribed", Path(tmp))
            routing = json.loads((out / "metrics.json").read_text())["routing"]
            self.assertEqual(set(routing["per_domain_adapter_router"]), {"code", "math", "tool"})

    def test_tuning_protocol_tb1_and_frozen_hparams_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tune_dir = Path(tmp) / "tune"
            # no --seed: tuning defaults to the first pinned seed; 4 steps -> 1 tuning step
            rc = m011.main(["--arm", "mixlora", "--out", str(tune_dir), "--tune",
                            "--tune-budget", "2", "--model", "tiny", "--device", "cpu",
                            "--max-steps", "4", "--eval-limit", "2"])
            self.assertEqual(rc, 0)
            tuning = json.loads((tune_dir / "tuning.json").read_text())
            run = json.loads((tune_dir / "run.json").read_text())
            self.assertEqual((tuning["tuning_seed"], run["seed"], run["tuning_seed"]), (17, 17, 17))
            self.assertEqual((tuning["tuning_steps"], run["tuning_steps"]), (1, 1))
            self.assertEqual(tuning["tuning_trials_used"], 2)
            self.assertEqual(run["tuning_trials_used"], 2)
            self.assertTrue(tuning["frozen_for_all_seeds"])
            self.assertEqual(run["mode"], "tune")
            self.assertEqual(run["evaluation"]["split"], "validation")
            # deviations from the pinned protocol are recorded (steps 500 -> 1, trials 8 -> 2)
            params = {d["parameter"]: d for d in tuning["deviations"]}
            self.assertEqual(set(params), {"tuning_steps", "tuning_trials_used"})
            self.assertEqual((params["tuning_steps"]["pinned"], params["tuning_steps"]["used"]), (500, 1))
            self.assertEqual(run["tuning_deviations"], tuning["deviations"])
            trials = m011.sample_trials(budget=2)
            self.assertEqual(tuning["trial_set_digest"], m011.trial_set_digest(trials))
            self.assertEqual(run["tuning_trial_set_digest"], tuning["trial_set_digest"])
            self.assertEqual(
                [(t["learning_rate"], t["batch_size"]) for t in tuning["trials"]],
                [(t["learning_rate"], t["batch_size"]) for t in trials],
            )
            self.assertEqual(run["tuning_selected"], tuning["selected"])
            self.assertFalse((tune_dir / "metrics.json").exists())

            # the same frozen hyperparameters are used for two different seeds, full budget
            for seed in (29, 43):
                final_dir = Path(tmp) / f"seed{seed}"
                rc = m011.main(["--arm", "mixlora", "--seed", str(seed), "--out", str(final_dir),
                                "--hparams", str(tune_dir / "tuning.json"), "--model", "tiny",
                                "--device", "cpu", "--max-steps", "3", "--eval-limit", "2",
                                "--micro-batch-size", "2"])
                self.assertEqual(rc, 0)
                final = json.loads((final_dir / "run.json").read_text())
                hp = final["hyperparameters"]
                self.assertEqual(hp["source"], "tuned")
                self.assertEqual(hp["learning_rate"], tuning["selected"]["learning_rate"])
                self.assertEqual(hp["batch_size"], tuning["selected"]["batch_size"])
                self.assertEqual(final["training_steps"]["run"], 3)
                self.assertEqual(final["seed"], seed)
                self.assertEqual(final["tuning_seed"], 17)
                self.assertEqual(final["tuning_steps"], 1)
                self.assertEqual(final["tuning_trials_used"], 2)
                self.assertEqual(final["tuning_trial_set_digest"], tuning["trial_set_digest"])
                self.assertEqual(final["tuning_selected"], tuning["selected"])

    def test_tune_steps_override_and_other_seed_are_recorded_as_deviations(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = m011.main(["--arm", "plain_lora", "--seed", "29", "--out", tmp, "--tune",
                            "--tune-budget", "1", "--tune-steps", "2", "--model", "tiny",
                            "--device", "cpu", "--max-steps", "4", "--eval-limit", "1"])
            self.assertEqual(rc, 0)
            tuning = json.loads((Path(tmp) / "tuning.json").read_text())
            self.assertEqual(tuning["tuning_steps"], 2)
            self.assertEqual(
                {d["parameter"] for d in tuning["deviations"]},
                {"tuning_seed", "tuning_steps", "tuning_trials_used"},
            )

    def test_all_seeds_driver_tunes_once_then_runs_every_pinned_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = m011.main(["--arm", "plain_lora", "--out", tmp, "--all-seeds", "--tune",
                            "--tune-budget", "1", "--model", "tiny", "--device", "cpu",
                            "--max-steps", "2", "--eval-limit", "1"])
            self.assertEqual(rc, 0)
            root = Path(tmp)
            tuning = json.loads((root / "tune" / "tuning.json").read_text())
            seeds = []
            for seed in m011.PINS["seeds"]:
                run = json.loads((root / f"seed{seed}" / "run.json").read_text())
                self.assertTrue((root / f"seed{seed}" / "metrics.json").exists())
                self.assertEqual(run["hyperparameters"]["learning_rate"], tuning["selected"]["learning_rate"])
                self.assertEqual(run["tuning_seed"], 17)
                seeds.append(run["seed"])
            self.assertEqual(seeds, [17, 29, 43, 71, 101])

    def test_all_seeds_requires_frozen_hparams_and_no_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                m011.main(["--arm", "mixlora", "--out", tmp, "--all-seeds", "--model", "tiny"])
            with self.assertRaises(SystemExit):
                m011.main(["--arm", "mixlora", "--out", tmp, "--all-seeds", "--seed", "1"])
            with self.assertRaises(SystemExit):
                m011.main(["--arm", "mixlora", "--out", tmp])  # no seed, no --tune

    def test_same_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ma = json.loads((self.run_cli("perft_r_emergent", Path(a)) / "metrics.json").read_text())
            mb = json.loads((self.run_cli("perft_r_emergent", Path(b)) / "metrics.json").read_text())
            self.assertEqual(ma["training"]["final_loss"], mb["training"]["final_loss"])
            self.assertEqual(ma["load_cv"], mb["load_cv"])


if __name__ == "__main__":
    unittest.main()
