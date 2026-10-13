"""M011 entrypoint: tiny end-to-end run per arm on CPU (2 steps), tuning, hyperparameter reuse."""

import json
import os
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptr_training.moe import scorers
from ptr_training.moe.pipeline import stack_available


def setUpModule():
    # The end-to-end runs score HumanEval; without bubblewrap they use the recorded opt-in.
    if not scorers._bwrap_usable():
        os.environ.setdefault(scorers.UNSANDBOXED_ENV, "1")

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
            # no --seed: tuning defaults to the pilot seed 1001; 4 steps -> 1 tuning step
            rc = m011.main(["--arm", "mixlora", "--out", str(tune_dir), "--tune",
                            "--tune-budget", "2", "--model", "tiny", "--device", "cpu",
                            "--max-steps", "4", "--eval-limit", "2"])
            self.assertEqual(rc, 0)
            tuning = json.loads((tune_dir / "tuning.json").read_text())
            run = json.loads((tune_dir / "run.json").read_text())
            self.assertEqual((tuning["tuning_seed"], run["seed"], run["tuning_seed"]), (1001, 1001, 1001))
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
                self.assertEqual(final["tuning_seed"], 1001)
                self.assertEqual(final["tuning_steps"], 1)
                self.assertEqual(final["tuning_trials_used"], 2)
                self.assertEqual(final["tuning_trial_set_digest"], tuning["trial_set_digest"])
                self.assertEqual(final["tuning_selected"], tuning["selected"])
                self.assertEqual(final["tuning_arm"], "mixlora")

    def test_tune_steps_override_and_other_seed_are_recorded_as_deviations(self):
        with tempfile.TemporaryDirectory() as tmp:
            # another pilot seed (>= 1001) is a recorded deviation; a confirmatory seed is refused
            # (test_moe_audit.TuningSeedTests)
            rc = m011.main(["--arm", "plain_lora", "--seed", "1002", "--out", tmp, "--tune",
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
                self.assertEqual(run["tuning_seed"], 1001)
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

    def test_a_tuning_file_of_another_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.json"
            path.write_text(json.dumps({"arm": "mixlora", "tuning_seed": 1001,
                                        "selected": {"learning_rate": 1e-3, "batch_size": 4}}))
            with self.assertRaises(SystemExit):
                m011._load_tuning(path, "perft_r_emergent")
            self.assertEqual(m011._load_tuning(path, "mixlora")["arm"], "mixlora")
            with self.assertRaises(SystemExit):
                m011.main(["--arm", "perft_r_emergent", "--out", tmp, "--seed", "17",
                           "--hparams", str(path), "--model", "tiny"])

    def test_same_seed_is_reproducible(self):
        """Same seed on CPU: every training, accuracy and routing number is identical (timings,
        which are wall-clock, excluded); a different seed changes the training."""
        def strip(m):
            m = json.loads(json.dumps(m))
            m.pop("generation_timing")
            m.pop("latency_per_token")
            return m

        for arm in ("perft_r_emergent", "mixlora"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as a, \
                    tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
                ma = json.loads((self.run_cli(arm, Path(a)) / "metrics.json").read_text())
                mb = json.loads((self.run_cli(arm, Path(b)) / "metrics.json").read_text())
                mc = json.loads((self.run_cli(arm, Path(c), seed=4) / "metrics.json").read_text())
                self.assertEqual(strip(ma), strip(mb))
                self.assertNotEqual(ma["training"]["first_loss"], mc["training"]["first_loss"])

    def test_run_json_records_provenance_and_measured_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_cli("perft_r_prescribed", Path(tmp))
            run = json.loads((out / "run.json").read_text())
            metrics = json.loads((out / "metrics.json").read_text())
            prov = run["provenance"]
            self.assertEqual(prov["git"]["commit"], run["git_commit"])
            self.assertIn(prov["git"]["dirty"], (True, False))
            for key in ("torch", "transformers", "peft", "trl", "accelerate", "datasets"):
                self.assertIn(key, prov["versions"])
            self.assertEqual(prov["device"]["device"], "cpu")
            self.assertIn("CUDA_VISIBLE_DEVICES", prov["device"]["env"])
            self.assertTrue(prov["device"]["bit_reproducible"])
            self.assertEqual(run["seeds"]["run"], 3)
            self.assertEqual(run["seeds"]["tuning_selection_seed"], 20240607)
            self.assertIn(run["sandbox"]["filesystem"], ("bwrap-readonly", "unsandboxed-opt-in"))
            # tiny model: compared by config digest, not against the OLMoE topology pin
            self.assertTrue(run["model"]["tiny_config_digest"].startswith("sha256:"))
            self.assertEqual(run["model"]["topology_observed"]["top_k"], 2)
            self.assertEqual(run["trainable_params"], run["trainable_params_closed_form"])
            for report in run["corpora"].values():
                self.assertEqual(report["training_outside_training_partition_count"], 0)
            self.assertEqual(metrics["training"]["examples_drawn"], 2 * 4)
            self.assertNotIn("drawn_item_ids", metrics["training"])
            self.assertEqual(metrics["training_data_outside_training_partition_count"], 0)
            self.assertEqual(metrics["task_accuracy_per_domain"], metrics["per_domain_accuracy"])
            for domain in ("code", "math", "tool"):
                timing = metrics["generation_timing"][domain]
                self.assertGreater(timing["generated_tokens"], 0)
                self.assertAlmostEqual(
                    metrics["latency_per_token"][domain], timing["seconds"] / timing["generated_tokens"]
                )
            self.assertIn("expert_specialisation_index", metrics["not_produced"])
            self.assertEqual(set(metrics["validation_overlap_count_by_corpus"]),
                             {"gsm8k_train", "humaneval", "bfcl_simple"})

    def test_training_flops_are_measured_over_every_step(self):
        """Codex P1: training_flops was the first step's FLOPs x steps; padded batches of
        different lengths make that an extrapolation, not the measured total the M011 config
        requires. Every step is counted now; the extrapolation has its own key."""
        with tempfile.TemporaryDirectory() as tmp:
            # one example per step, so steps differ in sequence length
            out = self.run_cli("plain_lora", Path(tmp), "--max-steps", "6", "--batch-size", "1")
            metrics = json.loads((out / "metrics.json").read_text())
        tr = metrics["training"]
        self.assertEqual(len(tr["flops_per_step"]), 6)
        self.assertEqual(tr["measured_total_flops"], sum(tr["flops_per_step"]))
        self.assertEqual(metrics["training_flops"], tr["measured_total_flops"])
        self.assertNotEqual(len(set(tr["flops_per_step"])), 1, "fixture batches differ in length")
        self.assertEqual(tr["extrapolated_total_flops_from_first_step"], tr["flops_per_step"][0] * 6)
        self.assertNotEqual(metrics["training_flops"], tr["extrapolated_total_flops_from_first_step"])
        self.assertNotIn("estimated_total_flops", tr)

    def test_per_corpus_outputs_use_the_pinned_corpus_names(self):
        import tomllib

        cfg = Path(__file__).resolve().parents[2] / "experiments/model/M011-routed-peft/config.toml"
        pinned = tomllib.loads(cfg.read_text())["pinned_thresholds"]
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_cli("plain_lora", Path(tmp))
            metrics = json.loads((out / "metrics.json").read_text())
            run = json.loads((out / "run.json").read_text())
            rc = m011.main(["--arm", "plain_lora", "--out", str(Path(tmp) / "tune"), "--tune",
                            "--tune-budget", "1", "--model", "tiny", "--device", "cpu",
                            "--max-steps", "4", "--eval-limit", "1"])
            self.assertEqual(rc, 0)
            tuning = json.loads((Path(tmp) / "tune" / "tuning.json").read_text())
        want_d = set(pinned["validation_digest_per_corpus"])
        want_o = set(pinned["validation_overlap_counters_per_corpus"])
        self.assertEqual(set(metrics["validation_set_digest_by_corpus"]), want_d)
        self.assertEqual(set(metrics["validation_overlap_count_by_corpus"]), want_o)
        self.assertEqual(set(run["validation_set_digest_by_corpus"]), want_d)
        self.assertEqual(set(run["validation_overlap_count_by_corpus"]), want_o)
        self.assertEqual(set(tuning["validation_digests"]), want_d)

    def test_a_multi_process_launch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaises(SystemExit):
                m011.main(["--arm", "mixlora", "--seed", "3", "--out", tmp, *SMOKE])
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
