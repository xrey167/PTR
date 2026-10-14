"""Audit regressions that need the `moe` extra (torch, trl, transformers, datasets, peft)."""

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptr_training.moe import config as moe_config
from ptr_training.moe.pipeline import stack_available

HAVE_STACK = stack_available(("torch", "trl", "transformers", "datasets", "peft"))
SMOKE = Path(__file__).resolve().parents[1] / "configs" / "stage-moe-smoke.toml"


class RlCheckpointingConfigTests(unittest.TestCase):
    """TRL's GRPOConfig defaults `gradient_checkpointing` to True. build_grpo_config did not pass
    it, so every RL arm ran with activation checkpointing while the README and GPU_HANDOVER said
    "RL does not use checkpointing" and metrics.json did not record it. It is now an explicit,
    recorded config key `moe.rl.gradient_checkpointing` (default true = the behaviour so far)."""

    def test_config_key_parses_with_default_true(self):
        cfg = moe_config.load(SMOKE)
        self.assertTrue(cfg.rl.gradient_checkpointing)
        raw = moe_config.tomllib.loads(SMOKE.read_text())
        raw["moe"]["rl"]["gradient_checkpointing"] = False
        self.assertFalse(moe_config.parse(raw, str(SMOKE)).rl.gradient_checkpointing)
        raw["moe"]["rl"]["gradient_checkpointing"] = "yes"
        with self.assertRaises(moe_config.ConfigError):
            moe_config.parse(raw, str(SMOKE))

    @unittest.skipUnless(HAVE_STACK, "moe extra not installed")
    def test_grpo_config_follows_the_config_key(self):
        from ptr_training.moe.rl import build_grpo_config

        cfg = moe_config.load(SMOKE)
        with tempfile.TemporaryDirectory() as tmp:
            for flag in (True, False):
                rl = dataclasses.replace(cfg.rl, gradient_checkpointing=flag)
                g = build_grpo_config("gspo", rl, seed=1, output_dir=tmp, bf16=False, device="cpu")
                self.assertIs(g.gradient_checkpointing, flag)


@unittest.skipUnless(HAVE_STACK, "moe extra not installed")
class PipelineProvenanceTests(unittest.TestCase):
    """metrics.json written through `run_config` (the direct entry) carried no git commit, and
    no run recorded the visible GPUs, the driver, CUDA_VISIBLE_DEVICES or whether the tracked
    tree was dirty."""

    @classmethod
    def setUpClass(cls):
        from ptr_training.moe import pipeline

        cls.pipeline = pipeline
        base = moe_config.load(SMOKE)
        cls.small = dataclasses.replace(
            base,
            sft=dataclasses.replace(base.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(base.rl, algorithms=()),
        )

    def test_metrics_carry_git_versions_and_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.pipeline.execute(self.small, Path(tmp))
        prov = data["provenance"]
        self.assertRegex(prov["git"]["commit"] or "", r"^[0-9a-f]{40}$")
        self.assertIn(prov["git"]["dirty"], (True, False))
        for key in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
            self.assertTrue(prov["versions"][key], key)
        self.assertEqual(prov["device"]["device"], "cpu")
        self.assertIn("CUDA_VISIBLE_DEVICES", prov["device"]["env"])
        self.assertEqual(data["run"]["seeds"]["tasks_sft"], self.small.seed)

    def test_multi_process_launch_is_refused(self):
        from ptr_training.moe import provenance

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaises(provenance.DistributedLaunchRefused):
                self.pipeline.execute(self.small, Path(tmp))


if __name__ == "__main__":
    unittest.main()
