import copy
import tomllib
import unittest
from pathlib import Path

from ptr_training.moe import config as moe_config
from ptr_training.moe.config import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _raw(name="stage-moe-smoke.toml") -> dict:
    return tomllib.loads((CONFIGS / name).read_text(encoding="utf-8"))


class MoeConfigTests(unittest.TestCase):
    def test_smoke_config_parses(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual(cfg.model.kind, "tiny-random")
        self.assertEqual((cfg.model.layers, cfg.model.experts, cfg.model.top_k), (2, 8, 2))
        self.assertEqual(cfg.rl.algorithms, ("gspo", "sapo"))
        self.assertGreaterEqual(cfg.rl.mini_steps, 2)
        self.assertFalse(cfg.lora.enabled)

    def test_olmoe_config_parses(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-olmoe.toml")
        self.assertEqual(cfg.model.kind, "pretrained")
        self.assertEqual(cfg.model.hf_id, "allenai/OLMoE-1B-7B-0125")
        self.assertEqual(cfg.model.precision, "bf16")
        self.assertTrue(cfg.lora.enabled)
        self.assertEqual(cfg.lora.rank, 16)
        self.assertIn("q_proj", cfg.lora.target_modules)
        self.assertIn("mlp.experts.gate_up_proj", cfg.lora.target_parameters)

    def _mutate(self, fn):
        raw = copy.deepcopy(_raw())
        fn(raw)
        return raw

    def test_wrong_backend_rejected(self):
        raw = self._mutate(lambda r: r["training"].update(backend="dry-run"))
        with self.assertRaisesRegex(ConfigError, "trl-moe"):
            moe_config.parse(raw)

    def test_unknown_model_kind(self):
        raw = self._mutate(lambda r: r["model"].update(kind="mystery"))
        with self.assertRaisesRegex(ConfigError, "model.kind"):
            moe_config.parse(raw)

    def test_pretrained_requires_hf_id(self):
        def mutate(r):
            r["model"]["kind"] = "pretrained"

        with self.assertRaisesRegex(ConfigError, "hf_id"):
            moe_config.parse(self._mutate(mutate))

    def test_top_k_exceeding_experts(self):
        raw = self._mutate(lambda r: r["model"].update(top_k=9))
        with self.assertRaisesRegex(ConfigError, "top_k"):
            moe_config.parse(raw)

    def test_unknown_algorithm(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(algorithms=["gspo", "ppo"]))
        with self.assertRaisesRegex(ConfigError, "unknown entries"):
            moe_config.parse(raw)

    def test_empty_and_duplicate_algorithms(self):
        with self.assertRaisesRegex(ConfigError, "non-empty"):
            moe_config.parse(self._mutate(lambda r: r["moe"]["rl"].update(algorithms=[])))
        with self.assertRaisesRegex(ConfigError, "duplicates"):
            moe_config.parse(
                self._mutate(lambda r: r["moe"]["rl"].update(algorithms=["sapo", "sapo"]))
            )

    def test_num_generations_minimum(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(num_generations=1))
        with self.assertRaisesRegex(ConfigError, "num_generations"):
            moe_config.parse(raw)

    def test_type_errors_are_clear(self):
        raw = self._mutate(lambda r: r["moe"]["sft"].update(steps="many"))
        with self.assertRaisesRegex(ConfigError, r"moe\.sft\.steps must be int"):
            moe_config.parse(raw)

    def test_missing_key(self):
        def mutate(r):
            del r["moe"]["sft"]["lr"]

        with self.assertRaisesRegex(ConfigError, r"moe\.sft\.lr"):
            moe_config.parse(self._mutate(mutate))

    def test_lora_enabled_needs_targets(self):
        raw = self._mutate(lambda r: r["moe"].update(lora={"enabled": True}))
        with self.assertRaisesRegex(ConfigError, "no target_modules"):
            moe_config.parse(raw)

    def test_gspo_epsilon_defaults_and_values(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual((cfg.rl.gspo_epsilon, cfg.rl.gspo_epsilon_high), (3e-4, 4e-4))
        raw = self._mutate(lambda r: [r["moe"]["rl"].pop(k) for k in ("gspo_epsilon", "gspo_epsilon_high")])
        cfg = moe_config.parse(raw)
        self.assertEqual((cfg.rl.gspo_epsilon, cfg.rl.gspo_epsilon_high), (3e-4, 4e-4))

    def test_gspo_epsilon_validation_names_key(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilon=0.0))
        with self.assertRaisesRegex(ConfigError, "gspo_epsilon"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilon=0.1, gspo_epsilon_high=0.01))
        with self.assertRaisesRegex(ConfigError, "gspo_epsilon_high"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilon_high="x"))
        with self.assertRaisesRegex(ConfigError, "gspo_epsilon_high"):
            moe_config.parse(raw)

    def test_smoke_logs_kl_and_olmoe_mini_steps(self):
        self.assertGreater(moe_config.load(CONFIGS / "stage-moe-smoke.toml").rl.beta, 0)
        self.assertEqual(moe_config.load(CONFIGS / "stage-moe-olmoe.toml").rl.mini_steps, 4)

    def test_epsilon_high_below_epsilon(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(epsilon=0.3, epsilon_high=0.2))
        with self.assertRaisesRegex(ConfigError, "epsilon_high"):
            moe_config.parse(raw)


class MoeManifestCompatTests(unittest.TestCase):
    def test_manifest_builds_for_both_stage_configs(self):
        from ptr_training.run import ROOT, build_manifest

        for name in ("stage-moe-smoke.toml", "stage-moe-olmoe.toml"):
            manifest = build_manifest(ROOT / "training" / "configs" / name)
            self.assertEqual(manifest["config"]["training"]["backend"], "trl-moe")
            self.assertEqual(len(manifest["uv_lock_sha256"]), 64)


class MoeDispatchTests(unittest.TestCase):
    def test_missing_extra_gives_clear_message(self):
        from unittest import mock

        from ptr_training import run
        from ptr_training.moe import pipeline

        with mock.patch.object(
            pipeline, "require_extra", side_effect=pipeline.MissingExtraError("needs the moe extra")
        ):
            self.assertEqual(run._execute_trl_moe({}, Path("x.toml")), "needs the moe extra")

    def test_require_extra_names_the_extra_when_stack_absent(self):
        from unittest import mock

        from ptr_training.moe import pipeline

        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(pipeline.MissingExtraError, "moe"):
                pipeline.require_extra()


if __name__ == "__main__":
    unittest.main()
