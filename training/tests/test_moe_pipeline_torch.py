import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

HAVE_STACK = all(
    importlib.util.find_spec(m) for m in ("torch", "trl", "transformers", "datasets", "peft")
)
SMOKE = Path(__file__).resolve().parents[1] / "configs" / "stage-moe-smoke.toml"


@unittest.skipUnless(HAVE_STACK, "torch/trl/transformers/datasets/peft not installed (moe extra)")
class MoeTorchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ptr_training.moe import config as moe_config
        from ptr_training.moe import model as model_mod

        cls.cfg = moe_config.load(SMOKE)
        cls.model_mod = model_mod
        cls.model, cls.tok = model_mod.build(cls.cfg.model, cls.cfg.lora)

    def test_builds_olmoe_architecture(self):
        self.assertEqual(type(self.model).__name__, "OlmoeForCausalLM")
        c = self.model.config
        self.assertEqual((c.num_hidden_layers, c.num_experts, c.num_experts_per_tok), (2, 8, 2))

    def test_tokenizer_roundtrip(self):
        text = 'Q:12+7=\nA:19 {"a":3}'
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        self.assertNotIn(self.tok.unk_token_id, ids)
        self.assertEqual(self.tok.decode(ids), text)

    def test_router_logits_extraction(self):
        from ptr_training.moe import routing_metrics as rm

        snap = self.model_mod.routing_snapshot(self.model, self.tok, ["Q:1+2=\nA:3", "Q:4-1=\nA:3"], 100)
        self.assertEqual(len(snap), self.cfg.model.layers)
        for layer in snap:
            self.assertTrue(layer)
            self.assertEqual(len(layer[0]), self.cfg.model.experts)
        metrics = rm.snapshot_metrics(snap, self.cfg.model.top_k)
        self.assertTrue(all(math.isfinite(m["entropy"]) for m in metrics))

    def test_one_sft_step_changes_weights_and_is_finite(self):
        import torch

        from ptr_training.moe.config import SftConfig
        from ptr_training.moe.sft import run_sft
        from ptr_training.moe.tasks import generate_tasks

        model, tok = self.model_mod.build(self.cfg.model, self.cfg.lora)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = run_sft(model, tok, generate_tasks(16, seed=1), SftConfig(steps=1, lr=1e-2, batch=4, examples=16), 1)
        self.assertEqual(len(result["losses"]), 1)
        self.assertFalse(result["nan"])
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_one_step_gspo_run(self):
        import dataclasses

        from ptr_training.moe.rl import run_rl
        from ptr_training.moe.tasks import generate_tasks

        rl = dataclasses.replace(
            self.cfg.rl, steps=1, mini_steps=1, num_generations=2, prompts_per_step=1,
            max_completion_length=6,
        )
        model, tok = self.model_mod.build(self.cfg.model, self.cfg.lora)
        out = run_rl(model, tok, generate_tasks(8, seed=2), "gspo", rl, seed=2)
        self.assertEqual(out["grpo_overrides"]["importance_sampling_level"], "sequence")
        self.assertEqual(len(out["reward_per_step"]), 1)
        self.assertFalse(out["nan"])

    def test_lora_targets_of_olmoe_config_resolve_on_olmoe_architecture(self):
        from ptr_training.moe import config as moe_config

        lora = moe_config.load(SMOKE.with_name("stage-moe-olmoe.toml")).lora
        peft_model = self.model_mod.apply_lora(self.model_mod.build(self.cfg.model, self.cfg.lora)[0], lora)
        names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        self.assertTrue(any("self_attn.q_proj" in n for n in names))
        self.assertTrue(any("experts" in n for n in names))
        self.assertFalse(any("mlp.gate." in n for n in names))

    def test_grpo_config_maps_algorithms(self):
        from ptr_training.moe.rl import build_grpo_config

        with tempfile.TemporaryDirectory() as tmp:
            g = build_grpo_config("gspo", self.cfg.rl, seed=1, output_dir=tmp, bf16=False)
            self.assertEqual((g.importance_sampling_level, g.loss_type), ("sequence", "grpo"))
            self.assertEqual(g.steps_per_generation, self.cfg.rl.mini_steps)
            s = build_grpo_config("sapo", self.cfg.rl, seed=1, output_dir=tmp, bf16=False)
            self.assertEqual(s.loss_type, "sapo")
            self.assertEqual(s.sapo_temperature_neg, self.cfg.rl.sapo_temperature_neg)

    def test_pipeline_writes_metrics_without_binding(self):
        import dataclasses

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=3, batch=4, examples=16),
            rl=dataclasses.replace(
                self.cfg.rl, steps=1, mini_steps=1, num_generations=2, prompts_per_step=1,
                max_completion_length=6,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.execute(small, Path(tmp))
            path = pipeline.write_metrics(result, Path(tmp))
            data = json.loads(path.read_text())
        self.assertEqual(data["evidence_class"], "pipeline-smoke")
        self.assertEqual(data["note"], "mechanism check only, not evidence")
        self.assertFalse(data["nan_check"]["nonfinite_values_found"])
        self.assertEqual(set(data["rl"]), {"gspo", "sapo"})
        for stage in ("init", "after_sft", "after_rl_gspo", "after_rl_sapo"):
            self.assertEqual(len(data["routing"][stage]["per_layer"]), 2)
        self.assertIn("init_to_after_sft", data["routing_churn"])
        if not data["ptr_py_installed"]:
            self.assertEqual(data["gates"]["init_to_sft"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
