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
            max_completion_length=14,
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

    def _lora(self):
        from ptr_training.moe.config import LoraConfig

        return LoraConfig(
            enabled=True, rank=4, alpha=8,
            target_modules=("q_proj", "v_proj"),
            target_parameters=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
        )

    def test_same_seed_builds_identical_params(self):
        import torch

        def snapshot(lora):
            torch.manual_seed(999)  # global RNG state must not leak into the build
            model, _ = self.model_mod.build(self.cfg.model, lora)
            return {k: v.clone() for k, v in model.state_dict().items()}

        for lora in (self.cfg.lora, self._lora()):
            a = snapshot(lora)
            torch.manual_seed(5)
            torch.rand(7)
            b = snapshot(lora)
            self.assertEqual(a.keys(), b.keys())
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        lora_keys = [k for k in a if "lora_A" in k]
        self.assertTrue(lora_keys)

    def test_different_seed_changes_lora_init(self):
        import dataclasses

        import torch

        a, _ = self.model_mod.build(self.cfg.model, self._lora())
        other = dataclasses.replace(self.cfg.model, seed=self.cfg.model.seed + 1)
        b, _ = self.model_mod.build(other, self._lora())
        sa, sb = a.state_dict(), b.state_dict()
        key = next(k for k in sa if "lora_A" in k)
        self.assertFalse(torch.equal(sa[key], sb[key]))

    def test_bf16_precision_applied_to_tiny_model(self):
        import dataclasses

        import torch

        cfg = dataclasses.replace(self.cfg.model, precision="bf16")
        model, _ = self.model_mod.build(cfg, self.cfg.lora)
        self.assertEqual({p.dtype for p in model.parameters()}, {torch.bfloat16})
        self.assertEqual(self.model_mod.param_dtypes(model), ["bfloat16"])
        self.assertEqual(self.model_mod.param_dtypes(self.model), ["float32"])

    def test_overlong_prompt_rejected(self):
        import dataclasses

        from ptr_training.moe.rl import check_prompt_lengths, run_rl
        from ptr_training.moe.tasks import generate_tasks

        tasks = generate_tasks(8, seed=2)
        longest = check_prompt_lengths(self.tok, tasks, self.cfg.rl.max_prompt_length)
        self.assertLessEqual(longest, self.cfg.rl.max_prompt_length)
        tight = dataclasses.replace(self.cfg.rl, max_prompt_length=3)
        with self.assertRaisesRegex(ValueError, "max_prompt_length"):
            check_prompt_lengths(self.tok, tasks, 3)
        with self.assertRaisesRegex(ValueError, "max_prompt_length"):
            run_rl(self.model, self.tok, tasks, "gspo", tight, seed=1)

    def test_peft_reference_assertion(self):
        from ptr_training.moe.rl import _assert_peft_reference

        peft_model, _ = self.model_mod.build(self.cfg.model, self._lora())
        with self.assertRaisesRegex(RuntimeError, "no 'ref' adapter"):
            _assert_peft_reference(peft_model, self.cfg.rl)  # beta = 0.04 in the smoke config
        _assert_peft_reference(peft_model, __import__("dataclasses").replace(self.cfg.rl, beta=0.0))

    def test_lora_reference_is_sft_adapter_not_base(self):
        import dataclasses

        import torch

        from ptr_training.moe import rl as rl_mod
        from ptr_training.moe.tasks import generate_tasks

        model, tok = self.model_mod.build(self.cfg.model, self._lora())
        with torch.no_grad():  # stand-in for SFT: make the adapter differ from its zero init
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.add_(0.05)
        cfg = dataclasses.replace(
            self.cfg.rl, steps=1, mini_steps=1, num_generations=2, prompts_per_step=1,
            max_completion_length=14, beta=0.04,
        )
        seen = {}
        original = rl_mod._assert_peft_reference

        def recording(m, c):
            original(m, c)
            params = dict(m.named_parameters())
            pairs = [(n, params[n.replace(".default.", ".ref.")]) for n in params if ".default." in n]
            seen["pairs"] = len(pairs)
            seen["ref_equals_sft"] = all(torch.equal(params[n].detach(), r.detach()) for n, r in pairs)
            seen["ref_nonzero"] = any(r.detach().abs().sum() > 0 for n, r in pairs if "lora_B" in n)

        rl_mod._assert_peft_reference = recording
        try:
            for arm in ("gspo", "sapo"):  # second arm exercises stale-"ref" cleanup
                seen.clear()
                out = rl_mod.run_rl(model, tok, generate_tasks(8, seed=2), arm, cfg, seed=2)
                self.assertGreater(seen["pairs"], 0)
                self.assertTrue(seen["ref_equals_sft"])
                self.assertTrue(seen["ref_nonzero"])  # reference carries the SFT state, not zero/base
                self.assertTrue(out["kl_per_step"])
                self.assertNotIn("ref", model.peft_config)
        finally:
            rl_mod._assert_peft_reference = original

    def test_model_class_is_base_architecture_for_peft(self):
        from ptr_training.moe import pipeline

        import dataclasses

        small = dataclasses.replace(
            self.cfg,
            lora=self._lora(),
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = pipeline.execute(small, Path(tmp))
        self.assertTrue(data["model"]["lora"])
        self.assertEqual(data["model"]["class"], "OlmoeForCausalLM")
        self.assertEqual(data["model"]["architecture"]["num_experts"], 8)

    def test_sft_collate_rejects_empty_label_example(self):
        from ptr_training.moe.sft import collate
        from ptr_training.moe.tasks import generate_tasks

        t = generate_tasks(2, seed=1)
        with self.assertRaisesRegex(ValueError, "empty-label"):
            collate(self.tok, t, max_length=3)
        collate(self.tok, t, max_length=self.cfg.sft.max_length)

    def test_completion_length_check_with_tokenizer(self):
        from ptr_training.moe.rl import check_completion_length
        from ptr_training.moe.tasks import generate_tasks

        tasks = generate_tasks(40, seed=3)
        self.assertLessEqual(check_completion_length(self.tok, tasks, 14), 14)
        with self.assertRaisesRegex(ValueError, "max_completion_length"):
            check_completion_length(self.tok, tasks, 5)

    def test_empty_algorithms_skips_rl(self):
        import dataclasses

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = pipeline.execute(small, Path(tmp))
        self.assertEqual(data["rl"], {})
        self.assertEqual(set(data["policy_eval"]), {"init", "after_sft"})
        self.assertFalse(data["nan_check"]["nonfinite_values_found"])

    def test_pipeline_writes_metrics_without_binding(self):
        import dataclasses

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=3, batch=4, examples=16),
            rl=dataclasses.replace(
                self.cfg.rl, steps=1, mini_steps=1, num_generations=2, prompts_per_step=1,
                max_completion_length=14,
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
        self.assertEqual(data["tasks"]["eval_overlap"], 0)
        self.assertEqual(data["model"]["param_dtypes"], ["float32"])
        self.assertEqual(data["model"]["precision"], "fp32")
        for stage in ("init", "after_sft", "after_rl_gspo", "after_rl_sapo"):
            self.assertEqual(len(data["routing"][stage]["per_layer"]), 2)
        self.assertIn("init_to_after_sft", data["routing_churn"])
        if not data["ptr_py_installed"]:
            self.assertEqual(data["gates"]["init_to_sft"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
