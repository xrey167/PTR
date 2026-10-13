"""GPU memory estimator (`ptr_training.moe.memory`): exact parameter counts, fitted activation
coefficients re-measured on CPU, and the fit verdicts for the shipped configs."""

import contextlib
import dataclasses
import io
import json
import sys
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from ptr_training.moe import config as moe_config
from ptr_training.moe import memory
from ptr_training.moe.pipeline import stack_available

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "training" / "configs"
M011_CONFIG = ROOT / "experiments" / "model" / "M011-routed-peft" / "config.toml"
OLMOE_TOTAL = 6_919_161_856  # sum(numel) of OlmoeForCausalLM at the pinned shapes (meta device)
OLMOE_LORA_R16 = 121_634_816  # PEFT 0.21 count for r=16 on q/k/v/o + fused experts
HAVE = stack_available(("torch", "transformers", "peft"))


class CountTests(unittest.TestCase):
    def setUp(self):
        self.arch = memory.architecture_from_config(memory.PINNED_OLMOE_CONFIG)

    def test_olmoe_total_and_lora_counts(self):
        self.assertEqual(memory.parameter_counts(self.arch)["total"], OLMOE_TOTAL)
        trainable = memory.peft_lora_trainable(
            self.arch, 16, ("q_proj", "k_proj", "v_proj", "o_proj"),
            ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
        )
        self.assertEqual(trainable, OLMOE_LORA_R16)

    def test_stage_configs_use_the_config_lora(self):
        for name in ("stage-moe-olmoe.toml", "stage-moe-olmoe-sft-smoke-rtx3090.toml"):
            with self.subTest(name=name):
                s = memory.stage_scenario(moe_config.load(CONFIGS / name))
                self.assertEqual(s.trainable, OLMOE_LORA_R16)
                self.assertEqual(memory.parameter_counts(s.arch)["total"], OLMOE_TOTAL)

    def test_tiny_count_equals_the_config_formula(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        s = memory.stage_scenario(cfg)
        m = cfg.model
        self.assertEqual(
            memory.parameter_counts(s.arch)["total"],
            moe_config.tiny_parameter_count(m.layers, m.hidden, m.intermediate, m.experts),
        )
        self.assertTrue(s.full_finetune)

    @unittest.skipUnless(M011_CONFIG.is_file(), "M011 config.toml not found")
    def test_m011_arms_equal_the_pinned_trainable_params(self):
        pinned = tomllib.loads(M011_CONFIG.read_text())["pinned_thresholds"]["arm_lora"]
        for arm, (method, n, r) in memory.M011_ARMS.items():
            with self.subTest(arm=arm):
                self.assertEqual((n, r), (pinned[arm]["num_experts"], pinned[arm]["rank"]))
                self.assertEqual(
                    memory.m011_trainable(self.arch, method, n, r), pinned[arm]["trainable_params"]
                )
                self.assertEqual(memory.m011_scenario(arm).trainable, pinned[arm]["trainable_params"])


class FitTests(unittest.TestCase):
    def report(self, name, **rl_sft):
        cfg = moe_config.load(CONFIGS / name)
        if rl_sft:
            cfg = dataclasses.replace(cfg, sft=dataclasses.replace(cfg.sft, **rl_sft))
        return memory.estimate(memory.stage_scenario(cfg))

    def test_olmoe_sft_smoke_fits_one_3090(self):
        rep = memory.report_for_config(CONFIGS / "stage-moe-olmoe-sft-smoke-rtx3090.toml")
        self.assertEqual(rep["fit"]["card_gib"], 24.0)
        self.assertTrue(rep["fit"]["fits"])
        self.assertEqual(rep["phases"]["sft"]["peft_merged_experts_saved_gib"], 0.0)

    def test_stage_olmoe_without_checkpointing_does_not_fit_and_with_it_does(self):
        rep = self.report("stage-moe-olmoe.toml")
        self.assertFalse(rep["phases"]["sft"]["checkpointing"])
        # 16 layers x 768 MiB of merged expert weights kept for backward
        self.assertAlmostEqual(rep["phases"]["sft"]["peft_merged_experts_saved_gib"], 12.0)
        self.assertFalse(memory.fit(rep, 24.0)["fits"])
        fixed = self.report("stage-moe-olmoe.toml", gradient_checkpointing=True)
        self.assertTrue(memory.fit(fixed, 24.0)["fits"])

    def test_merged_expert_constant_matches_the_formula(self):
        a = memory.architecture_from_config(memory.PINNED_OLMOE_CONFIG)
        mat = memory.peft_expert_materialization(
            a, ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"), 2,
            checkpointing=False, grad=True,
        )
        self.assertEqual(mat["per_layer_merged"], memory.MEASURED_MERGED_EXPERT_BYTES_PER_LAYER_BF16)

    def test_m011_arms_fit_one_card_at_defaults(self):
        for arm in memory.M011_ARMS:
            with self.subTest(arm=arm):
                self.assertTrue(memory.fit(memory.estimate(memory.m011_scenario(arm)), 24.0)["fits"])

    def test_generation_counts_the_routed_adapter_workspace(self):
        """Codex P2: the generation/prefill estimate held only the base layer workspace; the
        PERFT-R / MixLoRA adapters also run (no grad) on every prefill token."""
        for arm in ("perft_r_emergent", "mixlora"):
            with self.subTest(arm=arm):
                routed = memory.m011_scenario(arm)
                self.assertGreater(routed.adapter_bytes_per_token_layer, 0)
                bare = dataclasses.replace(routed, adapter_bytes_per_token_layer=0)
                gen_r = memory.estimate(routed)["phases"]["eval_generate"]
                gen_b = memory.estimate(bare)["phases"]["eval_generate"]
                tokens = 16 * 1024  # eval batch x assumed prompt tokens
                self.assertAlmostEqual(
                    (gen_r["peak_gib"] - gen_b["peak_gib"]) * memory.GIB,
                    tokens * routed.adapter_bytes_per_token_layer, delta=1.0,
                )
                self.assertGreater(gen_r["adapter_workspace_gib"], 0)
        plain = memory.estimate(memory.m011_scenario("plain_lora"))["phases"]["eval_generate"]
        self.assertEqual(plain["adapter_workspace_gib"], 0)

    def test_dual_profile_is_two_24_gib_cards(self):
        self.assertEqual(memory.card_gib_for_profile("hardware/dual-rtx3090.toml", ROOT), 24.0)
        self.assertEqual(memory.card_gib_for_profile("hardware/rtx3090.toml", ROOT), 24.0)

    def test_cli_and_ptr_train_run_flag_print_json(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(memory.main(["--m011-arm", "mixlora"]), 0)
        self.assertEqual(json.loads(out.getvalue())["trainable_parameters"], 4423680)
        from ptr_training import run

        out = io.StringIO()
        argv = ["ptr-train-run", "--config", "training/configs/stage-moe-olmoe-sft-smoke-rtx3090.toml",
                "--memory-report"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as stop:
                run.main()
        self.assertEqual(stop.exception.code, 0)
        self.assertTrue(json.loads(out.getvalue())["fit"]["fits"])
        with mock.patch.object(sys, "argv", ["ptr-train-run", "--memory-report"]):
            with self.assertRaises(SystemExit) as stop:
                run.main()  # run-default.toml is not a trl-moe config
        self.assertIn("trl-moe", str(stop.exception.code))


@unittest.skipUnless(HAVE, "torch/transformers/peft not installed (moe extra)")
class MeasuredTests(unittest.TestCase):
    """Counts against instantiated models; activation coefficients against saved tensors."""

    def olmoe(self, cfg_dict, dtype=None, device=None):
        import torch
        from transformers import OlmoeConfig, OlmoeForCausalLM

        cfg = OlmoeConfig(**{k: v for k, v in cfg_dict.items() if k != "torch_dtype"})
        if device == "meta":
            with torch.device("meta"):
                return OlmoeForCausalLM(cfg)
        torch.manual_seed(0)
        return OlmoeForCausalLM(cfg).to(dtype or torch.float32)

    def test_counts_equal_instantiated_models(self):
        from peft import LoraConfig, get_peft_model

        m = self.olmoe(memory.PINNED_OLMOE_CONFIG, device="meta")
        self.assertEqual(sum(p.numel() for p in m.parameters()), OLMOE_TOTAL)
        peft = get_peft_model(m, LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            task_type="CAUSAL_LM",
        ))
        self.assertEqual(sum(p.numel() for p in peft.parameters() if p.requires_grad), OLMOE_LORA_R16)

    def saved_bytes(self, model, tokens, exclude_params=True, **kwargs):
        import torch

        params = {p.untyped_storage().data_ptr() for p in model.parameters()} if exclude_params else set()
        seen, ndims = {}, {}

        def pack(t):
            st = t.untyped_storage()
            if st.data_ptr() not in params:
                seen[st.data_ptr()] = max(seen.get(st.data_ptr(), 0), st.nbytes())
                ndims[st.data_ptr()] = max(ndims.get(st.data_ptr(), 0), t.dim())
            return t

        ids = torch.randint(0, model.config.vocab_size, (2, tokens // 2))
        model.train()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(input_ids=ids, labels=ids, use_cache=False, **kwargs)
        self.ndims = ndims
        return seen

    def test_activation_coefficients_track_saved_tensors(self):
        import torch

        for hidden, inter, experts, k, dtype in (
            (64, 32, 8, 2, torch.float32), (256, 128, 16, 4, torch.bfloat16),
        ):
            with self.subTest(hidden=hidden, dtype=dtype):
                cfg = {"vocab_size": 59, "hidden_size": hidden, "intermediate_size": inter,
                       "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
                       "num_experts": experts, "num_experts_per_tok": k, "max_position_embeddings": 512}
                measured = sum(self.saved_bytes(self.olmoe(cfg, dtype), 256, output_router_logits=True).values())
                a = memory.architecture_from_config(cfg)
                b = 2 if dtype == torch.bfloat16 else 4
                est = a.layers * memory.layer_working_bytes(a, 256, b) + 256 * a.vocab * (b + 8)
                self.assertGreater(est / measured, 0.85)
                self.assertLess(est / measured, 1.15)

    def test_peft_keeps_each_layers_merged_experts_without_checkpointing(self):
        import torch
        from peft import LoraConfig, get_peft_model

        # Shapes chosen so no activation has the byte size of a merged expert tensor; attention
        # LoRA as in the stage configs (it makes every layer's MoE input require grad).
        cfg = {"vocab_size": 59, "hidden_size": 48, "intermediate_size": 40, "num_hidden_layers": 3,
               "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": 6,
               "num_experts_per_tok": 2, "max_position_embeddings": 128}
        model = get_peft_model(self.olmoe(cfg, torch.bfloat16), LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            task_type="CAUSAL_LM",
        ))
        saved = self.saved_bytes(model, 20)
        sizes = [n for ptr, n in saved.items() if self.ndims[ptr] == 3]
        a = memory.architecture_from_config(cfg)
        gate_up = a.experts * 2 * a.intermediate * a.hidden * 2
        down = a.experts * a.hidden * a.intermediate * 2
        self.assertEqual(sizes.count(gate_up), a.layers)
        self.assertEqual(sizes.count(down), a.layers)
        mat = memory.peft_expert_materialization(
            a, ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"), 2, checkpointing=False, grad=True
        )
        self.assertEqual(mat["saved_for_backward"], a.layers * (gate_up + down))


if __name__ == "__main__":
    unittest.main()
