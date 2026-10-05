import json
import math
import tempfile
import unittest
from pathlib import Path

from ptr_training.moe.pipeline import stack_available

# Distribution metadata, not find_spec: the repo's top-level `datasets/` namespace directory
# would make find_spec("datasets") succeed without the library installed.
HAVE_STACK = stack_available(("torch", "trl", "transformers", "datasets", "peft"))
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
                if arm == "sapo":
                    self.assertIsNone(out["clip_ratio_per_step"])
                    self.assertIn("sapo", out["clip_ratio_unavailable_reason"])
                else:
                    self.assertTrue(out["clip_ratio_per_step"])
                    self.assertNotIn("clip_ratio_unavailable_reason", out)
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

    def test_nonfinite_router_logits_reflected_in_nan_check(self):
        import dataclasses
        from unittest import mock

        from ptr_training.moe import pipeline

        real = self.model_mod.routing_snapshot_by_text
        calls = []

        def faulty(model, tok, texts, max_tokens):
            calls.append(1)
            snap = real(model, tok, texts, max_tokens)
            if len(calls) == 2:  # after_sft
                snap[0][0][0][0] = float("nan")  # [text][layer][token][expert]
            return snap

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            self.model_mod, "routing_snapshot_by_text", faulty
        ):
            data = pipeline.execute(small, Path(tmp))
        self.assertTrue(data["nan_check"]["nonfinite_values_found"])
        self.assertIn("after_sft", data["nan_check"]["router_errors"])
        self.assertIn("non-finite router logits in layer 0", data["routing"]["after_sft"]["error"])
        self.assertIsNone(data["routing"]["after_sft"]["per_layer"])
        self.assertIsNone(data["routing_churn"]["init_to_after_sft"])
        self.assertIsNotNone(data["routing"]["init"]["per_layer"])

    def test_routing_metrics_are_per_probe_text(self):
        texts = ["Q:1+2=\nA:3", "J:a=4,b=2\nA:{\"a\":4,\"b\":2}", "Q:9-9=\nA:0"]
        by_text = self.model_mod.routing_snapshot_by_text(self.model, self.tok, texts, 5)
        self.assertEqual(len(by_text), 3)
        for text in by_text:
            self.assertEqual(len(text), self.cfg.model.layers)
            self.assertTrue(all(0 < len(layer) <= 5 for layer in text))
        # the longest text is capped at 5 but the others are NOT crowded out by it
        lengths = [len(t[0]) for t in by_text]
        self.assertEqual(lengths[1], 5)
        self.assertEqual(lengths[0], min(5, len(texts[0])))
        flat = self.model_mod.routing_snapshot(self.model, self.tok, texts, 5)
        self.assertEqual(len(flat[0]), sum(lengths))

    def test_pipeline_records_per_text_routing_and_provenance_fields(self):
        import dataclasses

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            probe_tokens=6,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = pipeline.execute(small, Path(tmp))
        init = data["routing"]["init"]
        self.assertEqual(len(init["per_text"]), 8)
        self.assertEqual(len(init["per_text"][0]), self.cfg.model.layers)
        self.assertTrue(all(0 < n <= 6 for n in init["tokens_per_text"]))
        self.assertIn("lowest expert index", init["topk_tie_break"])
        self.assertEqual(data["data_source"], "synthetic-tasks")
        self.assertIn("not used", data["data_note"])
        self.assertIn("policy_ratio_variance_per_token", data["not_produced"])
        self.assertIsInstance(data["gates_evaluated"], bool)
        if data["gates_evaluated"]:
            self.assertIsNone(data["gates_not_evaluated_reason"])
        else:
            self.assertTrue(data["gates_not_evaluated_reason"])

    def test_gates_not_evaluated_is_prominent_without_binding(self):
        import dataclasses
        import sys
        from unittest import mock

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"ptr_py": None}):
            data = pipeline.execute(small, Path(tmp))
        self.assertFalse(data["gates_evaluated"])
        self.assertIn("binding not installed", data["gates_not_evaluated_reason"])

    def test_sft_check_examples_rejects_eos_truncation(self):
        from ptr_training.moe.sft import check_examples
        from ptr_training.moe import tasks

        every = [t for kind in tasks.KINDS for t in tasks._enumerate(kind, 9)]
        check_examples(self.tok, every, 26)  # exactly longest prompt + answer + EOS
        with self.assertRaisesRegex(ValueError, "truncate"):
            check_examples(self.tok, every, 25)

    def test_resolve_revision_verification(self):
        import dataclasses
        import sys
        import types
        from unittest import mock

        sha = "0123456789abcdef0123456789abcdef01234567"
        cfg = dataclasses.replace(self.cfg.model, kind="pretrained", hf_id="org/m", revision=sha)

        def fake_hub(returned=None, error=None):
            module = types.ModuleType("huggingface_hub")

            class Api:
                def model_info(self, repo, revision=None):
                    if error:
                        raise error
                    return types.SimpleNamespace(sha=returned)

            module.HfApi = Api
            return mock.patch.dict(sys.modules, {"huggingface_hub": module})

        with fake_hub(returned=sha):
            info = self.model_mod.resolve_revision(cfg)
        self.assertTrue(info["revision_verified"])
        self.assertEqual(info["resolved_sha"], sha)
        with fake_hub(returned="f" * 40):
            with self.assertRaisesRegex(ValueError, "does not identify"):
                self.model_mod.resolve_revision(cfg)
        with fake_hub(error=OSError("offline")):
            info = self.model_mod.resolve_revision(cfg)
        self.assertFalse(info["revision_verified"])
        self.assertIsNone(info["resolved_sha"])
        self.assertIn("offline", info["reason"])

    def test_trl_grpo_config_exposes_the_fields_we_pass(self):
        import dataclasses

        from trl import GRPOConfig

        names = {f.name for f in dataclasses.fields(GRPOConfig)}
        for field in ("importance_sampling_level", "loss_type", "sapo_temperature_neg",
                      "sapo_temperature_pos", "steps_per_generation", "num_generations",
                      "model_init_kwargs", "max_completion_length", "beta", "epsilon_high"):
            self.assertIn(field, names, field)
        self.assertNotIn("max_prompt_length", names)  # why rl.check_prompt_lengths exists

    def test_gspo_and_sapo_first_step_gradients_match_when_on_policy(self):
        """mini_steps = 1 => ratio == 1 at every update. Per TRL 1.14.1 `_compute_loss`:
        SAPO's gate has slope 1 at ratio 1 (4/tau * sigmoid'(0) * tau) and GSPO's sequence ratio
        has the same per-token gradient, with identical per-sequence token-mean aggregation, so
        the first-step gradient is the same; the *logged loss value* differs by construction
        (SAPO's gate value at ratio 1 is 2/tau, with tau differing for A > 0 and A < 0)."""
        import dataclasses
        from unittest import mock

        import torch

        from ptr_training.moe import rl
        from ptr_training.moe.tasks import generate_tasks

        def reward_fn(min_level, **kw):
            def reward_batch(completions, **ignored):
                return [float(sum(map(ord, c)) % 3) for c in completions]  # non-degenerate groups

            return reward_batch

        cfg = dataclasses.replace(
            self.cfg.rl, steps=1, mini_steps=1, num_generations=4, prompts_per_step=2, beta=0.04
        )
        out = {}
        with mock.patch.object(rl, "make_reward_fn", reward_fn):
            for algo in ("gspo", "sapo"):
                model, tok = self.model_mod.build(self.cfg.model, self.cfg.lora)
                torch.manual_seed(5)
                out[algo] = rl.run_rl(model, tok, generate_tasks(16, seed=3), algo, cfg, seed=5)
        g, s = out["gspo"], out["sapo"]
        self.assertEqual(g["reward_per_step"], s["reward_per_step"])  # same rollouts
        self.assertGreater(g["grad_norm_per_step"][0], 0.1)  # real signal, not a null batch
        self.assertAlmostEqual(g["grad_norm_per_step"][0], s["grad_norm_per_step"][0], delta=1e-4)
        self.assertNotAlmostEqual(g["loss_per_step"][0], s["loss_per_step"][0], places=3)

    def test_oversize_gate_snapshot_is_skipped_without_conversion(self):
        import dataclasses
        import sys
        import types
        from unittest import mock

        from ptr_training.moe import gates, pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        stub = types.ModuleType("ptr_py")
        for name in ("aimer_score", "rank_experts_for_pruning", "subspace_overlap", "chance_overlap"):
            setattr(stub, name, lambda *a: 0.0)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules, {"ptr_py": stub}
        ), mock.patch.object(gates, "MAX_TOTAL_GATE_ELEMENTS", 10), mock.patch.object(
            gates, "_flat", side_effect=AssertionError("converted")
        ) as spy:
            data = pipeline.execute(small, Path(tmp))
        spy.assert_not_called()
        self.assertEqual(data["gates"]["init_to_sft"]["status"], "skipped")
        self.assertIn("snapshot too large", data["gates"]["init_to_sft"]["reason"])
        self.assertFalse(data["gates_evaluated"])

    def test_grpo_config_disables_dropout(self):
        from ptr_training.moe.rl import build_grpo_config

        with tempfile.TemporaryDirectory() as tmp:
            for algo in ("gspo", "sapo"):
                g = build_grpo_config(algo, self.cfg.rl, seed=1, output_dir=tmp, bf16=False)
                self.assertTrue(g.disable_dropout)

    def test_sft_state_snapshot_only_for_multiple_arms(self):
        import dataclasses
        from unittest import mock

        from ptr_training.moe import pipeline, rl

        def small(algos):
            return dataclasses.replace(
                self.cfg,
                sft=dataclasses.replace(self.cfg.sft, steps=2, batch=2, examples=8),
                rl=dataclasses.replace(
                    self.cfg.rl, algorithms=algos, steps=1, mini_steps=1, num_generations=2,
                    prompts_per_step=1, beta=0.0,
                ),
            )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            self.model_mod, "trainable_state", side_effect=AssertionError("snapshot")
        ) as snap, mock.patch.object(
            self.model_mod, "load_trainable_state", side_effect=AssertionError("restore")
        ) as restore:
            pipeline.execute(small(("gspo",)), Path(tmp))  # one arm: no snapshot, no restore
        snap.assert_not_called()
        restore.assert_not_called()

        starts = []
        real_rl = rl.run_rl

        def recording(model, *a, **k):
            starts.append({n: p.detach().clone() for n, p in model.named_parameters()})
            return real_rl(model, *a, **k)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rl, "run_rl", recording):
            pipeline.execute(small(("gspo", "sapo")), Path(tmp))
        self.assertEqual(len(starts), 2)
        import torch

        # arm 2 starts from the SFT state (== arm 1's start), not from arm 1's trained weights
        self.assertTrue(all(torch.equal(starts[0][n], starts[1][n]) for n in starts[0]))

    def test_tiny_vocab_constant_and_parameter_formula_match_the_real_model(self):
        from ptr_training.moe import config as moe_config

        self.assertEqual(len(self.tok), moe_config.TINY_VOCAB_SIZE)
        m = self.cfg.model
        actual = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(
            actual, moe_config.tiny_parameter_count(m.layers, m.hidden, m.intermediate, m.experts)
        )

    def test_model_context_window_check(self):
        import types

        from ptr_training.moe import config as moe_config
        from ptr_training.moe import pipeline
        from ptr_training.moe.tasks import generate_tasks

        sft, ev = generate_tasks(16, seed=1), generate_tasks(8, seed=2)
        ok = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=64))
        info = pipeline.check_model_context(self.cfg, ok, self.tok, sft, [], ev)
        self.assertLessEqual(info["needed_positions"], 64)
        self.assertEqual(info["model_max_position_embeddings"], 64)
        self.assertLessEqual(info["sft_longest_tokens"], 26)
        short = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=20))
        with self.assertRaisesRegex(moe_config.ConfigError, r"needs \d+ positions.*is 20"):
            pipeline.check_model_context(self.cfg, short, self.tok, sft, [], ev)
        unknown = types.SimpleNamespace(config=types.SimpleNamespace())
        self.assertIsNone(
            pipeline.check_model_context(self.cfg, unknown, self.tok, sft, [], ev)[
                "model_max_position_embeddings"
            ]
        )

    def test_pipeline_records_context_check_and_no_pool_for_sft_only(self):
        import dataclasses

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = pipeline.execute(small, Path(tmp))
        self.assertEqual(data["context_check"]["model_max_position_embeddings"], 64)
        self.assertEqual(data["tasks"]["rl_pool_size"], 0)
        self.assertFalse(data["tasks"]["rl_pool_built"])

    def test_metrics_record_model_and_run_seed_separately(self):
        import dataclasses

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            seed=5,
            model=dataclasses.replace(self.cfg.model, seed=11),
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = pipeline.execute(small, Path(tmp))
        self.assertEqual(data["run"]["seed"], 5)
        self.assertEqual(data["model"]["seed"], 11)

    def test_bf16_full_finetune_reference_model_is_bf16(self):
        import dataclasses
        from unittest import mock

        import torch
        from trl import GRPOTrainer

        from ptr_training.moe.rl import run_rl
        from ptr_training.moe.tasks import generate_tasks

        cfg = dataclasses.replace(
            self.cfg.rl, steps=1, mini_steps=1, num_generations=2, prompts_per_step=1, beta=0.04,
        )
        mcfg = dataclasses.replace(self.cfg.model, precision="bf16")
        model, tok = self.model_mod.build(mcfg, self.cfg.lora)
        seen = {}
        real_train = GRPOTrainer.train

        def spy(trainer, *a, **k):
            seen["policy"] = next(trainer.model.parameters()).dtype
            seen["ref"] = {p.dtype for p in trainer.ref_model.parameters()}
            return real_train(trainer, *a, **k)

        with mock.patch.object(GRPOTrainer, "train", spy):
            run_rl(model, tok, generate_tasks(8, seed=2), "gspo", cfg, seed=2)
        self.assertEqual(seen["policy"], torch.bfloat16)
        self.assertEqual(seen["ref"], {torch.bfloat16})

    def test_missing_kl_is_null_with_reason(self):
        import dataclasses

        from ptr_training.moe.rl import run_rl
        from ptr_training.moe.tasks import generate_tasks

        rl = dataclasses.replace(
            self.cfg.rl, steps=1, mini_steps=1, num_generations=2, prompts_per_step=1,
            beta=0.0,
        )
        model, tok = self.model_mod.build(self.cfg.model, self.cfg.lora)
        out = run_rl(model, tok, generate_tasks(8, seed=2), "gspo", rl, seed=2)
        self.assertIsNone(out["kl_per_step"])
        self.assertIn("beta = 0", out["kl_unavailable_reason"])

    def test_sft_only_run_never_snapshots_trainable_state(self):
        import dataclasses
        from unittest import mock

        from ptr_training.moe import pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            self.model_mod, "trainable_state", side_effect=AssertionError("called")
        ) as spy:
            pipeline.execute(small, Path(tmp))
        spy.assert_not_called()

    def test_broken_binding_reported_as_error_and_flag(self):
        import dataclasses
        from unittest import mock

        from ptr_training.moe import gates, pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(self.cfg.rl, algorithms=()),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            gates.importlib, "import_module", side_effect=OSError("bad ELF")
        ):
            data = pipeline.execute(small, Path(tmp))
        self.assertEqual(data["ptr_py_installed"], "broken")
        self.assertEqual(data["gates"]["init_to_sft"]["status"], "error")
        self.assertIn("bad ELF", data["gates"]["init_to_sft"]["error"])
        self.assertEqual(data["backend_selection"]["status"], "unselected")

    def test_no_flattening_when_binding_absent(self):
        import dataclasses
        import sys
        from unittest import mock

        from ptr_training.moe import gates, pipeline

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(
                self.cfg.rl, algorithms=("gspo",), steps=1, mini_steps=1, num_generations=2,
                prompts_per_step=1,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules, {"ptr_py": None}
        ), mock.patch.object(gates, "_flat", side_effect=AssertionError("flattened")) as spy:
            data = pipeline.execute(small, Path(tmp))
        spy.assert_not_called()
        self.assertEqual(data["gates"]["init_to_sft"]["status"], "skipped")
        self.assertEqual(data["gates"]["sft_to_rl_gspo"]["status"], "skipped")

    def test_oversize_experts_skipped_before_conversion(self):
        from unittest import mock

        from ptr_training.moe import gates

        model, _ = self.model_mod.build(self.cfg.model, self.cfg.lora)
        with mock.patch.object(gates, "_flat", side_effect=AssertionError("flattened")) as spy:
            out = gates.flatten_experts(model, max_elements=10)
        spy.assert_not_called()
        self.assertEqual(out, [None] * self.cfg.model.layers)
        full = gates.flatten_experts(model)
        self.assertTrue(all(layer is not None for layer in full))
        self.assertEqual(gates.expert_deltas(out, full), [None] * self.cfg.model.layers)

    def test_nonfinite_parameters_detected_after_sft_and_each_rl_arm(self):
        import dataclasses
        from unittest import mock

        import torch

        from ptr_training.moe import pipeline, rl, sft

        def poison(model):
            name, p = next(iter(model.named_parameters()))
            with torch.no_grad():
                p.view(-1)[0] = float("nan")
            return name

        small = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(
                self.cfg.rl, algorithms=("gspo",), steps=1, mini_steps=1, num_generations=2,
                prompts_per_step=1,
            ),
        )
        real_sft, real_rl = sft.run_sft, rl.run_rl
        names = {}

        def bad_sft(model, *a, **k):
            out = real_sft(model, *a, **k)
            names["sft"] = poison(model)
            return out

        def bad_rl(model, *a, **k):
            out = real_rl(model, *a, **k)
            names["rl"] = poison(model)
            return out

        # a clean run records zero everywhere
        with tempfile.TemporaryDirectory() as tmp:
            clean = pipeline.execute(small, Path(tmp))
        checks = clean["nan_check"]["nonfinite_parameters"]
        self.assertEqual({k: v["count"] for k, v in checks.items()}, {"after_sft": 0, "after_rl_gspo": 0})
        self.assertFalse(clean["nan_check"]["nonfinite_values_found"])

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sft, "run_sft", bad_sft):
            data = pipeline.execute(small, Path(tmp))
        self.assertTrue(data["nan_check"]["nonfinite_values_found"])
        self.assertIn(names["sft"], data["nan_check"]["nonfinite_parameters"]["after_sft"]["names"])

        # NaN only in the RL arm's weights (SFT state restored afterwards) is still caught
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rl, "run_rl", bad_rl):
            data = pipeline.execute(small, Path(tmp))
        self.assertTrue(data["nan_check"]["nonfinite_values_found"])
        checks = data["nan_check"]["nonfinite_parameters"]
        self.assertEqual(checks["after_sft"]["count"], 0)
        self.assertGreaterEqual(checks["after_rl_gspo"]["count"], 1)

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
