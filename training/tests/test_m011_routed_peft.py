"""M011 arm topologies: shapes, parameter counts, freezing, routing (torch + transformers + peft)."""

import os
import tomllib
import unittest
from pathlib import Path

from ptr_training.moe.pipeline import stack_available

HAVE = stack_available(("torch", "transformers", "peft"))

if HAVE:
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    from ptr_training.moe import routed_peft as rp
    from ptr_training.moe import routing_metrics as rm


def tiny_model(seed=0, experts=8, top_k=2):
    torch.manual_seed(seed)
    cfg = OlmoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=experts,
        num_experts_per_tok=top_k, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return OlmoeForCausalLM(cfg).eval()


def real_shape_model():
    cfg = OlmoeConfig(
        vocab_size=50304, hidden_size=2048, intermediate_size=1024, num_hidden_layers=16,
        num_attention_heads=16, num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
        norm_topk_prob=False,
    )
    with torch.device("meta"):  # shapes only; nothing is allocated or downloaded
        return OlmoeForCausalLM(cfg)


@unittest.skipUnless(HAVE, "torch/transformers/peft not installed (moe extra)")
class ParameterCountTests(unittest.TestCase):
    def test_real_olmoe_counts_equal_pinned_arithmetic(self):
        expected = {
            "perft_r_emergent": 4456448,
            "perft_r_prescribed": 4456448,
            "plain_lora": 4456448,
            "mixlora": 4423680,
        }
        for arm, want in expected.items():
            with self.subTest(arm=arm):
                inst = rp.install_arm(real_shape_model(), arm)
                self.assertEqual(rp.count_trainable_parameters(inst.model), want)
                self.assertEqual(rp.expected_pinned_params(inst.spec, 2048, 1024, 16), want)
                self.assertEqual(inst.spec.pinned_trainable_params, want)
                self.assertTrue(rp.within_tolerance(want))
                self.assertEqual(rp.non_adapter_trainable(inst.model), [])

    def test_real_olmoe_tensor_shapes(self):
        shapes = lambda arm: sorted(  # noqa: E731
            set(rp.trainable_inventory(rp.install_arm(real_shape_model(), arm).model).values())
        )
        perft = shapes("perft_r_emergent")
        self.assertEqual(perft, sorted([(8, 2048, 8), (8, 8, 2048), (2048, 8)]))
        self.assertEqual(
            shapes("mixlora"),
            sorted([(9, 2048, 4), (9, 4, 2048), (9, 1024, 4), (2048, 9)]),
        )
        self.assertEqual(shapes("plain_lora"), [(17, 2048), (2048, 17)])

    def test_tiny_counts_follow_the_closed_form(self):
        for arm in rp.ARMS:
            with self.subTest(arm=arm):
                inst = rp.install_arm(tiny_model(), arm)
                self.assertEqual(
                    rp.count_trainable_parameters(inst.model),
                    rp.expected_pinned_params(inst.spec, 32, 16, 2),
                )

    def test_arm_table_matches_config_when_available(self):
        path = os.environ.get("PTR_M011_CONFIG")
        if not path or not Path(path).is_file():
            self.skipTest("PTR_M011_CONFIG not set")
        cfg = tomllib.loads(Path(path).read_text())["pinned_thresholds"]
        for arm, spec in rp.ARMS.items():
            pinned = cfg["arm_lora"][arm]
            self.assertEqual(
                (spec.num_experts, spec.router_top_k, spec.rank, spec.alpha, spec.pinned_trainable_params),
                (pinned["num_experts"], pinned["router_top_k"], pinned["rank"], pinned["alpha"], pinned["trainable_params"]),
            )
        self.assertEqual(rp.PRESCRIBED_ASSIGNMENT, cfg["prescribed_expert_domain_assignment"])
        self.assertEqual(
            rp.PRESCRIBED_ASSIGNMENT_DIGEST, cfg["expert_domain_assignment_expected_digest"]
        )

    def test_assignment_digest_is_the_pinned_one(self):
        self.assertEqual(
            rp.assignment_digest(rp.PRESCRIBED_ASSIGNMENT), rp.PRESCRIBED_ASSIGNMENT_DIGEST
        )
        # key order does not matter, content does
        shuffled = {"tool": [6, 7], "code": [0, 1, 2], "math": [3, 4, 5]}
        self.assertEqual(rp.assignment_digest(shuffled), rp.PRESCRIBED_ASSIGNMENT_DIGEST)
        self.assertNotEqual(
            rp.assignment_digest({"code": [0, 1], "math": [2, 3, 4, 5], "tool": [6, 7]}),
            rp.PRESCRIBED_ASSIGNMENT_DIGEST,
        )


@unittest.skipUnless(HAVE, "torch/transformers/peft not installed (moe extra)")
class FreezingTests(unittest.TestCase):
    def test_only_adapter_and_router_parameters_require_grad(self):
        for arm in rp.ARMS:
            with self.subTest(arm=arm):
                model = tiny_model()
                base_names = {n for n, _ in model.named_parameters()}
                inst = rp.install_arm(model, arm)
                trainable = [n for n, p in inst.model.named_parameters() if p.requires_grad]
                self.assertTrue(trainable)
                self.assertEqual(rp.non_adapter_trainable(inst.model), [])
                # every original tensor is still frozen (names may gain wrapper prefixes)
                for n, p in inst.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    self.assertFalse(
                        any(n.endswith(b) for b in base_names if "experts" in b or "gate.weight" in b),
                        n,
                    )

    def test_gradients_reach_adapters_not_base(self):
        for arm in rp.ARMS:
            with self.subTest(arm=arm):
                inst = rp.install_arm(tiny_model(), arm)
                if inst.ctx and inst.ctx.masked:
                    inst.ctx.set_domains(["code", "tool"])
                for _, p in inst.model.named_parameters():  # make the zero-init B/W_up matter
                    if p.requires_grad and p.dim() == 3:
                        torch.nn.init.normal_(p, std=0.05)
                ids = torch.randint(0, 64, (2, 6))
                inst.model(input_ids=ids, labels=ids, output_router_logits=True).loss.backward()
                for n, p in inst.model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, n)
                    else:
                        self.assertIsNone(p.grad, n)


@unittest.skipUnless(HAVE, "torch/transformers/peft not installed (moe extra)")
class BlockMathTests(unittest.TestCase):
    def test_adapters_start_as_exact_noops(self):
        ids = torch.randint(0, 64, (2, 7))
        ref = tiny_model()(input_ids=ids).logits
        for arm in ("perft_r_emergent", "mixlora", "plain_lora"):
            with self.subTest(arm=arm):
                inst = rp.install_arm(tiny_model(), arm)
                self.assertTrue(torch.allclose(inst.model(input_ids=ids).logits, ref, atol=1e-6))

    def test_perft_r_matches_a_per_token_reference(self):
        model = tiny_model()
        inst = rp.install_arm(model, "perft_r_emergent")
        block = model.model.layers[0].mlp
        with torch.no_grad():
            block.peft_expert_w_up.normal_(std=0.3)
            block.peft_router.normal_(std=1.0)
        h = torch.randn(2, 5, 32)
        with torch.no_grad():
            got = block(h)
            base = block.base(h)
        s = inst.spec.scaling
        flat = h.reshape(-1, 32)
        ref = base.reshape(-1, 32).clone()
        for t in range(flat.shape[0]):
            logits = flat[t] @ block.peft_router
            val, idx = torch.topk(logits, 2)
            w = torch.softmax(val, dim=0)
            for wi, e in zip(w, idx.tolist()):
                ref[t] += s * wi * (torch.nn.functional.silu(flat[t] @ block.peft_expert_w_down[e]) @ block.peft_expert_w_up[e])
        self.assertTrue(torch.allclose(got.reshape(-1, 32), ref, atol=1e-5))
        self.assertGreater(float((got - base).abs().max()), 1e-3)

    def test_mixlora_matches_a_per_token_reference(self):
        model = tiny_model()
        inst = rp.install_arm(model, "mixlora")
        block = model.model.layers[1].mlp
        with torch.no_grad():
            for p in (block.expert_gate_up_delta_b, block.expert_down_delta_b):
                p.normal_(std=0.3)
            block.lora_router.normal_(std=1.0)
        h = torch.randn(1, 6, 32)
        with torch.no_grad():
            got = block(h).reshape(-1, 32)
            flat = h.reshape(-1, 32)
            _, w_top, i_top = block.base.gate(flat)
        ex, s = block.base.experts, inst.spec.scaling
        ref = torch.zeros_like(flat)
        for t in range(flat.shape[0]):
            val, idx = torch.topk(flat[t] @ block.lora_router, inst.spec.router_top_k)
            g = torch.zeros(inst.spec.num_experts).scatter(0, idx, torch.softmax(val, 0))
            for slot in range(i_top.shape[1]):
                e = int(i_top[t, slot])
                gu = ex.gate_up_proj[e] @ flat[t]
                gu = gu + s * sum(
                    g[n] * (flat[t] @ block.expert_gate_up_delta_a[n]) @ block.expert_gate_up_delta_b[n]
                    for n in range(inst.spec.num_experts)
                )
                gate, up = gu.chunk(2)
                x = torch.nn.functional.silu(gate) * up
                y = ex.down_proj[e] @ x + s * sum(
                    g[n] * (x @ block.expert_down_delta_a[n]) @ block.expert_down_delta_b[n]
                    for n in range(inst.spec.num_experts)
                )
                ref[t] += w_top[t, slot] * y
        self.assertTrue(torch.allclose(got, ref, atol=1e-5), float((got - ref).detach().abs().max()))

    def test_mixlora_with_zero_delta_equals_the_base_block(self):
        model = tiny_model()
        h = torch.randn(2, 4, 32)
        base_out = model.model.layers[0].mlp(h)
        rp.install_arm(model, "mixlora")
        self.assertTrue(torch.allclose(model.model.layers[0].mlp(h), base_out, atol=1e-6))


@unittest.skipUnless(HAVE, "torch/transformers/peft not installed (moe extra)")
class RoutingTests(unittest.TestCase):
    def test_route_topk_matches_reference_top_k(self):
        torch.manual_seed(3)
        logits = torch.randn(40, 8)
        gates, idx, _ = rp.route_topk(logits, 2)
        for t in range(40):
            self.assertEqual(frozenset(idx[t].tolist()), rm.top_k_set(logits[t].tolist(), 2))
        self.assertTrue(torch.allclose(gates.sum(-1), torch.ones(40)))
        self.assertTrue(((gates > 0).sum(-1) == 2).all())
        with self.assertRaises(ValueError):
            rp.route_topk(logits, 9)

    def test_masked_routing_never_selects_out_of_domain_experts(self):
        torch.manual_seed(4)
        ctx = rp.RoutingContext(num_experts=8, top_k=2, assignment=rp.PRESCRIBED_ASSIGNMENT)
        for domain, ids in rp.PRESCRIBED_ASSIGNMENT.items():
            ctx.set_domains([domain] * 3)
            allowed = ctx.allowed_mask(3, 5, "cpu")
            for _ in range(20):
                logits = torch.randn(15, 8) * 5
                logits[:, [e for e in range(8) if e not in ids]] += 50  # tempt the router
                gates, idx, masked = rp.route_topk(logits, 2, allowed)
                self.assertTrue(set(idx.flatten().tolist()) <= set(ids), domain)
                self.assertEqual(float(gates[:, [e for e in range(8) if e not in ids]].abs().sum()), 0.0)
                self.assertTrue(torch.isfinite(masked).all())
                for t in range(15):  # top-2 *among the domain's experts*
                    in_dom = sorted(ids, key=lambda e: -logits[t, e])[:2]
                    self.assertEqual(set(idx[t].tolist()), set(in_dom))

    def test_masked_context_requires_labels_and_matching_batch(self):
        ctx = rp.RoutingContext(num_experts=8, top_k=2, assignment=rp.PRESCRIBED_ASSIGNMENT)
        with self.assertRaises(RuntimeError):
            ctx.allowed_mask(2, 3, "cpu")
        ctx.set_domains(["code"])
        with self.assertRaises(ValueError):
            ctx.allowed_mask(2, 3, "cpu")
        with self.assertRaises(ValueError):
            ctx.set_domains(["poetry"])
        self.assertIsNone(
            rp.RoutingContext(num_experts=8, top_k=2, assignment=None).allowed_mask(1, 1, "cpu")
        )

    def test_every_domain_has_at_least_top_k_experts(self):
        with self.assertRaises(ValueError):
            rp.RoutingContext(
                num_experts=8, top_k=2, assignment={"code": [0], "math": [1, 2], "tool": [3, 4]}
            )

    def test_prescribed_block_uses_only_domain_experts_in_training_and_eval(self):
        for mode in ("train", "eval"):
            model = tiny_model()
            inst = rp.install_arm(model, "perft_r_prescribed")
            model.train(mode == "train")
            block = model.model.layers[0].mlp
            with torch.no_grad():
                block.peft_router.normal_(std=2.0)
                block.peft_router[:, 6:] += 5.0  # the router prefers tool experts everywhere
            inst.ctx.recording = True
            for domain, ids in rp.PRESCRIBED_ASSIGNMENT.items():
                inst.ctx.set_domains([domain, domain])
                with torch.no_grad():
                    block(torch.randn(2, 6, 32))
                (rows,) = inst.ctx.take_records()
                for row in rows:
                    self.assertTrue(rm.top_k_set(row, 2) <= set(ids), (mode, domain))

    def test_prescribed_delta_only_comes_from_domain_experts(self):
        model = tiny_model()
        inst = rp.install_arm(model, "perft_r_prescribed")
        block = model.model.layers[0].mlp
        with torch.no_grad():
            block.peft_expert_w_up.normal_(std=0.5)
            block.peft_expert_w_up[6:] = 0.0  # tool experts contribute nothing
            block.peft_router.normal_(std=1.0)
        h = torch.randn(2, 5, 32)
        with torch.no_grad():
            base = block.base(h)
            inst.ctx.set_domains(["tool", "tool"])
            self.assertTrue(torch.allclose(block(h), base, atol=1e-6))
            inst.ctx.set_domains(["code", "math"])
            self.assertGreater(float((block(h) - base).abs().max()), 1e-4)

    def test_emergent_arm_is_unmasked_and_ignores_labels(self):
        model = tiny_model()
        inst = rp.install_arm(model, "perft_r_emergent")
        self.assertFalse(inst.ctx.masked)
        block = model.model.layers[0].mlp
        with torch.no_grad():
            block.peft_expert_w_up.normal_(std=0.5)
        h = torch.randn(1, 4, 32)
        with torch.no_grad():
            a = block(h)
            inst.ctx.set_domains(["tool"])
            self.assertTrue(torch.equal(a, block(h)))

    def test_recorded_logits_feed_routing_metrics(self):
        inst = rp.install_arm(tiny_model(), "mixlora")
        inst.ctx.recording = True
        inst.model(input_ids=torch.randint(0, 64, (1, 9)))
        layers = inst.ctx.take_records()
        self.assertEqual(len(layers), 2)
        stats = rm.snapshot_metrics(layers, inst.spec.router_top_k)
        self.assertTrue(all(0.0 <= s["entropy"] <= 2.2 for s in stats))
        self.assertEqual(sum(stats[0]["load"]), 9 * 2)
        self.assertEqual(inst.ctx.take_records(), [])


if __name__ == "__main__":
    unittest.main()
