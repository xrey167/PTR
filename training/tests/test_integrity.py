import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from ptr_training import integrity
from ptr_training.moe.pipeline import stack_available

HAVE_STACK = stack_available(("torch", "trl", "transformers", "datasets", "peft"))
SMOKE = Path(__file__).resolve().parents[1] / "configs" / "stage-moe-smoke.toml"


class FakeParam:
    """Stands in for a tensor: the digest logic only reads dtype, shape and bytes."""

    def __init__(self, data: bytes, requires_grad: bool, dtype="float32"):
        self.data = data
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.shape = (len(data),)


class FakeModel:
    def __init__(self, **params: FakeParam):
        self.params = params

    def named_parameters(self):
        return list(self.params.items())


def encode(param: FakeParam) -> bytes:
    return param.data


def lora_like() -> FakeModel:
    return FakeModel(
        base=FakeParam(b"base-weights", requires_grad=False),
        lora_a=FakeParam(b"a-init", requires_grad=True),
        lora_b=FakeParam(b"b-init", requires_grad=True),
    )


class DigestTests(unittest.TestCase):
    def test_frozen_digest_ignores_trainable_parameters(self):
        model = lora_like()
        before = integrity.frozen_parameter_digest(model, encode)
        model.params["lora_a"].data = b"a-trained"
        self.assertEqual(integrity.frozen_parameter_digest(model, encode), before)

    def test_trainable_digest_ignores_frozen_parameters(self):
        model = lora_like()
        before = integrity.trainable_parameter_digest(model, encode)
        model.params["base"].data = b"tampered"
        self.assertEqual(integrity.trainable_parameter_digest(model, encode), before)

    def test_digest_is_independent_of_parameter_order(self):
        first = FakeModel(a=FakeParam(b"1", True), b=FakeParam(b"2", True))
        second = FakeModel(b=FakeParam(b"2", True), a=FakeParam(b"1", True))
        self.assertEqual(
            integrity.trainable_parameter_digest(first, encode),
            integrity.trainable_parameter_digest(second, encode),
        )

    def test_digest_binds_name_and_dtype_not_only_bytes(self):
        base = FakeModel(a=FakeParam(b"xx", True))
        renamed = FakeModel(b=FakeParam(b"xx", True))
        retyped = FakeModel(a=FakeParam(b"xx", True, dtype="bfloat16"))
        digest = integrity.trainable_parameter_digest(base, encode)
        self.assertNotEqual(digest, integrity.trainable_parameter_digest(renamed, encode))
        self.assertNotEqual(digest, integrity.trainable_parameter_digest(retyped, encode))

    def test_adjacent_tensors_cannot_trade_bytes(self):
        a = FakeModel(p=FakeParam(b"ab", True), q=FakeParam(b"c", True))
        b = FakeModel(p=FakeParam(b"a", True), q=FakeParam(b"bc", True))
        self.assertNotEqual(
            integrity.trainable_parameter_digest(a, encode),
            integrity.trainable_parameter_digest(b, encode),
        )

    def test_parameter_counts_split_frozen_and_trainable(self):
        self.assertEqual(
            integrity.parameter_counts(lora_like()), {"frozen": 1, "trainable": 2}
        )

    def test_directory_digest_changes_with_content_and_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_model.bin").write_bytes(b"weights")
            first = integrity.directory_sha256(root)
            (root / "adapter_model.bin").write_bytes(b"other")
            self.assertNotEqual(integrity.directory_sha256(root), first)
            (root / "adapter_model.bin").write_bytes(b"weights")
            self.assertEqual(integrity.directory_sha256(root), first)
            (root / "adapter_model.bin").rename(root / "renamed.bin")
            self.assertNotEqual(integrity.directory_sha256(root), first)

    def test_directory_digest_refuses_an_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            integrity.directory_sha256(Path(tmp))


class AssertionTests(unittest.TestCase):
    def assertViolation(self, check, call, *args, **kwargs):
        with self.assertRaises(integrity.IntegrityError) as caught:
            call(*args, **kwargs)
        self.assertEqual(caught.exception.check, check)
        return caught.exception

    def test_a_changed_frozen_digest_is_a_violation(self):
        integrity.assert_frozen_unchanged("a", "a")
        error = self.assertViolation("frozen_unchanged", integrity.assert_frozen_unchanged, "a", "b")
        self.assertEqual((error.expected, error.actual), ("a", "b"))

    def test_an_unchanged_adapter_is_a_violation(self):
        integrity.assert_adapter_changed("a", "b")
        self.assertViolation("adapter_changed", integrity.assert_adapter_changed, "a", "a")

    def test_step_count_must_match_exactly_in_both_directions(self):
        self.assertEqual(integrity.assert_step_count(5, 5), 5)
        for actual in (4, 6, 0):
            error = self.assertViolation("steps_exact", integrity.assert_step_count, actual, 5)
            self.assertEqual((error.expected, error.actual), (5, actual))

    def test_step_count_reads_global_step_from_trainer_state(self):
        state = type("State", (), {"global_step": 3})()
        self.assertEqual(integrity.assert_step_count(state, 3, "RL"), 3)
        self.assertViolation("steps_exact", integrity.assert_step_count, state, 4, "RL")

    def test_a_boolean_is_not_a_step_count(self):
        self.assertViolation("steps_exact", integrity.assert_step_count, True, 1)

    def test_evaluation_must_leave_the_model_identical(self):
        integrity.assert_model_unchanged_during_eval("a", "a")
        self.assertViolation(
            "eval_identity_unchanged", integrity.assert_model_unchanged_during_eval, "a", "b"
        )


class IntegrityLogTests(unittest.TestCase):
    def passing_run(self, log: integrity.IntegrityLog, model: FakeModel) -> None:
        log.start(model)
        log.checked_eval("init", model, lambda: "scores")
        start = log.trainable_digest(model)
        model.params["lora_a"].data = b"a-trained"
        log.record_steps("sft", 3, 3)
        log.record_stage("after_sft", model, start)

    def test_a_clean_run_reports_every_property_true(self):
        log, model = integrity.IntegrityLog(encode), lora_like()
        self.passing_run(log, model)
        report = log.report()
        for name in integrity.REQUIRED_CHECKS:
            self.assertIs(report[name], True, name)
        self.assertEqual(report["frozen_tensors"], 1)
        self.assertEqual(report["trainable_tensors"], 2)
        self.assertEqual(report["schema"], integrity.SCHEMA)
        json.dumps(report, allow_nan=False)

    def test_mutating_a_frozen_weight_aborts_the_stage(self):
        log, model = integrity.IntegrityLog(encode), lora_like()
        log.start(model)
        start = log.trainable_digest(model)
        model.params["lora_a"].data = b"a-trained"
        model.params["base"].data = b"leaked-update"
        with self.assertRaises(integrity.IntegrityError) as caught:
            log.record_stage("after_sft", model, start)
        self.assertEqual(caught.exception.check, "frozen_unchanged")

    def test_a_no_op_training_stage_aborts(self):
        log, model = integrity.IntegrityLog(encode), lora_like()
        log.start(model)
        start = log.trainable_digest(model)
        with self.assertRaises(integrity.IntegrityError) as caught:
            log.record_stage("after_sft", model, start)
        self.assertEqual(caught.exception.check, "adapter_changed")

    def test_an_evaluation_that_touches_weights_aborts(self):
        log, model = integrity.IntegrityLog(encode), lora_like()
        log.start(model)

        def evaluate():
            model.params["lora_b"].data = b"drifted"

        with self.assertRaises(integrity.IntegrityError) as caught:
            log.checked_eval("after_sft", model, evaluate)
        self.assertEqual(caught.exception.check, "eval_identity_unchanged")

    def test_wrong_step_count_aborts(self):
        log = integrity.IntegrityLog(encode)
        with self.assertRaises(integrity.IntegrityError):
            log.record_steps("sft", 3, 2)

    def test_a_check_that_never_ran_fails_the_report(self):
        log, model = integrity.IntegrityLog(encode), lora_like()
        log.start(model)
        log.checked_eval("init", model, lambda: None)
        with self.assertRaises(integrity.IntegrityError) as caught:
            log.report()
        self.assertIn("frozen_unchanged", str(caught.exception))
        self.assertIn("steps_exact", str(caught.exception))

    def test_a_stage_before_start_is_refused(self):
        log = integrity.IntegrityLog(encode)
        with self.assertRaises(RuntimeError):
            log.record_stage("after_sft", lora_like(), "x")

    def test_a_full_fine_tune_has_nothing_frozen_and_says_so(self):
        model = FakeModel(w=FakeParam(b"w0", True))
        log = integrity.IntegrityLog(encode)
        log.start(model)
        log.checked_eval("init", model, lambda: None)
        start = log.trainable_digest(model)
        model.params["w"].data = b"w1"
        log.record_steps("sft", 1, 1)
        log.record_stage("after_sft", model, start)
        report = log.report()
        self.assertEqual(report["frozen_tensors"], 0)
        self.assertIs(report["frozen_unchanged"], True)


@unittest.skipUnless(HAVE_STACK, "torch/trl/transformers/datasets/peft not installed (moe extra)")
class IntegrityTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ptr_training.moe import config as moe_config

        cls.moe_config = moe_config
        cls.cfg = moe_config.load(SMOKE)

    def small(self, *algorithms, lora=False):
        cfg = dataclasses.replace(
            self.cfg,
            sft=dataclasses.replace(self.cfg.sft, steps=1, batch=2, examples=8),
            rl=dataclasses.replace(
                self.cfg.rl, algorithms=tuple(algorithms), steps=1, mini_steps=1,
                num_generations=2, prompts_per_step=1, max_completion_length=14,
            ),
        )
        if lora:
            cfg = dataclasses.replace(
                cfg,
                lora=self.moe_config.LoraConfig(
                    enabled=True, rank=4, alpha=8,
                    target_modules=("q_proj", "v_proj"),
                    target_parameters=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
                ),
            )
        return cfg

    def test_tensor_bytes_distinguishes_bfloat16_values(self):
        import torch

        a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0, 3.0], dtype=torch.bfloat16)
        self.assertNotEqual(integrity.tensor_bytes(a), integrity.tensor_bytes(b))
        self.assertEqual(integrity.tensor_bytes(a), integrity.tensor_bytes(a.clone()))

    def test_frozen_digest_follows_requires_grad_on_a_real_module(self):
        import torch

        module = torch.nn.Linear(3, 2)
        module.weight.requires_grad_(False)
        before = integrity.frozen_parameter_digest(module)
        with torch.no_grad():
            module.bias.add_(1.0)
        self.assertEqual(integrity.frozen_parameter_digest(module), before)
        with torch.no_grad():
            module.weight.add_(1.0)
        self.assertNotEqual(integrity.frozen_parameter_digest(module), before)

    def test_lora_run_writes_the_integrity_block_and_the_adapter(self):
        from ptr_training.moe import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.execute(self.small("gspo", lora=True), Path(tmp))
            integrity_block = result["integrity"]
            for name in integrity.REQUIRED_CHECKS:
                self.assertIs(integrity_block[name], True, name)
            self.assertGreater(integrity_block["frozen_tensors"], 0)
            self.assertEqual(result["rl"]["gspo"]["global_step"], 1)
            self.assertEqual(result["sft"]["optimizer_steps"], 1)
            self.assertEqual(set(result["adapters"]), {"after_sft", "after_rl_gspo"})
            final = result["adapters"]["after_rl_gspo"]
            self.assertEqual(result["adapter_sha256"], final["sha256"])
            self.assertEqual(
                integrity.directory_sha256(Path(tmp) / final["path"]), final["sha256"]
            )
            path = pipeline.write_metrics(result, Path(tmp))
            self.assertIn("integrity", json.loads(path.read_text()))

    def test_full_fine_tune_records_that_no_adapter_exists(self):
        from ptr_training.moe import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.execute(self.small(), Path(tmp))
        self.assertIsNone(result["adapter_sha256"])
        self.assertEqual(result["adapters"], {})
        self.assertEqual(result["integrity"]["frozen_tensors"], 0)

    def test_a_frozen_weight_changed_by_training_aborts_the_pipeline(self):
        import torch

        from ptr_training.moe import pipeline, sft

        real = sft.run_sft

        def leaky(model, *args, **kwargs):
            out = real(model, *args, **kwargs)
            frozen = next(p for p in model.parameters() if not p.requires_grad)
            with torch.no_grad():
                frozen.view(-1)[0] += 1.0
            return out

        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sft, "run_sft", leaky):
            with self.assertRaises(integrity.IntegrityError) as caught:
                pipeline.execute(self.small("gspo", lora=True), Path(tmp))
        self.assertEqual(caught.exception.check, "frozen_unchanged")

    def test_a_training_stage_that_moves_nothing_aborts_the_pipeline(self):
        from unittest import mock

        from ptr_training.moe import pipeline, sft

        def idle(model, tokenizer, tasks, cfg, seed):
            return {"optimizer_steps": cfg.steps, "losses": [0.0] * cfg.steps, "nan": False,
                    "aux_losses": []}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sft, "run_sft", idle):
            with self.assertRaises(integrity.IntegrityError) as caught:
                pipeline.execute(self.small("gspo", lora=True), Path(tmp))
        self.assertEqual(caught.exception.check, "adapter_changed")

    def test_an_evaluation_that_changes_weights_aborts_the_pipeline(self):
        import torch
        from unittest import mock

        from ptr_training.moe import pipeline, rl

        real = rl.evaluate_policy

        def mutating(model, *args, **kwargs):
            out = real(model, *args, **kwargs)
            with torch.no_grad():
                next(iter(model.parameters())).view(-1)[0] += 1.0
            return out

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rl, "evaluate_policy", mutating):
            with self.assertRaises(integrity.IntegrityError) as caught:
                pipeline.execute(self.small(), Path(tmp))
        self.assertEqual(caught.exception.check, "eval_identity_unchanged")

    def test_run_config_leaves_a_violation_record_and_no_metrics(self):
        from unittest import mock

        from ptr_training.moe import pipeline

        error = integrity.IntegrityError("frozen_unchanged", "x", expected="a", actual="b")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            pipeline, "execute", side_effect=error
        ):
            with self.assertRaises(integrity.IntegrityError):
                pipeline.run_config(SMOKE, tmp)
            runs = list(Path(tmp).iterdir())
            self.assertEqual(len(runs), 1)
            self.assertFalse((runs[0] / "metrics.json").exists())
            violation = json.loads((runs[0] / "integrity-violation.json").read_text())
        self.assertEqual(violation["check"], "frozen_unchanged")


if __name__ == "__main__":
    unittest.main()
