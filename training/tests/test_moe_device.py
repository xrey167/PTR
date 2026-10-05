"""Device selection, memory options (gradient checkpointing / accumulation) and the GPU configs.

Everything here runs without a GPU: CUDA availability is mocked and the end-to-end test uses
the tiny model on CPU. Torch-dependent tests skip when the moe extra is absent.
"""

import copy
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from ptr_training.moe import config as moe_config
from ptr_training.moe.config import ConfigError
from ptr_training.moe.pipeline import stack_available

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
HAVE_STACK = stack_available(("torch", "trl", "transformers", "datasets", "peft"))
HAVE_TORCH = stack_available(("torch",))


def _raw(name="stage-moe-smoke.toml") -> dict:
    return tomllib.loads((CONFIGS / name).read_text(encoding="utf-8"))


class DeviceConfigTests(unittest.TestCase):
    def test_device_defaults_to_auto(self):
        self.assertEqual(moe_config.load(CONFIGS / "stage-moe-smoke.toml").model.device, "auto")

    def test_known_devices_parse(self):
        for device in ("auto", "cpu", "cuda"):
            raw = _raw()
            raw["model"]["device"] = device
            self.assertEqual(moe_config.parse(raw).model.device, device)

    def test_unknown_device_rejected(self):
        for bad in ("gpu", "cuda:0", "CUDA", "", "mps"):
            raw = _raw()
            raw["model"]["device"] = bad
            with self.assertRaisesRegex(ConfigError, "model.device"):
                moe_config.parse(raw)

    def test_non_string_device_rejected(self):
        raw = _raw()
        raw["model"]["device"] = 0
        with self.assertRaisesRegex(ConfigError, "model.device"):
            moe_config.parse(raw)

    def test_device_not_accepted_under_other_tables(self):
        raw = _raw()
        raw["moe"]["sft"]["device"] = "cpu"
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            moe_config.parse(raw)

    def test_resolve_device_rejects_unknown(self):
        with self.assertRaisesRegex(ConfigError, "model.device"):
            moe_config.resolve_device("tpu")

    def test_resolve_cpu_never_needs_torch(self):
        self.assertEqual(moe_config.resolve_device("cpu"), "cpu")

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_cuda_rejected_when_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ConfigError, "CUDA is not available"):
                moe_config.resolve_device("cuda")

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_cuda_and_auto_resolve_when_available(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(moe_config.resolve_device("cuda"), "cuda")
            self.assertEqual(moe_config.resolve_device("auto"), "cuda")
            self.assertEqual(moe_config.resolve_device("cpu"), "cpu")

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_auto_is_cpu_when_cuda_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(moe_config.resolve_device("auto"), "cpu")


class MemoryOptionConfigTests(unittest.TestCase):
    def test_defaults(self):
        sft = moe_config.load(CONFIGS / "stage-moe-smoke.toml").sft
        self.assertFalse(sft.gradient_checkpointing)
        self.assertEqual(sft.gradient_accumulation_steps, 1)

    def test_values_parse(self):
        raw = _raw()
        raw["moe"]["sft"].update(gradient_checkpointing=True, gradient_accumulation_steps=4)
        sft = moe_config.parse(raw).sft
        self.assertTrue(sft.gradient_checkpointing)
        self.assertEqual(sft.gradient_accumulation_steps, 4)

    def test_accumulation_must_be_positive_int(self):
        for bad in (0, -1):
            raw = copy.deepcopy(_raw())
            raw["moe"]["sft"]["gradient_accumulation_steps"] = bad
            with self.assertRaisesRegex(ConfigError, "gradient_accumulation_steps"):
                moe_config.parse(raw)
        for bad in (1.5, "2", True):
            raw = copy.deepcopy(_raw())
            raw["moe"]["sft"]["gradient_accumulation_steps"] = bad
            with self.assertRaisesRegex(ConfigError, "gradient_accumulation_steps"):
                moe_config.parse(raw)

    def test_checkpointing_must_be_bool(self):
        raw = _raw()
        raw["moe"]["sft"]["gradient_checkpointing"] = "yes"
        with self.assertRaisesRegex(ConfigError, "gradient_checkpointing"):
            moe_config.parse(raw)

    def test_accumulation_ceiling_unless_allow_large(self):
        raw = _raw()
        raw["moe"]["sft"]["gradient_accumulation_steps"] = 5000
        with self.assertRaisesRegex(ConfigError, "sanity limit"):
            moe_config.parse(raw)
        raw["moe"]["allow_large"] = True
        self.assertEqual(moe_config.parse(raw).sft.gradient_accumulation_steps, 5000)


class GpuConfigTests(unittest.TestCase):
    def test_tiny_gpu_smoke_parses(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke-gpu.toml")
        cpu = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual(cfg.model.device, "cuda")
        self.assertTrue(cfg.allow_unselected_backend)
        self.assertEqual(cfg.model.precision, "fp32")
        # Same experiment as the CPU smoke apart from the device (and naming/output dir).
        self.assertEqual(cfg.model, type(cfg.model)(**{**cpu.model.__dict__, "device": "cuda"}))
        self.assertEqual((cfg.sft, cfg.rl), (cpu.sft, cpu.rl))
        self.assertEqual(_raw("stage-moe-smoke-gpu.toml")["run"]["hardware_profile"],
                         "hardware/rtx4090.toml")

    def test_olmoe_sft_smoke_parses(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-olmoe-sft-smoke.toml")
        self.assertEqual(cfg.model.hf_id, "allenai/OLMoE-1B-7B-0125")
        self.assertEqual(cfg.model.revision, "9b0c1aa87e34a20052389dce1f0cf01da783f654")
        self.assertEqual((cfg.model.precision, cfg.model.device), ("bf16", "cuda"))
        self.assertTrue(cfg.allow_unselected_backend)
        self.assertEqual(cfg.rl.algorithms, ())
        self.assertTrue(cfg.lora.enabled)
        self.assertEqual(cfg.lora, moe_config.load(CONFIGS / "stage-moe-olmoe.toml").lora)
        sft = cfg.sft
        self.assertEqual((sft.batch, sft.gradient_accumulation_steps, sft.max_length, sft.steps),
                         (2, 4, 256, 50))
        self.assertTrue(sft.gradient_checkpointing)
        self.assertEqual(_raw("stage-moe-olmoe-sft-smoke.toml")["run"]["hardware_profile"],
                         "hardware/rtx4090.toml")

    def test_olmoe_config_keeps_3090_profile_and_still_needs_opt_in(self):
        raw = _raw("stage-moe-olmoe.toml")
        self.assertEqual(raw["run"]["hardware_profile"], "hardware/rtx3090.toml")
        self.assertFalse(moe_config.parse(raw).allow_unselected_backend)


@unittest.skipUnless(HAVE_STACK, "torch/trl/transformers/datasets/peft not installed (moe extra)")
class DeviceTorchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")

    def test_build_tiny_on_cpu_respects_device(self):
        from ptr_training.moe import model as model_mod

        for device in ("cpu", "auto"):
            with mock.patch("torch.cuda.is_available", return_value=False):
                mcfg = type(self.cfg.model)(**{**self.cfg.model.__dict__, "device": device})
                model, _ = model_mod.build(mcfg, self.cfg.lora)
            self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_build_tiny_cuda_without_cuda_raises(self):
        from ptr_training.moe import model as model_mod

        mcfg = type(self.cfg.model)(**{**self.cfg.model.__dict__, "device": "cuda"})
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ConfigError, "CUDA is not available"):
                model_mod.build(mcfg, self.cfg.lora)

    def test_explicit_cpu_stays_on_cpu_even_if_cuda_is_reported(self):
        from ptr_training.moe import model as model_mod

        mcfg = type(self.cfg.model)(**{**self.cfg.model.__dict__, "device": "cpu"})
        with mock.patch("torch.cuda.is_available", return_value=True):
            model, _ = model_mod.build(mcfg, self.cfg.lora)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_use_cpu_follows_device(self):
        # GRPOConfig is replaced by a recorder: constructing a real one with use_cpu=False
        # while torch.cuda.is_available is mocked would make transformers touch CUDA.
        from ptr_training.moe.rl import build_grpo_config

        for cuda_available in (False, True):
            with mock.patch("torch.cuda.is_available", return_value=cuda_available):
                for device, use_cpu in (("cpu", True), ("cuda", False)):
                    with mock.patch("trl.GRPOConfig") as grpo_config:
                        build_grpo_config(
                            "gspo", self.cfg.rl, seed=1, output_dir="unused", bf16=False,
                            device=device,
                        )
                    self.assertEqual(
                        grpo_config.call_args.kwargs["use_cpu"], use_cpu, (cuda_available, device)
                    )

    def test_hardware_records_device_fields(self):
        from ptr_training.moe.pipeline import hardware

        cpu = hardware("cpu")
        self.assertEqual(cpu["device"], "cpu")
        self.assertIsNone(cpu["peak_cuda_memory_allocated_bytes"])
        self.assertIn("torch_cuda", cpu)
        self.assertIn("gpu", cpu)

    def test_gradient_accumulation_end_to_end_on_cpu(self):
        import torch

        from ptr_training.moe import model as model_mod
        from ptr_training.moe.config import SftConfig
        from ptr_training.moe.sft import run_sft
        from ptr_training.moe.tasks import generate_tasks

        model, tok = model_mod.build(self.cfg.model, self.cfg.lora)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        sft = SftConfig(steps=2, lr=1e-2, batch=2, examples=16, max_length=26,
                        gradient_accumulation_steps=2)
        result = run_sft(model, tok, generate_tasks(16, seed=1), sft, 1)
        self.assertEqual(len(result["losses"]), 2)  # one entry per optimizer step
        self.assertFalse(result["nan"])
        self.assertEqual(result["effective_batch"], 4)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_gradient_checkpointing_runs_and_is_switched_off_afterwards(self):
        from ptr_training.moe import model as model_mod
        from ptr_training.moe.config import SftConfig
        from ptr_training.moe.sft import run_sft
        from ptr_training.moe.tasks import generate_tasks

        model, tok = model_mod.build(self.cfg.model, self.cfg.lora)
        sft = SftConfig(steps=1, lr=1e-2, batch=2, examples=16, max_length=26,
                        gradient_checkpointing=True)
        result = run_sft(model, tok, generate_tasks(16, seed=1), sft, 1)
        self.assertFalse(result["nan"])
        self.assertTrue(result["gradient_checkpointing"])
        self.assertFalse(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_with_lora_updates_lora_weights(self):
        import torch

        from ptr_training.moe import model as model_mod
        from ptr_training.moe.config import SftConfig
        from ptr_training.moe.sft import run_sft
        from ptr_training.moe.tasks import generate_tasks

        cfg = moe_config.load(CONFIGS / "stage-moe-olmoe.toml")
        lora = type(cfg.lora)(**{**cfg.lora.__dict__, "rank": 4, "alpha": 8})
        mcfg = type(self.cfg.model)(**{**self.cfg.model.__dict__, "precision": "fp32"})
        model, tok = model_mod.build(mcfg, lora)
        before = model_mod.trainable_state(model)
        self.assertTrue(before)
        sft = SftConfig(steps=1, lr=1e-2, batch=2, examples=16, max_length=26,
                        gradient_checkpointing=True)
        result = run_sft(model, tok, generate_tasks(16, seed=1), sft, 1)
        self.assertFalse(result["nan"])
        after = model_mod.trainable_state(model)
        self.assertTrue(any(not torch.equal(after[k], v) for k, v in before.items()))


if __name__ == "__main__":
    unittest.main()
