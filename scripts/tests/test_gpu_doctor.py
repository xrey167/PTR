"""gpu_doctor must tell a working GPU box from an absent GPU and from a broken torch.

Everything external is mocked (`run`, `which`, disk usage, the network), so these tests
need no GPU, no venv and no network. The three scenarios the desktop will meet are the
GPU working, no GPU at all (`--require-gpu` decides warn vs fail), and a torch that
imports but cannot use the card.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpu_doctor", ROOT / "scripts/gpu_doctor.py")
gd = importlib.util.module_from_spec(spec)
sys.modules["gpu_doctor"] = gd  # dataclasses resolves string annotations through sys.modules
spec.loader.exec_module(gd)

Usage = namedtuple("Usage", "total used free")
SMI_OK = "NVIDIA GeForce RTX 4090, 580.65.06, 24564\n"
LOCK = """
[[package]]
name = "torch"
version = "2.14.1"
[[package]]
name = "transformers"
version = "5.18.0"
[[package]]
name = "peft"
version = "0.21.2"
[[package]]
name = "trl"
version = "1.14.1"
[[package]]
name = "datasets"
version = "5.0.1"
[[package]]
name = "accelerate"
version = "1.15.0"
"""
GOOD_PROBE = {
    "python": "3.11.15",
    "torch": "2.14.1+cu130",
    "torch_cuda_build": "13.0",
    "cuda_available": True,
    "device_name": "NVIDIA GeForce RTX 4090",
    "bf16": True,
    "matmul_ok": True,
    "transformers": "5.18.0",
    "peft": "0.21.2",
    "trl": "1.14.1",
    "datasets": "5.0.1",
    "accelerate": "1.15.0",
    "ptr_py": "0.1.0",
}


_FIXTURES = []


def tearDownModule():
    for fx in _FIXTURES:
        fx.dir.cleanup()


class Fixture:
    """A fake repo root with a venv python, lock file and toolchain pin."""

    def __init__(self, venv=True):
        self.dir = tempfile.TemporaryDirectory()
        _FIXTURES.append(self)
        self.root = Path(self.dir.name)
        (self.root / "training").mkdir()
        (self.root / "training" / "uv.lock").write_text(LOCK)
        (self.root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.99.0"\n')
        for rel in ("hardware/rtx3090.toml", "hardware/rtx4090.toml", "hardware/dual-rtx3090.toml"):
            (self.root / rel).parent.mkdir(exist_ok=True)
            (self.root / rel).write_text("")
        if venv:
            py = self.root / "training" / ".venv" / "bin" / "python"
            py.parent.mkdir(parents=True)
            py.write_text("")

    def options(self, **kw):
        return gd.Options(root=self.root, offline=True, **kw)


def fake_run(smi=(0, SMI_OK, ""), probe=None, rustup="stable-x86_64-unknown-linux-gnu (default)\n1.99.0-x86_64-unknown-linux-gnu\n",
             bwrap=(0, "", "")):
    def _run(cmd, timeout=60, cwd=None):
        if cmd[0] == "nvidia-smi":
            return smi
        if cmd[0].endswith("bwrap"):
            return bwrap
        if cmd[0] == "rustup":
            return 0, rustup, ""
        if cmd[0].endswith("python"):
            if probe is None:
                return 1, "", "Traceback\nModuleNotFoundError: no torch"
            return 0, "noise\n" + json.dumps(probe) + "\n", ""
        raise AssertionError(cmd)

    return _run


def collect(fx, opts, which_missing=(), **run_kw):
    def _which(name):
        return None if name in which_missing else "/usr/bin/" + name

    with mock.patch.object(gd, "run", fake_run(**run_kw)), mock.patch.object(
        gd, "which", _which
    ), mock.patch.object(gd.shutil, "disk_usage", return_value=Usage(10**12, 0, 500 * 10**9)), mock.patch.dict(
        os.environ, {"HF_HOME": str(fx.root / "hf")}
    ):
        return {c.name: c for c in gd.collect(opts)}


class GpuPresent(unittest.TestCase):
    def test_everything_ok(self):
        fx = Fixture()
        got = collect(fx, fx.options(require_gpu=True), probe=GOOD_PROBE)
        bad = {n: (c.status, c.detail) for n, c in got.items() if c.status != gd.OK and n != "hf hub"}
        self.assertEqual(bad, {})
        self.assertEqual(gd.summarize(list(got.values()))["exit_code"], 0)
        self.assertIn("580.65.06", got["nvidia-smi"].detail)
        self.assertIn("4090", got["gpu"].detail)

    def test_parse_nvidia_smi(self):
        self.assertEqual(
            gd.parse_nvidia_smi(SMI_OK),
            [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "driver": "580.65.06", "memory_mib": 24564,
              "compute_cap": None}],
        )
        self.assertEqual(gd.parse_nvidia_smi("garbage\n"), [])

    def test_lock_mismatch_warns(self):
        fx = Fixture()
        probe = dict(GOOD_PROBE, transformers="5.0.0")
        got = collect(fx, fx.options(), probe=probe)
        self.assertEqual(got["transformers"].status, gd.WARN)
        self.assertIn("5.18.0", got["transformers"].detail)

    def test_old_driver_for_cu13_fails(self):
        fx = Fixture()
        got = collect(fx, fx.options(), smi=(0, "NVIDIA GeForce RTX 4090, 535.1, 24564\n", ""), probe=GOOD_PROBE)
        self.assertEqual(got["driver vs torch"].status, gd.FAIL)

    def test_low_disk_warns(self):
        fx = Fixture()
        with mock.patch.object(gd, "run", fake_run(probe=GOOD_PROBE)), mock.patch.object(
            gd, "which", lambda n: "/usr/bin/" + n
        ), mock.patch.object(gd.shutil, "disk_usage", return_value=Usage(10**12, 0, 10 * 10**9)):
            got = {c.name: c for c in gd.collect(fx.options())}
        self.assertEqual(got["disk (repo)"].status, gd.WARN)
        self.assertTrue(got["disk (repo)"].hint)


SMI_3090 = "NVIDIA GeForce RTX 3090, 580.65.06, 24576, 8.6\n"
SMI_DUAL_3090 = SMI_3090 * 2
SMI_4090_CAP = "NVIDIA GeForce RTX 4090, 580.65.06, 24564, 8.9\n"


def device(index, name="NVIDIA GeForce RTX 3090", cap=(8, 6), **kw):
    return {"index": index, "name": name, "capability": list(cap), "total_memory_mib": 24576,
            "bf16": True, "matmul_ok": True, "alloc_ok": True, **kw}


def probe_for(*devices):
    return dict(GOOD_PROBE, device_name=devices[0]["name"], device_count=len(devices), devices=list(devices))


class RunNeeds(unittest.TestCase):
    """What the OLMoE and M011 runs need beyond a visible GPU: the M011 HumanEval sandbox
    (bubblewrap), the datasets/accelerate pins, a 24 GiB-class card and free memory on it."""

    def test_missing_bubblewrap_fails_when_gpus_are_required(self):
        fx = Fixture()
        got = collect(fx, fx.options(require_gpu=True), which_missing=("bwrap",), probe=GOOD_PROBE)
        self.assertEqual(got["bubblewrap"].status, gd.FAIL)
        self.assertIn("apt install bubblewrap", got["bubblewrap"].hint)
        got = collect(fx, fx.options(), which_missing=("bwrap",), probe=GOOD_PROBE)
        self.assertEqual(got["bubblewrap"].status, gd.WARN)

    def test_bubblewrap_that_cannot_create_a_sandbox_fails(self):
        fx = Fixture()
        got = collect(fx, fx.options(require_gpu=True), probe=GOOD_PROBE,
                      bwrap=(1, "", "bwrap: setting up uid map: Permission denied\n"))
        self.assertEqual(got["bubblewrap"].status, gd.FAIL)
        self.assertIn("Permission denied", got["bubblewrap"].detail)

    def test_datasets_and_accelerate_are_checked_against_the_lock(self):
        fx = Fixture()
        got = collect(fx, fx.options(), probe=dict(GOOD_PROBE, datasets=None, accelerate="1.0.0"))
        self.assertEqual(got["datasets"].status, gd.FAIL)
        self.assertEqual(got["accelerate"].status, gd.WARN)
        self.assertIn("1.15.0", got["accelerate"].detail)

    def test_small_card_fails_when_gpus_are_required(self):
        fx = Fixture()
        got = collect(fx, fx.options(require_gpu=True),
                      smi=(0, "NVIDIA GeForce RTX 3090, 580.65.06, 12288, 8.6\n", ""), probe=GOOD_PROBE)
        self.assertEqual(got["gpu 0"].status, gd.FAIL)

    def test_occupied_card_warns_with_its_index(self):
        fx = Fixture()
        busy = device(1, free_mib=6000)
        got = collect(fx, fx.options(require_gpu=True), smi=(0, SMI_DUAL_3090, ""),
                      probe=probe_for(device(0, free_mib=23000), busy))
        self.assertEqual(got["free vram"].status, gd.WARN)
        self.assertIn("cuda:1", got["free vram"].detail)
        got = collect(fx, fx.options(require_gpu=True), smi=(0, SMI_DUAL_3090, ""),
                      probe=probe_for(device(0, free_mib=23000), device(1, free_mib=23500)))
        self.assertEqual(got["free vram"].status, gd.OK)


TWO_VERSION_LOCK = LOCK + """
[[package]]
name = "datasets"
version = "4.8.4"
[[package]]
name = "accelerate"
version = "1.13.0"
[[package]]
name = "ptr-training"
version = "0.1.0"
source = { editable = "." }

[package.optional-dependencies]
moe = [
    { name = "accelerate", version = "1.15.0", source = { registry = "https://pypi.org/simple" } },
    { name = "datasets", version = "5.0.1", source = { registry = "https://pypi.org/simple" } },
    { name = "peft" },
    { name = "torch", version = "2.14.1", source = { registry = "https://pypi.org/simple" } },
    { name = "transformers", version = "5.18.0", source = { registry = "https://pypi.org/simple" } },
    { name = "trl", version = "1.14.1", source = { registry = "https://pypi.org/simple" } },
]
"""


class MoeExtraLockResolution(unittest.TestCase):
    """Codex P2: uv.lock holds several versions of a package (the cpt and moe extras fork the
    resolution); the doctor accepted any of them, e.g. datasets 4.8.4 and accelerate 1.13.0,
    although the moe extra resolves to 5.0.1 and 1.15.0. Only the moe extra's resolution of
    the root package counts now."""

    def fixture(self):
        fx = Fixture()
        (fx.root / "training" / "uv.lock").write_text(TWO_VERSION_LOCK)
        return fx

    def test_other_extras_versions_are_not_accepted(self):
        fx = self.fixture()
        got = collect(fx, fx.options(), probe=dict(GOOD_PROBE, datasets="4.8.4", accelerate="1.13.0"))
        self.assertEqual(got["datasets"].status, gd.WARN)
        self.assertEqual(got["accelerate"].status, gd.WARN)
        self.assertTrue(got["datasets"].detail.endswith("lock has 5.0.1"), got["datasets"].detail)
        self.assertTrue(got["accelerate"].detail.endswith("lock has 1.15.0"), got["accelerate"].detail)

    def test_moe_versions_pass_and_unversioned_entries_use_the_single_lock_version(self):
        fx = self.fixture()
        got = collect(fx, fx.options(), probe=GOOD_PROBE)
        for name in ("transformers", "peft", "trl", "datasets", "accelerate"):
            self.assertEqual(got[name].status, gd.OK, (name, got[name].detail))
        self.assertEqual(gd.lock_versions(fx.root)["datasets"], ["5.0.1"])
        self.assertEqual(gd.lock_versions(fx.root)["peft"], ["0.21.2"])


class HardwareProfiles(unittest.TestCase):
    def test_parse_with_and_without_compute_cap(self):
        got = gd.parse_nvidia_smi(SMI_DUAL_3090)
        self.assertEqual([g["index"] for g in got], [0, 1])
        self.assertEqual(got[1]["compute_cap"], (8, 6))
        self.assertEqual(got[0]["memory_mib"], 24576)
        self.assertIsNone(gd.parse_nvidia_smi(SMI_OK)[0]["compute_cap"])
        self.assertIsNone(gd.parse_capability("n/a"))
        self.assertEqual(gd.parse_capability([8, 9]), (8, 9))

    def test_card_kind_tested_and_untested(self):
        self.assertEqual(gd.card_kind("NVIDIA GeForce RTX 3090"), "rtx3090")
        self.assertEqual(gd.card_kind("NVIDIA GeForce RTX 4090"), "rtx4090")
        for other in ("NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4090 D", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 5090"):
            self.assertIsNone(gd.card_kind(other), other)

    def test_bf16_capability_covers_ampere_and_ada(self):
        self.assertTrue(gd.bf16_capable((8, 6)))  # RTX 3090
        self.assertTrue(gd.bf16_capable((8, 9)))  # RTX 4090
        self.assertFalse(gd.bf16_capable((7, 5)))
        self.assertIsNone(gd.bf16_capable(None))

    def test_select_profile(self):
        r3, r4 = "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"
        self.assertEqual(gd.select_profile([r4])[0], "hardware/rtx4090.toml")
        self.assertEqual(gd.select_profile([r3])[0], "hardware/rtx3090.toml")
        self.assertEqual(gd.select_profile([r3, r3])[0], "hardware/dual-rtx3090.toml")
        self.assertIn("single-GPU runs: hardware/rtx3090.toml", gd.select_profile([r3, r3])[1])
        self.assertEqual(gd.select_profile([r4, r4])[0], "hardware/rtx4090.toml")
        for names in ([r3, r4], ["NVIDIA RTX A6000"], []):
            profile, text = gd.select_profile(names)
            self.assertIsNone(profile)
            self.assertIn("untested profile", text)

    def test_profile_files_named_by_the_doctor_exist(self):
        for rel in (*gd.PROFILE_SINGLE.values(), gd.PROFILE_DUAL_3090):
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_dual_3090_all_ok(self):
        fx = Fixture()
        probe = probe_for(device(0), device(1))
        got = collect(fx, fx.options(require_gpu=True), smi=(0, SMI_DUAL_3090, ""), probe=probe)
        bad = {n: (c.status, c.detail) for n, c in got.items() if c.status != gd.OK and n != "hf hub"}
        self.assertEqual(bad, {})
        self.assertIn("2x NVIDIA GeForce RTX 3090", got["gpu"].detail)
        self.assertIn("gpu 0", got)
        self.assertIn("gpu 1", got)
        self.assertIn("sm_86", got["gpu 1"].detail)
        self.assertIn("dual-rtx3090.toml", got["hardware profile"].detail)
        self.assertIn("2 device(s)", got["torch cuda"].detail)
        self.assertEqual(gd.summarize(list(got.values()))["exit_code"], 0)

    def test_single_4090_names_its_profile(self):
        fx = Fixture()
        got = collect(fx, fx.options(), smi=(0, SMI_4090_CAP, ""), probe=GOOD_PROBE)
        self.assertEqual(got["hardware profile"].status, gd.OK)
        self.assertIn("hardware/rtx4090.toml", got["hardware profile"].detail)

    def test_missing_profile_file_warns(self):
        fx = Fixture()
        (fx.root / "hardware" / "rtx4090.toml").unlink()
        got = collect(fx, fx.options(), smi=(0, SMI_4090_CAP, ""), probe=GOOD_PROBE)
        self.assertEqual(got["hardware profile"].status, gd.WARN)

    def test_untested_card_warns_but_does_not_fail(self):
        fx = Fixture()
        smi = "NVIDIA RTX A6000, 580.65.06, 49140, 8.6\n"
        got = collect(fx, fx.options(require_gpu=True), smi=(0, smi, ""), probe=GOOD_PROBE)
        self.assertEqual(got["gpu"].status, gd.WARN)
        self.assertIn("untested profile", got["gpu"].hint)
        self.assertEqual(got["hardware profile"].status, gd.WARN)
        self.assertEqual(gd.summarize(list(got.values()))["fail"], 0)

    def test_second_card_failure_is_reported_with_its_index(self):
        fx = Fixture()
        bad = device(1, matmul_ok=False, alloc_ok=False, error="OutOfMemoryError: CUDA out of memory")
        got = collect(fx, fx.options(), smi=(0, SMI_DUAL_3090, ""), probe=probe_for(device(0), bad))
        self.assertEqual(got["cuda matmul"].status, gd.FAIL)
        self.assertIn("cuda:1", got["cuda matmul"].detail)
        self.assertNotIn("cuda:0", got["cuda matmul"].detail)

    def test_bf16_missing_on_one_card_names_it_and_hints_ampere_and_ada(self):
        fx = Fixture()
        probe = probe_for(device(0), device(1, bf16=False))
        got = collect(fx, fx.options(require_gpu=True), smi=(0, SMI_DUAL_3090, ""), probe=probe)
        self.assertEqual(got["bf16"].status, gd.FAIL)
        self.assertIn("cuda:1", got["bf16"].detail)
        self.assertIn("RTX 3090", got["bf16"].hint)
        self.assertIn("RTX 4090", got["bf16"].hint)

    def test_torch_seeing_fewer_cards_than_smi_warns(self):
        fx = Fixture()
        got = collect(fx, fx.options(), smi=(0, SMI_DUAL_3090, ""), probe=probe_for(device(0)))
        self.assertEqual(got["torch devices"].status, gd.WARN)
        self.assertIn("CUDA_VISIBLE_DEVICES", got["torch devices"].hint)

    def test_torch_seeing_fewer_cards_than_smi_fails_when_gpus_are_required(self):
        fx = Fixture()
        got = collect(
            fx, fx.options(require_gpu=True), smi=(0, SMI_DUAL_3090, ""), probe=probe_for(device(0))
        )
        self.assertEqual(got["torch devices"].status, gd.FAIL)

    def test_small_card_warns(self):
        fx = Fixture()
        got = collect(fx, fx.options(), smi=(0, "NVIDIA GeForce RTX 3090, 580.1, 12288, 8.6\n", ""), probe=GOOD_PROBE)
        self.assertEqual(got["gpu 0"].status, gd.WARN)
        self.assertIn("20 GiB", got["gpu 0"].hint)

    def test_compute_cap_query_falls_back_for_old_drivers(self):
        fx = Fixture()
        calls = []

        def _run(cmd, timeout=60, cwd=None):
            calls.append(cmd)
            if "compute_cap" in cmd[1]:
                return 2, "", 'Field "compute_cap" is not a valid field'
            return 0, SMI_OK, ""

        with mock.patch.object(gd, "run", _run), mock.patch.object(gd, "which", lambda n: "/usr/bin/" + n):
            checks, gpus = gd.check_nvidia_smi(fx.options())
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(gpus), 1)
        self.assertEqual(checks[0].status, gd.OK)


class GpuAbsent(unittest.TestCase):
    def run_absent(self, require):
        fx = Fixture()
        probe = dict(GOOD_PROBE, cuda_available=False, torch="2.14.1+cpu", torch_cuda_build=None)
        return collect(fx, fx.options(require_gpu=require), which_missing=("nvidia-smi",), probe=probe)

    def test_warns_without_require_gpu(self):
        got = self.run_absent(False)
        self.assertEqual(got["gpu"].status, gd.WARN)
        self.assertEqual(got["torch cuda"].status, gd.WARN)
        self.assertEqual(gd.summarize(list(got.values()))["exit_code"], 0)

    def test_fails_with_require_gpu(self):
        got = self.run_absent(True)
        self.assertEqual(got["gpu"].status, gd.FAIL)
        self.assertIn("nvidia-driver-580-open", got["gpu"].hint)
        self.assertEqual(gd.summarize(list(got.values()))["exit_code"], 1)

    def test_missing_venv_is_a_failure_and_skips_dependents(self):
        fx = Fixture(venv=False)
        got = collect(fx, fx.options(), which_missing=("nvidia-smi",))
        self.assertEqual(got["training venv"].status, gd.FAIL)
        self.assertEqual(got["torch"].status, gd.SKIP)
        self.assertEqual(got["ptr_py"].status, gd.SKIP)
        self.assertIn("make desktop-setup", got["training venv"].hint)


class BrokenTorch(unittest.TestCase):
    def test_import_error_fails(self):
        fx = Fixture()
        probe = {"python": "3.11.15", "torch_error": "ImportError: libcudnn.so.9", "transformers": "5.18.0",
                 "peft": "0.21.2", "trl": "1.14.1", "ptr_py_error": "ModuleNotFoundError: ptr_py"}
        got = collect(fx, fx.options(), probe=probe)
        self.assertEqual(got["torch"].status, gd.FAIL)
        self.assertEqual(got["ptr_py"].status, gd.FAIL)
        self.assertIn("maturin develop", got["ptr_py"].hint)

    def test_cuda_unavailable_with_driver_present(self):
        fx = Fixture()
        probe = dict(GOOD_PROBE, cuda_available=False)
        got = collect(fx, fx.options(require_gpu=True), probe=probe)
        self.assertEqual(got["nvidia-smi"].status, gd.OK)
        self.assertEqual(got["torch cuda"].status, gd.FAIL)
        self.assertEqual(gd.summarize(list(got.values()))["exit_code"], 1)

    def test_matmul_failure_fails(self):
        fx = Fixture()
        probe = dict(GOOD_PROBE, matmul_ok=False, matmul_error="RuntimeError: CUDA error: no kernel image")
        got = collect(fx, fx.options(), probe=probe)
        self.assertEqual(got["cuda matmul"].status, gd.FAIL)
        self.assertIn("no kernel image", got["cuda matmul"].detail)

    def test_nvidia_smi_failing_is_gpu_problem(self):
        fx = Fixture()
        got = collect(fx, fx.options(require_gpu=True), smi=(9, "", "NVIDIA-SMI has failed"), probe=None)
        self.assertEqual(got["gpu"].status, gd.FAIL)
        self.assertIn("has failed", got["gpu"].detail)

    def test_unprobeable_venv(self):
        fx = Fixture()
        got = collect(fx, fx.options(), probe=None)
        self.assertEqual(got["torch"].status, gd.FAIL)
        self.assertEqual(got["transformers"].status, gd.SKIP)


class Misc(unittest.TestCase):
    def test_rust_toolchain_missing_fails(self):
        fx = Fixture()
        got = collect(fx, fx.options(), probe=GOOD_PROBE, rustup="stable-x86_64 (default)\n")
        self.assertEqual(got["rust"].status, gd.FAIL)
        self.assertIn("1.99.0", got["rust"].hint)

    def test_hf_offline_is_skipped_and_online_failure_warns(self):
        fx = Fixture()
        self.assertEqual(gd.check_hf(fx.options()).status, gd.SKIP)
        online = gd.Options(root=fx.root, offline=False)
        with mock.patch.object(gd.urllib.request, "urlopen", side_effect=OSError("down")):
            self.assertEqual(gd.check_hf(online).status, gd.WARN)

    def test_main_writes_json_and_exit_code(self):
        fx = Fixture()
        out = fx.root / "out" / "doctor.json"
        with mock.patch.object(gd, "ROOT", fx.root), mock.patch.object(gd, "run", fake_run(probe=GOOD_PROBE)), \
                mock.patch.object(gd, "which", lambda n: "/usr/bin/" + n), \
                mock.patch.object(gd.shutil, "disk_usage", return_value=Usage(10**12, 0, 500 * 10**9)), \
                mock.patch.dict(os.environ, {"HF_HOME": str(fx.root / "hf")}), \
                mock.patch("builtins.print"):
            code = gd.main(["--offline", "--require-gpu", "--json", str(out)])
        data = json.loads(out.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(data["summary"]["fail"], 0)
        self.assertTrue(data["require_gpu"])
        self.assertIn("gpu", {c["name"] for c in data["checks"]})

    def test_hf_home_precedence(self):
        self.assertEqual(gd.hf_home({"HF_HOME": "/x"}), Path("/x"))
        self.assertEqual(gd.hf_home({"XDG_CACHE_HOME": "/c"}), Path("/c/huggingface"))


if __name__ == "__main__":
    unittest.main()
