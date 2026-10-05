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
        if venv:
            py = self.root / "training" / ".venv" / "bin" / "python"
            py.parent.mkdir(parents=True)
            py.write_text("")

    def options(self, **kw):
        return gd.Options(root=self.root, offline=True, **kw)


def fake_run(smi=(0, SMI_OK, ""), probe=None, rustup="stable-x86_64-unknown-linux-gnu (default)\n1.99.0-x86_64-unknown-linux-gnu\n"):
    def _run(cmd, timeout=60, cwd=None):
        if cmd[0] == "nvidia-smi":
            return smi
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
            [{"name": "NVIDIA GeForce RTX 4090", "driver": "580.65.06", "memory_mib": 24564}],
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
