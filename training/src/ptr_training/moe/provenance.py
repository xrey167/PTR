"""Run provenance shared by the `trl-moe` pipeline and the M011 entrypoint.

Records what a run needs to be reproduced or audited: the git commit and whether the tracked
tree was dirty, the installed package versions, the host, and the compute device (visible GPUs,
their names, capability and memory, the driver, the CUDA/cuDNN builds, the determinism flags).
Standard library only at import time; torch is touched only when it is installed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

PACKAGES = (
    "torch", "transformers", "trl", "peft", "datasets", "accelerate", "tokenizers",
    "huggingface_hub", "safetensors", "numpy", "ptr_py",
)
# Environment variables that decide which GPU a run lands on or how it is launched; their
# values are recorded (they hold no secret).
DEVICE_ENV = (
    "CUDA_VISIBLE_DEVICES", "PTR_CUDA_DEVICE", "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG", "WORLD_SIZE", "LOCAL_RANK", "RANK",
)
GPU_NONDETERMINISM_NOTE = (
    "CUDA runs are not expected to be bit-reproducible: with torch's default (deterministic "
    "algorithms off) OLMoE's expert loop (index_add_, and the backward of its token "
    "gathers) accumulates with CUDA atomics in a non-deterministic order; not measured here. "
    "CPU runs with the same seed reproduce (tested on the tiny model)"
)


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(root: Path = ROOT) -> dict:
    """{commit, dirty, tracked_diff_sha256}; values are None when git is unavailable.

    `dirty` considers tracked files only (untracked run outputs do not count), like the run
    manifest's `git_dirty`; `tracked_diff_sha256` identifies the uncommitted tracked changes.
    """
    head = _git(["rev-parse", "HEAD"], root)
    if head is None or head.returncode != 0:
        return {"commit": None, "dirty": None, "tracked_diff_sha256": None}
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    diff = _git(["diff", "--binary", "HEAD"], root)
    ok = status is not None and status.returncode == 0 and diff is not None
    return {
        "commit": head.stdout.decode().strip() or None,
        "dirty": bool(status.stdout.strip()) if ok else None,
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if ok else None,
    }


def package_versions(names=PACKAGES) -> dict:
    out = {"python": platform.python_version(), "python_implementation": sys.implementation.name}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def host_info() -> dict:
    ram_gib = None
    try:
        ram_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_gib": ram_gib,
    }


def nvidia_driver() -> str | None:
    """Driver version from nvidia-smi (None without it); never raises."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        res = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [x.strip() for x in res.stdout.splitlines() if x.strip()]
    return lines[0] if res.returncode == 0 and lines else None


def device_info(device: str | None) -> dict:
    """The resolved run device plus every CUDA device torch can see (index as torch numbers
    them, i.e. after CUDA_VISIBLE_DEVICES), the build versions and the determinism flags."""
    info: dict = {
        "device": device,
        "env": {k: os.environ.get(k) for k in DEVICE_ENV},
        "torch_available": False,
    }
    try:
        import torch
    except ImportError:
        return info
    info.update(
        torch_available=True,
        torch_cuda_build=torch.version.cuda,
        cudnn_version=(
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        cuda_available=bool(torch.cuda.is_available()),
        torch_num_threads=torch.get_num_threads(),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
    )
    gpus = []
    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_mib": int(props.total_memory // 2**20),
                "uuid": str(getattr(props, "uuid", "")) or None,
            })
        info["driver"] = nvidia_driver()
        info["current_device"] = torch.cuda.current_device()
    info["gpus"] = gpus
    info["bit_reproducible"] = device == "cpu"
    if device == "cuda":
        info["nondeterminism_note"] = GPU_NONDETERMINISM_NOTE
    return info


def collect(device: str | None, root: Path = ROOT) -> dict:
    """Everything above in one dict (`git`, `versions`, `host`, `device`)."""
    return {
        "git": git_state(root),
        "versions": package_versions(),
        "host": host_info(),
        "device": device_info(device),
    }


class DeviceSelectionError(RuntimeError):
    """PTR_CUDA_DEVICE is malformed or contradicts CUDA_VISIBLE_DEVICES."""


def apply_ptr_cuda_device(env: dict | None = None) -> str | None:
    """OD-3 GPU pinning: when `PTR_CUDA_DEVICE` (one card index) is set, make it the only
    visible card by setting `CUDA_VISIBLE_DEVICES` to it. Must run before torch initialises
    CUDA. Returns the index (None when unset); refuses a malformed value or a different,
    already set `CUDA_VISIBLE_DEVICES` instead of guessing which one was meant."""
    env = os.environ if env is None else env
    value = env.get("PTR_CUDA_DEVICE")
    if value is None or value == "":
        return None
    value = value.strip()
    if not value.isdigit():
        raise DeviceSelectionError(f"PTR_CUDA_DEVICE must be one card index, got {value!r}")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", value):
        raise DeviceSelectionError(
            f"PTR_CUDA_DEVICE={value} contradicts CUDA_VISIBLE_DEVICES={visible}; set one of them"
        )
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if getattr(torch.cuda, "is_initialized", lambda: False)():
            raise DeviceSelectionError("CUDA is already initialised; PTR_CUDA_DEVICE came too late")
    env["CUDA_VISIBLE_DEVICES"] = value
    return value


class DistributedLaunchRefused(RuntimeError):
    """The code path has no distributed support; a multi-process launch would duplicate it."""


def refuse_distributed_launch(env: dict | None = None) -> None:
    """Raise when started by `torchrun` / `accelerate launch` with more than one process.

    Neither the M011 loop nor the trl-moe SFT loop initialises a process group or shards data:
    under a multi-process launcher every rank would train the same full run on `cuda:0` (the
    default device), duplicating the work and the output directory on one card. Two cards are
    used as two independent runs pinned with CUDA_VISIBLE_DEVICES instead.
    """
    env = os.environ if env is None else env
    try:
        world = int(env.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    if world > 1:
        raise DistributedLaunchRefused(
            f"WORLD_SIZE={world}: this entrypoint is single-process (no process group, no data "
            "sharding). Run one process per GPU with CUDA_VISIBLE_DEVICES=<index> instead "
            "(training/src/ptr_training/moe/README.md, 'Two GPUs')."
        )


__all__ = [
    "DeviceSelectionError",
    "DistributedLaunchRefused",
    "apply_ptr_cuda_device",
    "GPU_NONDETERMINISM_NOTE",
    "collect",
    "device_info",
    "git_state",
    "host_info",
    "package_versions",
    "refuse_distributed_launch",
]
