#!/usr/bin/env python3
"""GPU desktop doctor: report and check that this machine can run PTR GPU experiments.

Standard library only. Anything that needs torch runs as a subprocess of the training
venv's python (`training/.venv/bin/python`), so this script itself imports nothing heavy
and works before the venv exists (it then says so and how to create it).

Every check returns one of `ok`, `warn`, `fail` (or `skip` when a prerequisite check
already failed, or `--offline` was given), plus a fix hint. The exit code is 1 when any
check is `fail`, otherwise 0. Without `--require-gpu` a missing or unusable GPU is a
warning, so the CPU smoke path can still be checked; with it, it is a failure.

    python3 scripts/gpu_doctor.py                  # report
    python3 scripts/gpu_doctor.py --require-gpu    # what `make desktop-setup` ends with
    python3 scripts/gpu_doctor.py --offline --json out.json

Every visible card is checked (name, memory, compute capability, free memory, a small torch
allocation and matmul on each device). RTX 3090 and RTX 4090 are the tested cards; any other card
is reported as an untested profile (a warning). The matching hardware profile file is printed:
hardware/rtx3090.toml, hardware/rtx4090.toml, or hardware/dual-rtx3090.toml for two 3090.

Beyond the GPU it checks what the OLMoE and M011 runs need: the `moe` extra pins (transformers,
peft, trl, datasets, accelerate against training/uv.lock), bubblewrap able to create a sandbox
(M011 scores HumanEval by running generated code and refuses without it), a card of at least
20 GiB (a failure with --require-gpu) and at least 22 GiB free on every visible card (the
estimated peaks are 17 to 21 GiB plus the CUDA context: `python -m ptr_training.moe.memory`).

`ptrctl doctor` checks no GPU; this is the GPU-side counterpart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 11)
MIN_FREE_GB = 60.0
# Minimum driver major version for the CUDA major version of the torch wheel.
MIN_DRIVER_FOR_CUDA = {13: 580, 12: 525}
MIN_VRAM_MIB = 20 * 1024
# Estimated peaks of the OLMoE SFT smoke and the M011 arms are about 17-21 GiB plus the CUDA
# context (ptr_training.moe.memory); a card with less free memory is likely shared.
MIN_FREE_VRAM_MIB = 22 * 1024
# Packages of the `moe` extra compared with training/uv.lock (torch is reported separately).
VERSIONED = ("transformers", "peft", "trl", "datasets", "accelerate")
HF_URL = "https://huggingface.co"
# bf16 tensor-core math needs compute capability 8.0 or newer: Ampere (RTX 3090 is sm_86) and
# Ada (RTX 4090 is sm_89) both qualify.
MIN_BF16_CAPABILITY = (8, 0)
# Hardware profiles the repo ships for the tested desktop cards. Anything else is reported as an
# untested profile (a warning, never a failure).
PROFILE_SINGLE = {"rtx3090": "hardware/rtx3090.toml", "rtx4090": "hardware/rtx4090.toml"}
PROFILE_DUAL_3090 = "hardware/dual-rtx3090.toml"
_CARD_PATTERNS = {
    "rtx3090": re.compile(r"\bRTX 3090\b(?! ?Ti)", re.IGNORECASE),
    "rtx4090": re.compile(r"\bRTX 4090\b(?! ?(D|Ti)\b)", re.IGNORECASE),
}

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# Run inside the training venv. Prints exactly one JSON object; each probe is isolated so
# a broken torch does not hide whether transformers or ptr_py import.
VENV_PROBE = r"""
import json, sys
from importlib import metadata
out = {"python": sys.version.split()[0]}
for name in ("transformers", "peft", "trl", "datasets", "accelerate"):
    try:
        out[name] = metadata.version(name)
    except Exception as exc:
        out[name] = None
        out[name + "_error"] = type(exc).__name__
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device_count"] = int(torch.cuda.device_count())
        devices = []
        for i in range(out["device_count"]):
            d = {"index": i, "name": torch.cuda.get_device_name(i)}
            try:
                d["capability"] = list(torch.cuda.get_device_capability(i))
                d["total_memory_mib"] = int(torch.cuda.get_device_properties(i).total_memory // (1024 * 1024))
                d["free_mib"] = int(torch.cuda.mem_get_info(i)[0] // (1024 * 1024))
                with torch.cuda.device(i):
                    d["bf16"] = bool(torch.cuda.is_bf16_supported())
                dev = "cuda:%d" % i
                a = torch.randn(256, 256, device=dev)
                d["matmul_ok"] = bool(torch.isfinite((a @ a).sum()).item())
                blob = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
                blob.fill_(1)
                torch.cuda.synchronize(i)
                d["alloc_ok"] = True
                del blob
            except Exception as exc:
                d.setdefault("matmul_ok", False)
                d["alloc_ok"] = False
                d["error"] = type(exc).__name__ + ": " + str(exc)[:200]
            devices.append(d)
        out["devices"] = devices
        # Device 0 keys kept for older consumers of this JSON.
        if devices:
            out["device_name"] = devices[0]["name"]
            out["bf16"] = devices[0].get("bf16", False)
            out["matmul_ok"] = devices[0].get("matmul_ok", False)
            if "error" in devices[0]:
                out["matmul_error"] = devices[0]["error"]
except Exception as exc:
    out["torch_error"] = type(exc).__name__ + ": " + str(exc)[:200]
try:
    import ptr_py
    out["ptr_py"] = getattr(ptr_py, "__version__", "importable")
except Exception as exc:
    out["ptr_py_error"] = type(exc).__name__ + ": " + str(exc)[:200]
print(json.dumps(out))
"""


@dataclass
class Check:
    """One diagnosed item: a status, what was seen, and what to do about it."""

    name: str
    status: str
    detail: str
    hint: str = ""


@dataclass
class Options:
    """Command-line options, as a typed bundle."""

    require_gpu: bool = False
    offline: bool = False
    root: Path = ROOT
    venv: Path | None = None
    min_free_gb: float = MIN_FREE_GB

    @property
    def venv_dir(self) -> Path:
        return self.venv if self.venv is not None else self.root / "training" / ".venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"


def run(cmd: list[str], timeout: float = 60, cwd: Path | None = None):
    """Run a command; return (returncode, stdout, stderr) or (None, "", reason) if it cannot start."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "", f"{type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout, proc.stderr


def which(name: str) -> str | None:
    return shutil.which(name)


def gpu_missing(opts: Options, detail: str, hint: str, name: str = "gpu") -> Check:
    """A missing GPU capability: a failure when required, otherwise a warning."""
    return Check(name, FAIL if opts.require_gpu else WARN, detail, hint)


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse `--query-gpu=name,driver_version,memory.total[,compute_cap] --format=csv,noheader,nounits`.

    The fourth column is optional (old drivers lack `compute_cap`); the result then has
    `compute_cap` None. Cards keep the order nvidia-smi printed them in (their index).
    """
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, driver = parts[0], parts[1]
        try:
            mem = int(float(parts[2]))
        except ValueError:
            continue
        cap = parse_capability(parts[3]) if len(parts) > 3 else None
        gpus.append({"index": len(gpus), "name": name, "driver": driver, "memory_mib": mem, "compute_cap": cap})
    return gpus


def parse_capability(text) -> tuple[int, int] | None:
    """`"8.6"` or `[8, 6]` to `(8, 6)`; None when it is not a capability."""
    if isinstance(text, (list, tuple)) and len(text) == 2:
        return int(text[0]), int(text[1])
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(text))
    return (int(match.group(1)), int(match.group(2))) if match else None


def card_kind(name: str) -> str | None:
    """`rtx3090` or `rtx4090` for the tested cards, None for any other name (including Ti/D variants)."""
    for kind, pattern in _CARD_PATTERNS.items():
        if pattern.search(name):
            return kind
    return None


def bf16_capable(capability: tuple[int, int] | None) -> bool | None:
    """Whether a compute capability supports bf16 tensor math; None when unknown."""
    return None if capability is None else capability >= MIN_BF16_CAPABILITY


def select_profile(names: list[str]) -> tuple[str | None, str]:
    """The hardware profile file matching the visible cards, and a one-line description.

    One 3090 or one 4090 select their single-card file; two 3090 select the dual profile
    (their single-card file stays valid for single-GPU runs on one of them). Any other mix is
    an untested profile and returns None.
    """
    kinds = [card_kind(n) for n in names]
    if not names or any(k is None for k in kinds):
        return None, "untested profile: " + (", ".join(sorted(set(names))) or "no GPU")
    if set(kinds) == {"rtx3090"} and len(kinds) == 2:
        return PROFILE_DUAL_3090, f"{PROFILE_DUAL_3090} (2x RTX 3090, PCIe only); single-GPU runs: {PROFILE_SINGLE['rtx3090']}"
    if len(set(kinds)) == 1 and len(kinds) == 1:
        return PROFILE_SINGLE[kinds[0]], PROFILE_SINGLE[kinds[0]]
    if len(set(kinds)) == 1:
        return PROFILE_SINGLE[kinds[0]], f"{PROFILE_SINGLE[kinds[0]]} per card ({len(kinds)}x, no multi-GPU profile)"
    return None, "untested profile: mixed cards " + ", ".join(sorted(set(names)))


def check_nvidia_smi(opts: Options) -> tuple[list[Check], list[dict]]:
    install = (
        "install driver 580: sudo apt install nvidia-driver-580-open, reboot "
        "(see docs/DESKTOP_SETUP.md, Secure Boot needs MOK enrolment)"
    )
    if which("nvidia-smi") is None:
        return [gpu_missing(opts, "nvidia-smi not found", install)], []
    query = "--query-gpu=name,driver_version,memory.total,compute_cap"
    code, out, err = run(["nvidia-smi", query, "--format=csv,noheader,nounits"], timeout=30)
    if code != 0:
        # Drivers older than 510 do not know compute_cap; retry without it before giving up.
        code, out, err = run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            timeout=30,
        )
    if code != 0:
        reason = (err or out).strip().splitlines()[:1]
        return [
            gpu_missing(
                opts,
                "nvidia-smi failed: " + (reason[0] if reason else f"exit {code}"),
                "driver not loaded: reboot, check `sudo mokutil --sb-state` and `dmesg | grep -i nvidia`; " + install,
            )
        ], []
    gpus = parse_nvidia_smi(out)
    if not gpus:
        return [gpu_missing(opts, "nvidia-smi reported no GPU", install)], []
    checks = [Check("nvidia-smi", OK, f"driver {gpus[0]['driver']}")]
    tested = all(card_kind(g["name"]) for g in gpus)
    for g in gpus:
        cap = g["compute_cap"]
        detail = f"{g['name']}, {g['memory_mib']} MiB" + (f", sm_{cap[0]}{cap[1]}" if cap else "")
        small = g["memory_mib"] < MIN_VRAM_MIB
        status = WARN if small or not card_kind(g["name"]) else OK
        if small and opts.require_gpu:
            status = FAIL
        hint = ""
        if small:
            hint = "less than 20 GiB: the OLMoE SFT smoke config targets a 24 GiB card"
        elif not card_kind(g["name"]):
            hint = "untested profile: tested cards are RTX 3090 and RTX 4090; record the real GPU in the run manifest"
        checks.append(Check(f"gpu {g['index']}", status, detail, hint))
    names = ", ".join(sorted({g["name"] for g in gpus}))
    checks.append(
        Check(
            "gpu",
            OK if tested and all(g["memory_mib"] >= MIN_VRAM_MIB for g in gpus) else WARN,
            f"{len(gpus)}x {names}, {gpus[0]['memory_mib']} MiB" + (" each" if len(gpus) > 1 else ""),
            "" if tested else "untested profile: tested cards are RTX 3090 and RTX 4090; "
            "record the real GPU in the run manifest",
        )
    )
    profile, text = select_profile([g["name"] for g in gpus])
    if profile is None:
        checks.append(
            Check(
                "hardware profile",
                WARN,
                text,
                "no hardware/*.toml matches; pick the nearest by hand and record the real GPU in the run manifest",
            )
        )
    elif not (opts.root / profile).is_file():
        checks.append(Check("hardware profile", WARN, f"{text} (file missing)", f"restore {profile}"))
    else:
        checks.append(Check("hardware profile", OK, text))
    return checks, gpus


def check_python() -> Check:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        return Check("python", OK, text)
    return Check("python", FAIL, f"{text}, need >= 3.11", "sudo apt install python3.11 or run uv python install 3.11")


def check_uv() -> Check:
    path = which("uv")
    if path:
        return Check("uv", OK, path)
    return Check(
        "uv", FAIL, "not found", "curl -LsSf https://astral.sh/uv/install.sh | sh (make desktop-setup does this)"
    )


def check_venv(opts: Options) -> Check:
    if opts.venv_python.exists():
        return Check("training venv", OK, str(opts.venv_dir))
    return Check(
        "training venv",
        FAIL,
        f"{opts.venv_python} missing",
        "make desktop-setup (uv sync --project training --extra moe --locked)",
    )


def lock_versions(root: Path) -> dict[str, list[str]]:
    """Versions of the packages we compare, as training/uv.lock resolves them for the `moe`
    extra of the root package `ptr-training`.

    The lock holds several versions of some packages (the `cpt` and `moe` extras fork the
    resolution, e.g. datasets 4.8.4 and 5.0.1), so collecting every occurrence would accept a
    version the moe extra does not allow. The root package's `optional-dependencies.moe`
    entries name the version when the name is ambiguous; an entry without one means the lock
    has a single version of that package. A lock without that table (older format) falls
    back to every occurrence."""
    path = root / "training" / "uv.lock"
    found: dict[str, list[str]] = {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return found
    wanted = ("torch", *VERSIONED)
    every: dict[str, list[str]] = {}
    for pkg in data.get("package", []):
        if pkg.get("name") in wanted:
            every.setdefault(pkg["name"], []).append(pkg.get("version", ""))
    root_pkg = next((p for p in data.get("package", []) if p.get("name") == "ptr-training"), None)
    moe = (root_pkg or {}).get("optional-dependencies", {}).get("moe")
    if not moe:
        return every
    for dep in moe:
        name = dep.get("name")
        if name not in wanted:
            continue
        if dep.get("version"):
            found[name] = [dep["version"]]
        elif len(set(every.get(name, []))) == 1:
            found[name] = [every[name][0]]
    return found


def probe_venv(opts: Options) -> tuple[dict | None, str]:
    code, out, err = run([str(opts.venv_python), "-c", VENV_PROBE], timeout=180)
    if code != 0:
        return None, (err or out).strip().splitlines()[-1] if (err or out).strip() else f"exit {code}"
    try:
        return json.loads(out.strip().splitlines()[-1]), ""
    except (ValueError, IndexError):
        return None, "probe printed no JSON"


def base_version(v: str) -> str:
    return v.split("+")[0]


def check_torch(opts: Options, probe: dict | None, why: str, gpus: list[dict]) -> list[Check]:
    if probe is None:
        return [Check("torch", FAIL, f"venv python could not be probed: {why}", "make desktop-setup")]
    if "torch_error" in probe:
        return [
            Check(
                "torch",
                FAIL,
                "import failed: " + probe["torch_error"],
                "reinstall: uv sync --project training --extra moe --locked --reinstall-package torch",
            )
        ]
    checks = [Check("torch", OK, probe["torch"])]
    cuda_build = probe.get("torch_cuda_build")
    if not probe.get("cuda_available"):
        if cuda_build is None:
            hint = (
                "this is a CPU-only torch wheel; resync from the lock on the desktop: "
                "uv sync --project training --extra moe --locked"
            )
        else:
            hint = "driver/torch mismatch or no GPU: compare `nvidia-smi` driver with the CUDA build above"
        checks.append(
            gpu_missing(opts, f"torch.cuda.is_available() is False (CUDA build {cuda_build})", hint, "torch cuda")
        )
        return checks
    devices = probe.get("devices") or [
        {"index": 0, "name": probe.get("device_name", "?"), "bf16": probe.get("bf16"),
         "matmul_ok": probe.get("matmul_ok"), "alloc_ok": probe.get("matmul_ok"),
         **({"error": probe["matmul_error"]} if "matmul_error" in probe else {})}
    ]
    names = ", ".join(d["name"] for d in devices)
    checks.append(
        Check("torch cuda", OK, f"available, CUDA build {cuda_build}, {len(devices)} device(s): {names}")
    )
    if gpus and len(devices) < len(gpus):
        checks.append(
            Check(
                "torch devices",
                FAIL if opts.require_gpu else WARN,
                f"torch sees {len(devices)} of {len(gpus)} GPUs",
                "check CUDA_VISIBLE_DEVICES; the dual-GPU profile needs both cards visible",
            )
        )
    major = int(str(cuda_build).split(".")[0]) if cuda_build else 0
    need = MIN_DRIVER_FOR_CUDA.get(major)
    if need and gpus:
        try:
            have = int(gpus[0]["driver"].split(".")[0])
        except ValueError:
            have = None
        if have is not None and have < need:
            checks.append(
                Check(
                    "driver vs torch",
                    FAIL,
                    f"driver {gpus[0]['driver']} too old for CUDA {major} wheels (need >= {need})",
                    "sudo apt install nvidia-driver-580-open and reboot",
                )
            )
    no_bf16 = [d for d in devices if not d.get("bf16")]
    if not no_bf16:
        checks.append(Check("bf16", OK, "supported" + (f" on all {len(devices)} devices" if len(devices) > 1 else "")))
    else:
        checks.append(
            Check(
                "bf16",
                FAIL if opts.require_gpu else WARN,
                "not supported on " + ", ".join(f"cuda:{d['index']} {d['name']}" for d in no_bf16),
                "bf16 needs compute capability >= 8.0; RTX 3090 (Ampere, sm_86) and RTX 4090 "
                "(Ada, sm_89) both support it, so check the driver and the torch CUDA build",
            )
        )
    known = [d for d in devices if d.get("free_mib") is not None]
    if known:
        low = [d for d in known if d["free_mib"] < MIN_FREE_VRAM_MIB]
        if low:
            checks.append(
                Check(
                    "free vram",
                    WARN,
                    "; ".join(f"cuda:{d['index']} has {d['free_mib']} MiB free" for d in low),
                    "another process holds memory on that card (nvidia-smi lists it); the OLMoE runs "
                    "are estimated at 17-21 GiB plus the CUDA context (python -m ptr_training.moe.memory)",
                )
            )
        else:
            checks.append(
                Check("free vram", OK, ", ".join(f"cuda:{d['index']} {d['free_mib']} MiB" for d in known))
            )
    bad = [d for d in devices if not (d.get("matmul_ok") and d.get("alloc_ok", True))]
    if not bad:
        checks.append(Check("cuda matmul", OK, f"256x256 float32 finite and 64 MiB allocation on {len(devices)} device(s)"))
    else:
        reason = "; ".join(
            f"cuda:{d['index']} {d.get('error') or d.get('matmul_error') or 'result not finite'}" for d in bad
        )
        checks.append(
            Check(
                "cuda matmul",
                FAIL,
                reason,
                "driver/torch mismatch or GPU in a bad state: reboot, then rerun; see docs/DESKTOP_SETUP.md",
            )
        )
    return checks


def check_versions(opts: Options, probe: dict | None) -> list[Check]:
    if probe is None:
        return [Check(n, SKIP, "venv not probed") for n in VERSIONED]
    locked = lock_versions(opts.root)
    checks = []
    for name in VERSIONED:
        have = probe.get(name)
        want = locked.get(name, [])
        if have is None:
            checks.append(Check(name, FAIL, "not installed", "uv sync --project training --extra moe --locked"))
        elif not want:
            checks.append(Check(name, WARN, f"{have} (training/uv.lock unreadable or has no entry)"))
        elif base_version(have) in want:
            checks.append(Check(name, OK, f"{have} (lock: {', '.join(sorted(set(want)))})"))
        else:
            checks.append(
                Check(
                    name,
                    WARN,
                    f"{have}, lock has {', '.join(sorted(set(want)))}",
                    "uv sync --project training --extra moe --locked",
                )
            )
    return checks


def check_ptr_py(probe: dict | None) -> Check:
    hint = (
        "uv pip install --python training/.venv/bin/python maturin; with the venv active: "
        "maturin develop --locked -m bindings/ptr-py/Cargo.toml (make desktop-setup does this)"
    )
    if probe is None:
        return Check("ptr_py", SKIP, "venv not probed")
    if "ptr_py_error" in probe:
        return Check("ptr_py", FAIL, probe["ptr_py_error"], hint)
    return Check("ptr_py", OK, str(probe["ptr_py"]))


def bwrap_probe_command(exe: str) -> list[str]:
    """A minimal sandbox like the M011 HumanEval one: only /usr, fresh /proc and /dev, every
    namespace unshared (no network)."""
    cmd = [exe, "--ro-bind", "/usr", "/usr"]
    for top in ("bin", "lib", "lib64", "sbin"):
        cmd += ["--symlink", f"usr/{top}", f"/{top}"]
    return cmd + ["--proc", "/proc", "--dev", "/dev", "--unshare-all", "--die-with-parent",
                  "--", "/usr/bin/true"]


def check_bwrap(opts: Options) -> Check:
    """M011 scores HumanEval by running model-generated code; without a working bubblewrap
    sandbox the entrypoint refuses to start (unless PTR_M011_ALLOW_UNSANDBOXED=1)."""
    hint = (
        "sudo apt install bubblewrap; if it is installed but cannot sandbox, check "
        "`sysctl kernel.apparmor_restrict_unprivileged_userns` and the bwrap AppArmor profile "
        "(M011 refuses to score HumanEval without it)"
    )
    exe = which("bwrap")
    if exe is None:
        return gpu_missing(opts, "bwrap not found", hint, "bubblewrap")
    code, out, err = run(bwrap_probe_command(exe), timeout=20)
    if code == 0:
        return Check("bubblewrap", OK, f"{exe} creates a sandbox (no network)")
    reason = (err or out).strip().splitlines()[:1]
    return gpu_missing(
        opts, f"{exe} cannot create a sandbox: " + (reason[0] if reason else f"exit {code}"), hint,
        "bubblewrap",
    )


def nearest_existing(path: Path) -> Path:
    path = path.expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def hf_home(env: dict | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"])
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def check_disk(opts: Options) -> list[Check]:
    checks = []
    for label, path in (("disk (repo)", opts.root), ("disk (HF cache)", hf_home())):
        target = nearest_existing(path)
        try:
            free = shutil.disk_usage(target).free / 1e9
        except OSError as exc:
            checks.append(Check(label, WARN, f"cannot stat {target}: {exc}"))
            continue
        detail = f"{free:.0f} GB free at {path}"
        if free < opts.min_free_gb:
            checks.append(
                Check(
                    label,
                    WARN,
                    detail,
                    f"below {opts.min_free_gb:.0f} GB: model and run artifacts need room; "
                    "free space or set HF_HOME to a bigger disk",
                )
            )
        else:
            checks.append(Check(label, OK, detail))
    return checks


def check_hf(opts: Options) -> Check:
    if opts.offline:
        return Check("hf hub", SKIP, "skipped (--offline)")
    try:
        with urllib.request.urlopen(HF_URL, timeout=5) as resp:  # noqa: S310 fixed https URL
            return Check("hf hub", OK, f"{HF_URL} reachable (HTTP {resp.status})")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Check(
            "hf hub",
            WARN,
            f"{HF_URL} unreachable: {exc}",
            "check network/proxy; the first OLMoE download needs it (use --offline to skip)",
        )


def pinned_channel(root: Path) -> str | None:
    try:
        data = tomllib.loads((root / "rust-toolchain.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return data.get("toolchain", {}).get("channel")


def check_rust(opts: Options) -> Check:
    channel = pinned_channel(opts.root)
    if channel is None:
        return Check("rust", WARN, "rust-toolchain.toml unreadable")
    hint = (
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh, then "
        f"rustup toolchain install {channel} --profile minimal -c rustfmt -c clippy"
    )
    if which("rustup") is None:
        return Check("rust", FAIL, f"rustup not found (pinned {channel})", hint)
    code, out, _ = run(["rustup", "toolchain", "list"], timeout=30)
    if code != 0:
        return Check("rust", FAIL, "rustup toolchain list failed", hint)
    installed = [line.split()[0] for line in out.splitlines() if line.strip()]
    if any(t == channel or t.startswith(channel + "-") for t in installed):
        return Check("rust", OK, f"toolchain {channel} installed")
    return Check("rust", FAIL, f"toolchain {channel} not installed (have: {', '.join(installed) or 'none'})", hint)


def collect(opts: Options) -> list[Check]:
    checks: list[Check] = []
    smi_checks, gpus = check_nvidia_smi(opts)
    checks += smi_checks
    checks.append(check_python())
    checks.append(check_uv())
    venv = check_venv(opts)
    checks.append(venv)
    probe, why = (None, "venv missing")
    if venv.status == OK:
        probe, why = probe_venv(opts)
    if venv.status == OK:
        checks += check_torch(opts, probe, why, gpus)
    else:
        checks.append(Check("torch", SKIP, "no training venv"))
    checks += check_versions(opts, probe)
    checks.append(check_bwrap(opts))
    checks.append(check_ptr_py(probe))
    checks += check_disk(opts)
    checks.append(check_hf(opts))
    checks.append(check_rust(opts))
    return checks


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{'check'.ljust(width)}  status  detail", f"{'-' * width}  ------  ------"]
    for c in checks:
        lines.append(f"{c.name.ljust(width)}  {c.status.upper().ljust(6)}  {c.detail}")
        if c.hint and c.status in (WARN, FAIL):
            lines.append(f"{' ' * width}          fix: {c.hint}")
    return "\n".join(lines)


def summarize(checks: list[Check]) -> dict:
    counts = {s: sum(1 for c in checks if c.status == s) for s in (OK, WARN, FAIL, SKIP)}
    return {**counts, "exit_code": 1 if counts[FAIL] else 0}


def parse_args(argv: list[str] | None = None) -> tuple[Options, Path | None]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--require-gpu", action="store_true", help="a missing/unusable GPU is a failure")
    p.add_argument("--offline", action="store_true", help="skip the Hugging Face Hub reachability check")
    p.add_argument("--json", type=Path, metavar="PATH", help="also write the results as JSON")
    p.add_argument("--venv", type=Path, help="training venv (default: training/.venv)")
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    a = p.parse_args(argv)
    return Options(a.require_gpu, a.offline, ROOT, a.venv, a.min_free_gb), a.json


def main(argv: list[str] | None = None) -> int:
    opts, json_path = parse_args(argv)
    checks = collect(opts)
    print(render(checks))
    summary = summarize(checks)
    print(
        f"\n{summary[OK]} ok, {summary[WARN]} warn, {summary[FAIL]} fail, {summary[SKIP]} skipped"
        f" (require_gpu={opts.require_gpu})"
    )
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checks": [asdict(c) for c in checks],
            "summary": summary,
            "require_gpu": opts.require_gpu,
            "offline": opts.offline,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
