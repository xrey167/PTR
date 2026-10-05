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

`ptrctl doctor` checks no GPU; this is the GPU-side counterpart.
"""

from __future__ import annotations

import argparse
import json
import os
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
HF_URL = "https://huggingface.co"

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# Run inside the training venv. Prints exactly one JSON object; each probe is isolated so
# a broken torch does not hide whether transformers or ptr_py import.
VENV_PROBE = r"""
import json, sys
from importlib import metadata
out = {"python": sys.version.split()[0]}
for name in ("transformers", "peft", "trl"):
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
        out["device_name"] = torch.cuda.get_device_name(0)
        out["bf16"] = bool(torch.cuda.is_bf16_supported())
        try:
            a = torch.randn(256, 256, device="cuda")
            out["matmul_ok"] = bool(torch.isfinite((a @ a).sum()).item())
            torch.cuda.synchronize()
        except Exception as exc:
            out["matmul_ok"] = False
            out["matmul_error"] = type(exc).__name__ + ": " + str(exc)[:200]
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
    """Parse `--query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits`."""
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
        gpus.append({"name": name, "driver": driver, "memory_mib": mem})
    return gpus


def check_nvidia_smi(opts: Options) -> tuple[list[Check], dict | None]:
    install = (
        "install driver 580: sudo apt install nvidia-driver-580-open, reboot "
        "(see docs/DESKTOP_SETUP.md, Secure Boot needs MOK enrolment)"
    )
    if which("nvidia-smi") is None:
        return [gpu_missing(opts, "nvidia-smi not found", install)], None
    code, out, err = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
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
        ], None
    gpus = parse_nvidia_smi(out)
    if not gpus:
        return [gpu_missing(opts, "nvidia-smi reported no GPU", install)], None
    g = gpus[0]
    checks = [
        Check("nvidia-smi", OK, f"driver {g['driver']}"),
        Check(
            "gpu",
            OK if "4090" in g["name"] else WARN,
            f"{g['name']}, {g['memory_mib']} MiB x{len(gpus)}",
            "" if "4090" in g["name"] else "expected an RTX 4090 (hardware/rtx4090.toml); "
            "record the real GPU in the run manifest",
        ),
    ]
    if g["memory_mib"] < MIN_VRAM_MIB:
        checks[-1] = Check(
            "gpu",
            WARN,
            f"{g['name']}, {g['memory_mib']} MiB",
            "less than 20 GiB: the OLMoE SFT smoke config targets a 24 GiB card",
        )
    return checks, g


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
    """Versions of the packages we compare, as locked in training/uv.lock (several per name)."""
    path = root / "training" / "uv.lock"
    found: dict[str, list[str]] = {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return found
    for pkg in data.get("package", []):
        if pkg.get("name") in ("torch", "transformers", "peft", "trl"):
            found.setdefault(pkg["name"], []).append(pkg.get("version", ""))
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


def check_torch(opts: Options, probe: dict | None, why: str, gpu: dict | None) -> list[Check]:
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
    checks.append(Check("torch cuda", OK, f"available, CUDA build {cuda_build}, {probe.get('device_name', '?')}"))
    major = int(str(cuda_build).split(".")[0]) if cuda_build else 0
    need = MIN_DRIVER_FOR_CUDA.get(major)
    if need and gpu:
        try:
            have = int(gpu["driver"].split(".")[0])
        except ValueError:
            have = None
        if have is not None and have < need:
            checks.append(
                Check(
                    "driver vs torch",
                    FAIL,
                    f"driver {gpu['driver']} too old for CUDA {major} wheels (need >= {need})",
                    "sudo apt install nvidia-driver-580-open and reboot",
                )
            )
    checks.append(
        Check("bf16", OK, "supported")
        if probe.get("bf16")
        else Check("bf16", FAIL if opts.require_gpu else WARN, "not supported", "RTX 4090 supports bf16; check driver")
    )
    if probe.get("matmul_ok"):
        checks.append(Check("cuda matmul", OK, "256x256 float32 finite"))
    else:
        checks.append(
            Check(
                "cuda matmul",
                FAIL,
                probe.get("matmul_error", "result not finite"),
                "driver/torch mismatch or GPU in a bad state: reboot, then rerun; see docs/DESKTOP_SETUP.md",
            )
        )
    return checks


def check_versions(opts: Options, probe: dict | None) -> list[Check]:
    if probe is None:
        return [Check(n, SKIP, "venv not probed") for n in ("transformers", "peft", "trl")]
    locked = lock_versions(opts.root)
    checks = []
    for name in ("transformers", "peft", "trl"):
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
    smi_checks, gpu = check_nvidia_smi(opts)
    checks += smi_checks
    checks.append(check_python())
    checks.append(check_uv())
    venv = check_venv(opts)
    checks.append(venv)
    probe, why = (None, "venv missing")
    if venv.status == OK:
        probe, why = probe_venv(opts)
    if venv.status == OK:
        checks += check_torch(opts, probe, why, gpu)
    else:
        checks.append(Check("torch", SKIP, "no training venv"))
    checks += check_versions(opts, probe)
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
