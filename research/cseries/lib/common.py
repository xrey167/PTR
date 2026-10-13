"""Pieces every stage-A harness shares: the environment check, the environment record, seeding, per mille and the
torch-free audit of the plateau protocol.

The harnesses use the locked environment of research/consciousness (torch 2.7.1, CPU, no numpy). The environment lives outside
the repository (the runner refuses a repository that holds git-ignored files); see the preregistration, section 8."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

PINNED_TORCH = "2.7.1"
PINNED_PYTHON = (3, 11)
LOCK_DIR = "research/consciousness"  # the locked environment (pyproject.toml, uv.lock) is C001's; it is only read here


def environment_problems(config_path: Path) -> list[str]:
    """What is missing from the environment, before anything trains. The harness refuses (exit 2) rather than crash with
    Python's exit 1, which the runner would read as a falsification."""
    problems = []
    if sys.version_info[:2] != PINNED_PYTHON:
        problems.append(f"python {sys.version_info[0]}.{sys.version_info[1]}, pinned {PINNED_PYTHON[0]}.{PINNED_PYTHON[1]}")
    try:
        import torch
    except ImportError as error:
        problems.append(f"torch cannot be imported: {error}")
    else:
        if not torch.__version__.startswith(PINNED_TORCH):
            problems.append(f"torch {torch.__version__}, pinned {PINNED_TORCH}")
    if not config_path.is_file():
        problems.append(f"{config_path} is missing")
    return problems


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def environment_record(root: Path, torch) -> dict:
    """What the result was computed with, recorded in every result so that the aggregator can refuse a mix. The runner records only
    the first token of the command (env), so the interpreter, the torch build, the CPU and the lock files are written here."""
    cpu_model, flags = "unknown", ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name") and cpu_model == "unknown":
                cpu_model = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not flags:
                flags = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "cpu_model": cpu_model,
        "cpu_flags_sha256": hashlib.sha256(flags.encode()).hexdigest(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "uv_lock_sha256": file_sha256(root / LOCK_DIR / "uv.lock"),
        "pyproject_sha256": file_sha256(root / LOCK_DIR / "pyproject.toml"),
    }


def arm_seed(experiment: str, seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{experiment}:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def permille(num: int, den: int) -> int:
    return (1000 * num) // den


def plateau_reached(curve: list, patience: int, min_delta_permille: int) -> bool:
    """Whether a validation curve (accuracy per mille at each evaluation) contains `patience` consecutive evaluations without an
    improvement of more than `min_delta_permille` over the best so far: an independent recomputation of the stopping rule of
    `train.train_to_plateau`, used to audit the convergence flag the harness records."""
    best, since = -1, 0
    for value in curve:
        if value > best + min_delta_permille:
            best, since = value, 0
        else:
            since += 1
        if since >= patience:
            return True
    return False


def param_mismatch_arms(counts: dict, reference: str, tolerance_permille: int) -> int:
    """Arms whose parameter count differs from the reference arm's by more than the tolerance (per mille of the reference)."""
    ref = counts[reference]
    return sum(1 for arm, n in counts.items() if arm != reference and 1000 * abs(n - ref) > tolerance_permille * ref)


def flop_mismatch_arms(flops: dict, reference: str, tolerance_permille: int) -> int:
    """Arms whose analytic FLOP count differs from the reference arm's by more than the tolerance (per mille of the reference)."""
    ref = flops[reference]
    return sum(1 for arm, n in flops.items() if arm != reference and 1000 * abs(n - ref) > tolerance_permille * ref)


def holds(result: dict, rule: str) -> bool:
    left, op, right = rule.split(" ")
    a, b = result[left], int(right)
    return a >= b if op == ">=" else a <= b


def judge(result: dict, coverage: list[str], falsifiers: list[str]) -> tuple[list[str], list[str]]:
    """(coverage rules that fail, falsifier rules that fail)."""
    return [r for r in coverage if not holds(result, r)], [r for r in falsifiers if not holds(result, r)]


def counter_rules(counters: tuple) -> list[str]:
    return [f"{counter} <= 0" for counter in counters]
