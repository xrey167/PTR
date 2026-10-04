"""End-to-end orchestration of the `trl-moe` backend.

build model -> routing snapshot -> SFT -> snapshot -> RL per algorithm (fresh SFT state,
seeded) -> snapshot -> gates -> metrics JSON. `run_config` is the direct entry point
(no manifest); `run` is what `ptr_training.run --execute` dispatches to.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from importlib import metadata
from pathlib import Path

from ptr_training.moe import config as moe_config
from ptr_training.moe import gates, routing_metrics
from ptr_training.moe.tasks import generate_tasks

EVIDENCE_CLASS = "pipeline-smoke"
EVIDENCE_NOTE = "mechanism check only, not evidence"
ROOT = Path(__file__).resolve().parents[4]


class MissingExtraError(RuntimeError):
    """The `moe` extra (torch, transformers, trl, ...) is not installed."""


def require_extra() -> None:
    import importlib.util

    missing = [m for m in ("torch", "transformers", "trl", "datasets", "peft", "accelerate")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise MissingExtraError(
            f"backend 'trl-moe' needs the optional 'moe' extra (missing: {', '.join(missing)}); "
            "install with: pip install -e 'training[moe]'"
        )


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "tokenizers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        out["ptr_py"] = metadata.version("ptr_py")
    except metadata.PackageNotFoundError:
        out["ptr_py"] = None
    return out


def hardware() -> dict:
    import torch

    ram_gib = None
    try:
        ram_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (ValueError, OSError, AttributeError):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "cpu_count": os.cpu_count(),
        "ram_gib": ram_gib,
        "gpu": gpu,
        "platform": platform.platform(),
    }


def _sanitize(obj, flags: list):
    """Replace non-finite floats with None (recording that it happened)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        flags.append(True)
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v, flags) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, flags) for v in obj]
    return obj


class _Timer:
    def __init__(self, sink: dict, name: str):
        self.sink, self.name = sink, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.sink[self.name] = round(time.perf_counter() - self.t0, 3)


def _gate_status(report: dict) -> dict:
    """Mark the gate report 'error' if any individual binding call failed."""
    if report.get("status") != "ok":
        return report
    errors = 0

    def walk(x):
        nonlocal errors
        if isinstance(x, dict):
            if x.get("status") == "error":
                errors += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(report["layers"])
    if errors:
        report["status"] = "error"
        report["failed_calls"] = errors
    return report


def execute(cfg: moe_config.MoeConfig, out_dir: Path, extra: dict | None = None) -> dict:
    require_extra()
    import torch

    from ptr_training.moe import model as model_mod
    from ptr_training.moe.rl import evaluate_policy, run_rl
    from ptr_training.moe.sft import run_sft

    torch.set_num_threads(os.cpu_count() or 1)
    timings: dict = {}
    t_start = time.perf_counter()
    k = cfg.model.top_k if cfg.model.kind == "tiny-random" else None

    with _Timer(timings, "build_model"):
        model, tokenizer = model_mod.build(cfg.model, cfg.lora)
        if k is None:
            k = int(getattr(model.config, "num_experts_per_tok"))
    if cfg.model.kind == "pretrained":
        model_mod.reset_requires_grad(model)

    seed = cfg.seed
    sft_tasks = generate_tasks(cfg.sft.examples, seed=seed)
    rl_n = max(64, cfg.rl.steps * cfg.rl.prompts_per_step * cfg.rl.mini_steps)
    rl_tasks = generate_tasks(rl_n, seed=seed + 1)
    eval_tasks = generate_tasks(32, seed=seed + 2)
    probe_texts = [t.prompt + t.answer for t in eval_tasks[:8]]
    eval_new = cfg.rl.max_completion_length

    snapshots: dict = {}
    routing: dict = {}
    evals: dict = {}

    def snap(name: str):
        s = model_mod.routing_snapshot(model, tokenizer, probe_texts, cfg.probe_tokens)
        snapshots[name] = s
        routing[name] = {"per_layer": routing_metrics.snapshot_metrics(s, k), "tokens": len(s[0])}

    with _Timer(timings, "snapshot_init"):
        snap("init")
        evals["init"] = evaluate_policy(model, tokenizer, eval_tasks, eval_new)

    track_gates = cfg.model.kind == "tiny-random"
    w_init = gates.flatten_experts(model) if track_gates else None

    torch.manual_seed(seed)
    with _Timer(timings, "sft"):
        sft = run_sft(model, tokenizer, sft_tasks, cfg.sft, seed)
    with _Timer(timings, "snapshot_after_sft"):
        snap("after_sft")
        evals["after_sft"] = evaluate_policy(model, tokenizer, eval_tasks, eval_new)
    w_sft = gates.flatten_experts(model) if track_gates else None
    sft_state = model_mod.trainable_state(model)

    rl_results: dict = {}
    churn = {"init_to_after_sft": routing_metrics.snapshot_churn(snapshots["init"], snapshots["after_sft"], k)}
    gate_reports: dict = {}
    for algo in cfg.rl.algorithms:
        model_mod.load_trainable_state(model, sft_state)
        torch.manual_seed(seed)
        with _Timer(timings, f"rl_{algo}"):
            rl_results[algo] = run_rl(model, tokenizer, rl_tasks, algo, cfg.rl, seed)
        with _Timer(timings, f"snapshot_after_rl_{algo}"):
            snap(f"after_rl_{algo}")
            evals[f"after_rl_{algo}"] = evaluate_policy(model, tokenizer, eval_tasks, eval_new)
        churn[f"after_sft_to_after_rl_{algo}"] = routing_metrics.snapshot_churn(
            snapshots["after_sft"], snapshots[f"after_rl_{algo}"], k
        )
        if track_gates:
            with _Timer(timings, f"gates_{algo}"):
                gate_reports[f"sft_to_rl_{algo}"] = _gate_status(
                    gates.run_gates(w_sft, gates.flatten_experts(model))
                )
    if track_gates:
        with _Timer(timings, "gates_sft"):
            gate_reports["init_to_sft"] = _gate_status(gates.run_gates(w_init, w_sft))
    else:
        gate_reports = {"status": "skipped", "reason": "gates not run for pretrained/LoRA models"}
    timings["total"] = round(time.perf_counter() - t_start, 3)

    result = {
        "schema": "ptr.moe-pipeline-metrics.v1",
        "evidence_class": EVIDENCE_CLASS,
        "note": EVIDENCE_NOTE,
        "run": {"name": cfg.name, "seed": seed, "config_path": cfg.source_path},
        "model": {
            "kind": cfg.model.kind,
            "class": type(getattr(model, "base_model", model)).__name__
            if cfg.lora.enabled else type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "experts": getattr(model.config, "num_experts", None),
            "top_k": k,
            "layers": getattr(model.config, "num_hidden_layers", None),
            "lora": cfg.lora.enabled,
        },
        "versions": versions(),
        "hardware": hardware(),
        "wall_time_s": timings,
        "sft": sft,
        "rl": rl_results,
        "policy_eval": evals,
        "routing": routing,
        "routing_churn": churn,
        "gates": gate_reports,
        "ptr_py_installed": gates.load_binding() is not None,
        **(extra or {}),
    }
    flags: list = []
    result = _sanitize(result, flags)
    result["nan_check"] = {
        "nonfinite_values_found": bool(flags) or sft["nan"] or any(r["nan"] for r in rl_results.values()),
    }
    return result


def write_metrics(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_config(config_path: str | Path, out: str | Path | None = None, extra: dict | None = None) -> Path:
    """Run the pipeline from a stage TOML. `out` is an output directory (default: config's)."""
    cfg = moe_config.load(_resolve(config_path))
    out_dir = _resolve(out if out is not None else cfg.output_dir)
    result = execute(cfg, out_dir, extra)
    return write_metrics(result, out_dir)


def run(manifest: dict, config_path: str | Path) -> Path:
    """Entry used by `ptr_training.run --execute`; records the run-manifest provenance."""
    extra = {
        "manifest": {
            "git_sha": manifest.get("git_sha"),
            "git_dirty": manifest.get("git_dirty"),
            "input_fingerprint_sha256": manifest.get("input_fingerprint_sha256"),
            "config_sha256": manifest.get("config_sha256"),
            "uv_lock_sha256": manifest.get("uv_lock_sha256"),
        }
    }
    return run_config(config_path, None, extra)
