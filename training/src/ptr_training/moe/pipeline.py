"""End-to-end orchestration of the `trl-moe` backend.

build model -> routing snapshot -> SFT -> snapshot -> RL per algorithm (fresh SFT state,
seeded) -> snapshot -> gates -> metrics JSON. `run_config` is the direct entry point
(no manifest); `run` is what `ptr_training.run --execute` dispatches to.
"""

from __future__ import annotations

import hashlib
import secrets
import json
import math
import os
import platform
import re
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ptr_training.moe import config as moe_config
from ptr_training.moe import gates, routing_metrics
from ptr_training.moe.tasks import generate_heldout, generate_tasks

EVIDENCE_CLASS = "pipeline-smoke"
EVIDENCE_NOTE = "mechanism check only, not evidence"
# Quantities the M010/M012 experiments name that this backend does not compute (yet).
NOT_PRODUCED = [
    "policy_ratio_variance_per_token",
    "crash_step",
    "train_inference_kl",
    "rollout_vs_training_expert_churn",
]
BACKEND_SELECTION = {
    "status": "unselected",
    "component": "training-backend",
    "candidate": "trl-peft",
    "note": "unselected backend, pipeline evidence only",
}
ROOT = Path(__file__).resolve().parents[4]


class MissingExtraError(RuntimeError):
    """The `moe` extra (torch, transformers, trl, ...) is not installed."""


# Minimum versions of the `moe` extra; tests/test_moe_config.py checks this against
# training/pyproject.toml so the two cannot drift.
MOE_MINIMUMS = {
    "torch": "2.14.1",
    "transformers": "5.18.0",
    "peft": "0.21.2",
    "trl": "1.14.1",
    "datasets": "5.0.1",
    "accelerate": "1.15.0",
}


# Exclusive upper bounds (pyproject.toml `moe` extra; checked by a test): the backend reads
# trl / peft internals, so a newer minor is unverified.
MOE_UPPER_BOUNDS = {
    "torch": "3",
    "transformers": "6",
    "peft": "0.22",
    "trl": "1.15",
    "datasets": "6",
    "accelerate": "2",
}


def _release(version: str) -> tuple[int, ...]:
    """Release segment as ints ('2.14.1+cpu' -> (2, 14, 1)); `packaging` is not required."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", version)
    if match is None:
        raise ValueError(f"unparseable version {version!r}")
    parts = [int(x) for x in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def version_at_least(installed: str, required: str) -> bool:
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed) >= Version(required)
        except InvalidVersion:
            pass
    except ImportError:
        pass
    return _release(installed) >= _release(required)


def missing_distributions(names) -> list[str]:
    """Names without an installed distribution (importlib.metadata, never find_spec: the
    repo's top-level `datasets/` directory is an importable namespace package)."""
    missing = []
    for name in names:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return missing


def stack_available(names=("torch", "transformers", "peft", "trl", "datasets")) -> bool:
    """True when every named distribution is installed (used to gate torch-dependent tests)."""
    return not missing_distributions(names)


def require_extra() -> None:
    # Check installed distributions, not module names: the repo's top-level `datasets/`
    # directory is importable as a namespace package and would satisfy find_spec("datasets").
    missing, too_old, too_new = [], [], []
    for dist, minimum in MOE_MINIMUMS.items():
        try:
            installed = metadata.version(dist)
        except metadata.PackageNotFoundError:
            missing.append(dist)
            continue
        try:
            ok = version_at_least(installed, minimum)
        except ValueError:
            ok = False
        if not ok:
            too_old.append(f"{dist} {installed} < {minimum}")
        elif version_at_least(installed, MOE_UPPER_BOUNDS[dist]):
            too_new.append(f"{dist} {installed} >= {MOE_UPPER_BOUNDS[dist]}")
    if missing or too_old or too_new:
        problems = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if too_old:
            problems.append(f"too old: {', '.join(too_old)}")
        if too_new:
            problems.append(f"too new (unverified): {', '.join(too_new)}")
        raise MissingExtraError(
            f"backend 'trl-moe' needs the optional 'moe' extra ({'; '.join(problems)}); "
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


def hardware(device: str | None = None) -> dict:
    """Host and device facts. `device` is the resolved device of the run ("cpu"/"cuda"); on
    CUDA the peak allocation since `torch.cuda.reset_peak_memory_stats()` is recorded."""
    import torch

    ram_gib = None
    try:
        ram_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (ValueError, OSError, AttributeError):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    on_cuda = device == "cuda" and torch.cuda.is_available()
    return {
        "cpu_count": os.cpu_count(),
        "ram_gib": ram_gib,
        "gpu": gpu,
        "platform": platform.platform(),
        "device": device,
        "torch_cuda": torch.version.cuda,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if on_cuda else None
        ),
        "peak_cuda_memory_allocated_gib": (
            round(torch.cuda.max_memory_allocated() / 2**30, 3) if on_cuda else None
        ),
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
    if report.get("status") not in ("ok", "partial"):
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


EVAL_TASKS = 32


def build_task_splits(cfg: moe_config.MoeConfig):
    """(sft, rl, eval, stats). Eval prompts are held out first; SFT/RL never sample them."""
    eval_tasks = generate_heldout(EVAL_TASKS, seed=cfg.seed + 2)
    held = frozenset(t.prompt for t in eval_tasks)
    sft_tasks = generate_tasks(cfg.sft.examples, seed=cfg.seed, exclude_prompts=held)
    if cfg.rl.algorithms:
        # TRL (steps_per_generation = mini_steps) draws one generation batch of
        # mini_steps * prompts_per_step NEW prompts (RepeatSampler batch_size =
        # generation_batch_size // num_generations) and slices it across `mini_steps` optimizer
        # steps, so each optimizer step consumes prompts_per_step fresh prompts. A run needs
        # ceil(steps / mini_steps) whole generation batches (the last may be partial); floor 64.
        generation_batches = -(-cfg.rl.steps // cfg.rl.mini_steps)
        rl_n = max(64, generation_batches * cfg.rl.mini_steps * cfg.rl.prompts_per_step)
        rl_tasks = generate_tasks(rl_n, seed=cfg.seed + 1, exclude_prompts=held)
    else:
        rl_tasks = []  # no RL arm will run: do not build a pool
    train = {t.prompt for t in sft_tasks} | {t.prompt for t in rl_tasks}
    overlap = len(held & train)
    if overlap or len(held) != len(eval_tasks):
        raise RuntimeError(
            f"evaluation tasks are not disjoint from training tasks (overlap={overlap})"
        )
    stats = {
        "sft_tasks": len(sft_tasks),
        "sft_unique_prompts": len({t.prompt for t in sft_tasks}),
        "rl_tasks": len(rl_tasks),
        "rl_pool_size": len(rl_tasks),
        "rl_pool_built": bool(rl_tasks),
        "rl_unique_prompts": len({t.prompt for t in rl_tasks}),
        "eval_tasks": len(eval_tasks),
        "eval_unique_prompts": len(held),
        "eval_overlap": overlap,
    }
    return sft_tasks, rl_tasks, eval_tasks, stats


def check_model_context(cfg, model, tokenizer, sft_tasks, rl_tasks, eval_tasks) -> dict:
    """Tokenizer-measured sequence lengths vs the loaded model's context window.

    Longest SFT example (prompt + answer + EOS) and longest task prompt + max_completion_length
    (RL rollouts and the policy evaluation both generate that many tokens).
    """
    sft_longest = max(
        len(tokenizer(t.prompt, add_special_tokens=False)["input_ids"])
        + len(tokenizer(t.answer, add_special_tokens=False)["input_ids"])
        + 1
        for t in sft_tasks
    ) if sft_tasks else 0
    prompts = [t.prompt for t in [*eval_tasks, *rl_tasks]]
    prompt_longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    needed = max(sft_longest, prompt_longest + cfg.rl.max_completion_length)
    window = getattr(model.config, "max_position_embeddings", None)
    info = {
        "sft_longest_tokens": sft_longest,
        "prompt_longest_tokens": prompt_longest,
        "needed_positions": needed,
        "model_max_position_embeddings": window,
    }
    if window is not None and needed > window:
        raise moe_config.ConfigError(
            f"the run needs {needed} positions (longest SFT example {sft_longest} tokens incl. "
            f"EOS; longest prompt {prompt_longest} + moe.rl.max_completion_length "
            f"{cfg.rl.max_completion_length}) but the loaded model's max_position_embeddings "
            f"is {window}"
        )
    return info


def execute(cfg: moe_config.MoeConfig, out_dir: Path, extra: dict | None = None) -> dict:
    moe_config.require_backend_opt_in(cfg.allow_unselected_backend)
    require_extra()
    import torch

    from ptr_training.moe import model as model_mod
    from ptr_training.moe.rl import (
        check_completion_length,
        check_prompt_lengths,
        evaluate_policy,
        run_rl,
    )
    from ptr_training.moe.sft import check_examples, run_sft

    torch.set_num_threads(os.cpu_count() or 1)
    device = moe_config.resolve_device(cfg.model.device)  # ConfigError for cuda without CUDA
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model_mod.seed_everything(cfg.seed)  # before any model / LoRA initialisation
    timings: dict = {}
    t_start = time.perf_counter()
    k = cfg.model.top_k if cfg.model.kind == "tiny-random" else None

    # Cheap, pure-Python guards first: fail before any hub call, download or model build.
    sft_tasks, rl_tasks, eval_tasks, task_stats = build_task_splits(cfg)
    model_source = (
        model_mod.resolve_revision(cfg.model) if cfg.model.kind == "pretrained" else None
    )
    with _Timer(timings, "build_model"):
        model, tokenizer = model_mod.build(cfg.model, cfg.lora)
        if k is None:
            k = int(getattr(model.config, "num_experts_per_tok"))
    if cfg.model.kind == "pretrained":
        model_mod.reset_requires_grad(model)
    # Tokenizer-dependent guards, before the first eval / SFT step spends any compute.
    context_check = check_model_context(cfg, model, tokenizer, sft_tasks, rl_tasks, eval_tasks)
    check_examples(tokenizer, sft_tasks, cfg.sft.max_length)
    if cfg.rl.algorithms:
        check_prompt_lengths(tokenizer, rl_tasks, cfg.rl.max_prompt_length)
        check_completion_length(tokenizer, rl_tasks, cfg.rl.max_completion_length)

    seed = cfg.seed
    probe_texts = [t.prompt + t.answer for t in eval_tasks[:8]]
    eval_new = cfg.rl.max_completion_length

    snapshots: dict = {}
    routing: dict = {}
    evals: dict = {}

    router_errors: dict = {}
    param_checks: dict = {}

    def snap(name: str):
        try:
            by_text = model_mod.routing_snapshot_by_text(
                model, tokenizer, probe_texts, cfg.probe_tokens
            )
            s = [[row for text in by_text for row in text[layer]] for layer in range(len(by_text[0]))]
            routing[name] = {
                "per_layer": routing_metrics.snapshot_metrics(s, k),
                "per_text": routing_metrics.per_text_metrics(by_text, k),
                "tokens": len(s[0]),
                "tokens_per_text": [len(text[0]) for text in by_text],
                "max_tokens_per_text": cfg.probe_tokens,
                "topk_tie_break": routing_metrics.TOP_K_TIE_BREAK,
            }
        except routing_metrics.NonFiniteRouterLogits as error:
            # Recorded, not hidden: nan_check.nonfinite_values_found is forced true below.
            router_errors[name] = str(error)
            snapshots[name] = None
            routing[name] = {"per_layer": None, "tokens": 0, "error": str(error)}
            return
        snapshots[name] = s

    def churn_between(a: str, b: str):
        if snapshots[a] is None or snapshots[b] is None:
            return None
        return routing_metrics.snapshot_churn(snapshots[a], snapshots[b], k)

    with _Timer(timings, "snapshot_init"):
        snap("init")
        evals["init"] = evaluate_policy(model, tokenizer, eval_tasks, eval_new, cfg.rl.min_verification_level)

    track_gates = cfg.model.kind == "tiny-random" and not cfg.lora.enabled
    gate_cap = None if cfg.allow_large else gates.MAX_TOTAL_GATE_ELEMENTS
    gate_problem = None
    if track_gates:
        gate_problem = gates.binding_problem() or gates.snapshot_problem(model, gate_cap)
    run_gate_calls = track_gates and gate_problem is None  # never flatten weights nobody reads
    w_init = gates.flatten_experts(model, max_total=gate_cap) if run_gate_calls else None

    torch.manual_seed(seed)
    with _Timer(timings, "sft"):
        sft = run_sft(model, tokenizer, sft_tasks, cfg.sft, seed)
    param_checks["after_sft"] = model_mod.nonfinite_parameters(model)
    with _Timer(timings, "snapshot_after_sft"):
        snap("after_sft")
        evals["after_sft"] = evaluate_policy(model, tokenizer, eval_tasks, eval_new, cfg.rl.min_verification_level)
    w_sft = gates.flatten_experts(model, max_total=gate_cap) if run_gate_calls else None
    # The SFT state is restored only before arms 2..n (arm 1 starts from it already), so it is
    # snapshotted only when there is more than one arm.
    sft_state = model_mod.trainable_state(model) if len(cfg.rl.algorithms) > 1 else None

    rl_results: dict = {}
    churn = {"init_to_after_sft": churn_between("init", "after_sft")}
    gate_reports: dict = {}
    for arm, algo in enumerate(cfg.rl.algorithms):
        if arm > 0:
            model_mod.load_trainable_state(model, sft_state)
        torch.manual_seed(seed)
        with _Timer(timings, f"rl_{algo}"):
            rl_results[algo] = run_rl(model, tokenizer, rl_tasks, algo, cfg.rl, seed)
        # before the next arm restores the SFT state over these weights
        param_checks[f"after_rl_{algo}"] = model_mod.nonfinite_parameters(model)
        with _Timer(timings, f"snapshot_after_rl_{algo}"):
            snap(f"after_rl_{algo}")
            evals[f"after_rl_{algo}"] = evaluate_policy(model, tokenizer, eval_tasks, eval_new, cfg.rl.min_verification_level)
        churn[f"after_sft_to_after_rl_{algo}"] = churn_between("after_sft", f"after_rl_{algo}")
        if track_gates:
            with _Timer(timings, f"gates_{algo}"):
                if run_gate_calls:
                    gate_reports[f"sft_to_rl_{algo}"] = _gate_status(
                        gates.run_gates(
                            w_sft,
                            gates.flatten_experts(model, max_total=gate_cap),
                            subspace_rank=cfg.gates.subspace_rank,
                        )
                    )
                else:
                    gate_reports[f"sft_to_rl_{algo}"] = dict(gate_problem)
    if track_gates:
        with _Timer(timings, "gates_sft"):
            gate_reports["init_to_sft"] = (
                _gate_status(
                    gates.run_gates(w_init, w_sft, subspace_rank=cfg.gates.subspace_rank)
                )
                if run_gate_calls
                else dict(gate_problem)
            )
    else:
        gate_reports = {"status": "skipped", "reason": "gates not run for pretrained/LoRA models (no gate is exercised by this config)"}
    gates_evaluated = any(
        isinstance(r, dict) and r.get("status") in ("ok", "partial")
        for r in (gate_reports.values() if "status" not in gate_reports else [gate_reports])
    )
    if gates_evaluated:
        gates_not_evaluated_reason = None
    elif "status" in gate_reports and gate_reports["status"] == "skipped" and not track_gates:
        gates_not_evaluated_reason = gate_reports["reason"]
    elif gate_problem is not None:
        gates_not_evaluated_reason = gate_problem.get("reason") or gate_problem.get("error")
    else:
        gates_not_evaluated_reason = "no gate report completed"
    timings["total"] = round(time.perf_counter() - t_start, 3)

    result = {
        "schema": "ptr.moe-pipeline-metrics.v1",
        "evidence_class": EVIDENCE_CLASS,
        "note": EVIDENCE_NOTE,
        "run": {"name": cfg.name, "seed": seed, "config_path": cfg.source_path},
        "model": {
            "kind": cfg.model.kind,
            "device_requested": cfg.model.device,
            "device": next(model.parameters()).device.type,
            "seed": cfg.model.seed,  # init seed actually used (run.seed above is separate)
            "class": type(
                model.get_base_model() if hasattr(model, "get_base_model") else model
            ).__name__,
            "architecture": {
                key: getattr(model.config, key, None)
                for key in (
                    "hidden_size", "intermediate_size", "num_attention_heads", "num_experts",
                    "num_experts_per_tok", "num_hidden_layers", "max_position_embeddings",
                    "router_aux_loss_coef",
                )
            },
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "experts": getattr(model.config, "num_experts", None),
            "top_k": k,
            "layers": getattr(model.config, "num_hidden_layers", None),
            "lora": cfg.lora.enabled,
            "precision": cfg.model.precision,
            "param_dtypes": model_mod.param_dtypes(model),
            "source": model_source,
        },
        "tasks": task_stats,
        "context_check": context_check,
        "data_source": "synthetic-tasks",
        "data_note": (
            "trl-moe trains and evaluates on seeded synthetic verifiable tasks "
            "(ptr_training.moe.tasks); the run manifest's [data] dataset/fingerprint is not "
            "used by this backend"
        ),
        "not_produced": NOT_PRODUCED,
        "gates_evaluated": gates_evaluated,
        "gates_not_evaluated_reason": gates_not_evaluated_reason,
        "versions": versions(),
        "hardware": hardware(device),
        "wall_time_s": timings,
        "sft": sft,
        "rl": rl_results,
        "policy_eval": evals,
        "routing": routing,
        "routing_churn": churn,
        "router_errors": router_errors,
        "gates": gate_reports,
        "ptr_py_installed": {"installed": True, "absent": False, "broken": "broken"}[
            gates.binding_state()[0]
        ],
        "backend_selection": BACKEND_SELECTION,
        **(extra or {}),
    }
    flags: list = []
    result = _sanitize(result, flags)
    result["nan_check"] = {
        "nonfinite_values_found": bool(flags)
        or bool(router_errors)
        or any(c["count"] for c in param_checks.values())
        or sft["nan"]
        or any(r["nan"] for r in rl_results.values()),
    }
    result["nan_check"]["nonfinite_parameters"] = param_checks
    if router_errors:
        result["nan_check"]["router_errors"] = router_errors
    return result


def write_metrics(result: dict, out_dir: Path) -> Path:
    """Write metrics.json into `out_dir`; refuses to overwrite an existing file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]


def unique_run_dir(base: Path, name: str, cfg_hash: str, now: datetime | None = None) -> Path:
    """Create `<base>/<name>-<UTC timestamp(us)>-<config hash>-<random>`; never reuses a dir."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
    path = base / f"{safe}-{stamp}-{cfg_hash}-{secrets.token_hex(3)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)  # FileExistsError rather than overwriting a previous run
    return path


def run_config(config_path: str | Path, out: str | Path | None = None, extra: dict | None = None) -> Path:
    """Run the pipeline from a stage TOML.

    `out` is the base output directory (default: the config's `moe.output.dir`); every run
    writes into its own fresh subdirectory under it.
    """
    config_path = _resolve(config_path)
    cfg = moe_config.load(config_path)
    moe_config.require_backend_opt_in(cfg.allow_unselected_backend)  # before any directory
    base = _resolve(out if out is not None else cfg.output_dir)
    out_dir = unique_run_dir(base, cfg.name, config_hash(config_path))
    result = execute(cfg, out_dir, extra)
    result["run"]["output_dir"] = str(out_dir)
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
