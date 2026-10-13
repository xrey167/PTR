"""Preflight: prove the heaviest step of a run fits before the run is allowed to start.

`python -m ptr_training.preflight --config <stage.toml>` builds the model the run would
build and takes one bounded optimizer step on the longest training row, then generates the
longest rollout, on the device the run will use. The result is a record bound to three
hashes:

* `protocol_sha256`: the config file the run is judged by (the manifest's `config_sha256`),
* `runner_sha256`: the `ptr_training` source that executes it,
* `device`: the device it ran on (`cpu`, or `cuda:<gpu name>`).

`ptr_training.run --execute` refuses to start without a record whose three hashes match
the run it is about to make. A record from another config, another version of the runner
or another GPU proves nothing about this run. The one exception is a run whose evidence
class is `pipeline-smoke`, and only when `--allow-missing-preflight` is passed.

Records live in `training/runs/preflight/`, which is ignored like every run artifact: the
record stays local and only its `record_sha256` is versioned, in the run manifest and in
`metrics.json`. `run.py` resolves the record before it writes the manifest, so a refused run
leaves no manifest.

The record logic here is plain Python and needs no torch; only `run_probe` does.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ptr_training.integrity import canonical_sha256

SCHEMA = "ptr.training-preflight.v1"
SMOKE_EVIDENCE_CLASS = "pipeline-smoke"
ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_DIR = ROOT / "training" / "runs" / "preflight"
PACKAGE_DIR = Path(__file__).resolve().parent
BOUND_FIELDS = ("protocol_sha256", "runner_sha256", "device")


class PreflightError(RuntimeError):
    """The preflight record is missing, tampered with, or does not match this run."""


def runner_sha256(package_dir: Path = PACKAGE_DIR) -> str:
    """SHA-256 over the relative path and bytes of every `ptr_training` source file."""
    hasher = hashlib.sha256()
    sources = sorted(
        path for path in package_dir.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not sources:
        raise PreflightError(f"no runner sources under {package_dir}")
    for path in sources:
        name = path.relative_to(package_dir).as_posix().encode("utf-8")
        data = path.read_bytes()
        hasher.update(len(name).to_bytes(8, "big") + name)
        hasher.update(len(data).to_bytes(8, "big") + data)
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def device_identity(device: str, gpu_name: str | None) -> str:
    """`cpu`, or `cuda:<gpu name>`: a record made on one GPU does not cover another."""
    if device == "cuda":
        if not gpu_name:
            raise PreflightError("cuda device without a GPU name cannot be identified")
        return f"cuda:{gpu_name}"
    if device == "cpu":
        return "cpu"
    raise PreflightError(f"unknown device {device!r}")


def _body_sha256(record: dict) -> str:
    return canonical_sha256({k: v for k, v in record.items() if k != "record_sha256"})


def build_record(
    *,
    protocol_sha256: str,
    runner_sha256: str,
    device: str,
    input_fingerprint_sha256: str,
    probe: dict,
    created_at: str | None = None,
) -> dict:
    record = {
        "schema": SCHEMA,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256,
        "runner_sha256": runner_sha256,
        "device": device,
        "input_fingerprint_sha256": input_fingerprint_sha256,
        "probe": probe,
    }
    record["record_sha256"] = _body_sha256(record)
    return record


def verify_record(record: Any) -> dict:
    """Return `record` if it is well-formed and untampered, else raise `PreflightError`."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise PreflightError(f"not a {SCHEMA} record")
    for field in (*BOUND_FIELDS, "input_fingerprint_sha256", "record_sha256"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise PreflightError(f"preflight record lacks {field}")
    if record["record_sha256"] != _body_sha256(record):
        raise PreflightError("preflight record was modified after it was written")
    return record


def mismatches(record: dict, *, protocol_sha256: str, runner_sha256: str, device: str) -> list[str]:
    wanted = {
        "protocol_sha256": protocol_sha256,
        "runner_sha256": runner_sha256,
        "device": device,
    }
    return [name for name, value in wanted.items() if record.get(name) != value]


def write_record(directory: Path, record: dict) -> Path:
    verify_record(record)
    directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{record['protocol_sha256'][:16]}-{record['runner_sha256'][:16]}-"
        f"{record['record_sha256'][:12]}"
    )
    path = directory / f"preflight-{stem}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def _load_records(directory: Path) -> tuple[list[dict], list[str]]:
    """(valid records, problems with the unreadable or tampered ones)."""
    records, problems = [], []
    if not directory.is_dir():
        return records, problems
    for path in sorted(directory.glob("preflight-*.json")):
        try:
            records.append(verify_record(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, PreflightError) as error:
            problems.append(f"{path.name}: {error}")
    return records, problems


def require_preflight(
    *,
    protocol_sha256: str,
    runner_sha256: str,
    device: str,
    evidence_class: str,
    allow_missing: bool,
    directory: Path = PREFLIGHT_DIR,
) -> dict | None:
    """The matching record, or `None` when a pipeline-smoke run waived it; else raises."""
    records, problems = _load_records(directory)
    wanted = {
        "protocol_sha256": protocol_sha256,
        "runner_sha256": runner_sha256,
        "device": device,
    }
    for record in records:
        if not mismatches(record, **wanted):
            return record
    if allow_missing:
        if evidence_class != SMOKE_EVIDENCE_CLASS:
            raise PreflightError(
                "--allow-missing-preflight is only valid for evidence_class "
                f"{SMOKE_EVIDENCE_CLASS!r}, this run is {evidence_class!r}"
            )
        return None
    detail = []
    if records:
        near = min((mismatches(r, **wanted) for r in records), key=len)
        detail.append(f"closest record differs in: {', '.join(near)}")
    detail.extend(f"ignored {p}" for p in problems)
    suffix = f" ({'; '.join(detail)})" if detail else ""
    raise PreflightError(
        "no preflight record matches this run "
        f"(protocol {protocol_sha256[:12]}, runner {runner_sha256[:12]}, device {device})"
        f"{suffix}; run `python -m ptr_training.preflight --config <config>` first"
        f" or pass --allow-missing-preflight for a {SMOKE_EVIDENCE_CLASS} run"
    )


def current_device_identity(model_device: str) -> str:
    """Resolve the configured device on this host (needs torch)."""
    import torch

    from ptr_training.moe import config as moe_config

    device = moe_config.resolve_device(model_device)
    return device_identity(
        device, torch.cuda.get_device_name(0) if device == "cuda" else None
    )


def run_probe(cfg) -> tuple[dict, str]:
    """One bounded step on the longest row. Returns (probe facts, input fingerprint)."""
    import torch

    from ptr_training.moe import config as moe_config
    from ptr_training.moe import model as model_mod
    from ptr_training.moe import pipeline
    from ptr_training.moe.sft import _encode_example, check_examples, run_sft

    pipeline.require_extra()
    device = moe_config.resolve_device(cfg.model.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model_mod.seed_everything(cfg.seed)
    sft_tasks, rl_tasks, eval_tasks, _ = pipeline.build_task_splits(cfg)
    model, tokenizer = model_mod.build(cfg.model, cfg.lora)
    if cfg.model.kind == "pretrained":
        model_mod.reset_requires_grad(model)
    pipeline.check_model_context(cfg, model, tokenizer, sft_tasks, rl_tasks, eval_tasks)
    check_examples(tokenizer, sft_tasks, cfg.sft.max_length)

    def sft_length(task) -> int:
        return len(_encode_example(tokenizer, task, cfg.sft.max_length)[0])

    longest_sft = max(sft_tasks, key=sft_length)
    prompts = [t.prompt for t in [*eval_tasks, *rl_tasks]]

    def prompt_length(prompt: str) -> int:
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    longest_prompt = max(prompts, key=prompt_length)

    # One optimizer step over `batch` copies of the longest row: the real SFT loop, bounded.
    one_step = dataclasses.replace(cfg.sft, steps=1, gradient_accumulation_steps=1)
    started = time.perf_counter()
    result = run_sft(model, tokenizer, [longest_sft], one_step, cfg.seed)
    sft_seconds = round(time.perf_counter() - started, 3)
    if result["nan"]:
        raise PreflightError(f"preflight SFT step produced a non-finite loss: {result['losses']}")

    new_tokens = cfg.rl.max_completion_length
    target = next(model.parameters()).device
    enc = tokenizer(longest_prompt, return_tensors="pt", add_special_tokens=False).to(target)
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    produced = int(generated.shape[1] - enc["input_ids"].shape[1])

    probe = {
        "sft_batch": cfg.sft.batch,
        "longest_sft_tokens": sft_length(longest_sft),
        "longest_prompt_tokens": prompt_length(longest_prompt),
        "rollout_new_tokens": produced,
        "sft_step_loss": result["losses"][0],
        "sft_step_seconds": sft_seconds,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
    }
    fingerprint = canonical_sha256(
        {
            "longest_sft": [longest_sft.prompt, longest_sft.answer],
            "longest_prompt": longest_prompt,
            "sft_batch": cfg.sft.batch,
            "sft_max_length": cfg.sft.max_length,
            "rollout_new_tokens": new_tokens,
            "seed": cfg.seed,
        }
    )
    return probe, fingerprint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PREFLIGHT_DIR)
    args = parser.parse_args(argv)

    from ptr_training.moe import config as moe_config

    config_path = (args.config if args.config.is_absolute() else ROOT / args.config).resolve()
    cfg = moe_config.load(config_path)
    moe_config.require_backend_opt_in(cfg.allow_unselected_backend)
    device = current_device_identity(cfg.model.device)
    probe, fingerprint = run_probe(cfg)
    record = build_record(
        protocol_sha256=file_sha256(config_path),
        runner_sha256=runner_sha256(),
        device=device,
        input_fingerprint_sha256=fingerprint,
        probe=probe,
    )
    print(write_record(args.out_dir, record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
