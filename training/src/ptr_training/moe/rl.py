"""RL with TRL's GRPOTrainer configured for GSPO and SAPO.

Verified against trl 1.14.1 (see README.md):
  * GSPO = importance_sampling_level="sequence" + loss_type="grpo" (the setup TRL's own
    warning names as the GSPO paper reproduction) with `gspo_epsilon`/`gspo_epsilon_high`
    (default 3e-4/4e-4, the GSPO paper's sequence-level clip range; a config may loosen
    them, which then is not the paper's setting). `epsilon`/`epsilon_high` are
    grpo-style token-level clipping values and are not used by either algorithm here.
  * SAPO = loss_type="sapo" (token-level ratios by default) with sapo_temperature_pos/neg;
    epsilon is unused by SAPO's soft gate.
  * `mini_steps` maps to `steps_per_generation`: optimizer steps taken per generation batch.
    With 1 the first update is exactly on-policy (ratio == 1) and neither clipping nor the
    SAPO gate can act, so smoke configs use >= 2.
  * GRPOConfig has no `max_prompt_length` (removed upstream), so `check_prompt_lengths`
    enforces `moe.rl.max_prompt_length` here: an over-long prompt is rejected, never
    silently truncated (truncating would corrupt the task).
  * Reference policy dtype: TRL reloads the full-fine-tune reference via
    `create_model_from_path(**model_init_kwargs)` whose `dtype` defaults to float32, so the
    policy dtype is passed through `GRPOConfig.model_init_kwargs` (TRL warns it is "ignored"
    for the instantiated policy but still uses it for the reference; verified by a test).
  * Reference policy (beta > 0). Full fine-tune: a copy of the SFT state saved to disk.
    PEFT: TRL (peft >= 0.20) copies the live "default" adapter into a frozen "ref" adapter
    when the trainer is built and scores the reference under it; with the SFT adapter state
    loaded at that point, the reference is the SFT state, not the pretrained base. TRL falls
    back to the base (adapter disabled) when that is impossible; `_assert_peft_reference`
    makes that case an error instead of a silent wrong KL.
"""

from __future__ import annotations

import math
import tempfile

from ptr_training.integrity import assert_step_count
from ptr_training.moe.config import RlConfig
from ptr_training.moe.reward import (
    LEVEL,
    is_truncated,
    make_reward_fn,
    rl_reward,
    truncated_report,
    verify,
)
from ptr_training.moe.tasks import Task


def grpo_overrides(algorithm: str, cfg: RlConfig) -> dict:
    """GRPOConfig keyword arguments that distinguish the algorithms."""
    if algorithm == "gspo":
        return {
            "importance_sampling_level": "sequence",
            "loss_type": "grpo",
            "epsilon": cfg.gspo_epsilon,
            "epsilon_high": cfg.gspo_epsilon_high,
        }
    if algorithm == "sapo":
        return {
            "importance_sampling_level": "token",
            "loss_type": "sapo",
            "sapo_temperature_pos": cfg.sapo_temperature_pos,
            "sapo_temperature_neg": cfg.sapo_temperature_neg,
        }
    raise ValueError(f"unknown RL algorithm {algorithm!r}")


def build_grpo_config(
    algorithm: str, cfg: RlConfig, *, seed: int, output_dir: str, bf16: bool,
    device: str | None = None,
):
    """`device` is the device the model lives on ("cpu" or "cuda"); `use_cpu` follows it.
    None keeps the old behaviour (CUDA when available)."""
    import torch
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.prompts_per_step * cfg.num_generations,
        steps_per_generation=cfg.mini_steps,
        num_generations=cfg.num_generations,
        num_iterations=1,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        learning_rate=cfg.lr,
        max_steps=cfg.steps,
        beta=cfg.beta,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        seed=seed,
        bf16=bf16,
        # trl 1.14.1: GRPOConfig.disable_dropout (grpo_config.py ~452, default False) is applied
        # in GRPOTrainer.__init__ (grpo_trainer.py ~988: disable_dropout_in_model). Policy and
        # reference must be deterministic for the logprob ratio / KL to mean anything.
        disable_dropout=True,
        # TRL reloads the (non-PEFT) reference model with these kwargs and defaults to float32,
        # which would make a bf16 policy's KL be measured against an fp32 reference.
        model_init_kwargs={"dtype": "bfloat16" if bf16 else "float32"} if cfg.beta > 0 else None,
        use_cpu=(device == "cpu") if device is not None else not torch.cuda.is_available(),
        **grpo_overrides(algorithm, cfg),
    )


def check_prompt_lengths(tokenizer, tasks: list[Task], max_prompt_length: int) -> int:
    """Longest prompt in tokens; raises ValueError if any exceeds `max_prompt_length`."""
    longest = 0
    for task in tasks:
        n = len(tokenizer(task.prompt, add_special_tokens=False)["input_ids"])
        if n > max_prompt_length:
            raise ValueError(
                f"prompt of {n} tokens exceeds moe.rl.max_prompt_length={max_prompt_length}: "
                f"{task.prompt!r}"
            )
        longest = max(longest, n)
    return longest


def check_completion_length(tokenizer, tasks: list[Task], max_completion_length: int) -> int:
    """Longest canonical answer + EOS in tokens; ValueError if it cannot fit."""
    longest = 0
    for task in tasks:
        n = len(tokenizer(task.answer, add_special_tokens=False)["input_ids"]) + 1
        if n > max_completion_length:
            raise ValueError(
                f"answer {task.answer!r} needs {n} tokens (with EOS) but "
                f"moe.rl.max_completion_length={max_completion_length}"
            )
        longest = max(longest, n)
    return longest


def _assert_peft_reference(model, cfg: RlConfig) -> None:
    """With PEFT and beta > 0 the KL reference must be a frozen copy of the SFT adapter."""
    if cfg.beta <= 0 or not hasattr(model, "peft_config"):
        return
    if "ref" not in model.peft_config:
        raise RuntimeError(
            "PEFT model with beta > 0 but no 'ref' adapter: TRL would compute the reference "
            "logprobs with the adapter disabled (the pretrained base, not the SFT state). "
            "Needs peft >= 0.20; refusing to report KL against the wrong reference."
        )
    own = dict(model.named_parameters())
    for name, param in own.items():
        if ".default." in name:
            ref = own[name.replace(".default.", ".ref.")]
            if not bool((param.detach() == ref.detach()).all()):
                raise RuntimeError(f"'ref' adapter differs from the SFT adapter at {name}")


def evaluate_policy(
    model, tokenizer, tasks: list[Task], max_new_tokens: int, min_level: str = LEVEL
) -> dict:
    """Greedy-decode every task and score it with the verifier."""
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rewards, scores, statuses = [], [], {}
    by_kind: dict = {}
    sample = None
    for task in tasks:
        enc = tokenizer(task.prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        generated = out[0, enc["input_ids"].shape[1] :].tolist()
        cut = is_truncated(generated, tokenizer.eos_token_id, max_new_tokens)
        report = truncated_report() if cut else verify(task.kind, task.spec, text)
        rewards.append(rl_reward(report, min_level))
        scores.append(report.score)
        statuses[report.status] = statuses.get(report.status, 0) + 1
        counts = by_kind.setdefault(task.kind, {"n": 0, "pass": 0})
        counts["n"] += 1
        counts["pass"] += report.status == "Pass"
        if sample is None:
            sample = {"prompt": task.prompt, "completion": text, "report": report.to_dict()}
    model.train(was_training)
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_report_score": sum(scores) / len(scores),
        "pass_rate": statuses.get("Pass", 0) / len(scores),
        "status_counts": statuses,
        "by_kind": by_kind,
        "n": len(scores),
        "sample": sample,
    }


def _clean(value):
    return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)


def merge_log_history(history: list[dict]) -> dict[int, dict]:
    """Group TRL log entries by optimizer `step`, merging all keys.

    TRL does not log every metric in every entry (with `steps_per_generation` > 1 the reward
    is logged only on generation steps while kl/clip_ratio are logged on every step), so a
    per-step series must never be read from a pre-filtered list of entries. Entries without a
    `step` (the end-of-training summary) are not per-step data.
    """
    merged: dict[int, dict] = {}
    for entry in history:
        step = entry.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            continue
        merged.setdefault(int(step), {}).update(entry)
    return dict(sorted(merged.items()))


def _series(out: dict, merged: dict[int, dict], name: str, key: str, reason: str) -> None:
    """`<name>_per_step` = values of `key` at the steps that logged it (null + reason if none);
    the matching step indices are recorded in `out["series_steps"][name]`."""
    points = [(step, h[key]) for step, h in merged.items() if key in h]
    out[f"{name}_per_step"] = [v for _, v in points] or None
    out.setdefault("series_steps", {})[name] = [s for s, _ in points]
    if not points:
        out[f"{name}_unavailable_reason"] = reason


def run_rl(model, tokenizer, tasks: list[Task], algorithm: str, cfg: RlConfig, seed: int) -> dict:
    """Train `model` in place with `algorithm`; returns logs and summary statistics."""
    import torch
    from datasets import Dataset
    from trl import GRPOTrainer

    check_prompt_lengths(tokenizer, tasks, cfg.max_prompt_length)
    check_completion_length(tokenizer, tasks, cfg.max_completion_length)
    bf16 = next(model.parameters()).dtype == torch.bfloat16
    is_peft = hasattr(model, "peft_config")
    original_name_or_path = model.config._name_or_path
    if is_peft and "ref" in model.peft_config:
        model.delete_adapter("ref")  # stale copy from a previous arm; TRL re-creates it
    with tempfile.TemporaryDirectory(prefix=f"ptr-moe-{algorithm}-") as tmp:
        args = build_grpo_config(
            algorithm, cfg, seed=seed, output_dir=tmp, bf16=bf16,
            device=next(model.parameters()).device.type,
        )
        if cfg.beta > 0 and not is_peft:
            # TRL builds the reference policy by reloading `config._name_or_path`; a randomly
            # initialised model has no hub path, so snapshot the RL starting point (the SFT
            # state) to disk and point the config at it. KL is then measured against it.
            ref_dir = f"{tmp}/reference-policy"
            model.save_pretrained(ref_dir)
            model.config._name_or_path = ref_dir
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=make_reward_fn(
                cfg.min_verification_level,
                eos_token_id=tokenizer.eos_token_id,
                max_completion_length=cfg.max_completion_length,
            ),
            args=args,
            train_dataset=Dataset.from_list([t.to_row() for t in tasks]),
            processing_class=tokenizer,
        )
        try:
            _assert_peft_reference(model, cfg)
            trainer.train()
        finally:
            model.config._name_or_path = original_name_or_path  # undo the reference-dir redirect
            if is_peft and "ref" in model.peft_config:
                model.delete_adapter("ref")
        global_step = assert_step_count(trainer.state, cfg.steps, f"RL {algorithm}")
        history = [
            {k: _clean(v) for k, v in entry.items()} for entry in trainer.state.log_history
        ]
    model.eval()
    merged = merge_log_history(history)
    numeric = [v for h in history for v in h.values() if isinstance(v, float)]
    keys = sorted({k for h in merged.values() for k in h if isinstance(h[k], (int, float))})
    out = {
        "algorithm": algorithm,
        "global_step": global_step,
        "grpo_overrides": grpo_overrides(algorithm, cfg),
        "log_history": history,
        "logged_metric_keys": keys,
        "nan": any(math.isnan(v) or math.isinf(v) for v in numeric),
    }
    # Every series is built from the step-merged history. A metric TRL did not log is `null`
    # plus a reason, never a silent empty list.
    _series(out, merged, "reward", "reward", "trl logged no reward")
    out["reward_per_step"] = out["reward_per_step"] or []  # reward always exists when training ran
    _series(out, merged, "kl", "kl",
            "beta = 0: TRL logs no kl" if cfg.beta == 0 else "trl logged no kl")
    _series(out, merged, "clip_ratio", "clip_ratio/region_mean",
            "trl 1.14.1 logs no clip_ratio for loss_type=sapo" if algorithm == "sapo"
            else "trl logged no clip_ratio/region_mean")
    _series(out, merged, "entropy", "entropy", "trl logged no entropy")
    _series(out, merged, "grad_norm", "grad_norm", "trl logged no grad_norm")
    _series(out, merged, "loss", "loss", "trl logged no loss")
    return out
