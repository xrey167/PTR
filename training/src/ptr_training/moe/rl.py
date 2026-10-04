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

from ptr_training.moe.config import RlConfig
from ptr_training.moe.reward import LEVEL, make_reward_fn, rl_reward, verify
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


def build_grpo_config(algorithm: str, cfg: RlConfig, *, seed: int, output_dir: str, bf16: bool):
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
        use_cpu=not torch.cuda.is_available(),
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
        report = verify(task.kind, task.spec, text)
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


def run_rl(model, tokenizer, tasks: list[Task], algorithm: str, cfg: RlConfig, seed: int) -> dict:
    """Train `model` in place with `algorithm`; returns logs and summary statistics."""
    import torch
    from datasets import Dataset
    from trl import GRPOTrainer

    check_prompt_lengths(tokenizer, tasks, cfg.max_prompt_length)
    check_completion_length(tokenizer, tasks, cfg.max_completion_length)
    bf16 = next(model.parameters()).dtype == torch.bfloat16
    is_peft = hasattr(model, "peft_config")
    if is_peft and "ref" in model.peft_config:
        model.delete_adapter("ref")  # stale copy from a previous arm; TRL re-creates it
    with tempfile.TemporaryDirectory(prefix=f"ptr-moe-{algorithm}-") as tmp:
        args = build_grpo_config(algorithm, cfg, seed=seed, output_dir=tmp, bf16=bf16)
        if cfg.beta > 0 and not is_peft:
            # TRL builds the reference policy by reloading `config._name_or_path`; a randomly
            # initialised model has no hub path, so snapshot the RL starting point (the SFT
            # state) to disk and point the config at it. KL is then measured against it.
            ref_dir = f"{tmp}/reference-policy"
            model.save_pretrained(ref_dir)
            model.config._name_or_path = ref_dir
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=make_reward_fn(cfg.min_verification_level),
            args=args,
            train_dataset=Dataset.from_list([t.to_row() for t in tasks]),
            processing_class=tokenizer,
        )
        try:
            _assert_peft_reference(model, cfg)
            trainer.train()
        finally:
            if is_peft and "ref" in model.peft_config:
                model.delete_adapter("ref")
        history = [
            {k: _clean(v) for k, v in entry.items()} for entry in trainer.state.log_history
        ]
    model.eval()
    steps = [h for h in history if "reward" in h]
    numeric = [v for h in history for v in h.values() if isinstance(v, float)]
    keys = sorted({k for h in steps for k in h if isinstance(h[k], (int, float))})
    return {
        "algorithm": algorithm,
        "grpo_overrides": grpo_overrides(algorithm, cfg),
        "log_history": history,
        "reward_per_step": [h["reward"] for h in steps],
        "kl_per_step": [h["kl"] for h in steps if "kl" in h],
        "clip_ratio_per_step": [h["clip_ratio/region_mean"] for h in steps if "clip_ratio/region_mean" in h],
        "logged_metric_keys": keys,
        "nan": any(math.isnan(v) or math.isinf(v) for v in numeric),
    }
