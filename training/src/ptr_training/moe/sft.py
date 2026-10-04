"""Plain supervised fine-tuning loop (completion-only loss, MoE aux loss included)."""

from __future__ import annotations

import math
import random

from ptr_training.moe.config import SftConfig
from ptr_training.moe.tasks import Task, sft_text


def _encode_example(tokenizer, task: Task, max_length: int):
    prompt_ids = tokenizer(task.prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(task.answer, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids + [tokenizer.eos_token_id])[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id])[:max_length]
    return ids, labels


def collate(tokenizer, tasks: list[Task], max_length: int):
    import torch

    rows = [_encode_example(tokenizer, t, max_length) for t in tasks]
    width = max(len(ids) for ids, _ in rows)
    pad = tokenizer.pad_token_id
    input_ids = [ids + [pad] * (width - len(ids)) for ids, _ in rows]
    labels = [lab + [-100] * (width - len(lab)) for _, lab in rows]
    attn = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }


def run_sft(model, tokenizer, tasks: list[Task], cfg: SftConfig, seed: int) -> dict:
    """Train in place; returns {"losses": [...], "aux_losses": [...], "nan": bool}."""
    import torch

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = random.Random(seed)
    losses: list[float] = []
    aux: list[float] = []
    model.train()
    for _ in range(cfg.steps):
        batch_tasks = [rng.choice(tasks) for _ in range(cfg.batch)]
        batch = {k: v.to(device) for k, v in collate(tokenizer, batch_tasks, cfg.max_length).items()}
        out = model(**batch, output_router_logits=True, use_cache=False)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if getattr(out, "aux_loss", None) is not None:
            aux.append(float(out.aux_loss.detach()))
    model.eval()
    return {
        "losses": losses,
        "aux_losses": aux,
        "nan": any(math.isnan(x) or math.isinf(x) for x in losses + aux),
    }


__all__ = ["run_sft", "collate", "sft_text"]
