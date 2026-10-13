"""M011 (routed PEFT on OLMoE) entrypoint.

    python -m ptr_training.moe.m011 --arm <arm> --seed <seed> --out <dir> \
        [--model tiny|olmoe] [--max-steps N] [--device auto|cpu|cuda] [--tune]

Trains one arm with SFT on the three domain corpora and evaluates per-domain task accuracy,
writing `run.json` and `metrics.json` (and `tuning.json` for `--tune`) into `--out`.

Importing this module needs only the standard library; torch, transformers, peft and datasets
are imported when a run starts. Pins below mirror the M011 config
(`experiments/model/M011-routed-peft/config.toml`; `tests/test_m011_*` compare them); anything
the config does not pin is listed in `UNPINNED_CHOICES` and recorded in `run.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from ptr_training.moe import corpora, provenance

SCHEMA_VERSION = 1
DOMAINS = ("code", "math", "tool")

PINS = {
    "experiment": "M011-routed-peft",
    "model_checkpoint_repo": "allenai/OLMoE-1B-7B-0125",
    "model_checkpoint_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
    "tokenizer_repo": "allenai/OLMoE-1B-7B-0125",
    "tokenizer_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
    "moe_topology": {
        "layers": 16,
        "num_experts": 64,
        "top_k": 8,
        "router_layout": "per-layer linear router, softmax then top-k, norm_topk_prob = false",
        "router_aux_loss_coef": 0.01,
    },
    "training_steps_per_arm": 2000,
    "domains": list(DOMAINS),
    "min_seeds": 5,
    "seeds": [17, 29, 43, 71, 101],
    # TB1: tune on the pilot seed only (>= 1001, never one of the confirmatory seeds), each trial
    # at 25% of the pinned step budget (rounded down, at least 1); the selected hyperparameters
    # are frozen for every confirmatory seed.
    "tuning_seed": 1001,
    "tuning_protocol": {"seed_rule": "pilot seed, never confirmatory", "steps_fraction": 0.25},
    "corpora": {
        k: {"repo": p.repo, "revision": p.revision, "config": p.config, "file": p.file}
        for k, p in corpora.PINNED_CORPORA.items()
    },
    "tuning_search_spec": {
        "space": {
            "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
            "batch_size": {"type": "choice", "values": [16, 32, 64]},
        },
        "trial_budget": 8,
        "sampler": "seeded random search",
        "selection_seed": 20240607,
        "objective": "macro-average validation task accuracy over code, math and tool",
        "tie_break": "lowest trial index",
    },
}

# Choices the M011 config does not pin (recorded verbatim in run.json).
UNPINNED_CHOICES = {
    "optimizer": "AdamW, weight_decay 0, constant learning rate, grad-norm clip 1.0",
    "training_sampler": "domain-balanced: domain uniform, then example uniform, with replacement",
    "loss": "completion-only cross entropy + router_aux_loss_coef * HF aux loss (base router)",
    "perft_activation": "silu; W_up zero-init so the adapter starts as a no-op",
    "perft_gate_weights": "softmax over the top-k selected adapter-router logits",
    "adapter_router_aux_loss": "none (the adapter router has no balancing loss)",
    "lora_dropout": 0.0,
    "max_length_tokens": {"tiny": 160, "olmoe": 1024},
    "prompt_truncation": "left-truncate the prompt so the answer + EOS always fits",
    "generation": "greedy; max_new_tokens math 256 / code 384 / tool 128 (tiny: 8)",
    "gsm8k_target": "reference answer text with calculator annotations, leading space",
    "bfcl_target": "name(param=value, ...) first allowed value, optional params omitted",
    "bfcl_scorer": "simplified single-call AST check (see scorers.py), not the official evaluator",
    "humaneval_scorer": "pass@1 greedy, subprocess, 10 s timeout, unshare net namespace if usable",
    "routing_probe": "teacher-forced eval items, first 8 per domain, batch of 1, <=256 tokens",
    "routing_metrics": "load CV / entropy via routing_metrics.py, mean over layers",
    "tuning_trial_set": "same trial set for every arm (one selection_seed), prefix-stable in budget",
    "default_untuned_hparams": {"olmoe": {"lr": 1e-4, "batch": 16}, "tiny": {"lr": 1e-3, "batch": 4}},
    "micro_batch_size": "gradient accumulation; default 4 (olmoe) / whole batch (tiny)",
    "flops": "torch FlopCounterMode on the first optimizer step; total = first step x steps",
    "latency_per_token": "wall seconds of the batched greedy evaluation generation of a domain "
    "divided by the tokens it generated (up to and including EOS, padding excluded); eval batch "
    "16 (olmoe) / 4 (tiny)",
}

# Manifest metrics this entrypoint does not compute, with the reason (never a silent gap).
NOT_PRODUCED = {
    "expert_specialisation_index": "no definition is pinned in the M011 manifest or config; "
    "per-domain adapter-router statistics are recorded under routing.per_domain_adapter_router",
}

# Tiny-random model (`--model tiny`): its config is part of the record (digest in run.json).
TINY_MODEL_CONFIG = {
    "hidden_size": 32, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 4, "num_experts": 8, "num_experts_per_tok": 2,
    "max_position_embeddings": 512, "tie_word_embeddings": False,
}

MAX_NEW_TOKENS = {"math": 256, "code": 384, "tool": 128}
PROBE_ITEMS = 8
PROBE_MAX_TOKENS = 256


# ------------------------------------------------------------------------------------ tuning


def sample_trials(spec: dict = PINS["tuning_search_spec"], budget: int | None = None) -> list[dict]:
    """Seeded random search: trial i draws `learning_rate` (log-uniform) then `batch_size`.

    One `random.Random(selection_seed)` stream, so a smaller budget is a prefix of a larger one.
    """
    budget = spec["trial_budget"] if budget is None else budget
    rng = random.Random(spec["selection_seed"])
    lr = spec["space"]["learning_rate"]
    batches = spec["space"]["batch_size"]["values"]
    trials = []
    for i in range(budget):
        u = rng.random()
        trials.append(
            {
                "index": i,
                "learning_rate": math.exp(
                    math.log(lr["low"]) + u * (math.log(lr["high"]) - math.log(lr["low"]))
                ),
                "batch_size": rng.choice(batches),
            }
        )
    return trials


def trial_set_digest(trials: list[dict]) -> str:
    canon = json.dumps(
        [[t["index"], t["learning_rate"], t["batch_size"]] for t in trials], separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def select_best(results: list[dict]) -> dict:
    """Highest macro validation accuracy; ties go to the lowest trial index."""
    return max(results, key=lambda r: (r["macro_accuracy"], -r["index"]))


def tuning_steps_for(budget_steps: int) -> int:
    """TB1: floor(0.25 * pinned per-arm budget), at least 1."""
    return max(1, math.floor(PINS["tuning_protocol"]["steps_fraction"] * budget_steps))


def tuning_deviations(*, seed: int, steps: int, trials: int) -> list[dict]:
    """Departures of a tuning run from the pinned protocol (empty for a conforming run)."""
    pinned = {
        "tuning_seed": PINS["tuning_seed"],
        "tuning_steps": tuning_steps_for(PINS["training_steps_per_arm"]),
        "tuning_trials_used": PINS["tuning_search_spec"]["trial_budget"],
    }
    used = {"tuning_seed": seed, "tuning_steps": steps, "tuning_trials_used": trials}
    return [
        {"parameter": k, "pinned": pinned[k], "used": used[k]} for k in pinned if used[k] != pinned[k]
    ]


def check_tuning_seed(seed: int) -> None:
    """TB1 / AGENTS.md: tune only on a pilot seed (>= the pinned tuning seed 1001), never on a
    confirmatory seed. Anything else is refused, not recorded as a deviation."""
    if seed in PINS["seeds"] or seed < PINS["tuning_seed"]:
        raise SystemExit(
            f"m011: tuning seed {seed} refused: tuning runs only on a pilot seed >= "
            f"{PINS['tuning_seed']} (pinned {PINS['tuning_seed']}), never on a confirmatory "
            f"seed {PINS['seeds']}"
        )


_TOPOLOGY_FIELDS = ("layers", "num_experts", "top_k", "norm_topk_prob", "router_aux_loss_coef")


def pinned_topology(pinned: dict = PINS["moe_topology"]) -> dict:
    """The comparable fields of the pinned `moe_topology` (norm_topk_prob is read from the
    pinned router layout text)."""
    out = {k: pinned[k] for k in ("layers", "num_experts", "top_k", "router_aux_loss_coef")}
    m = re.search(r"norm_topk_prob = (true|false)", pinned.get("router_layout", ""))
    if m:
        out["norm_topk_prob"] = m.group(1) == "true"
    return out


def observed_topology(config) -> dict:
    """The same fields read from a loaded model's config."""
    return {
        "layers": getattr(config, "num_hidden_layers", None),
        "num_experts": getattr(config, "num_experts", None),
        "top_k": getattr(config, "num_experts_per_tok", None),
        "norm_topk_prob": getattr(config, "norm_topk_prob", None),
        "router_aux_loss_coef": getattr(config, "router_aux_loss_coef", None),
    }


def topology_mismatches(observed: dict, pinned: dict = PINS["moe_topology"]) -> list[dict]:
    """Fields where the loaded model's topology differs from the pin (empty when it matches)."""
    want = pinned_topology(pinned)
    return [
        {"field": f, "pinned": want[f], "observed": observed.get(f)}
        for f in _TOPOLOGY_FIELDS
        if f in want and observed.get(f) != want[f]
    ]


def outside_training_counts(drawn_item_ids, parts: dict) -> dict:
    """Per corpus: how many drawn training examples are not in that corpus's training
    partition (M011 `training_data_outside_training_partition_count`, measured, not assumed)."""
    owner, train_ids = {}, {}
    for key, p in parts.items():
        train_ids[key] = {ex.item_id for ex in p.train}
        for ex in (*p.train, *p.validation, *p.evaluation):
            owner[ex.item_id] = key
    counts = {key: 0 for key in parts}
    for item in drawn_item_ids:
        key = owner.get(item, "unknown")
        if key == "unknown" or item not in train_ids[key]:
            counts[key] = counts.get(key, 0) + 1
    return counts


# ------------------------------------------------------------------------------------ helpers


def _write_json(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)  # NaN/inf must fail loudly
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _versions() -> dict:
    return provenance.package_versions()


def _git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return res.stdout.strip() or None if res.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def resolve_device(choice: str) -> str:
    import torch

    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return choice


# ------------------------------------------------------------------------------------ model


def build_model(kind: str, device: str, seed: int):
    """(model, tokenizer, info). `tiny`: random 2-layer OLMoE with the local char tokenizer."""
    import torch

    from ptr_training.moe.model import build_tokenizer, seed_everything

    seed_everything(seed)
    if kind == "tiny":
        from transformers import OlmoeConfig, OlmoeForCausalLM

        tok = build_tokenizer()
        tiny = {
            **TINY_MODEL_CONFIG,
            "vocab_size": len(tok),
            "router_aux_loss_coef": PINS["moe_topology"]["router_aux_loss_coef"],
        }
        cfg = OlmoeConfig(
            **tiny,
            output_router_logits=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            bos_token_id=tok.bos_token_id,
        )
        model = OlmoeForCausalLM(cfg)
        digest = hashlib.sha256(
            json.dumps(tiny, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        info = {
            "kind": "tiny-random", "hidden": 32, "layers": 2, "experts": 8, "top_k": 2,
            "tiny_config": tiny, "tiny_config_digest": "sha256:" + digest,
            "topology_observed": observed_topology(model.config),
            # The pinned topology is OLMoE's; the tiny model is compared by its config digest.
            "model_pin_mismatches": None,
        }
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            PINS["tokenizer_repo"], revision=PINS["tokenizer_revision"]
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            PINS["model_checkpoint_repo"], revision=PINS["model_checkpoint_revision"], dtype=dtype
        )
        # Observed before the aux-loss coefficient is set, so a checkpoint that differs from the
        # pin is reported, not overwritten silently.
        topology = observed_topology(model.config)
        model.config.router_aux_loss_coef = PINS["moe_topology"]["router_aux_loss_coef"]
        info = {
            "kind": "olmoe",
            "repo": PINS["model_checkpoint_repo"],
            "revision": PINS["model_checkpoint_revision"],
            "tokenizer_repo": PINS["tokenizer_repo"],
            "tokenizer_revision": PINS["tokenizer_revision"],
            "dtype": str(dtype).removeprefix("torch."),
            "topology_observed": topology,
            "topology_mismatches": topology_mismatches(topology),
            # transformers stores the hub commit the files were resolved from.
            "checkpoint_commit_observed": getattr(model.config, "_commit_hash", None),
        }
        commit = info["checkpoint_commit_observed"]
        info["checkpoint_commit_matches_pin"] = (
            None if commit is None else commit == PINS["model_checkpoint_revision"]
        )
        info["model_pin_mismatches"] = len(info["topology_mismatches"]) + (
            info["checkpoint_commit_matches_pin"] is False
        )
    return model.to(device), tok, info


# ------------------------------------------------------------------------------------ data


def encode_example(tok, ex, max_length: int) -> tuple[list[int], list[int], bool]:
    """(input_ids, labels, truncated): completion-only labels; the prompt is left-truncated so
    the answer and EOS always survive (answer capped at half the window)."""
    prompt = tok(ex.prompt, add_special_tokens=False)["input_ids"]
    answer = tok(ex.answer, add_special_tokens=False)["input_ids"][: max_length // 2 - 1]
    answer = answer + [tok.eos_token_id]
    room = max_length - len(answer)
    truncated = len(prompt) > room
    prompt = prompt[-room:] if truncated else prompt
    return prompt + answer, [-100] * len(prompt) + answer, truncated


def collate(tok, examples, max_length: int):
    import torch

    rows = [encode_example(tok, ex, max_length) for ex in examples]
    width = max(len(ids) for ids, _, _ in rows)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([ids + [pad] * (width - len(ids)) for ids, _, _ in rows]),
        "attention_mask": torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i, _, _ in rows]),
        "labels": torch.tensor([lab + [-100] * (width - len(lab)) for _, lab, _ in rows]),
    }, sum(t for _, _, t in rows)


# ------------------------------------------------------------------------------------ training


def train(
    inst,
    tok,
    train_by_domain: dict,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    micro_batch: int,
    seed: int,
    max_length: int,
    device: str,
    log_every: int = 50,
) -> dict:
    import torch

    params = [p for p in inst.model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    domains = [d for d in DOMAINS if train_by_domain.get(d)]
    losses, truncated, first_flops = [], 0, None
    drawn: list[str] = []
    inst.model.train()
    for step in range(steps):
        batch = []
        for _ in range(batch_size):
            d = rng.choice(domains)
            batch.append(rng.choice(train_by_domain[d]))
            drawn.append(batch[-1].item_id)
        opt.zero_grad(set_to_none=True)
        counter = None
        if step == 0:
            try:
                from torch.utils.flop_counter import FlopCounterMode

                counter = FlopCounterMode(display=False)
                counter.__enter__()
            except Exception:  # pragma: no cover - flop counting is best effort
                counter = None
        total = 0.0
        chunks = [batch[i : i + micro_batch] for i in range(0, len(batch), micro_batch)]
        for chunk in chunks:
            enc, trunc = collate(tok, chunk, max_length)
            truncated += trunc
            enc = {k: v.to(device) for k, v in enc.items()}
            if inst.ctx is not None:
                inst.ctx.set_domains([ex.domain for ex in chunk])
            out = inst.model(**enc, output_router_logits=True, use_cache=False)
            loss = out.loss * (len(chunk) / len(batch))
            loss.backward()
            total += float(loss.detach())
        if counter is not None:
            counter.__exit__(None, None, None)
            first_flops = int(counter.get_total_flops())
        if not math.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(total)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1}/{steps} loss {total:.4f}", file=sys.stderr, flush=True)
    inst.model.eval()
    if inst.ctx is not None:
        inst.ctx.set_domains(None)
    return {
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "first_loss": losses[0] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "truncated_examples": truncated,
        "first_step_flops": first_flops,
        "estimated_total_flops": None if first_flops is None else first_flops * steps,
        "examples_drawn": len(drawn),
        # Item ids of every drawn example; run() reduces them to the partition counters and
        # removes the list before writing JSON.
        "drawn_item_ids": drawn,
    }


# ------------------------------------------------------------------------------------ evaluation


def generate(
    inst, tok, prompts: list[str], domain: str, max_new: int, device: str, timing: dict | None = None
) -> list[str]:
    """Greedy batched generation. `timing`, when given, accumulates `seconds` and
    `generated_tokens` (up to and including EOS, padding excluded)."""
    import torch

    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if inst.ctx is not None:
        inst.ctx.set_domains([domain] * len(prompts))
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = inst.model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if inst.ctx is not None:
        inst.ctx.set_domains(None)
    new = out[:, enc["input_ids"].shape[1] :]
    if timing is not None:
        timing["seconds"] = timing.get("seconds", 0.0) + elapsed
        timing["generated_tokens"] = timing.get("generated_tokens", 0) + generated_token_count(
            new.tolist(), tok.eos_token_id
        )
    return tok.batch_decode(new, skip_special_tokens=True)


def generated_token_count(rows: list[list[int]], eos_id: int) -> int:
    """Tokens a batched greedy generation produced per row: up to and including the first EOS
    (generate pads finished rows after their EOS), else the whole row."""
    total = 0
    for row in rows:
        total += row.index(eos_id) + 1 if eos_id in row else len(row)
    return total


def evaluate_accuracy(inst, tok, by_domain: dict, *, limit, max_new: dict, batch: int, device: str):
    from ptr_training.moe import scorers

    per_domain, counts, latency = {}, {}, {}
    for domain in DOMAINS:
        items = by_domain[domain][:limit] if limit else by_domain[domain]
        if not items:
            raise RuntimeError(f"no {domain} items to evaluate (an accuracy of 0 would be invented)")
        correct, timing = 0, {}
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            gens = generate(
                inst, tok, [e.prompt for e in chunk], domain, max_new[domain], device, timing
            )
            correct += sum(scorers.score(domain, g, e.reference) for g, e in zip(gens, chunk))
        counts[domain] = len(items)
        per_domain[domain] = correct / len(items)
        tokens = timing.get("generated_tokens", 0)
        latency[domain] = {
            "seconds": timing.get("seconds", 0.0),
            "generated_tokens": tokens,
            "latency_per_token": timing["seconds"] / tokens if tokens else None,
            "batch": batch,
        }
    return per_domain, counts, latency


def routing_probe(inst, tok, by_domain: dict, device: str) -> dict:
    """Teacher-forced router statistics on a few evaluation items per domain."""
    import torch

    from ptr_training.moe import routing_metrics as rm

    top_k = inst.model.config.num_experts_per_tok
    base_layers: list | None = None
    adapter_layers: list | None = None
    per_domain_adapter = {}

    def merge(acc, new):
        return new if acc is None else [a + n for a, n in zip(acc, new)]

    if inst.ctx is not None:
        inst.ctx.recording = True
    try:
        for domain in DOMAINS:
            dom_adapter = None
            for ex in by_domain[domain][:PROBE_ITEMS]:
                ids = tok(ex.prompt + ex.answer, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([ids[:PROBE_MAX_TOKENS]], device=device)
                if inst.ctx is not None:
                    inst.ctx.set_domains([domain])
                with torch.no_grad():
                    out = inst.model(input_ids=ids, output_router_logits=True, use_cache=False)
                base_layers = merge(base_layers, rm.router_logits_to_lists(out.router_logits))
                if inst.ctx is not None:
                    got = inst.ctx.take_records()
                    adapter_layers = merge(adapter_layers, got)
                    dom_adapter = merge(dom_adapter, got)
            if dom_adapter is not None:
                per_domain_adapter[domain] = dom_adapter
    finally:
        if inst.ctx is not None:
            inst.ctx.recording = False
            inst.ctx.set_domains(None)

    def summarise(layers, k):
        stats = rm.snapshot_metrics(layers, k)
        n = len(stats)
        return {
            "load_cv": sum(s["load_cv"] for s in stats) / n,
            "entropy": sum(s["entropy"] for s in stats) / n,
            "dead_experts": sum(s["dead_experts"] for s in stats) / n,
            "tokens_per_layer": len(layers[0]),
        }

    result = {"base_router": summarise(base_layers, top_k), "adapter_router": None}
    if adapter_layers is not None:
        k = inst.spec.router_top_k
        result["adapter_router"] = summarise(adapter_layers, k)
        result["per_domain_adapter_router"] = {
            d: summarise(v, k) for d, v in per_domain_adapter.items()
        }
    return result


# ------------------------------------------------------------------------------------ run


def _arm_report(spec) -> dict:
    return {
        "arm": spec.name,
        "method": spec.method,
        "specialisation": spec.specialisation,
        "num_experts": spec.num_experts,
        "router_top_k": spec.router_top_k,
        "rank": spec.rank,
        "alpha": spec.alpha,
        "pinned_trainable_params_olmoe": spec.pinned_trainable_params,
    }


def _load_tuning(path, arm: str) -> dict:
    """Read a tuning.json and refuse one that another arm's --tune run produced."""
    tuned = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuned.get("arm") != arm:
        raise SystemExit(
            f"m011: {path} was tuned for arm {tuned.get('arm')!r}, not {arm!r}; "
            "hyperparameters are selected per arm"
        )
    seed = tuned.get("tuning_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SystemExit(f"m011: {path} records no integer tuning_seed")
    check_tuning_seed(seed)
    return tuned


def _hparams(args, kind: str) -> tuple[dict, dict | None]:
    """(hyperparameters, tuning-source record); explicit flags beat --hparams beat defaults."""
    default = UNPINNED_CHOICES["default_untuned_hparams"][kind]
    source, tuned = "default-untuned", None
    lr, bs = default["lr"], default["batch"]
    if args.hparams:
        tuned = _load_tuning(args.hparams, args.arm)
        lr, bs = tuned["selected"]["learning_rate"], tuned["selected"]["batch_size"]
        source = "tuned"
    if args.lr is not None or args.batch_size is not None:
        lr = args.lr if args.lr is not None else lr
        bs = args.batch_size if args.batch_size is not None else bs
        source = "explicit"
    return {"learning_rate": lr, "batch_size": bs, "source": source}, tuned


def run(args) -> int:
    t0 = time.time()
    if args.tune:
        check_tuning_seed(args.seed)  # before anything is imported, built or written
    try:
        provenance.refuse_distributed_launch()
    except provenance.DistributedLaunchRefused as error:
        raise SystemExit(f"m011: {error}") from error
    from ptr_training.moe import routed_peft as rp
    from ptr_training.moe import scorers

    # HumanEval scoring runs model-generated code: refuse before any training, not after it.
    if scorers.sandbox_info()["filesystem"] == "refused":
        raise SystemExit(
            "m011: no bubblewrap sandbox for HumanEval (apt install bubblewrap); set "
            f"{scorers.UNSANDBOXED_ENV}=1 only to accept running generated code as this user"
        )
    # A tuning file of another arm is refused before any model is built.
    if args.hparams:
        _load_tuning(args.hparams, args.arm)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = args.model
    device = resolve_device(args.device)
    model, tok, model_info = build_model(kind, device, args.seed)
    use_gc = args.gradient_checkpointing if args.gradient_checkpointing is not None else kind == "olmoe"
    if use_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rp_arm = rp.ARMS[args.arm]
    inst = rp.install_arm(model, rp_arm)
    if inst.ctx is not None and inst.ctx.masked:
        inst.ctx.set_domains(None)
    trainable = rp.count_trainable_parameters(inst.model)
    leaked = rp.non_adapter_trainable(inst.model)
    if leaked:
        raise RuntimeError(f"non-adapter parameters are trainable: {leaked[:5]}")
    cfg = inst.model.config
    expected = rp.expected_pinned_params(
        rp_arm, cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    )
    if trainable != expected or (kind == "olmoe" and trainable != rp_arm.pinned_trainable_params):
        raise RuntimeError(
            f"{args.arm}: {trainable} trainable parameters, closed form {expected}, pinned "
            f"{rp_arm.pinned_trainable_params} (olmoe); the arm does not match its pinned topology"
        )

    source = args.corpora or ("fixture" if kind == "tiny" else "hf")
    parts = corpora.fixture_partitions() if source == "fixture" else corpora.load_pinned()
    corpus_reports = {k: corpora.corpus_report(k, p) for k, p in parts.items()}
    max_length = UNPINNED_CHOICES["max_length_tokens"][kind]
    max_new = {
        d: args.max_new_tokens or (8 if kind == "tiny" else MAX_NEW_TOKENS[d]) for d in DOMAINS
    }
    micro = args.micro_batch_size
    steps = args.max_steps or PINS["training_steps_per_arm"]
    hp, tuned_in = _hparams(args, kind)
    eval_split = "validation" if args.tune else "evaluation"
    eval_by_domain = corpora.by_domain(parts, eval_split)
    train_by_domain = corpora.by_domain(parts, "train")
    eval_batch = 16 if kind == "olmoe" else 4

    init_state = {n: p.detach().clone() for n, p in inst.model.named_parameters() if p.requires_grad}

    def reset():
        import torch

        own = dict(inst.model.named_parameters())
        with torch.no_grad():
            for n, v in init_state.items():
                own[n].copy_(v)

    def one_training(lr, bs, nsteps):
        reset()
        return train(
            inst, tok, train_by_domain, steps=nsteps, lr=lr, batch_size=bs,
            micro_batch=micro or (4 if kind == "olmoe" else bs), seed=args.seed,
            max_length=max_length, device=device,
        )

    corpus_reports_for = corpus_reports  # filled with the measured partition counters below

    def record_training(tr: dict) -> dict:
        """Fold one training's drawn item ids into the per-corpus counters, drop the id list."""
        counts = outside_training_counts(tr.pop("drawn_item_ids"), parts)
        for key, n in counts.items():
            report = corpus_reports_for.setdefault(key, {})
            report["training_outside_training_partition_count"] = (
                (report.get("training_outside_training_partition_count") or 0) + n
            )
        return tr

    run_json = {
        "schema_version": SCHEMA_VERSION,
        "experiment": PINS["experiment"],
        "mode": "tune" if args.tune else "final",
        "seed": args.seed,
        # Every RNG consumer of the run: model and adapter init (torch/python/numpy seeded
        # before the model is built), the domain-balanced training sampler, the trial sampler.
        "seeds": {
            "run": args.seed,
            "model_and_adapter_init": args.seed,
            "training_sampler": args.seed,
            "tuning_selection_seed": PINS["tuning_search_spec"]["selection_seed"],
        },
        **_arm_report(rp_arm),
        "device": device,
        "pins": PINS,
        "unpinned_choices": UNPINNED_CHOICES,
        "model": model_info,
        "corpora_source": source,
        "corpora": corpus_reports,
        "expert_domain_assignment": {
            "mapping": rp.PRESCRIBED_ASSIGNMENT,
            "digest": rp.assignment_digest(rp.PRESCRIBED_ASSIGNMENT),
            "matches_pinned_digest": rp.assignment_digest(rp.PRESCRIBED_ASSIGNMENT)
            == rp.PRESCRIBED_ASSIGNMENT_DIGEST,
            "applied": bool(inst.ctx is not None and inst.ctx.masked),
        },
        "trainable_params": trainable,
        "trainable_params_closed_form": expected,
        "hyperparameters": hp,
        "training_config": {
            "micro_batch_size": micro or (4 if kind == "olmoe" else hp["batch_size"]),
            "gradient_checkpointing": use_gc,
            "max_length_tokens": max_length,
            "eval_batch": eval_batch,
            "max_new_tokens": max_new,
        },
        "training_steps": {
            "pinned": PINS["training_steps_per_arm"],
            "run": steps,
            "smoke_override": args.max_steps is not None,
        },
        "evaluation": {"split": eval_split, "limit_per_domain": args.eval_limit},
        "versions": _versions(),
        "sandbox": scorers.sandbox_info(),
        "git_commit": _git_commit(),
        "provenance": provenance.collect(device),
        "model_pin_mismatches": model_info.get("model_pin_mismatches"),
        "tuning_trials_used": 0,
        "tuning_trial_set_digest": None,
    }
    if tuned_in is not None:
        run_json["tuning_trials_used"] = tuned_in["tuning_trials_used"]
        run_json["tuning_trial_set_digest"] = tuned_in["trial_set_digest"]
        run_json["tuning_source"] = str(args.hparams)
        run_json["tuning_arm"] = tuned_in["arm"]
        run_json.update(
            tuning_seed=tuned_in["tuning_seed"], tuning_steps=tuned_in["tuning_steps"],
            tuning_selected=tuned_in["selected"], tuning_deviations=tuned_in["deviations"],
        )

    if args.tune:
        trials = sample_trials(PINS["tuning_search_spec"], args.tune_budget)
        results = []
        tune_steps = args.tune_steps or tuning_steps_for(steps)
        for t in trials:
            tr = record_training(one_training(t["learning_rate"], t["batch_size"], tune_steps))
            per, counts, _ = evaluate_accuracy(
                inst, tok, eval_by_domain, limit=args.eval_limit, max_new=max_new,
                batch=eval_batch, device=device,
            )
            results.append({**t, "per_domain_accuracy": per, "counts": counts,
                            "macro_accuracy": sum(per.values()) / len(per),
                            "final_loss": tr["final_loss"]})
            print(f"trial {t['index']}: macro {results[-1]['macro_accuracy']:.4f}", file=sys.stderr)
        best = select_best(results)
        deviations = tuning_deviations(seed=args.seed, steps=tune_steps, trials=len(results))
        tuning = {
            "arm": args.arm,
            "seed": args.seed,
            "tuning_seed": args.seed,
            "tuning_steps": tune_steps,
            "tuning_steps_rule": "floor(0.25 * training steps), at least 1",
            "frozen_for_all_seeds": True,
            "deviations": deviations,
            "search_spec": PINS["tuning_search_spec"],
            "tuning_trials_used": len(results),
            "trial_set_digest": trial_set_digest(trials),
            "trials": results,
            "selected_trial": best["index"],
            "selected": {"learning_rate": best["learning_rate"], "batch_size": best["batch_size"]},
            "validation_digests": {k: r["selection_digest"]["validation"] for k, r in corpus_reports.items()},
        }
        _write_json(out_dir / "tuning.json", tuning)
        run_json["tuning_trials_used"] = len(results)
        run_json["tuning_trial_set_digest"] = tuning["trial_set_digest"]
        run_json.update(
            tuning_seed=args.seed, tuning_steps=tune_steps, tuning_selected=tuning["selected"],
            tuning_deviations=deviations,
        )
        run_json["wall_seconds"] = time.time() - t0
        _write_json(out_dir / "run.json", run_json)
        print(json.dumps({"selected": tuning["selected"], "trials": len(results)}))
        return 0

    tr = record_training(one_training(hp["learning_rate"], hp["batch_size"], steps))
    per, counts, latency = evaluate_accuracy(
        inst, tok, eval_by_domain, limit=args.eval_limit, max_new=max_new,
        batch=eval_batch, device=device,
    )
    probe = routing_probe(inst, tok, eval_by_domain, device)
    primary = probe["adapter_router"] or probe["base_router"]
    metrics = {
        "arm": args.arm,
        "seed": args.seed,
        "split": eval_split,
        "per_domain_accuracy": per,
        # The manifest's name for the same values.
        "task_accuracy_per_domain": per,
        "counts": counts,
        "mean_task_accuracy": sum(per.values()) / len(per),
        "latency_per_token": {d: v["latency_per_token"] for d, v in latency.items()},
        "generation_timing": latency,
        "training_flops": tr["estimated_total_flops"],
        "training_flops_source": "first optimizer step x steps (FlopCounterMode)",
        "training_data_outside_training_partition_count": sum(
            r["training_outside_training_partition_count"] for r in corpus_reports.values()
        ),
        "validation_set_digest_by_corpus": {
            k: r["selection_digest"]["validation"] for k, r in corpus_reports.items()
        },
        "validation_overlap_count_by_corpus": {
            k: r["validation_overlap_count"] for k, r in corpus_reports.items()
        },
        "model_pin_mismatches": model_info.get("model_pin_mismatches"),
        "not_produced": NOT_PRODUCED,
        "load_cv": primary["load_cv"],
        "routing_entropy": primary["entropy"],
        "routing_source": "adapter_router" if probe["adapter_router"] else "base_router",
        "routing": probe,
        "trainable_params": trainable,
        "training": tr,
    }
    run_json["wall_seconds"] = time.time() - t0
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "run.json", run_json)
    print(json.dumps({"arm": args.arm, "mean_task_accuracy": metrics["mean_task_accuracy"]}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ptr_training.moe.m011", description=__doc__)
    p.add_argument(
        "--arm", required=True,
        choices=["perft_r_emergent", "perft_r_prescribed", "plain_lora", "mixlora"],
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="run seed; for --tune defaults to the pilot tuning seed (1001)",
    )
    p.add_argument(
        "--all-seeds", action="store_true",
        help="run every pinned seed at the full budget into <out>/seed<N> (with --tune: tune "
        "first into <out>/tune, then use its frozen hyperparameters; else needs --hparams)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=["tiny", "olmoe"], default="tiny")
    p.add_argument("--max-steps", type=int, default=None, help="override the pinned 2000 steps")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tune", action="store_true", help="seeded random search on validation")
    p.add_argument("--tune-budget", type=int, default=None, help="trials (pinned: 8)")
    p.add_argument("--tune-steps", type=int, default=None, help="steps per trial (default floor(0.25*steps); recorded as a deviation)")
    p.add_argument("--hparams", default=None, help="tuning.json from a --tune run")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None, help="items per domain (smoke)")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--corpora", choices=["fixture", "hf"], default=None)
    p.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    return p


def run_all_seeds(args) -> int:
    """Desktop driver: (optionally tune on the pilot seed) then one full-budget run per pinned seed."""
    import copy

    out = Path(args.out)
    hparams = args.hparams
    if args.tune:
        t_args = copy.copy(args)
        t_args.all_seeds, t_args.seed, t_args.out = False, PINS["tuning_seed"], str(out / "tune")
        if (rc := run(t_args)) != 0:
            return rc
        hparams = str(out / "tune" / "tuning.json")
    if not hparams:
        raise SystemExit("--all-seeds needs --tune or --hparams (hyperparameters are frozen first)")
    for seed in PINS["seeds"]:
        s_args = copy.copy(args)
        s_args.tune, s_args.all_seeds, s_args.seed = False, False, seed
        s_args.hparams, s_args.out = hparams, str(out / f"seed{seed}")
        if (rc := run(s_args)) != 0:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:  # OD-3: before torch is imported anywhere in this process
        provenance.apply_ptr_cuda_device()
    except provenance.DeviceSelectionError as error:
        raise SystemExit(f"m011: {error}") from error
    if args.all_seeds:
        if args.seed is not None:
            parser.error("--all-seeds runs the pinned seeds; do not pass --seed")
        return run_all_seeds(args)
    if args.seed is None:
        if not args.tune:
            parser.error("--seed is required unless --tune or --all-seeds")
        args.seed = PINS["tuning_seed"]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
