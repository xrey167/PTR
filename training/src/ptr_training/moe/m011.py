"""M011 (routed PEFT on OLMoE) entrypoint.

    python -m ptr_training.moe.m011 --arm <arm> --seed <seed> --out <dir> \
        [--model tiny|olmoe] [--max-steps N] [--device auto|cpu|cuda] [--tune]

Trains one arm with SFT on the three domain corpora and evaluates per-domain task accuracy,
writing `run.json` and `metrics.json` (and `tuning.json` for `--tune`) into `--out`.

Importing this module needs only the standard library; torch, transformers, peft and datasets
are imported when a run starts. Pins below mirror the M011 config (`experiment.toml` /
`config.toml` live in an experiment PR, not in this tree); anything the config does not pin is
listed in `UNPINNED_CHOICES` and recorded in `run.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from ptr_training.moe import corpora

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


# ------------------------------------------------------------------------------------ helpers


def _write_json(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)  # NaN/inf must fail loudly
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for dist in ("torch", "transformers", "peft", "datasets", "huggingface_hub", "tokenizers"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


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
        cfg = OlmoeConfig(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            router_aux_loss_coef=PINS["moe_topology"]["router_aux_loss_coef"],
            output_router_logits=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            bos_token_id=tok.bos_token_id,
            tie_word_embeddings=False,
        )
        model = OlmoeForCausalLM(cfg)
        info = {"kind": "tiny-random", "hidden": 32, "layers": 2, "experts": 8, "top_k": 2}
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
        model.config.router_aux_loss_coef = PINS["moe_topology"]["router_aux_loss_coef"]
        info = {
            "kind": "olmoe",
            "repo": PINS["model_checkpoint_repo"],
            "revision": PINS["model_checkpoint_revision"],
            "tokenizer_repo": PINS["tokenizer_repo"],
            "tokenizer_revision": PINS["tokenizer_revision"],
            "dtype": str(dtype).removeprefix("torch."),
        }
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
    inst.model.train()
    for step in range(steps):
        batch = []
        for _ in range(batch_size):
            d = rng.choice(domains)
            batch.append(rng.choice(train_by_domain[d]))
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
    }


# ------------------------------------------------------------------------------------ evaluation


def generate(inst, tok, prompts: list[str], domain: str, max_new: int, device: str) -> list[str]:
    import torch

    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if inst.ctx is not None:
        inst.ctx.set_domains([domain] * len(prompts))
    with torch.no_grad():
        out = inst.model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
    if inst.ctx is not None:
        inst.ctx.set_domains(None)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def evaluate_accuracy(inst, tok, by_domain: dict, *, limit, max_new: dict, batch: int, device: str):
    from ptr_training.moe import scorers

    per_domain, counts = {}, {}
    for domain in DOMAINS:
        items = by_domain[domain][:limit] if limit else by_domain[domain]
        correct = 0
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            gens = generate(inst, tok, [e.prompt for e in chunk], domain, max_new[domain], device)
            correct += sum(scorers.score(domain, g, e.reference) for g, e in zip(gens, chunk))
        counts[domain] = len(items)
        per_domain[domain] = correct / len(items) if items else 0.0
    return per_domain, counts


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

    run_json = {
        "schema_version": SCHEMA_VERSION,
        "experiment": PINS["experiment"],
        "mode": "tune" if args.tune else "final",
        "seed": args.seed,
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
        "hyperparameters": hp,
        "training_steps": {
            "pinned": PINS["training_steps_per_arm"],
            "run": steps,
            "smoke_override": args.max_steps is not None,
        },
        "evaluation": {"split": eval_split, "limit_per_domain": args.eval_limit},
        "versions": _versions(),
        "sandbox": scorers.sandbox_info(),
        "git_commit": _git_commit(),
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
            tr = one_training(t["learning_rate"], t["batch_size"], tune_steps)
            per, counts = evaluate_accuracy(
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

    tr = one_training(hp["learning_rate"], hp["batch_size"], steps)
    per, counts = evaluate_accuracy(
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
        "counts": counts,
        "mean_task_accuracy": sum(per.values()) / len(per),
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
