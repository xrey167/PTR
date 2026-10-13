"""GPU memory estimator for the `trl-moe` stage configs and the M011 arms (pure Python).

Parameter counts are exact functions of the model config (`config.json` keys; tested equal to
the instantiated `OlmoeForCausalLM` and to PEFT's own count). Everything that depends on
activations is an ESTIMATE from the formulas below, not a measurement; the one input measured
here is the per-layer size of the merged expert weights PEFT keeps for backward (see
`peft_expert_materialization`). The real peak of a run lands in `metrics.json`
(`hardware.peak_cuda_memory_allocated_bytes`); compare and correct this module with it.

    python -m ptr_training.moe.memory --config training/configs/stage-moe-olmoe-sft-smoke-rtx3090.toml
    python -m ptr_training.moe.memory --m011-arm mixlora
    ptr-train-run --config <stage toml> --memory-report

What the estimate covers, per phase (the peak is the largest phase):

* weights: every parameter in the model dtype (bf16 2 bytes, fp32 4).
* trainable state: adapter parameters are fp32 (PEFT keeps LoRA weights fp32 on a bf16 base;
  the M011 adapters are created fp32): weight + grad + AdamW exp_avg + exp_avg_sq = 16 bytes per
  trainable parameter. A full fine-tune adds grad + two AdamW states in the model dtype.
* PEFT expert materialization: `target_parameters` on the fused 3-D expert tensors is applied as
  a parametrization: every forward builds the merged tensor `W + scale * lhs @ rhs` for all
  experts of a layer in the model dtype (peft 0.21 `_LoraFactorsProxy`, one `baddbmm`, for a
  single active adapter on a bf16/fp32 base). Without activation checkpointing autograd keeps
  the merged tensor of every layer for backward (measured on CPU for one real-width OLMoE layer
  in bf16 with torch saved-tensor hooks: 512 MiB gate_up + 256 MiB down = 768 MiB per layer);
  with checkpointing only the layer being recomputed holds it. The transient (the merged tensor,
  and its gradient in backward) is counted once.
* activations: with checkpointing, every layer's input (tokens x hidden) plus one layer's working
  set; without, every layer's working set. Working set per token and layer (`layer_working_bytes`,
  coefficients fitted to saved-tensor measurements on CPU, re-checked by the tests): attention,
  norms and residuals 14 x hidden in the model dtype (+ fp32 norm upcasts in bf16); router
  logits and probabilities in fp32 and the top-k indices; per routed copy (top_k of them) the
  gathered input, gate_up output, activation, product and down output (2 x hidden +
  4 x intermediate). CUDA attention kernels (SDPA) may save less than the CPU path measured.
* logits: tokens x vocab x (model dtype + fp32 upcast in the loss + fp32 gradient + fp32
  log-softmax) for training forward/backward.
* generation (eval and RL rollouts): KV cache = 2 x layers x kv_heads x head_dim x dtype bytes
  x sequences x (prompt + new tokens), plus one forward's working set for the prompt prefill.
* not inside torch's allocator (shown separately, not added to the peak): the CUDA context and
  library workspaces (CONTEXT_OVERHEAD_GIB, an assumption) and allocator fragmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 2**30
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
ADAPTER_STATE_BYTES = 16  # fp32 weight + fp32 grad + AdamW exp_avg + exp_avg_sq
CONTEXT_OVERHEAD_GIB = 0.75  # CUDA context + cuBLAS/cuDNN workspaces (assumption, not measured)
USABLE_FRACTION = 0.95  # leave 5% of the card for fragmentation (assumption)
# Measured on CPU (scratch measurement, torch saved-tensor hooks, bf16, one real-width layer):
# without checkpointing PEFT's merged expert weights of one layer stay alive for backward.
MEASURED_MERGED_EXPERT_BYTES_PER_LAYER_BF16 = 768 * 2**20

# config.json of the pinned OLMoE checkpoint (repo and revision as in
# training/configs/stage-moe-olmoe*.toml and the M011 config), read from the hub file at that
# revision on 2026-10-13 (sha256 21dc5723...1482). Used only when no config.json is given or
# cached locally.
PINNED_OLMOE_CONFIG = {
    "vocab_size": 50304, "hidden_size": 2048, "intermediate_size": 1024,
    "num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 16,
    "num_experts": 64, "num_experts_per_tok": 8, "tie_word_embeddings": False,
    "max_position_embeddings": 4096, "torch_dtype": "float32",
}
PINNED_OLMOE_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"


@dataclass(frozen=True)
class Architecture:
    vocab: int
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    experts: int
    top_k: int
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


def architecture_from_config(cfg: dict) -> Architecture:
    """From OLMoE `config.json` keys (or a `transformers.OlmoeConfig.to_dict()`)."""
    return Architecture(
        vocab=int(cfg["vocab_size"]),
        hidden=int(cfg["hidden_size"]),
        intermediate=int(cfg["intermediate_size"]),
        layers=int(cfg["num_hidden_layers"]),
        heads=int(cfg["num_attention_heads"]),
        kv_heads=int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"]),
        experts=int(cfg["num_experts"]),
        top_k=int(cfg["num_experts_per_tok"]),
        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
    )


# ------------------------------------------------------------------------- parameter counts


def parameter_counts(a: Architecture) -> dict:
    """Exact parameter counts of `OlmoeForCausalLM` (transformers 5.18 layout)."""
    hd = a.head_dim
    q, kv = a.heads * hd, a.kv_heads * hd
    attention = a.hidden * q + 2 * a.hidden * kv + q * a.hidden
    qk_norm = q + kv  # OLMoE normalises q and k (RMSNorm weights)
    layer_norms = 2 * a.hidden
    router = a.experts * a.hidden
    gate_up = a.experts * 2 * a.intermediate * a.hidden
    down = a.experts * a.hidden * a.intermediate
    per_layer = attention + qk_norm + layer_norms + router + gate_up + down
    embed = a.vocab * a.hidden
    lm_head = 0 if a.tie_embeddings else a.vocab * a.hidden
    total = embed + a.layers * per_layer + a.hidden + lm_head
    active_per_token = (
        a.layers * (attention + qk_norm + layer_norms + router + a.top_k * (gate_up + down) // a.experts)
        + a.hidden + (a.vocab * a.hidden)
    )
    return {
        "total": total,
        "embedding": embed,
        "lm_head": lm_head,
        "per_layer": per_layer,
        "per_layer_attention": attention,
        "per_layer_router": router,
        "per_layer_experts_gate_up": gate_up,
        "per_layer_experts_down": down,
        "active_per_token": active_per_token,
    }


def peft_lora_trainable(
    a: Architecture, rank: int, target_modules=(), target_parameters=()
) -> int:
    """Trainable parameters of PEFT LoRA (peft 0.21): `nn.Linear` targets get A [r, in] and
    B [out, r]; a 3-D expert tensor [E, d0, d1] targeted via `target_parameters` gets
    A [r*E, d0] and B [d1, r*E]."""
    hd = a.head_dim
    linear = {
        "q_proj": (a.hidden, a.heads * hd),
        "k_proj": (a.hidden, a.kv_heads * hd),
        "v_proj": (a.hidden, a.kv_heads * hd),
        "o_proj": (a.heads * hd, a.hidden),
    }
    experts = {
        "mlp.experts.gate_up_proj": (2 * a.intermediate, a.hidden),
        "mlp.experts.down_proj": (a.hidden, a.intermediate),
    }
    per_layer = 0
    for name in target_modules:
        if name not in linear:
            raise ValueError(f"unknown LoRA target module {name!r}")
        fan_in, fan_out = linear[name]
        per_layer += rank * (fan_in + fan_out)
    for name in target_parameters:
        if name not in experts:
            raise ValueError(f"unknown LoRA target parameter {name!r}")
        d0, d1 = experts[name]
        per_layer += rank * a.experts * (d0 + d1)
    return per_layer * a.layers


def m011_trainable(a: Architecture, method: str, num_experts: int, rank: int) -> int:
    """Closed form of the M011 arms (same arithmetic as the config's `arm_lora_computation`)."""
    n, r, h, i = num_experts, rank, a.hidden, a.intermediate
    if method == "perft_r":
        per_layer = n * (h * r + r * h) + h * n
    elif method == "mixlora":
        per_layer = n * (r * (h + 2 * i) + r * (i + h)) + h * n
    elif method == "lora":
        per_layer = 4 * r * (h + h)
    else:
        raise ValueError(f"unknown M011 method {method!r}")
    return per_layer * a.layers


# ------------------------------------------------------------------------- activations


def layer_working_bytes(a: Architecture, tokens: int, dtype_bytes: int) -> int:
    """Bytes one OLMoE decoder layer keeps for backward. Coefficients fitted to torch
    saved-tensor measurements of transformers 5.18 OLMoE on CPU (eager attention path,
    `tests/test_moe_memory.py` re-measures them): attention, q/k/RMS norms and residuals
    14 x hidden in the model dtype (+ 2 x hidden fp32 norm upcasts below fp32), router logits
    and probabilities 2 x experts fp32 plus top_k int64 indices, and per routed copy
    2 x hidden + 4 x intermediate."""
    attention = 14 * a.hidden * dtype_bytes + (8 * a.hidden if dtype_bytes < 4 else 0)
    router = 2 * a.experts * 4 + a.top_k * 8
    experts = a.top_k * (2 * a.hidden + 4 * a.intermediate) * dtype_bytes
    return tokens * (attention + router + experts)


def logits_bytes(a: Architecture, tokens: int, dtype_bytes: int) -> int:
    return tokens * a.vocab * (dtype_bytes + 12)


def kv_cache_bytes(a: Architecture, sequences: int, length: int, dtype_bytes: int) -> int:
    return 2 * a.layers * a.kv_heads * a.head_dim * dtype_bytes * sequences * length


def peft_expert_materialization(
    a: Architecture, target_parameters, dtype_bytes: int, *, checkpointing: bool, grad: bool
) -> dict:
    """Merged-weight memory of PEFT `target_parameters` LoRA on the fused expert tensors."""
    numel = 0
    for name in target_parameters:
        if name == "mlp.experts.gate_up_proj":
            numel += a.experts * 2 * a.intermediate * a.hidden
        elif name == "mlp.experts.down_proj":
            numel += a.experts * a.hidden * a.intermediate
    merged = numel * dtype_bytes
    transient = merged * (2 if grad else 1)  # merged tensor (+ its gradient in backward)
    saved = merged * a.layers if (grad and not checkpointing) else 0
    return {"per_layer_merged": merged, "saved_for_backward": saved, "transient": transient}


# ------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class TrainPhase:
    name: str
    tokens: int  # tokens in one forward/backward micro-batch
    checkpointing: bool


@dataclass(frozen=True)
class GenPhase:
    name: str
    sequences: int
    prompt_tokens: int
    new_tokens: int


@dataclass(frozen=True)
class Scenario:
    label: str
    arch: Architecture
    precision: str
    trainable: int
    full_finetune: bool = False
    lora_target_parameters: tuple[str, ...] = ()
    adapter_bytes_per_token_layer: int = 0  # extra activations of M011 adapters
    reference_copy: str = "none"  # "none" | "adapter" (PEFT ref) | "model" (full copy)
    train: tuple[TrainPhase, ...] = ()
    generate: tuple[GenPhase, ...] = ()
    arch_source: str = ""


def estimate(s: Scenario) -> dict:
    a, b = s.arch, DTYPE_BYTES[s.precision]
    counts = parameter_counts(a)
    weights = counts["total"] * b
    if s.full_finetune:
        state = s.trainable * 3 * b  # grad + exp_avg + exp_avg_sq in the model dtype
    else:
        state = s.trainable * ADAPTER_STATE_BYTES
    reference = {"none": 0, "adapter": s.trainable * 4, "model": weights}[s.reference_copy]
    resident = weights + state + reference
    phases = {}
    for p in s.train:
        work = layer_working_bytes(a, p.tokens, b) + p.tokens * s.adapter_bytes_per_token_layer
        acts = (a.layers * p.tokens * a.hidden * b + work) if p.checkpointing else a.layers * work
        mat = peft_expert_materialization(
            a, s.lora_target_parameters, b, checkpointing=p.checkpointing, grad=True
        )
        logits = logits_bytes(a, p.tokens, b)
        total = resident + acts + logits + mat["saved_for_backward"] + mat["transient"]
        phases[p.name] = {
            "kind": "train",
            "tokens_per_micro_batch": p.tokens,
            "checkpointing": p.checkpointing,
            "activations_gib": acts / GIB,
            "logits_gib": logits / GIB,
            "peft_merged_experts_saved_gib": mat["saved_for_backward"] / GIB,
            "peft_merged_experts_transient_gib": mat["transient"] / GIB,
            "peak_gib": total / GIB,
        }
    for p in s.generate:
        kv = kv_cache_bytes(a, p.sequences, p.prompt_tokens + p.new_tokens, b)
        prefill = layer_working_bytes(a, p.sequences * p.prompt_tokens, b) // 2  # no grad
        mat = peft_expert_materialization(
            a, s.lora_target_parameters, b, checkpointing=True, grad=False
        )
        total = resident + kv + prefill + mat["transient"]
        phases[p.name] = {
            "kind": "generate",
            "sequences": p.sequences,
            "length": p.prompt_tokens + p.new_tokens,
            "kv_cache_gib": kv / GIB,
            "prefill_working_gib": prefill / GIB,
            "peft_merged_experts_transient_gib": mat["transient"] / GIB,
            "peak_gib": total / GIB,
        }
    peak_phase = max(phases, key=lambda k: phases[k]["peak_gib"]) if phases else None
    peak = phases[peak_phase]["peak_gib"] if phases else resident / GIB
    return {
        "label": s.label,
        "evidence": "estimate (formulas in ptr_training/moe/memory.py), not a measurement",
        "architecture": asdict(a),
        "architecture_source": s.arch_source,
        "precision": s.precision,
        "parameters_total": counts["total"],
        "parameters_active_per_token": counts["active_per_token"],
        "trainable_parameters": s.trainable,
        "weights_gib": weights / GIB,
        "trainable_state_gib": state / GIB,
        "reference_copy": s.reference_copy,
        "reference_gib": reference / GIB,
        "phases": phases,
        "peak_phase": peak_phase,
        "peak_allocated_gib": peak,
        "outside_allocator_gib": CONTEXT_OVERHEAD_GIB,
    }


def fit(report: dict, card_gib: float) -> dict:
    """Verdict for one card of `card_gib` (estimate): fits when peak + context overhead stays
    within USABLE_FRACTION of the card."""
    need = report["peak_allocated_gib"] + report["outside_allocator_gib"]
    budget = card_gib * USABLE_FRACTION
    return {
        "card_gib": card_gib,
        "needed_gib": round(need, 2),
        "budget_gib": round(budget, 2),
        "headroom_gib": round(budget - need, 2),
        "fits": need <= budget,
    }


# ------------------------------------------------------------------------- builders


def _pretrained_architecture(hf_id: str | None, revision: str | None, config_json: Path | None):
    if config_json is not None:
        return architecture_from_config(json.loads(Path(config_json).read_text())), str(config_json)
    if hf_id:
        try:  # offline: only a config.json already in the local HF cache, never a download
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(hf_id, "config.json", revision=revision)
            if isinstance(cached, str):
                return architecture_from_config(json.loads(Path(cached).read_text())), cached
        except ImportError:
            pass
    if revision == PINNED_OLMOE_REVISION:
        return architecture_from_config(PINNED_OLMOE_CONFIG), (
            "PINNED_OLMOE_CONFIG (config.json of the pinned revision, copied into memory.py)"
        )
    raise ValueError(
        f"no architecture for {hf_id}@{revision}: pass --model-config <config.json>"
    )


def stage_scenario(cfg, config_json: Path | None = None) -> Scenario:
    """Scenario for a parsed `MoeConfig` (`ptr_training.moe.config`)."""
    from ptr_training.moe import config as moe_config

    m = cfg.model
    if m.kind == "tiny-random":
        arch = Architecture(
            vocab=moe_config.TINY_VOCAB_SIZE, hidden=m.hidden, intermediate=m.intermediate,
            layers=m.layers, heads=m.heads, kv_heads=m.heads, experts=m.experts, top_k=m.top_k,
        )
        source = "tiny-random [model] keys of the config"
    else:
        arch, source = _pretrained_architecture(m.hf_id, m.revision, config_json)
    lora = cfg.lora
    if lora.enabled:
        trainable = peft_lora_trainable(arch, lora.rank, lora.target_modules, lora.target_parameters)
        full = False
    else:
        trainable = parameter_counts(arch)["total"]
        full = True
    sft = cfg.sft
    train = [TrainPhase("sft", sft.batch * sft.max_length, sft.gradient_checkpointing)]
    generate = [GenPhase("policy_eval", 1, cfg.rl.max_prompt_length, cfg.rl.max_completion_length)]
    reference = "none"
    if cfg.rl.algorithms:
        rl = cfg.rl
        seq = rl.max_prompt_length + rl.max_completion_length
        train.append(
            TrainPhase("rl_update", rl.prompts_per_step * rl.num_generations * seq,
                       rl.gradient_checkpointing)
        )
        generate.append(
            GenPhase("rl_rollout", rl.mini_steps * rl.prompts_per_step * rl.num_generations,
                     rl.max_prompt_length, rl.max_completion_length)
        )
        if rl.beta > 0:
            reference = "adapter" if lora.enabled else "model"
    return Scenario(
        label=f"{cfg.name} ({cfg.source_path or 'config'})",
        arch=arch,
        precision=m.precision,
        trainable=trainable,
        full_finetune=full,
        lora_target_parameters=tuple(lora.target_parameters) if lora.enabled else (),
        reference_copy=reference,
        train=tuple(train),
        generate=tuple(generate),
        arch_source=source,
    )


M011_ARMS = {  # method, adapter experts, rank (mirrors routed_peft.ARMS; tested equal)
    "perft_r_emergent": ("perft_r", 8, 8),
    "perft_r_prescribed": ("perft_r", 8, 8),
    "plain_lora": ("lora", 1, 17),
    "mixlora": ("mixlora", 9, 4),
}


def m011_scenario(
    arm: str,
    *,
    micro_batch: int = 4,
    max_length: int = 1024,
    eval_batch: int = 16,
    eval_prompt_tokens: int = 1024,
    max_new_tokens: int = 384,
    checkpointing: bool = True,
    config_json: Path | None = None,
) -> Scenario:
    """M011 on OLMoE with the entrypoint's defaults (bf16 on CUDA, micro-batch 4, 1024-token
    training window, eval batch 16, up to 384 new tokens). `eval_prompt_tokens` is an
    assumption: evaluation prompts are not truncated and their length is not measured here."""
    arch, source = _pretrained_architecture(
        "allenai/OLMoE-1B-7B-0125", PINNED_OLMOE_REVISION, config_json
    )
    method, n, r = M011_ARMS[arm]
    extra = 0
    if method == "perft_r":  # fp32 copy of the block input, per-expert bottleneck, delta
        extra = 4 * (3 * arch.hidden + 3 * n * r)
    elif method == "mixlora":  # per routed copy: fp32 inputs and both deltas
        extra = arch.top_k * 4 * (2 * arch.hidden + 3 * arch.intermediate + 2 * n * r)
    return Scenario(
        label=f"M011 {arm} (olmoe, micro-batch {micro_batch} x {max_length} tokens)",
        arch=arch,
        precision="bf16",
        trainable=m011_trainable(arch, method, n, r),
        adapter_bytes_per_token_layer=extra,
        train=(TrainPhase("sft", micro_batch * max_length, checkpointing),),
        generate=(GenPhase("eval_generate", eval_batch, eval_prompt_tokens, max_new_tokens),),
        arch_source=source,
    )


def card_gib_for_profile(profile: str | None, root: Path) -> float | None:
    """Per-card memory of a hardware profile (the dual profile's 48 GiB are two 24 GiB cards)."""
    if not profile:
        return None
    import tomllib

    path = root / profile
    if not path.is_file():
        return None
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    gib = data.get("gpu_memory_gib")
    if gib is None:
        return None
    count = 2 if str(data.get("gpu", "")).startswith("2x") else 1
    return float(gib) / count


def report_for_config(path: Path, card_gib: float | None = None, config_json: Path | None = None):
    from ptr_training.moe import config as moe_config

    root = Path(__file__).resolve().parents[4]
    cfg = moe_config.load(path)
    rep = estimate(stage_scenario(cfg, config_json))
    import tomllib

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    profile = raw.get("run", {}).get("hardware_profile")
    card = card_gib if card_gib is not None else card_gib_for_profile(profile, root)
    rep["hardware_profile"] = profile
    rep["fit"] = fit(rep, card) if card else None
    return rep


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ptr_training.moe.memory", description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="trl-moe stage TOML")
    src.add_argument("--m011-arm", choices=sorted(M011_ARMS))
    p.add_argument("--card-gib", type=float, default=None,
                   help="memory of one GPU (default: the config's hardware profile; M011: 24)")
    p.add_argument("--model-config", type=Path, default=None, help="config.json to read shapes from")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--eval-prompt-tokens", type=int, default=1024)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    a = p.parse_args(argv)
    if a.config is not None:
        rep = report_for_config(a.config, a.card_gib, a.model_config)
    else:
        rep = estimate(m011_scenario(
            a.m011_arm, micro_batch=a.micro_batch_size, max_length=a.max_length,
            eval_batch=a.eval_batch, eval_prompt_tokens=a.eval_prompt_tokens,
            checkpointing=not a.no_gradient_checkpointing, config_json=a.model_config,
        ))
        rep["fit"] = fit(rep, a.card_gib or 24.0)
    json.dump(_round(rep), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
