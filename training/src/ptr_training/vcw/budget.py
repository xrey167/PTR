"""VRAM and wall-time ESTIMATES for the VCW-MoE GPU plan (pure arithmetic, no torch).

Every number this module returns is an estimate from stated assumptions, not a measurement.
The assumptions are the module-level constants below; the first real run replaces them with
`metrics.json` values (hardware.peak_cuda_memory_allocated_bytes and wall time). Wall time is
returned as a (low, mid, high) range because the dominant unknowns (achieved MFU of a
HuggingFace-eager MoE, host-bound decode step time) are guesses spanning several x.

Model: allenai/OLMoE-1B-7B-0125 (16 layers, hidden 2048, expert intermediate 1024, 64 experts,
top-8, vocab 50304, 6.919B parameters, about 1.3B active per token).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 2**30

# ---- model shape (config.json of the pinned revision; also the pins of M011) -------------
LAYERS = 16
HIDDEN = 2048
EXPERT_INTERMEDIATE = 1024
EXPERTS = 64
TOP_K = 8
VOCAB = 50304
TOTAL_PARAMS = 6_919_200_000  # Hub metadata: 6919.2M
ACTIVE_PARAMS = 1_300_000_000  # OLMoE paper: 1.3B active (approximate)
EXPERT_PARAMS = LAYERS * EXPERTS * 3 * HIDDEN * EXPERT_INTERMEDIATE
NON_EXPERT_PARAMS = TOTAL_PARAMS - EXPERT_PARAMS  # attention, embeddings, head, norms, routers
KV_BYTES_PER_TOKEN = 2 * LAYERS * HIDDEN * 2  # K and V, bf16, kv heads == heads

# ---- GPU assumptions (vendor datasheet values as remembered; VERIFY before relying) -------
@dataclass(frozen=True)
class GpuAssumption:
    profile_id: str
    memory_gib: float  # from hardware/*.toml
    bf16_dense_tflops: float  # tensor-core bf16 with fp32 accumulate, dense (assumed datasheet)
    mem_bw_gbs: float  # assumed datasheet


GPUS = {
    "rtx4090-reference": GpuAssumption("rtx4090-reference", 24.0, 165.2, 1008.0),
    "rtx5090-reference": GpuAssumption("rtx5090-reference", 32.0, 209.5, 1792.0),
}

# ---- memory model assumptions -------------------------------------------------------------
CUDA_CONTEXT_GIB = 0.6  # context and cuBLAS workspaces, not in the allocator peak
DYNAMIC_OVERHEAD = 1.12  # allocator fragmentation on activation-like memory
REQUIRED_HEADROOM_GIB = 1.5  # a plan must leave this free of the card's nominal memory
ADAPTER_BYTES_PER_PARAM = 16  # fp32 weight + fp32 grad + two fp32 AdamW moments
# Saved activations per token per layer without checkpointing, bf16: attention (q,k,v,o,norms,
# softmax inputs) about 16*H elements, routed experts about 6*top_k*I elements.
ACT_ELEMS_PER_TOKEN_LAYER = 16 * HIDDEN + 6 * TOP_K * EXPERT_INTERMEDIATE
LOGITS_BYTES_PER_TOKEN = VOCAB * (2 + 4 + 4)  # bf16 logits, fp32 upcast, softmax/grad copy

# ---- speed model assumptions: (low, mid, high) = pessimistic, central, optimistic ---------
MFU = (0.04, 0.10, 0.20)  # achieved fraction of peak for HF-eager adapter training of a MoE
DECODE_STEP_OVERHEAD_S = (0.12, 0.07, 0.04)  # host-bound HF generate step, any GPU
VERIFY_S_PER_CANDIDATE = (0.08, 0.03, 0.01)  # sandboxed code tests dominate; 8 workers assumed
LABELS = ("low", "mid", "high")  # low = slowest


def adapter_state_gib(trainable_params: int) -> float:
    return trainable_params * ADAPTER_BYTES_PER_PARAM / GIB


def weights_gib() -> float:
    return TOTAL_PARAMS * 2 / GIB


def train_activation_gib(tokens: int, checkpointing: bool) -> float:
    """Activation memory of one forward/backward over `tokens` tokens (bf16)."""
    if checkpointing:
        boundary = tokens * HIDDEN * 2 * LAYERS
        one_layer = tokens * ACT_ELEMS_PER_TOKEN_LAYER * 2 * 2  # recompute plus its gradients
        return (boundary + one_layer) / GIB
    return tokens * ACT_ELEMS_PER_TOKEN_LAYER * 2 * LAYERS / GIB


def train_peak_gib(tokens: int, checkpointing: bool, trainable_params: int) -> float:
    dynamic = train_activation_gib(tokens, checkpointing) + tokens * LOGITS_BYTES_PER_TOKEN / GIB
    return weights_gib() + adapter_state_gib(trainable_params) + dynamic * DYNAMIC_OVERHEAD + CUDA_CONTEXT_GIB


def forward_peak_gib(tokens: int, trainable_params: int = 0) -> float:
    """Inference-only forward (feature extraction): no saved activations, one layer working set."""
    working = tokens * ACT_ELEMS_PER_TOKEN_LAYER * 2 + tokens * VOCAB * 2
    return weights_gib() + adapter_state_gib(trainable_params) * 0 + working * DYNAMIC_OVERHEAD / GIB + CUDA_CONTEXT_GIB


def generate_peak_gib(sequences: int, tokens_per_sequence: int, trainable_params: int) -> float:
    kv = sequences * tokens_per_sequence * KV_BYTES_PER_TOKEN / GIB
    logits = sequences * LOGITS_BYTES_PER_TOKEN / GIB
    return weights_gib() + adapter_state_gib(trainable_params) + (kv + logits) * DYNAMIC_OVERHEAD + CUDA_CONTEXT_GIB


def fits(peak_gib: float, profile_id: str) -> bool:
    return peak_gib <= GPUS[profile_id].memory_gib - REQUIRED_HEADROOM_GIB


def _flops_per_token(checkpointing: bool) -> float:
    # adapter-only training: forward 2N, input-gradient backward 2N (frozen weights need no
    # weight gradient), plus a second forward 2N when checkpointing.
    return (6 if checkpointing else 4) * ACTIVE_PARAMS


def train_seconds(tokens: float, checkpointing: bool, profile_id: str) -> tuple[float, float, float]:
    peak = GPUS[profile_id].bf16_dense_tflops * 1e12
    f = tokens * _flops_per_token(checkpointing)
    return tuple(f / (peak * m) for m in MFU)


def forward_seconds(tokens: float, profile_id: str) -> tuple[float, float, float]:
    peak = GPUS[profile_id].bf16_dense_tflops * 1e12
    f = tokens * 2 * ACTIVE_PARAMS
    return tuple(f / (peak * m) for m in MFU)


def decode_step_seconds(batch: int, profile_id: str) -> tuple[float, float, float]:
    """One decode step for `batch` sequences: max(host overhead, weight bytes / bandwidth)."""
    touched = 1.0 - (1.0 - TOP_K / EXPERTS) ** batch
    bytes_read = NON_EXPERT_PARAMS * 2 + touched * EXPERT_PARAMS * 2
    bw = bytes_read / (GPUS[profile_id].mem_bw_gbs * 1e9)
    return tuple(max(o, bw) for o in DECODE_STEP_OVERHEAD_S)


def generate_seconds(
    sequences: int, new_tokens: int, batch: int, profile_id: str
) -> tuple[float, float, float]:
    rounds = math.ceil(sequences / batch)
    step = decode_step_seconds(min(batch, sequences), profile_id)
    return tuple(rounds * new_tokens * s for s in step)


def verify_seconds(candidates: int) -> tuple[float, float, float]:
    return tuple(candidates * v for v in VERIFY_S_PER_CANDIDATE)


def _add(*parts: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(p[i] for p in parts) for i in range(3))


def hours(seconds: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(s / 3600 for s in seconds)


@dataclass(frozen=True)
class SftWorkload:
    steps: int
    micro_batch: int
    accumulation: int
    seq_tokens: int  # padded length per sequence
    checkpointing: bool
    trainable_params: int


@dataclass(frozen=True)
class RlWorkload:
    steps: int  # optimizer steps
    prompts_per_step: int
    num_generations: int
    mini_steps: int
    prompt_tokens: int
    completion_tokens: int
    trainable_params: int
    checkpointing: bool = False  # the trl-moe backend turns checkpointing off during RL


def sft_peak_gib(w: SftWorkload) -> float:
    return train_peak_gib(w.micro_batch * w.seq_tokens, w.checkpointing, w.trainable_params)


def sft_seconds(w: SftWorkload, profile_id: str):
    tokens = w.steps * w.micro_batch * w.accumulation * w.seq_tokens
    return train_seconds(tokens, w.checkpointing, profile_id)


def rl_peak_gib(w: RlWorkload) -> float:
    seqs = w.prompts_per_step * w.num_generations  # TRL per-device train batch
    train = train_peak_gib(seqs * (w.prompt_tokens + w.completion_tokens), w.checkpointing, w.trainable_params)
    gen = generate_peak_gib(seqs * w.mini_steps, w.prompt_tokens + w.completion_tokens, w.trainable_params)
    return max(train, gen)


def rl_seconds(w: RlWorkload, profile_id: str):
    seqs = w.prompts_per_step * w.num_generations
    gen_batches = math.ceil(w.steps / w.mini_steps)
    gen = generate_seconds(seqs * w.mini_steps, w.completion_tokens, seqs * w.mini_steps, profile_id)
    gen = tuple(g * gen_batches for g in gen)
    train = train_seconds(w.steps * seqs * (w.prompt_tokens + w.completion_tokens), w.checkpointing, profile_id)
    ver = verify_seconds(gen_batches * seqs * w.mini_steps)
    return _add(gen, train, ver)


def eval_seconds(
    tasks: int, samples_per_task: int, prompt_tokens: int, completion_tokens: int, batch: int, profile_id: str
):
    """Sampling `samples_per_task` completions per task, each verified once."""
    n = tasks * samples_per_task
    pre = forward_seconds(n * prompt_tokens, profile_id)
    return _add(pre, generate_seconds(n, completion_tokens, batch, profile_id), verify_seconds(n))


def rng_hours(h: tuple[float, float, float]) -> str:
    return f"{h[2]:.1f} / {h[1]:.1f} / {h[0]:.1f}"  # high (fastest) / mid / low (slowest)


def main() -> None:  # pragma: no cover - convenience printout used to fill docs
    from ptr_training.vcw.plan import print_plan

    print_plan()


if __name__ == "__main__":  # pragma: no cover
    main()
