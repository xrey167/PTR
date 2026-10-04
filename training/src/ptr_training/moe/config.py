"""Typed parsing of the MoE stage TOML (pure Python)."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ptr_training.moe.reward import LEVEL, LEVELS
from ptr_training.moe.tasks import longest_answer_chars, longest_prompt_chars

MODEL_KINDS = ("tiny-random", "pretrained")
ALGORITHMS = ("gspo", "sapo")
PRECISIONS = ("fp32", "bf16")

# Keys each table may carry. [model] also holds manifest keys (`family`, `config`) that the
# run manifest reads; shared tables ([run], [data], [training], [evaluation], [codebook])
# are owned by the manifest and are not validated here.
_MODEL_KEYS = {
    "family", "config", "kind", "hf_id", "revision", "seed", "layers", "hidden", "intermediate",
    "heads", "experts", "top_k", "max_positions", "precision", "router_aux_loss_coef",
}
_MOE_KEYS = {"sft", "rl", "lora", "output"}
_SFT_KEYS = {"steps", "lr", "batch", "examples", "max_length"}
_RL_KEYS = {
    "algorithms", "steps", "num_generations", "mini_steps", "beta", "epsilon", "epsilon_high",
    "gspo_epsilon", "gspo_epsilon_high", "sapo_temperature_pos", "sapo_temperature_neg", "lr",
    "prompts_per_step", "max_prompt_length", "max_completion_length", "temperature",
    "min_verification_level",
}
_LORA_KEYS = {"enabled", "rank", "alpha", "dropout", "target_modules", "target_parameters"}
_OUTPUT_KEYS = {"dir", "probe_tokens"}
# Architecture keys that only configure the tiny-random model; a pretrained model's
# architecture comes from the Hub config and these would be silently ignored.
_TINY_ONLY_KEYS = (
    "layers", "hidden", "intermediate", "heads", "experts", "top_k", "max_positions",
    "router_aux_loss_coef",
)
# Mutable refs: resolving them would make a "pinned" run drift.
_FLOATING_REVISIONS = {"main", "master", "HEAD"}


class ConfigError(ValueError):
    """Invalid MoE stage configuration."""


def _table(raw: dict, name: str, *, required: bool = True) -> dict:
    value = raw.get(name)
    if value is None:
        if required:
            raise ConfigError(f"missing required table [{name}]")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _check_keys(table: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(table) - allowed)
    if extra:
        raise ConfigError(f"unknown key(s) {extra} in [{where}]; allowed: {sorted(allowed)}")


def _get(table: dict, key: str, typ, where: str, default=None):
    if key not in table:
        if default is None:
            raise ConfigError(f"missing required key {where}.{key}")
        return default
    value = table[key]
    ok = isinstance(value, typ) and not (typ is int and isinstance(value, bool))
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        value, ok = float(value), True
    if not ok:
        raise ConfigError(f"{where}.{key} must be {typ.__name__}, got {value!r}")
    if typ is float and not math.isfinite(value):
        raise ConfigError(f"{where}.{key} must be finite, got {value!r}")
    return value


def _seed(value: int, where: str) -> int:
    if not 0 <= value < 2**32:  # numpy.random.seed range
        raise ConfigError(f"{where} must satisfy 0 <= seed < 2**32, got {value}")
    return value


def _str_list(table: dict, key: str, where: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{where}.{key} must be a list of non-empty strings, got {value!r}")
    return tuple(value)


def _positive(value, where: str):
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{where} must be a finite number > 0, got {value}")
    return value


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    seed: int
    hf_id: str | None = None
    revision: str | None = None
    layers: int = 2
    hidden: int = 64
    intermediate: int = 64
    heads: int = 4
    experts: int = 8
    top_k: int = 2
    max_positions: int = 128
    precision: str = "fp32"
    router_aux_loss_coef: float = 0.01


@dataclass(frozen=True)
class SftConfig:
    steps: int
    lr: float
    batch: int
    examples: int
    max_length: int = 32


@dataclass(frozen=True)
class RlConfig:
    algorithms: tuple[str, ...]
    steps: int
    num_generations: int
    mini_steps: int
    beta: float
    epsilon: float
    epsilon_high: float
    gspo_epsilon: float
    gspo_epsilon_high: float
    sapo_temperature_pos: float
    sapo_temperature_neg: float
    lr: float
    prompts_per_step: int
    max_prompt_length: int
    max_completion_length: int
    temperature: float = 1.0
    min_verification_level: str = LEVEL


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ()
    target_parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class MoeConfig:
    name: str
    seed: int
    model: ModelConfig
    sft: SftConfig
    rl: RlConfig
    lora: LoraConfig
    output_dir: str
    probe_tokens: int = 64
    source_path: str | None = field(default=None, compare=False)


def parse(raw: dict, source_path: str | None = None) -> MoeConfig:
    run = _table(raw, "run")
    seed = _seed(_get(run, "seed", int, "run"), "run.seed")
    name = _get(run, "name", str, "run")
    training = _table(raw, "training")
    if training.get("backend") != "trl-moe":
        raise ConfigError(f"[training].backend must be 'trl-moe', got {training.get('backend')!r}")

    train_precision = training.get("precision")
    m = _table(raw, "model")
    _check_keys(m, _MODEL_KEYS, "model")
    kind = _get(m, "kind", str, "model")
    if kind not in MODEL_KINDS:
        raise ConfigError(f"model.kind must be one of {MODEL_KINDS}, got {kind!r}")
    precision = _get(m, "precision", str, "model", "fp32")
    if precision not in PRECISIONS:
        raise ConfigError(f"model.precision must be one of {PRECISIONS}, got {precision!r}")
    if train_precision is not None and train_precision != precision:
        raise ConfigError(
            f"[training].precision ({train_precision!r}) differs from model.precision "
            f"({precision!r}); the backend uses model.precision only"
        )
    hf_id = m.get("hf_id")
    if kind == "pretrained":
        if not isinstance(hf_id, str) or not hf_id:
            raise ConfigError("model.hf_id is required when model.kind = 'pretrained'")
    revision = m.get("revision")
    if kind == "pretrained":
        tiny_only = [k for k in _TINY_ONLY_KEYS if k in m]
        if tiny_only:
            raise ConfigError(
                f"{['model.' + k for k in tiny_only]} only apply to model.kind = 'tiny-random'; a pretrained "
                "model's architecture comes from the Hub config (recorded in the metrics)"
            )
        if not isinstance(revision, str) or not revision.strip():
            raise ConfigError(
                "model.revision is required when model.kind = 'pretrained' "
                "(a commit sha or tag; the hub default branch is mutable)"
            )
        if revision in _FLOATING_REVISIONS:
            raise ConfigError(
                f"model.revision {revision!r} is a floating ref; pin a commit sha or tag"
            )
    elif revision is not None:
        raise ConfigError("model.revision is only valid when model.kind = 'pretrained'")
    model = ModelConfig(
        kind=kind,
        seed=_seed(_get(m, "seed", int, "model", seed), "model.seed"),
        hf_id=hf_id,
        revision=revision,
        layers=_positive(_get(m, "layers", int, "model", 2), "model.layers"),
        hidden=_positive(_get(m, "hidden", int, "model", 64), "model.hidden"),
        intermediate=_positive(_get(m, "intermediate", int, "model", 64), "model.intermediate"),
        heads=_positive(_get(m, "heads", int, "model", 4), "model.heads"),
        experts=_positive(_get(m, "experts", int, "model", 8), "model.experts"),
        top_k=_positive(_get(m, "top_k", int, "model", 2), "model.top_k"),
        max_positions=_positive(_get(m, "max_positions", int, "model", 128), "model.max_positions"),
        precision=precision,
        router_aux_loss_coef=_get(m, "router_aux_loss_coef", float, "model", 0.01),
    )
    if model.router_aux_loss_coef < 0:
        raise ConfigError(
            f"model.router_aux_loss_coef must be >= 0, got {model.router_aux_loss_coef}"
        )
    if kind == "tiny-random":
        if model.top_k > model.experts:
            raise ConfigError(f"model.top_k ({model.top_k}) exceeds model.experts ({model.experts})")
        if model.hidden % model.heads != 0:
            raise ConfigError("model.hidden must be divisible by model.heads")
        if (model.hidden // model.heads) % 2 != 0:
            raise ConfigError("model.hidden / model.heads (head dim) must be even for rotary")

    moe = _table(raw, "moe")
    _check_keys(moe, _MOE_KEYS, "moe")

    s = _table(moe, "sft")
    _check_keys(s, _SFT_KEYS, "moe.sft")
    sft = SftConfig(
        steps=_get(s, "steps", int, "moe.sft"),
        lr=_positive(_get(s, "lr", float, "moe.sft"), "moe.sft.lr"),
        batch=_positive(_get(s, "batch", int, "moe.sft"), "moe.sft.batch"),
        examples=_positive(_get(s, "examples", int, "moe.sft", 256), "moe.sft.examples"),
        max_length=_positive(_get(s, "max_length", int, "moe.sft", 32), "moe.sft.max_length"),
    )
    if sft.steps < 0:
        raise ConfigError("moe.sft.steps must be >= 0")
    longest_prompt = longest_prompt_chars()
    if sft.max_length <= longest_prompt + 1:
        raise ConfigError(
            f"moe.sft.max_length ({sft.max_length}) must exceed the longest task prompt "
            f"({longest_prompt} chars) + 1, or truncation can leave no supervised label"
        )

    r = _table(moe, "rl")
    _check_keys(r, _RL_KEYS, "moe.rl")
    algos = r.get("algorithms")
    # An empty list disables RL (SFT-only run); steps = 0 does not (see the steps check below).
    if not isinstance(algos, list) or not all(isinstance(a, str) for a in algos):
        raise ConfigError("moe.rl.algorithms must be a list of strings (empty list disables RL)")
    bad = [a for a in algos if a not in ALGORITHMS]
    if bad:
        raise ConfigError(f"moe.rl.algorithms has unknown entries {bad}; allowed {ALGORITHMS}")
    if len(set(algos)) != len(algos):
        raise ConfigError("moe.rl.algorithms has duplicates")
    epsilon = _get(r, "epsilon", float, "moe.rl", 0.2)
    rl = RlConfig(
        algorithms=tuple(algos),
        steps=_get(r, "steps", int, "moe.rl"),
        num_generations=_get(r, "num_generations", int, "moe.rl"),
        mini_steps=_positive(_get(r, "mini_steps", int, "moe.rl", 1), "moe.rl.mini_steps"),
        beta=_get(r, "beta", float, "moe.rl", 0.0),
        epsilon=_positive(epsilon, "moe.rl.epsilon"),
        epsilon_high=_positive(_get(r, "epsilon_high", float, "moe.rl", epsilon), "moe.rl.epsilon_high"),
        gspo_epsilon=_positive(
            _get(r, "gspo_epsilon", float, "moe.rl", 3e-4), "moe.rl.gspo_epsilon"
        ),
        gspo_epsilon_high=_positive(
            _get(r, "gspo_epsilon_high", float, "moe.rl", 4e-4), "moe.rl.gspo_epsilon_high"
        ),
        sapo_temperature_pos=_positive(
            _get(r, "sapo_temperature_pos", float, "moe.rl", 1.0), "moe.rl.sapo_temperature_pos"
        ),
        sapo_temperature_neg=_positive(
            _get(r, "sapo_temperature_neg", float, "moe.rl", 1.05), "moe.rl.sapo_temperature_neg"
        ),
        lr=_positive(_get(r, "lr", float, "moe.rl", 1e-5), "moe.rl.lr"),
        prompts_per_step=_positive(_get(r, "prompts_per_step", int, "moe.rl", 1), "moe.rl.prompts_per_step"),
        max_prompt_length=_positive(_get(r, "max_prompt_length", int, "moe.rl", 16), "moe.rl.max_prompt_length"),
        max_completion_length=_positive(
            _get(r, "max_completion_length", int, "moe.rl", 12), "moe.rl.max_completion_length"
        ),
        temperature=_positive(_get(r, "temperature", float, "moe.rl", 1.0), "moe.rl.temperature"),
        min_verification_level=_get(r, "min_verification_level", str, "moe.rl", LEVEL),
    )
    if rl.min_verification_level not in LEVELS:
        raise ConfigError(
            f"moe.rl.min_verification_level must be one of {LEVELS}, "
            f"got {rl.min_verification_level!r}"
        )
    if rl.steps < 1:
        # transformers only honours a positive max_steps; 0 would silently fall back to the
        # default epoch count. Disable RL with `algorithms = []` instead.
        raise ConfigError(
            f"moe.rl.steps must be >= 1, got {rl.steps} (to disable RL set moe.rl.algorithms = [])"
        )
    longest_answer = longest_answer_chars() + 1  # + EOS
    if rl.max_completion_length < longest_answer:
        raise ConfigError(
            f"moe.rl.max_completion_length ({rl.max_completion_length}) is below the longest "
            f"canonical answer + EOS over the task space ({longest_answer} tokens)"
        )
    if kind == "tiny-random":
        # positions needed: SFT sequences, and RL prompt + completion (also used by eval)
        needed = max(sft.max_length, rl.max_prompt_length + rl.max_completion_length)
        if needed > model.max_positions:
            raise ConfigError(
                f"model.max_positions ({model.max_positions}) is below the longest sequence "
                f"the run builds ({needed}: max of moe.sft.max_length and "
                "moe.rl.max_prompt_length + moe.rl.max_completion_length)"
            )
    if rl.num_generations < 2:
        raise ConfigError("moe.rl.num_generations must be >= 2 (group-relative advantages)")
    if rl.beta < 0:
        raise ConfigError("moe.rl.beta must be >= 0")
    if rl.epsilon_high < rl.epsilon and "epsilon_high" in r:
        raise ConfigError("moe.rl.epsilon_high must be >= moe.rl.epsilon")
    if rl.gspo_epsilon_high < rl.gspo_epsilon:
        raise ConfigError("moe.rl.gspo_epsilon_high must be >= moe.rl.gspo_epsilon")

    lo = _table(moe, "lora", required=False)
    _check_keys(lo, _LORA_KEYS, "moe.lora")
    enabled = _get(lo, "enabled", bool, "moe.lora", False)
    lora = LoraConfig(
        enabled=enabled,
        rank=_positive(_get(lo, "rank", int, "moe.lora", 16), "moe.lora.rank"),
        alpha=_positive(_get(lo, "alpha", int, "moe.lora", 32), "moe.lora.alpha"),
        dropout=_get(lo, "dropout", float, "moe.lora", 0.0),
        target_modules=_str_list(lo, "target_modules", "moe.lora"),
        target_parameters=_str_list(lo, "target_parameters", "moe.lora"),
    )
    if not 0.0 <= lora.dropout < 1.0:
        raise ConfigError(f"moe.lora.dropout must be in [0, 1), got {lora.dropout}")
    if enabled and not (lora.target_modules or lora.target_parameters):
        raise ConfigError("moe.lora is enabled but has no target_modules or target_parameters")

    out = _table(moe, "output")
    _check_keys(out, _OUTPUT_KEYS, "moe.output")
    output_dir = _get(out, "dir", str, "moe.output")
    return MoeConfig(
        name=name,
        seed=seed,
        model=model,
        sft=sft,
        rl=rl,
        lora=lora,
        output_dir=output_dir,
        probe_tokens=_positive(_get(out, "probe_tokens", int, "moe.output", 64), "moe.output.probe_tokens"),
        source_path=source_path,
    )


def load(path: str | Path) -> MoeConfig:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error
    return parse(raw, source_path=str(path))
