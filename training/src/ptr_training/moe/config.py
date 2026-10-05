"""Typed parsing of the MoE stage TOML (pure Python)."""

from __future__ import annotations

import math
import re
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
_MOE_KEYS = {"sft", "rl", "lora", "output", "gates", "allow_full_finetune", "allow_large"}
# [training] keys this backend reads; everything else there is ignored by trl-moe (the
# hyperparameters live in [moe.*]) and is rejected instead of silently doing nothing.
_TRAINING_KEYS = {"backend", "precision", "allow_unselected_backend"}
# Resource-key ceilings; `[moe] allow_large = true` lifts them.
_MAXIMA = {
    "moe.sft.steps": 100_000, "moe.sft.examples": 100_000, "moe.sft.batch": 1024,
    "moe.rl.steps": 10_000, "moe.rl.mini_steps": 64, "moe.rl.num_generations": 64, "moe.rl.prompts_per_step": 64,
    "moe.rl.max_completion_length": 512, "moe.output.probe_tokens": 4096,
}
MAX_GENERATION_BATCH = 4096  # mini_steps * prompts_per_step * num_generations rollouts at once
_SHA40 = re.compile(r"[0-9a-f]{40}")
_GATES_KEYS = {"subspace_rank"}
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
class GatesConfig:
    # Rank of the expert-delta subspaces compared by the interference gate. Full-rank
    # subspaces of the ambient space always overlap ~1.0 (chance 1.0): uninformative.
    subspace_rank: int = 4


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
    gates: GatesConfig = field(default_factory=GatesConfig)
    allow_unselected_backend: bool = False
    allow_full_finetune: bool = False
    allow_large: bool = False
    source_path: str | None = field(default=None, compare=False)


def parse(raw: dict, source_path: str | None = None) -> MoeConfig:
    run = _table(raw, "run")
    seed = _seed(_get(run, "seed", int, "run"), "run.seed")
    name = _get(run, "name", str, "run")
    training = _table(raw, "training")
    if training.get("backend") != "trl-moe":
        raise ConfigError(f"[training].backend must be 'trl-moe', got {training.get('backend')!r}")

    ignored = sorted(set(training) - _TRAINING_KEYS)
    if ignored:
        raise ConfigError(
            f"[training] keys {ignored} are ignored by backend 'trl-moe' (hyperparameters are "
            f"read from [moe.sft] / [moe.rl]); remove them. Allowed: {sorted(_TRAINING_KEYS)}"
        )
    evaluation = raw.get("evaluation") or {}
    if evaluation.get("every", 0) not in (0,) or evaluation.get("suites", []) != []:
        raise ConfigError(
            "[evaluation] every/suites are ignored by backend 'trl-moe' (evaluation is built "
            "in); set every = 0 and suites = [] or remove the table"
        )
    allow_unselected = _get(training, "allow_unselected_backend", bool, "training", False)
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
        if not _SHA40.fullmatch(revision):
            raise ConfigError(
                f"model.revision must be a full 40-hex commit sha (tags and branches are "
                f"mutable), got {revision!r}"
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
    need_sft = longest_prompt + longest_answer_chars() + 1  # prompt + answer + EOS, in tokens
    if sft.max_length < need_sft:
        raise ConfigError(
            f"moe.sft.max_length ({sft.max_length}) must be >= longest prompt ({longest_prompt}) "
            f"+ longest answer ({need_sft - longest_prompt - 1}) + EOS = {need_sft}, or "
            "truncation drops the EOS label or the answer"
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
    # Execution-only RL validations apply only when an RL arm will actually run.
    if algos and rl.steps < 1:
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
    longest_prompt_tokens = longest_prompt_chars()  # 1 char = 1 token for the tiny tokenizer
    if kind == "tiny-random":
        # Longest sequence the run builds: SFT sequences, and any task prompt plus a full
        # generation (RL rollouts and the policy evaluation both generate max_completion_length).
        needed = max(sft.max_length, longest_prompt_tokens + rl.max_completion_length)
        if needed > model.max_positions:
            raise ConfigError(
                f"model.max_positions ({model.max_positions}) is below the longest sequence "
                f"the run builds ({needed}: max of moe.sft.max_length and the longest task "
                f"prompt ({longest_prompt_tokens}) + moe.rl.max_completion_length)"
            )
    if algos and rl.max_prompt_length < longest_prompt_tokens and kind == "tiny-random":
        raise ConfigError(
            f"moe.rl.max_prompt_length ({rl.max_prompt_length}) is below the longest task "
            f"prompt ({longest_prompt_tokens} tokens)"
        )
    if rl.num_generations < 2:
        raise ConfigError("moe.rl.num_generations must be >= 2 (group-relative advantages)")
    if rl.beta < 0:
        raise ConfigError("moe.rl.beta must be >= 0")
    if rl.epsilon_high < rl.epsilon and "epsilon_high" in r:
        raise ConfigError("moe.rl.epsilon_high must be >= moe.rl.epsilon")
    if algos and rl.mini_steps > rl.steps:
        raise ConfigError(
            f"moe.rl.mini_steps ({rl.mini_steps}) must not exceed moe.rl.steps ({rl.steps}): "
            "TRL would build a generation batch larger than the whole run"
        )
    if rl.sapo_temperature_neg <= rl.sapo_temperature_pos:
        raise ConfigError(
            "moe.rl.sapo_temperature_neg must be > moe.rl.sapo_temperature_pos "
            f"(got {rl.sapo_temperature_neg} <= {rl.sapo_temperature_pos})"
        )
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

    g = _table(moe, "gates", required=False)
    _check_keys(g, _GATES_KEYS, "moe.gates")
    gates_cfg = GatesConfig(
        subspace_rank=_positive(_get(g, "subspace_rank", int, "moe.gates", 4), "moe.gates.subspace_rank")
    )
    if kind == "tiny-random" and gates_cfg.subspace_rank >= min(model.hidden, model.intermediate):
        raise ConfigError(
            f"moe.gates.subspace_rank ({gates_cfg.subspace_rank}) must be below the expert delta "
            f"dimensions (min(model.hidden, model.intermediate) = "
            f"{min(model.hidden, model.intermediate)}): a full-rank comparison is uninformative"
        )

    out = _table(moe, "output")
    _check_keys(out, _OUTPUT_KEYS, "moe.output")
    output_dir = _get(out, "dir", str, "moe.output")
    probe_tokens = _positive(_get(out, "probe_tokens", int, "moe.output", 64), "moe.output.probe_tokens")

    allow_full_finetune = _get(moe, "allow_full_finetune", bool, "moe", False)
    allow_large = _get(moe, "allow_large", bool, "moe", False)
    if kind == "pretrained" and not lora.enabled and not allow_full_finetune:
        raise ConfigError(
            "a pretrained model with moe.lora.enabled = false is a full fine-tune of every "
            "weight (memory and compute of the whole model); set [moe] allow_full_finetune = "
            "true to confirm"
        )
    if not allow_large:
        values = {
            "moe.sft.steps": sft.steps, "moe.sft.examples": sft.examples, "moe.sft.batch": sft.batch,
            "moe.rl.steps": rl.steps, "moe.rl.mini_steps": rl.mini_steps,
            "moe.rl.num_generations": rl.num_generations,
            "moe.rl.prompts_per_step": rl.prompts_per_step,
            "moe.rl.max_completion_length": rl.max_completion_length,
            "moe.output.probe_tokens": probe_tokens,
        }
        for key, value in values.items():
            if key.startswith("moe.rl.") and not algos:
                continue  # RL never runs
            if value > _MAXIMA[key]:
                raise ConfigError(
                    f"{key} = {value} exceeds the sanity limit {_MAXIMA[key]}; "
                    "set [moe] allow_large = true to override"
                )
        generation_batch = rl.mini_steps * rl.prompts_per_step * rl.num_generations
        if algos and generation_batch > MAX_GENERATION_BATCH:
            raise ConfigError(
                f"generation batch mini_steps * prompts_per_step * num_generations = "
                f"{generation_batch} exceeds {MAX_GENERATION_BATCH} rollouts; "
                "set [moe] allow_large = true to override"
            )
    return MoeConfig(
        name=name,
        seed=seed,
        model=model,
        sft=sft,
        rl=rl,
        lora=lora,
        output_dir=output_dir,
        probe_tokens=probe_tokens,
        gates=gates_cfg,
        allow_full_finetune=allow_full_finetune,
        allow_large=allow_large,
        allow_unselected_backend=allow_unselected,
        source_path=source_path,
    )


def load(path: str | Path) -> MoeConfig:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error
    return parse(raw, source_path=str(path))
