"""Loader and validator for the VCW-MoE GPU plan configs (`training/configs/vcw-*.toml`).

A VCW config reuses the repository's stage schema and adds a strict `[vcw]` extension:

* manifest tables (`[codebook]`, `[run]`, `[data]`, `[model]`, `[training]`, `[evaluation]`) as in
  every stage config; `[training].backend` is `vcw-<stage>`, which `ptr_training.run --execute`
  does not implement, so a VCW config cannot be mistaken for the synthetic-task `trl-moe` run;
* for stages that train weights, `[moe.*]` in the existing `trl-moe` schema. It is validated by
  `ptr_training.moe.config.parse` with the backend name swapped to `trl-moe` for that call only;
* `[vcw.*]`: pins, adapter or repair or probe description, verifier contract, data schema (checked
  by `guard`), welfare block and a declared VRAM / wall-time estimate that is recomputed from the
  config by `budget` and must agree.

Nothing here imports torch. Nothing here runs a model.
"""

from __future__ import annotations

import copy
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ptr_training.moe import config as moe_config
from ptr_training.moe.config import ConfigError
from ptr_training.moe.reward import LEVELS
from ptr_training.vcw import budget, guard

ROOT = Path(__file__).resolve().parents[4]
STAGES = ("perft-r-sft", "rlvr-router", "repair-expert", "self-model-probe")
BACKEND_PREFIX = "vcw-"
DOMAINS = ("math", "code", "tool")
_SHA40 = re.compile(r"[0-9a-f]{40}")
FINAL_SEEDS = (17, 29, 43, 71, 101)
MIN_PILOT_SEED = 1001  # tuning happens only on seeds >= 1001, never on a final seed
REQUIRED_ROUTING_METRICS = (
    "expert_churn_per_layer", "routing_entropy", "load_cv", "dead_experts",
    "train_inference_kl", "clip_fraction", "crash_step",
)
_VCW_KEYS = {
    "stage", "status", "experiments", "doc", "pins", "adapter", "verifier", "routing", "repair",
    "probe", "data", "estimate", "welfare", "tuning",
}
_ESTIMATE_KEYS = {"profile", "vram_gib", "wall_hours_mid", "wall_hours_range", "basis"}
_WELFARE_KEYS = {"self_report_training_signals", "self_report_guard", "integrated_system", "stop_rules"}
_ESTIMATE_TOLERANCE = 0.05


def _keys(table: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(table) - allowed)
    if extra:
        raise ConfigError(f"unknown key(s) {extra} in [{where}]; allowed: {sorted(allowed)}")


def _need(table: dict, key: str, where: str):
    if key not in table:
        raise ConfigError(f"missing required key {where}.{key}")
    return table[key]


def _sha(value, where: str) -> str:
    if not isinstance(value, str) or not _SHA40.fullmatch(value):
        raise ConfigError(f"{where} must be a full 40-hex commit sha, got {value!r}")
    return value


@dataclass(frozen=True)
class VcwConfig:
    stage: str
    name: str
    profile_id: str
    moe: moe_config.MoeConfig | None
    raw: dict
    estimate_vram_gib: float
    estimate_hours: tuple[float, float, float]  # (low-speed, mid, high-speed) hours, slowest first


def _profile(raw: dict, root: Path) -> tuple[str, float]:
    rel = _need(raw.get("run", {}), "hardware_profile", "run")
    path = root / rel
    if not path.is_file():
        raise ConfigError(f"run.hardware_profile {rel!r} does not exist")
    prof = tomllib.loads(path.read_text(encoding="utf-8"))
    pid = prof["id"]
    if pid not in budget.GPUS:
        raise ConfigError(f"hardware profile {pid!r} has no GPU assumptions in vcw.budget")
    if float(prof["gpu_memory_gib"]) != budget.GPUS[pid].memory_gib:
        raise ConfigError(f"{rel}: gpu_memory_gib differs from the budget assumption")
    return pid, float(prof["gpu_memory_gib"])


def _check_pins(vcw: dict, model: dict) -> None:
    pins = vcw["pins"]
    _keys(pins, {"model_repo", "model_revision", "tokenizer_repo", "tokenizer_revision", "datasets", "seeds", "pilot_seeds"}, "vcw.pins")
    if pins["model_repo"] != model.get("hf_id") or pins["model_revision"] != model.get("revision"):
        raise ConfigError("vcw.pins model repo/revision differ from [model] hf_id/revision")
    _sha(pins["model_revision"], "vcw.pins.model_revision")
    _sha(_need(pins, "tokenizer_revision", "vcw.pins"), "vcw.pins.tokenizer_revision")
    if tuple(pins["seeds"]) != FINAL_SEEDS:
        raise ConfigError(f"vcw.pins.seeds must be {list(FINAL_SEEDS)}")
    pilot = pins["pilot_seeds"]
    if not pilot or any(not isinstance(s, int) or s < MIN_PILOT_SEED for s in pilot):
        raise ConfigError(f"vcw.pins.pilot_seeds must all be >= {MIN_PILOT_SEED}")
    datasets = pins["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("vcw.pins.datasets must be a non-empty array of tables")
    for i, d in enumerate(datasets):
        _keys(d, {"repo", "revision", "role", "domain"}, f"vcw.pins.datasets[{i}]")
        _sha(_need(d, "revision", f"vcw.pins.datasets[{i}]"), f"vcw.pins.datasets[{i}].revision")
        if d.get("domain") not in DOMAINS + ("general",):
            raise ConfigError(f"vcw.pins.datasets[{i}].domain must be one of {DOMAINS + ('general',)}")


def _check_data(vcw: dict) -> None:
    data = _need(vcw, "data", "vcw")
    _keys(data, {"fields", "targets", "text_screen_rows"}, "vcw.data")
    try:
        guard.check_fields(_need(data, "fields", "vcw.data"), _need(data, "targets", "vcw.data"))
    except guard.SelfReportError as error:
        raise ConfigError(f"vcw.data refused by the self-report guard: {error}") from error


def _check_welfare(vcw: dict) -> None:
    w = _need(vcw, "welfare", "vcw")
    _keys(w, _WELFARE_KEYS, "vcw.welfare")
    if w.get("self_report_training_signals") != 0:
        raise ConfigError("vcw.welfare.self_report_training_signals must be exactly 0")
    if w.get("self_report_guard") is not True:
        raise ConfigError("vcw.welfare.self_report_guard must be true")
    if w.get("integrated_system") is not False:
        raise ConfigError(
            "vcw.welfare.integrated_system must be false: a config that trains one system "
            "combining workspace, self-model, goal-directed proposals and persistent state is an "
            "owner decision under MODEL_WELFARE section 3, not a plan default"
        )
    if not w.get("stop_rules"):
        raise ConfigError("vcw.welfare.stop_rules must name the applicable MODEL_WELFARE rules")


def _check_verifier(vcw: dict, moe: moe_config.MoeConfig) -> None:
    v = _need(vcw, "verifier", "vcw")
    _keys(v, {"reward_source", "gate", "min_level", "non_pass_reward", "hard_finding_forces_fail",
              "report_score_for_non_pass", "learned_score_in_reward", "finding_classes"}, "vcw.verifier")
    if v.get("reward_source") != "verification-report":
        raise ConfigError("vcw.verifier.reward_source must be 'verification-report'")
    if v.get("gate") != "weakest-level":
        raise ConfigError("vcw.verifier.gate must be 'weakest-level' (doc 37 section 5)")
    if v.get("non_pass_reward") != 0.0:
        raise ConfigError("vcw.verifier.non_pass_reward must be 0.0 (Fail, Disputed, Unknown)")
    for key in ("hard_finding_forces_fail",):
        if v.get(key) is not True:
            raise ConfigError(f"vcw.verifier.{key} must be true")
    for key in ("report_score_for_non_pass", "learned_score_in_reward"):
        if v.get(key) is not False:
            raise ConfigError(f"vcw.verifier.{key} must be false (invariant 11)")
    levels = _need(v, "min_level", "vcw.verifier")
    if set(levels) != set(DOMAINS) or any(x not in LEVELS for x in levels.values()):
        raise ConfigError(f"vcw.verifier.min_level needs exactly {DOMAINS} with values in {LEVELS}")
    floor = min(levels.values(), key=LEVELS.index)
    if moe.rl.min_verification_level != floor:
        raise ConfigError(
            f"moe.rl.min_verification_level ({moe.rl.min_verification_level}) must equal the "
            f"weakest per-domain level in vcw.verifier.min_level ({floor})"
        )
    if "Unverified" in levels.values() or "LatentAgreement" in levels.values():
        raise ConfigError("a reward-bearing domain may not accept Unverified or LatentAgreement")


def _trainable(vcw: dict, stage: str) -> int:
    if stage in ("perft-r-sft", "rlvr-router"):
        a = _need(vcw, "adapter", "vcw")
        _keys(a, {"method", "specialisation", "num_experts", "router_top_k", "rank", "alpha",
                  "trainable_params", "base_router_frozen", "init_from", "router_scope"}, "vcw.adapter")
        h, r, n, layers = budget.HIDDEN, a["rank"], a["num_experts"], budget.LAYERS
        expected = layers * (n * (h * r + r * h) + h * n)
        if a["method"] != "perft_r" or a["trainable_params"] != expected:
            raise ConfigError(f"vcw.adapter.trainable_params {a.get('trainable_params')} != {expected} (PERFT-R arithmetic)")
        if a.get("base_router_frozen") is not True:
            raise ConfigError("vcw.adapter.base_router_frozen must be true")
        return expected
    r = _need(vcw, "repair", "vcw")
    _keys(r, {"classes", "rank_per_class", "arms", "generic_rank", "shuffled_map_seed", "routing"}, "vcw.repair")
    if len(set(r["classes"])) != len(r["classes"]) or len(r["classes"]) < 2:
        raise ConfigError("vcw.repair.classes must be distinct and at least 2")
    if len(r["classes"]) * r["rank_per_class"] != r["generic_rank"]:
        raise ConfigError("generic repair expert must match the total rank of the per-class experts")
    if r["routing"] != "finding-code":
        raise ConfigError("vcw.repair.routing must be 'finding-code' (FindingRouter, not learned)")
    for arm in ("per_class", "generic", "regenerate_only", "shuffled_map"):
        if arm not in r["arms"]:
            raise ConfigError(f"vcw.repair.arms must include {arm!r}")
    return budget.LAYERS * 4 * (2 * budget.HIDDEN * r["generic_rank"])


def _estimate(vcw: dict, stage: str, moe, trainable: int, pid: str, mem: float):
    e = _need(vcw, "estimate", "vcw")
    _keys(e, _ESTIMATE_KEYS, "vcw.estimate")
    if e["profile"] != pid:
        raise ConfigError("vcw.estimate.profile differs from the hardware profile id")
    if stage == "self-model-probe":
        p = vcw["probe"]
        peak = budget.forward_peak_gib(p["micro_batch"] * p["seq_tokens"], trainable)
        secs = budget.forward_seconds(p["extract_sequences"] * p["seq_tokens"], pid)
        secs = tuple(s + p["probe_train_seconds"] for s in secs)
    elif stage == "rlvr-router":
        rl = moe.rl
        w = budget.RlWorkload(rl.steps, rl.prompts_per_step, rl.num_generations, rl.mini_steps,
                              rl.max_prompt_length, rl.max_completion_length, trainable)
        peak, secs = budget.rl_peak_gib(w), budget.rl_seconds(w, pid)
        secs = tuple(s * len(rl.algorithms) for s in secs)
    else:
        s = moe.sft
        w = budget.SftWorkload(s.steps, s.batch, s.gradient_accumulation_steps, s.max_length,
                               s.gradient_checkpointing, trainable)
        peak, secs = budget.sft_peak_gib(w), budget.sft_seconds(w, pid)
    hrs = budget.hours(secs)
    if not budget.fits(peak, pid):
        raise ConfigError(
            f"estimated peak {peak:.1f} GiB leaves less than {budget.REQUIRED_HEADROOM_GIB} GiB "
            f"free on {pid} ({mem} GiB); change batch, length or checkpointing"
        )
    for key, value in (("vram_gib", peak), ("wall_hours_mid", hrs[1])):
        if abs(e[key] - value) > _ESTIMATE_TOLERANCE * value:
            raise ConfigError(f"vcw.estimate.{key} = {e[key]} but the config recomputes to {value:.2f}")
    if not e.get("basis"):
        raise ConfigError("vcw.estimate.basis must state the assumptions (these are estimates)")
    return peak, hrs


def parse(raw: dict, root: Path = ROOT) -> VcwConfig:
    vcw = raw.get("vcw")
    if not isinstance(vcw, dict):
        raise ConfigError("missing required table [vcw]")
    _keys(vcw, _VCW_KEYS, "vcw")
    stage = _need(vcw, "stage", "vcw")
    if stage not in STAGES:
        raise ConfigError(f"vcw.stage must be one of {STAGES}, got {stage!r}")
    if vcw.get("status") != "plan-not-run":
        raise ConfigError("vcw.status must be 'plan-not-run' until a measured run exists")
    training = raw.get("training", {})
    if training.get("backend") != BACKEND_PREFIX + stage:
        raise ConfigError(f"[training].backend must be {BACKEND_PREFIX + stage!r} for stage {stage!r}")
    if set(training) - {"backend", "precision"}:
        raise ConfigError("[training] may only hold backend and precision in a VCW config")
    pid, mem = _profile(raw, root)
    model = raw.get("model", {})
    if model.get("kind") != "pretrained":
        raise ConfigError("VCW configs run the pretrained OLMoE only (model.kind = 'pretrained')")
    _check_pins(vcw, model)
    _check_data(vcw)
    _check_welfare(vcw)
    tuning = _need(vcw, "tuning", "vcw")
    _keys(tuning, {"status", "note"}, "vcw.tuning")
    if tuning.get("status") != "pilot-start-not-tuned":
        raise ConfigError("vcw.tuning.status must be 'pilot-start-not-tuned': values are tuned on pilot seeds only")

    moe = None
    if stage == "self-model-probe":
        trainable = _need(_need(vcw, "probe", "vcw"), "trainable_params", "vcw.probe")
        if "moe" in raw:
            raise ConfigError("[moe] is not used by the self-model-probe stage (no LLM weight training)")
    else:
        view = copy.deepcopy(raw)
        view["training"] = dict(training, backend="trl-moe", allow_unselected_backend=True)
        view.pop("vcw")
        moe = moe_config.parse(view)
        if not moe.lora.enabled:
            raise ConfigError("moe.lora must be enabled (adapter-only training)")
        if stage in ("rlvr-router",):
            if tuple(moe.rl.algorithms) != ("gspo", "sapo"):
                raise ConfigError("rlvr-router runs the GSPO and SAPO arms: moe.rl.algorithms = ['gspo','sapo']")
            _check_verifier(vcw, moe)
            r = _need(vcw, "routing", "vcw")
            _keys(r, {"metrics", "routing_replay", "l2_propensity_log"}, "vcw.routing")
            missing = [m for m in REQUIRED_ROUTING_METRICS if m not in r["metrics"]]
            if missing:
                raise ConfigError(f"vcw.routing.metrics is missing {missing}")
        else:
            if moe.rl.algorithms:
                raise ConfigError(f"stage {stage!r} is SFT only: moe.rl.algorithms = []")
            if stage == "repair-expert":
                _check_verifier(vcw, moe)
        trainable = _trainable(vcw, stage)
    if stage == "perft-r-sft" and moe is not None and moe.sft.steps < 1:
        raise ConfigError("moe.sft.steps must be >= 1")
    if stage == "self-model-probe":
        p = vcw["probe"]
        _keys(p, {"arms", "inputs", "targets", "trainable_params", "extract_sequences", "seq_tokens",
                  "micro_batch", "probe_train_seconds", "write_paths"}, "vcw.probe")
        if p["write_paths"] != []:
            raise ConfigError("vcw.probe.write_paths must be empty: observe and propose only (invariants 4, 5, 11)")
        for a in ("self_model", "info_matched_probe", "io_only_probe", "model_confidence", "constant"):
            if a not in p["arms"]:
                raise ConfigError(f"vcw.probe.arms must include {a!r}")
    peak, hrs = _estimate(vcw, stage, moe, trainable, pid, mem)
    return VcwConfig(stage, raw["run"]["name"], pid, moe, raw, peak, hrs)


def load(path: str | Path, root: Path = ROOT) -> VcwConfig:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error
    return parse(raw, root)


def audit(paths) -> dict:
    """Self-report audit over VCW configs: `self_report_training_signals` is the number of
    configs the loader refuses (any refusal, not only a guard refusal), so a clean tree is 0."""
    refused = {}
    for path in paths:
        try:
            load(path)
        except (ConfigError, OSError) as error:
            refused[str(path)] = str(error)
    return {"configs": len(list(paths)), "self_report_training_signals": len(refused), "refused": refused}


def main(argv: list[str] | None = None) -> int:
    import json
    import sys

    paths = sorted(Path(a) for a in (argv if argv is not None else sys.argv[1:]))
    if not paths:
        paths = sorted((ROOT / "training" / "configs").glob("vcw-*.toml"))
    report = audit(paths)
    print(json.dumps(report, indent=2))
    return 0 if report["self_report_training_signals"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
