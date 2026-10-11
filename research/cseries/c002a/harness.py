"""C007 harness: a self-model that sees a mixture-of-experts classifier's routing traces against an information-matched probe and the model's own
confidence, one seed.

    python research/cseries/c002a/harness.py --benchmark c007 --seed 6001

Prints one JSON object on its last line (the result the aggregator judges).
Exit 0: every coverage rule and counter holds. Exit 1: a coverage rule or a STRUCTURAL counter failed (the self-model changed the classifier, its
advice reached what the verifier or a gate reads, nondeterminism; never a statistical comparison). Exit 2: harness integrity: the environment is
incomplete, an exception, a negative control that did not trip, bad arguments, or an INTEGRITY counter above 0 (matching of the self-model and the
probe, splits, audits, bookkeeping), which makes the seed, and so the experiment, INVALID rather than FALSIFIED.
The comparison across seeds, the guards and the sensitivity controls are the aggregator's.

Nothing here reads, trains on or scores a self-report: the self-model reads the classifier's routing traces and is scored against the verifier's
deterministic outcomes (the prediction is wrong or right against the true class)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIB = HERE.parent / "lib"
EXPERIMENT_DIR = ROOT / "experiments/consciousness/C007-metacognitive-self-model-cpu"
CONFIG_PATH = EXPERIMENT_DIR / "config.toml"
BENCHMARK = "c007"
EXPERIMENT = "C007"

sys.path.insert(0, str(LIB))
sys.path.insert(0, str(HERE))
import c007_rules as rules  # noqa: E402
import common  # noqa: E402
from common import arm_seed as _arm_seed  # noqa: E402
from common import permille  # noqa: E402

torch = task = model = train = None

# trained heads: name -> (world, features, auxiliary weight key, labels); the probe's auxiliary weight is zero, the self-model's is pinned
ARM_SPECS = {
    "self_model": ("task", "trace", "aux_weight_milli", "true"),
    "probe_matched": ("task", "trace", None, "true"),
    "probe_io": ("task", "io", None, "true"),
    "self_model_perm": ("task", "trace", "aux_weight_milli", "permuted"),
    "self_model_null": ("null", "trace", "aux_weight_milli", "true"),
    "probe_matched_null": ("null", "trace", None, "true"),
}
JUDGED = ("self_model", "probe_matched")


def arm_seed(seed: int, name: str) -> int:
    return _arm_seed(EXPERIMENT, seed, name)


def _load_torch():
    global torch, task, model, train
    import torch as _torch

    torch = _torch
    import c007_model as _model
    import c007_task as _task
    import train as _train

    task, model, train = _task, _model, _train


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = dict(tomllib.loads(path.read_text(encoding="utf-8"))["pinned_thresholds"])
    cfg["late_width"] = 0  # filled once torch is loaded
    return cfg


@dataclass
class Environment:
    """What the verifier and the gate read: the outcome of every example, the accept decision of the gate, the gate's threshold and the digest of the
    classifier. The self-model's advice must never write any of it."""

    outcomes: "torch.Tensor"
    gate: "torch.Tensor"
    threshold: float
    model_digest: str

    def digest(self) -> str:
        return hashlib.sha256(repr((self.outcomes.tolist(), self.gate.tolist(), self.threshold, self.model_digest)).encode()).hexdigest()


def propose(advice, env: Environment):
    """The advice sink: the self-model may only propose which examples to abstain on. It reads the environment and writes nothing."""
    return advice > 0.5


def advice_leak(env: Environment, advice) -> int:
    """1 iff consuming adversarial advice changed what the verifier or the gate reads."""
    before = env.digest()
    propose(advice, env)
    propose(torch.rand_like(advice), env)
    propose(torch.zeros_like(advice), env)
    return int(env.digest() != before)


def feature_digest(x) -> str:
    """A digest of a feature matrix: two heads trained on the same features have the same one."""
    return hashlib.sha256(repr(x.flatten().tolist()).encode()).hexdigest()[:16]


def mismatches(a: list, b: list) -> int:
    """Positions at which two equally long lists differ."""
    return sum(1 for x, y in zip(a, b) if x != y)


def differ(a, b) -> int:
    """1 iff two quantities differ."""
    return int(a != b)


def row_hashes(x) -> list:
    return [hashlib.sha256(repr([round(v, 5) for v in row[:6]]).encode()).digest()[:8] for row in x.tolist()]


def overlap_count(splits: dict) -> int:
    """Inputs that occur in two different splits."""
    seen, total = {}, 0
    for name, x in splits.items():
        for h in row_hashes(x):
            if h in seen and seen[h] != name:
                total += 1
            seen.setdefault(h, name)
    return total


def preflight() -> int:
    problems = common.environment_problems(CONFIG_PATH)
    if problems:
        print("preflight failed: " + "; ".join(problems), file=sys.stderr)
        return 2
    _load_torch()
    try:
        cfg = load_config()
        torch.manual_seed(0)
        net = model.Classifier(cfg)
        trace = net.trace(torch.randn(4, cfg["dim"]))
        width = model.feature_width(cfg, model.EARLY + model.LATE)
        feats = model.features(trace, model.EARLY + model.LATE)
        model.Predictor(width + 1, model.feature_width(cfg, model.LATE) + 1, cfg, base=(1.0, 0.0))(torch.cat([feats, model.confidence_score(trace["probs"]).unsqueeze(-1)], dim=-1))
        print(json.dumps({"preflight": "ok", "environment": common.environment_record(ROOT, torch)}, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 2


def train_classifier(cfg: dict, seed: int, name: str, x, y):
    """The classifier under study, trained once on its world's training split and frozen."""
    torch.manual_seed(arm_seed(seed, "classifier:" + name))
    net = model.Classifier(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["model_lr"])
    gen = torch.Generator().manual_seed(arm_seed(seed, "classifier-batches:" + name))
    finite = True
    for _ in range(cfg["model_steps"]):
        idx = torch.randint(0, x.shape[0], (cfg["model_batch"],), generator=gen)
        loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        finite = finite and math.isfinite(float(loss))
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval(), finite


def make_world(cfg: dict, seed: int, world: str, gens: dict):
    """Splits of one world: the classifier's training set, the calibration set the self-model and the probes learn from, their validation set, and the
    in-distribution and shifted test sets."""
    null = world == "null"
    means = task.class_means(gens["means"], cfg)
    direction = task.shift_direction(gens["means"], cfg)
    draw = lambda n, shifted=False, g="data": task.sample(gens[g], means, n, cfg, shifted=shifted, direction=direction, null=null)  # noqa: E731
    return {
        "train": draw(cfg["train_size"]), "cal": draw(cfg["cal_size"]), "val": draw(cfg["val_size"]),
        "test_id": draw(cfg["test_size"]), "test_shift": draw(cfg["test_size"], shifted=True),
    }


def first_steps_loss(cfg, seed, feats, y, steps: int = 20) -> float:
    torch.manual_seed(arm_seed(seed, "determinism"))
    net = model.Predictor(feats.shape[1], cfg["late_width"], cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["probe_lr"], weight_decay=cfg["probe_weight_decay_milli"] / 1000)
    loss = None
    for k in range(steps):
        idx = torch.arange(k * cfg["probe_batch"], (k + 1) * cfg["probe_batch"]) % feats.shape[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(feats[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def run(seed: int, overrides: dict, only_arms: tuple | None = None) -> dict:
    """One seed. `only_arms` (the pilot tuning scripts only) trains just those heads and returns their records."""
    t0 = time.time()
    cfg = load_config()
    cfg.update(overrides)
    scale = "pinned" if not overrides else "custom"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = seed * 1000
    cfg["late_width"] = model.feature_width(cfg, model.LATE)
    worlds, classifiers, finite_all = {}, {}, True
    for index, world in enumerate(("task", "null")):
        gens = {k: torch.Generator().manual_seed(base + cfg["stream_offset"] + 10 * index + i) for i, k in enumerate(("means", "data", "probe", "perm", "noise"))}
        worlds[world] = make_world(cfg, seed, world, gens)
        net, finite = train_classifier(cfg, seed, world, *worlds[world]["train"])
        classifiers[world] = net
        finite_all = finite_all and finite
    digests_before = {w: model.params_digest(n) for w, n in classifiers.items()}

    # -- traces: everything the classifier exposes about its computation on each split, and the verifier's outcomes
    all_parts = model.EARLY + model.LATE
    data, lates = {}, {}
    for world, splits in worlds.items():
        for split, (x, labels) in splits.items():
            if split == "train":
                continue
            with torch.no_grad():
                trace = classifiers[world].trace(x)
            y = task.outcomes(trace["logits"], labels)
            data[(world, split)] = {"x": x, "labels": labels, "trace": trace, "y": y, "feats": model.features(trace, all_parts),
                                    "io": torch.cat([x, trace["probs"], trace["entropy"]], dim=-1)}
    for d in data.values():  # the classifier's own failure score is the last column of every view
        z = model.confidence_score(d["trace"]["probs"]).unsqueeze(-1)
        d["feats"], d["io"] = torch.cat([d["feats"], z], dim=-1), torch.cat([d["io"], z], dim=-1)
    cfg["late_width"] += 1
    task_cal, task_val, task_id, task_shift = (data[("task", s)] for s in ("cal", "val", "test_id", "test_shift"))
    null_cal, null_val, null_id, null_shift = (data[("null", s)] for s in ("cal", "val", "test_id", "test_shift"))
    perm_gen = torch.Generator().manual_seed(base + cfg["stream_offset"] + 77)
    permuted = {"cal": task.permute_outcomes(task_cal["y"], perm_gen), "val": task.permute_outcomes(task_val["y"], perm_gen)}
    perm_same = int((permuted["cal"] == task_cal["y"]).sum())
    # the base of the heads: the classifier's own calibrated confidence, a two-parameter logistic fit of its failure score on the calibration and validation labels
    # (of the world the head lives in; of the permuted outcomes for the head trained on them), the same fit the own-confidence baseline is
    fit_scores = torch.cat([model.confidence_score(task_cal["trace"]["probs"]), model.confidence_score(task_val["trace"]["probs"])])
    y_fit = torch.cat([task_cal["y"], task_val["y"]])  # the baseline is calibrated on every labelled example the heads were given: calibration and validation
    a, b = model.platt_fit(fit_scores, y_fit)
    with torch.no_grad():  # an independent recomputation of what the fit saw, from the named splits' inputs, made now (before anything can change the classifier)
        refit = torch.cat([model.confidence_score(classifiers["task"].trace(worlds["task"][s][0])["probs"]) for s in ("cal", "val")])
    refit_digest = feature_digest(refit)
    bases = {("task", "true"): (a, b)}
    if cfg["base_term"]:
        null_z = torch.cat([model.confidence_score(null_cal["trace"]["probs"]), model.confidence_score(null_val["trace"]["probs"])])
        bases[("null", "true")] = model.platt_fit(null_z, torch.cat([null_cal["y"], null_val["y"]]))
        bases[("task", "permuted")] = model.platt_fit(fit_scores, torch.cat([permuted["cal"], permuted["val"]]))

    # -- the heads
    heads, records, nonfinite, checkpoint_mismatch, convergence_mismatch = {}, {}, 0, 0, 0
    aux_steps, calibs, aux_grads, calib_digests = {}, {}, {}, {}
    for name, (world, view, aux_key, label_kind) in ARM_SPECS.items():
        if only_arms is not None and name not in only_arms:
            continue
        cal, val = data[(world, "cal")], data[(world, "val")]
        key = "feats" if view == "trace" else "io"
        fx, vx = cal[key], val[key]
        fy = permuted["cal"] if label_kind == "permuted" else cal["y"]
        vy = permuted["val"] if label_kind == "permuted" else val["y"]
        aux_weight = cfg[aux_key] / 1000 if aux_key else 0.0
        torch.manual_seed(arm_seed(seed, "predictor"))  # the self-model and the probes start from the same weights
        net = model.Predictor(fx.shape[1], cfg["late_width"] if view == "trace" else 0, cfg, base=bases.get((world, "permuted" if label_kind == "permuted" else "true")) if cfg["base_term"] else None)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["probe_lr"], weight_decay=cfg["probe_weight_decay_milli"] / 1000)
        gen = torch.Generator().manual_seed(arm_seed(seed, "predictor-batches"))
        expert_target = cal["trace"]["p2"].argmax(-1)
        entropy_target = cal["trace"]["entropy"].squeeze(-1) / math.log(cfg["classes"])

        aux_steps[name] = 0

        def step_loss(step, net=net, fx=fx, fy=fy, gen=gen, view=view, aux_weight=aux_weight, name=name):
            idx = torch.randint(0, fx.shape[0], (cfg["probe_batch"],), generator=gen)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(fx[idx]), fy[idx])
            if aux_weight > 0 and view == "trace":
                expert_logits, entropy = net.auxiliary(fx[idx])
                loss = loss + aux_weight * (torch.nn.functional.cross_entropy(expert_logits, expert_target[idx]) + torch.nn.functional.mse_loss(entropy, entropy_target[idx]))
                aux_steps[name] += 1
            return loss

        def validate(n, vx=vx, vy=vy):
            return score_counts(torch.sigmoid(n(vx)).detach(), vy)

        cap = cfg["max_steps"] if name in JUDGED or name == "probe_io" else cfg["max_steps_control"]
        net, info, loss, finite = train.train_to_plateau(net, opt, step_loss, validate, cfg, max_steps=cap)
        nonfinite += 0 if finite else 1
        cm, vm = train.audit(net, info, validate, cfg)
        checkpoint_mismatch += cm
        convergence_mismatch += vm
        heads[name] = net
        with torch.no_grad():
            val_logits = net(vx)
        calib_digests[name] = feature_digest(val_logits)
        calibs[name] = model.platt_fit(val_logits, vy)  # the same post-hoc recalibration on the held-out validation split for every head
        aux_grads[name] = aux_gradient_norm_micro(net, fx, expert_target, entropy_target, cfg) if view == "trace" else 0
        records[name] = {
            "world": world, "view": view, "aux_weight_milli": round(1000 * aux_weight), "params": model.count_params(net), "converged": int(info["converged"]),
            "steps": info["steps"], "step_cap": info["cap"], "best_val_permille": info["best_val"], "curve_permille": info["curve"],
            "train_loss_window_micro": info["loss_window_micro"], "input_digest": feature_digest(fx),
            "aux_steps": aux_steps[name], "aux_grad_norm_micro": aux_grads[name], "calibration_a_milli": finite_milli(calibs[name][0]), "calibration_b_milli": finite_milli(calibs[name][1]),
        }
    if only_arms is not None:
        return {"arms": {n: {**records[n], **evaluate_head(heads[n], ARM_SPECS[n], data, cfg, calibs[n])} for n in records}, "own_confidence": own_confidence_scores(data, cfg)}

    # -- scoring: every head on the in-distribution and the shifted split; the baselines that need no training of a head
    arms = {n: {**records[n], **evaluate_head(heads[n], ARM_SPECS[n], data, cfg, calibs[n])} for n in records}
    constant = float(y_fit.mean())
    own = lambda d: torch.sigmoid(a * model.confidence_score(d["trace"]["probs"]) + b)  # noqa: E731
    shuffle_gen = torch.Generator().manual_seed(base + cfg["stream_offset"] + 88)
    noise_gen = torch.Generator().manual_seed(base + cfg["stream_offset"] + 99)
    shuffled_rows = torch.randperm(task_shift["feats"].shape[0], generator=shuffle_gen)
    shuffled_id_rows = torch.randperm(task_id["feats"].shape[0], generator=shuffle_gen)
    with torch.no_grad():
        p_self = {"id": head_probs(heads["self_model"], calibs["self_model"], task_id["feats"]), "shift": head_probs(heads["self_model"], calibs["self_model"], task_shift["feats"])}
        p_shuffled = {"id": head_probs(heads["self_model"], calibs["self_model"], task_id["feats"][shuffled_id_rows]),
                      "shift": head_probs(heads["self_model"], calibs["self_model"], task_shift["feats"][shuffled_rows])}
        p_probe_shuffled = {"id": head_probs(heads["probe_matched"], calibs["probe_matched"], task_id["feats"][shuffled_id_rows]),
                            "shift": head_probs(heads["probe_matched"], calibs["probe_matched"], task_shift["feats"][shuffled_rows])}
    noise = {k: torch.randn(p_self[k].shape, generator=noise_gen) * cfg["noise_sigma_milli"] / 1000 for k in p_self}
    p_noisy = {k: (p_self[k] + noise[k]).clamp(0, 1) for k in p_self}
    baselines = {
        "own_confidence": {"id": own(task_id), "shift": own(task_shift)},
        "own_confidence_raw": {"id": 1 - task_id["trace"]["probs"].max(-1).values, "shift": 1 - task_shift["trace"]["probs"].max(-1).values},
        "constant": {"id": torch.full_like(task_id["y"], constant), "shift": torch.full_like(task_shift["y"], constant)},
        "self_model_shuffled": p_shuffled,
        "probe_matched_shuffled": p_probe_shuffled,
        "self_model_noisy": p_noisy,
    }
    for name, preds in baselines.items():
        arms[name] = {"world": "task", "view": "none", **metrics(preds["id"], task_id["y"], "id_"), **metrics(preds["shift"], task_shift["y"], "")}
    arms["platt"] = {"a_milli": round(1000 * a), "b_milli": round(1000 * b)}

    # -- counters
    sm, pm = heads["self_model"], heads["probe_matched"]
    divergence = trunk_divergence_micro(sm, pm)
    inert = (int(records["self_model"]["aux_steps"] != records["self_model"]["steps"]) + int(records["probe_matched"]["aux_steps"] != 0)
             + int(records["self_model"]["aux_grad_norm_micro"] == 0) + int(divergence == 0))
    with torch.no_grad():
        head_fits = sum(differ(calib_digests[n], feature_digest(heads[n](data[(ARM_SPECS[n][0], "val")]["feats" if ARM_SPECS[n][1] == "trace" else "io"]))) for n in heads)
    fit_errors = differ(feature_digest(fit_scores), refit_digest) + head_fits
    head_gap = max(rules.gap_permille(arms[n]["id_brier_correct"], arms["own_confidence"]["id_brier_correct"], arms[n]["id_brier_total"])
                   for n in rules.JUDGED_ARMS)
    capacity = differ(model.count_params(sm), model.count_params(pm))
    inputs = differ(records["self_model"]["input_digest"], records["probe_matched"]["input_digest"])
    aux_errors = int(records["probe_matched"]["aux_weight_milli"] != 0) + int(records["self_model"]["aux_weight_milli"] != cfg["aux_weight_milli"]) + int(records["probe_io"]["aux_weight_milli"] != 0)
    splits = {f"{w}:{s}": x for w, sp in worlds.items() for s, (x, _) in sp.items()}
    overlap = overlap_count(splits)
    shifted_rate = permille(int(task_shift["y"].sum()), task_shift["y"].numel())
    id_rate = permille(int(task_id["y"].sum()), task_id["y"].numel())
    rate_errors = rules.band_error(id_rate, cfg["id_error_low_permille"], cfg["id_error_high_permille"])
    outcome_audit = mismatches(task.outcomes_independent(task_shift["trace"]["logits"][:256], task_shift["labels"][:256]), task_shift["y"][:256].int().tolist())
    metric_audit = differ(model.brier_sum_micro(baselines["own_confidence"]["shift"][:512], task_shift["y"][:512]), model.brier_reference_micro(baselines["own_confidence"]["shift"][:512].tolist(), task_shift["y"][:512].tolist()))
    perm_error = rules.agreement_error(perm_same, task_cal["y"].numel(), permille(int((task_cal["y"] == 1).sum()) ** 2 + int((task_cal["y"] == 0).sum()) ** 2, task_cal["y"].numel() ** 2), cfg["perm_slack_permille"])
    same_rows = int((shuffled_rows == torch.arange(shuffled_rows.numel())).sum())
    shuffle_error = rules.unchanged_error(same_rows, shuffled_rows.numel(), cfg["unchanged_ceiling_permille"])
    unchanged = int((p_noisy["shift"] == p_self["shift"]).sum())
    noise_error = rules.unchanged_error(unchanged, p_self["shift"].numel(), cfg["unchanged_ceiling_permille"])
    digests_after = {w: model.params_digest(n) for w, n in classifiers.items()}
    write_paths = sum(int(digests_before[w] != digests_after[w]) for w in digests_before)
    gate = (own(task_shift) < cfg["gate_threshold_milli"] / 1000)
    env = Environment(task_shift["y"].clone(), gate.clone(), cfg["gate_threshold_milli"] / 1000, digests_after["task"])
    leaks = advice_leak(env, p_self["shift"])
    advice_hits = int((propose(p_self["shift"], env) != propose(1 - p_self["shift"], env)).sum())
    nondet = int(first_steps_loss(cfg, seed, task_cal["feats"], task_cal["y"]) != first_steps_loss(cfg, seed, task_cal["feats"], task_cal["y"]))

    # -- negative controls: each doctored input (never the arm under test) must trip the check it exists for
    leaky = Environment(env.outcomes.clone(), env.gate.clone(), env.threshold, env.model_digest)
    doctored_digest = model.params_digest(classifiers["task"])
    controls = {
        "split_overlap": {"required_rule_tripped": overlap_count({"a": task_cal["x"], "b": torch.cat([task_cal["x"][:3], task_id["x"][:3]])}) > 0},
        "capacity": {"required_rule_tripped": differ(model.count_params(sm), model.count_params(heads["probe_io"])) > 0},
        "input_hash": {"required_rule_tripped": differ(feature_digest(task_cal["io"]), records["self_model"]["input_digest"]) > 0},
        "outcome_audit": {"required_rule_tripped": mismatches([0, 1, 1], [1, 1, 0]) > 0},
        "metric_audit": {"required_rule_tripped": differ(model.brier_sum_micro(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0])), model.brier_reference_micro([0.0, 1.0], [1.0, 0.0])) > 0},
        "write_probe": {"required_rule_tripped": write_probe(classifiers["task"], doctored_digest)},
        "advice_probe": {"required_rule_tripped": advice_probe(leaky)},
        "treatment_probe": {"required_rule_tripped": trunk_divergence_micro(sm, sm) == 0},
        "fit_probe": {"required_rule_tripped": differ(feature_digest(fit_scores), feature_digest(model.confidence_score(task_shift["trace"]["probs"]))) > 0},
        "shuffle_probe": {"required_rule_tripped": rules.unchanged_error(shuffled_rows.numel(), shuffled_rows.numel(), cfg["unchanged_ceiling_permille"]) == 1},
    }

    shifted_n = int(task_shift["y"].numel())
    best_trace = min(arms["self_model"]["brier_correct"], arms["probe_matched"]["brier_correct"])
    result = {
        "benchmark": BENCHMARK, "experiment": EXPERIMENT, "seed": seed, "scale": scale,
        "procedure": {"seed_is_declared": seed in cfg["seeds"], "shape_is_pinned": scale == "pinned", "test_size": cfg["test_size"], "cal_size": cfg["cal_size"]},
        "arms": arms,
        "arms_trained": len(records),
        "id_eval_instances": int(task_id["y"].numel()),
        "shifted_eval_instances": shifted_n,
        "null_eval_instances": int(null_shift["y"].numel()),
        "id_error_permille": id_rate,
        "shifted_error_permille": shifted_rate,
        "null_error_permille": permille(int(null_shift["y"].sum()), null_shift["y"].numel()),
        "signal_permille": rules.signal(arms["constant"]["brier_correct"], best_trace, arms["constant"]["brier_total"]),
        "unconverged_judged_arms": rules.unconverged_guard_arms({n: records[n]["converged"] for n in rules.JUDGED_ARMS}),
        "unconverged_arms": [n for n in records if not records[n]["converged"]],
        "environment": common.environment_record(ROOT, torch),
        "advice_probe_hits": advice_hits,
        "model_write_paths": write_paths,
        "advice_leaks": leaks,
        "nondeterminism": nondet,
        "head_gap_id_permille": head_gap,
        "treatment_divergence_micro": divergence,
        "inert_treatment_arms": inert,
        "fit_split_errors": fit_errors,
        "capacity_mismatch_arms": capacity,
        "input_mismatches": inputs,
        "aux_weight_errors": aux_errors,
        "split_overlap": overlap,
        "error_rate_errors": rate_errors,
        "outcome_audit_mismatches": outcome_audit,
        "metric_audit_mismatches": metric_audit,
        "permutation_leaks": perm_error,
        "shuffle_errors": shuffle_error,
        "noise_errors": noise_error,
        "nonfinite_arms": nonfinite + (0 if finite_all else 1),
        "checkpoint_mismatch_arms": checkpoint_mismatch,
        "convergence_mismatch_arms": convergence_mismatch,
        "controls": controls,
        "harness_gaps": [],
        "timing_s": int(time.time() - t0),
        "cpu_s": int(time.process_time()),
        "summary": {n: {"brier_milli": round(1000 * a["brier_correct"] / a["brier_total"]), "id_brier_milli": round(1000 * a["id_brier_correct"] / a["id_brier_total"]) if "id_brier_total" in a else None}
                    for n, a in arms.items() if "brier_correct" in a},
    }
    coverage_failed, falsifiers_failed = common.judge(result, rules.coverage_rules(cfg), rules.falsifier_rules())
    result["coverage_failed"] = coverage_failed
    result["falsifiers_failed"] = falsifiers_failed
    result["integrity_failed"] = rules.integrity_failed(result)
    result["guards_failed"] = [r for r in rules.guard_rules(cfg) if not common.holds(result, r)]
    result["hard_failures"] = len(falsifiers_failed) + len(result["integrity_failed"])
    return result


def write_probe(net, digest_before: str) -> bool:
    """The write detector: it must see a change when one weight of a COPY of the classifier is altered."""
    import copy

    clone = copy.deepcopy(net)
    with torch.no_grad():
        next(clone.parameters()).add_(1.0)
    return model.params_digest(clone) != digest_before


def advice_probe(leaky: Environment) -> bool:
    """The advice detector: it must see a change when a consumer of advice writes the environment."""
    before = leaky.digest()
    leaky.threshold = leaky.threshold + 0.1
    return leaky.digest() != before


def finite_milli(value: float) -> int:
    """round(1000 * value), or -1 for a non-finite value (an arm with non-finite weights is counted by nonfinite_arms, not allowed to crash the harness)."""
    return round(1000 * value) if math.isfinite(value) else -1


def head_probs(net, calib, x):
    """The head's failure probability: its logit recalibrated by the held-out fit (a, b) every head gets."""
    return torch.sigmoid(calib[0] * net(x) + calib[1])


def own_confidence_scores(data, cfg) -> dict:
    """The model's own calibrated confidence on the in-distribution and shifted test splits (pilot tuning: how far below it the heads are)."""
    cal, val = data[("task", "cal")], data[("task", "val")]
    z = torch.cat([model.confidence_score(cal["trace"]["probs"]), model.confidence_score(val["trace"]["probs"])])
    a, b = model.platt_fit(z, torch.cat([cal["y"], val["y"]]))
    out = {}
    for split, prefix in (("test_id", "id_"), ("test_shift", "")):
        d = data[("task", split)]
        out.update(metrics(torch.sigmoid(a * model.confidence_score(d["trace"]["probs"]) + b), d["y"], prefix))
    return out


def aux_gradient_norm_micro(net, fx, expert_target, entropy_target, cfg) -> int:
    """The norm (millionths) of the gradient of the auxiliary self-modelling loss with respect to the trunk, on a fixed slice of the calibration set:
    zero would mean the auxiliary term cannot move the shared representation."""
    n = min(512, fx.shape[0])
    expert_logits, entropy = net.auxiliary(fx[:n])
    loss = torch.nn.functional.cross_entropy(expert_logits, expert_target[:n]) + torch.nn.functional.mse_loss(entropy, entropy_target[:n])
    grads = torch.autograd.grad(loss, list(net.trunk.parameters()), allow_unused=True)
    total = sum(float((g ** 2).sum()) for g in grads if g is not None)
    return int(round(1e6 * math.sqrt(total))) if math.isfinite(total) else -1  # non-finite weights are counted by nonfinite_arms, not allowed to crash the harness


def trunk_divergence_micro(a, b) -> int:
    """The largest absolute difference (millionths) between two networks' trunk parameters: the heads start identical, so only the training differs."""
    pairs = list(zip(a.trunk.parameters(), b.trunk.parameters()))
    if any(p.shape != q.shape for p, q in pairs):  # networks of different shapes are not the same head (the capacity counter says so); they diverge
        return 10**9
    largest = max(float((p - q).abs().max()) for p, q in pairs)
    return int(round(1e6 * largest)) if math.isfinite(largest) else -1


def score_counts(p, y) -> tuple[int, int]:
    """(correct, total) in the plateau protocol's terms: 1 - Brier, in millionths per example (higher is better)."""
    n = int(y.numel())
    return n * 1_000_000 - model.brier_sum_micro(p, y), n * 1_000_000


def metrics(p, y, prefix: str) -> dict:
    n = int(y.numel())
    return {f"{prefix}brier_correct": model.brier_sum_micro(p, y), f"{prefix}brier_total": n * 1_000_000, f"{prefix}auroc_permille": model.auroc_permille(p, y),
            f"{prefix}ece_permille": model.ece_permille(p, y), f"{prefix}rc_area_permille": model.risk_coverage_area_permille(p, y)}


def evaluate_head(net, spec, data, cfg, calib) -> dict:
    world, view, _, _ = spec
    key = "feats" if view == "trace" else "io"
    out = {}
    with torch.no_grad():
        for split, prefix in (("test_id", "id_"), ("test_shift", "")):
            d = data[(world, split)]
            out.update(metrics(head_probs(net, calib, d[key]), d["y"], prefix))
        if view == "trace":
            d = data[(world, "test_shift")]
            logits, _ = net.auxiliary(d[key])
            out["expert_accuracy_permille"] = permille(int((logits.argmax(-1) == d["trace"]["p2"].argmax(-1)).sum()), d["y"].numel())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="must be c007; the runner checks this token in the command")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight", action="store_true", help="check the environment and exit; reads no seed, trains nothing")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a pinned value (scale 'custom', never evidence)")
    args = parser.parse_args()
    if args.benchmark != BENCHMARK:
        print(f"unknown benchmark {args.benchmark!r}", file=sys.stderr)
        return 2
    if args.preflight:
        return preflight()
    if args.seed is None:
        print("--seed is required", file=sys.stderr)
        return 2
    problems = common.environment_problems(CONFIG_PATH)
    if problems:
        print("the environment is incomplete, refusing to run: " + "; ".join(problems), file=sys.stderr)
        return 2
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    _load_torch()
    try:
        result = run(args.seed, overrides)
    except Exception:  # harness integrity
        traceback.print_exc()
        return 2
    failed_controls = [k for k, v in result["controls"].items() if not v["required_rule_tripped"]]
    broken = [rule.split(" ")[0] for rule in result["integrity_failed"]]
    print(json.dumps(result, sort_keys=True))
    if failed_controls or broken:
        print(f"harness integrity: negative controls did not trip: {failed_controls}", file=sys.stderr)
        print(f"harness integrity: integrity counters above 0: {broken}", file=sys.stderr)
        return 2
    return 1 if (result["coverage_failed"] or result["falsifiers_failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
