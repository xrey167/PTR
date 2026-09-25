"""Apply the A0 ablation study's preregistered criteria to its run records.

Mechanical by design: every number comes from exact counts, every threshold from
research/falsification/A0-ablations-v1/criteria.toml, and every comparison is
exact on float64. The core (`decide`) takes a table of per-arm, per-seed scores
and the gate results and returns every verdict with its reason; the I/O layer
builds that table from the run records through benchmarks/operator-routing/
score.py (which recounts every prediction independently of the binary) and checks
gates G0-G6 and the leak alarm first.

Outputs: results/a0_internal_metrics.json in each of M001-M004 (deliberately not
metrics.json, which stays reserved for the manifests' own baseline comparison)
and research/falsification/A0-ablations-v1/results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "research/falsification/A0-ablations-v1"
CRITERIA = STUDY_DIR / "criteria.toml"
PREREG_TAG = "a0-ablation-prereg-v1"
DATA_DIR = ROOT / "datasets/generated/operator_routing_v1"
LOCK = ROOT / "benchmarks/operator-routing/splits.lock.json"
EXPERIMENTS = {
    "M001": "model/M001-semantic-slots",
    "M002": "model/M002-typed-attention",
    "M003": "model/M003-latent-recurrence",
    "M004": "model/M004-operator-router",
}
TEST_SPLITS = (
    "test_iid", "ood_compose_epi", "ood_compose_regime",
    "ood_distractors", "ood_validity", "ood_payload",
)


# ------------------------------------------------------------------------ statistics


def paired(comparator: list[float], ablated: list[float], t_critical: float) -> dict:
    """delta_s = comparator - ablated per seed, with the mean, sample SD and t-interval."""
    deltas = [c - a for c, a in zip(comparator, ablated)]
    n = len(deltas)
    m = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    half = t_critical * sd / math.sqrt(n)
    return {
        "deltas": deltas,
        "mean": m,
        "sd": sd,
        "min": min(deltas),
        "max": max(deltas),
        "positive_seeds": sum(1 for d in deltas if d > 0),
        "ci": [m - half, m + half],
    }


def common_rule(stats: dict, delta_min: float, blocked: str | None) -> tuple[str, str]:
    """SUPPORTS / FALSIFIES (HARMFUL) / INCONCLUSIVE under the common decision rule."""
    if blocked:
        return "INCONCLUSIVE", blocked
    lower, upper = stats["ci"]
    if stats["mean"] >= delta_min and lower > 0 and stats["positive_seeds"] == len(stats["deltas"]):
        return "SUPPORTS", "mean >= delta_min, CI lower > 0 and every seed positive"
    if upper < delta_min:
        if upper < 0:
            return "HARMFUL", "CI upper < 0: the ablated arm is better"
        return "FALSIFIES", "CI upper < delta_min"
    return "INCONCLUSIVE", "neither SUPPORTS nor FALSIFIES holds"


# ------------------------------------------------------------------------ the table


def endpoint(table: dict, arm: str, seed: int, spec: str, criteria: dict) -> float:
    """The contrast endpoint for one arm and seed.

    `split:<name>` is that split's accuracy; `composite:<X>` the unweighted mean of
    its splits' accuracies; `subset:<split>:<tag>` a decisive subset's accuracy."""
    scores = table[arm][seed]
    kind, _, rest = spec.partition(":")
    if kind == "split":
        return scores[rest]["accuracy"]
    if kind == "composite":
        splits = criteria["composites"][rest]
        return statistics.fmean(scores[s]["accuracy"] for s in splits)
    if kind == "subset":
        split, _, tag = rest.partition(":")
        value = scores[split]["subsets"][tag]["accuracy"]
        if value is None:
            raise ValueError(f"{arm} seed {seed}: subset {rest} is empty")
        return value
    raise ValueError(f"unknown endpoint {spec!r}")


def mean_over_seeds(table: dict, arm: str, split: str, seeds: list[int]) -> float:
    return statistics.fmean(table[arm][s][split]["accuracy"] for s in seeds)


def learnability(table: dict, references: dict, criteria: dict, seeds: list[int]) -> dict:
    """Per arm: whether its 5-seed mean test_iid reaches its bar."""
    rules = criteria["learnability"]
    refs = references["references"]
    bound = refs["bound_additive"]["test_iid"]
    restricted_bar = refs["train_majority"]["test_iid"] + rules["restricted_margin"]
    out = {}
    for arm in table:
        mean = mean_over_seeds(table, arm, "test_iid", seeds)
        if arm in rules["complete_path"]:
            bar, kind = bound, "complete-path"
        elif arm in rules["restricted"]:
            bar, kind = restricted_bar, "restricted"
        else:
            continue
        out[arm] = {"kind": kind, "mean_test_iid": mean, "bar": bar, "passes": mean >= bar}
    return out


# ------------------------------------------------------------------------ the core


def decide(
    table: dict,
    references: dict,
    criteria: dict,
    gates: dict,
    contingency: dict | None = None,
) -> dict:
    """Every verdict, from a table {arm: {seed: {split: {accuracy, subsets}}}}.

    `gates` holds G0, G2-G6 as {"pass": bool, "detail": ...}; G1 and the leak alarm
    are computed here from the table. `contingency` is a table of the same shape at
    4000 steps, holding only the arms the learnability contingency re-ran."""
    seeds = criteria["seeds"]
    t = criteria["t_critical"]
    refs = references["references"]
    ceiling = refs["raw_blind_exact_bayes"]["test_iid"]
    gate_rules = criteria["gates"]
    result: dict = {"gates": dict(gates), "verdicts": {}, "reported": {}}

    full_mean = mean_over_seeds(table, "full", "test_iid", seeds)
    competence_bar = max(gate_rules["competence_min"], ceiling + gate_rules["competence_margin"])
    result["gates"]["G1"] = {
        "pass": full_mean >= competence_bar,
        "detail": {"full_mean_test_iid": full_mean, "bar": competence_bar},
    }
    leak = None
    if "raw-blind" in table:
        raw_blind_mean = mean_over_seeds(table, "raw-blind", "test_iid", seeds)
        leak = raw_blind_mean > ceiling + gate_rules["leak_alarm_margin"]
        result["leak_alarm"] = {"active": leak, "raw_blind_mean_test_iid": raw_blind_mean, "ceiling": ceiling}

    if not result["gates"]["G1"]["pass"]:
        result["summary"] = (
            f"A0 (D=32, one block) does not learn operator-routing v1: the full arm's mean "
            f"test_iid {full_mean:.4f} is below {competence_bar:.4f}. No mechanism verdict is issued."
        )
        for contrast in criteria["contrast"]:
            result["verdicts"][contrast["id"]] = {"verdict": "NOT ISSUED", "reason": "G1 (competence) failed"}
        return result

    failed_gates = [name for name, gate in result["gates"].items() if not gate["pass"]]
    learn = learnability(table, references, criteria, seeds)
    result["learnability"] = learn

    def blocked(arms: list[str]) -> str | None:
        if leak:
            return "HOLD: the raw-blind leak alarm is active"
        if failed_gates:
            return f"gates failed: {', '.join(sorted(failed_gates))}"
        for arm in arms:
            status = learn.get(arm)
            if status and not status["passes"]:
                if status["kind"] == "restricted":
                    return f"{arm} failed to train (below its learnability bar)"
                if not (contingency and arm in contingency):
                    return f"{arm} is below its learnability bar; the 4000-step contingency decides"
        return None

    def source(arms: list[str]) -> tuple[dict, str]:
        """The contingency table decides when a complete-path arm failed learnability."""
        needs = [a for a in arms if learn.get(a) and not learn[a]["passes"] and learn[a]["kind"] == "complete-path"]
        if needs and contingency and all(a in contingency for a in arms):
            return contingency, "decided at 4000 steps"
        return table, ""

    def contrast_stats(c: dict, comparator: str, ablated: str, data: dict) -> dict:
        return paired(
            [endpoint(data, comparator, s, c["endpoint"], criteria) for s in seeds],
            [endpoint(data, ablated, s, c["endpoint"], criteria) for s in seeds],
            t,
        )

    checks: dict[str, str] = {}
    for c in criteria["contrast"]:
        cid = c["id"]
        kind = c["kind"]
        if kind == "not-tested":
            result["verdicts"][cid] = {"verdict": "NOT TESTED", "reason": c["note"]}
            continue
        arms = [c["comparator"], c["ablated"]] + ([c["attribution_ablated"]] if "attribution_ablated" in c else [])
        if any(arm not in table for arm in arms):
            result["verdicts"][cid] = {"verdict": "NOT RUN", "reason": "not run (budget rule)"}
            continue
        data, note = source(arms)
        stats = contrast_stats(c, c["comparator"], c["ablated"], data)
        entry: dict = {"endpoint": c["endpoint"], "stats": stats}
        if note:
            entry["note"] = note

        if kind == "manipulation-check":
            passed = stats["mean"] >= c["pass_min"] and stats["positive_seeds"] == len(seeds)
            checks[cid] = "PASS" if passed else "FAIL"
            entry.update(verdict=checks[cid], reason=f"mean >= {c['pass_min']} and every seed positive" if passed else "the raw path is not shown to be used")
        elif kind == "negative-control":
            lower, upper = stats["ci"]
            eq = c["equivalence"]
            reason_block = blocked(arms)
            if reason_block:
                entry.update(verdict="INCONCLUSIVE", reason=reason_block)
            elif -eq < lower and upper < eq:
                entry.update(verdict="EQUIVALENT", reason=f"the CI lies inside (-{eq}, {eq})")
            else:
                verdict, reason = common_rule(stats, c["delta_min"], None)
                if verdict == "SUPPORTS":
                    entry.update(verdict="SUPPORTS", reason=c["supports"])
                elif verdict == "HARMFUL":
                    entry.update(verdict="HARMFUL", reason=reason)
                else:
                    entry.update(verdict="INCONCLUSIVE", reason="neither EQUIVALENT, SUPPORTS nor HARMFUL")
            hand = refs[c["deterministic_reference"]]["test_iid"]
            full = [data["full"][s]["test_iid"]["accuracy"] for s in seeds]
            if min(full) >= hand + c["deterministic_margin"]:
                claim = "holds"
            elif max(full) < hand:
                claim = "fails"
            else:
                claim = "mixed"
            entry["deterministic_reference"] = {"hand_weighted_test_iid": hand, "full_min": min(full), "full_max": max(full), "learned_beats_deterministic": claim}
        elif kind == "mechanism-pair":
            reason_block = blocked(arms)
            first = common_rule(stats, c["delta_min"], reason_block)
            attribution = contrast_stats(c, c["comparator"], c["attribution_ablated"], data)
            second = common_rule(attribution, c["delta_min"], reason_block)
            entry["attribution_stats"] = attribution
            entry["component_verdicts"] = {c["ablated"]: first[0], c["attribution_ablated"]: second[0]}
            if first[0] in ("FALSIFIES", "HARMFUL"):
                entry.update(verdict=first[0], reason=c["falsifies"])
            elif first[0] == "SUPPORTS" and second[0] == "SUPPORTS":
                entry.update(verdict="SUPPORTS-NONLINEARITY", reason=c["supports"])
            elif first[0] == "SUPPORTS":
                entry.update(verdict="INCONCLUSIVE", reason="the gain is not attributable to the nonlinearity")
            else:
                entry.update(verdict="INCONCLUSIVE", reason=first[1])
            entry["linear_minus_zero"] = statistics.fmean(
                endpoint(data, c["attribution_ablated"], s, c["endpoint"], criteria)
                - endpoint(data, c["ablated"], s, c["endpoint"], criteria)
                for s in seeds
            )
        else:  # mechanism
            reason_block = blocked(arms)
            # A leak or a failed gate means no number here can be trusted, so it
            # outranks every other outcome, NOT EXERCISED included.
            if reason_block:
                entry.update(verdict="INCONCLUSIVE", reason=reason_block)
            elif c.get("requires_pass") and checks.get(c["requires_pass"]) == "FAIL":
                entry.update(verdict="NOT EXERCISED", reason=f"{c['requires_pass']} failed")
            elif "blindness_margin" in c and mean_over_seeds(table, c["ablated"], "test_iid", seeds) > refs["bound_additive"]["test_iid"] + c["blindness_margin"]:
                entry.update(verdict="INCONCLUSIVE", reason="blindness failed: the blind arm exceeds the bound additive reference")
            else:
                verdict, reason = common_rule(stats, c["delta_min"], reason_block)
                text = c["supports"] if verdict == "SUPPORTS" else c["falsifies"] if verdict in ("FALSIFIES", "HARMFUL") else reason
                entry.update(verdict=verdict, reason=text)
            if c.get("report_mask_share") and "no-semantic-slots-masked" in data:
                entry["mask_share"] = statistics.fmean(
                    endpoint(data, "no-semantic-slots-masked", s, c["endpoint"], criteria)
                    - endpoint(data, "no-semantic-slots", s, c["endpoint"], criteria)
                    for s in seeds
                )
            dose = c.get("dose_response_arm")
            if dose:
                if dose in data:
                    ahead = sum(
                        1 for s in seeds
                        if endpoint(data, dose, s, c["endpoint"], criteria) - endpoint(data, "full", s, c["endpoint"], criteria) >= 0
                    )
                    entry["dose_response"] = {"seeds_at_or_above_full": ahead, "monotone": ahead >= c["dose_response_min_seeds"]}
                else:
                    entry["dose_response"] = "not run (budget rule)"
        result["verdicts"][cid] = entry

    if "no-semantic-slots" in table and "no-semantic-slots-masked" in table:
        result["reported"]["mask_effect"] = {
            "validity_decisive_test_iid": statistics.fmean(
                endpoint(table, "no-semantic-slots-masked", s, "subset:test_iid:validity", criteria)
                - endpoint(table, "no-semantic-slots", s, "subset:test_iid:validity", criteria)
                for s in seeds
            ),
            "ood_validity": statistics.fmean(
                table["no-semantic-slots-masked"][s]["ood_validity"]["accuracy"]
                - table["no-semantic-slots"][s]["ood_validity"]["accuracy"]
                for s in seeds
            ),
        }
    rules = criteria["transfer"]
    heldout_ref = references["no_transfer"]["ood_compose_epi"][rules["heldout_reference"]]
    transfer = {}
    for arm in table:
        strict = [table[arm][s]["ood_compose_regime"]["subsets"]["transfer"]["accuracy"] for s in seeds]
        held = [table[arm][s]["ood_compose_epi"]["subsets"]["heldout"]["accuracy"] for s in seeds]
        transfer[arm] = {
            "new_role_regime_cell": "transfers" if min(strict) >= rules["strict_transfer_min"]
            else "does not transfer" if max(strict) < rules["strict_transfer_min"] else "mixed",
            "heldout_pairs": "transfers" if min(held) >= heldout_ref
            else "does not transfer" if max(held) < heldout_ref else "mixed",
            "strict_transfer": strict,
            "heldout": held,
        }
    result["reported"]["transfer"] = transfer
    return result


# ------------------------------------------------------------------------ I/O


def load_score_module():
    path = ROOT / "benchmarks/operator-routing/score.py"
    spec = importlib.util.spec_from_file_location("operator_routing_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_records(experiment: str, entrypoint: str) -> list[Path]:
    results = ROOT / "experiments" / EXPERIMENTS[experiment] / "results"
    out = []
    for path in sorted(results.glob("run-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("entrypoint") == entrypoint:
            out.append(path)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def at_tag(relative: str) -> bytes | None:
    try:
        return subprocess.run(["git", "show", f"{PREREG_TAG}:{relative}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None


def build_table(paths: list[Path], score) -> tuple[dict, list[str], list[dict]]:
    document, problems = score.score_records(paths, DATA_DIR)
    table: dict = {}
    for entry in document["results"]:
        # score.py names a split's accuracy route_accuracy; the core reads `accuracy`.
        entry["accuracy"] = entry["route_accuracy"]
        table.setdefault(entry["arm"], {}).setdefault(entry["seed"], {})[entry["split"]] = entry
    return table, problems, document["results"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    frozen = at_tag("research/falsification/A0-ablations-v1/criteria.toml")
    if frozen is None:
        raise SystemExit(f"no {PREREG_TAG} tag: there is nothing preregistered to apply")
    if hashlib.sha256(frozen).hexdigest() != sha256(CRITERIA):
        raise SystemExit("criteria.toml differs from the preregistered file; refusing")
    criteria = tomllib.loads(CRITERIA.read_text(encoding="utf-8"))
    references = json.loads((STUDY_DIR / "references.json").read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    score = load_score_module()
    seeds = criteria["seeds"]

    paths = {e: eval_records(e, "a0_ablation_entrypoint") for e in EXPERIMENTS}
    records = {p: json.loads(p.read_text(encoding="utf-8")) for ps in paths.values() for p in ps}
    all_paths = [p for ps in paths.values() for p in ps]
    table, problems, scored = build_table(all_paths, score)
    gates: dict = {}

    # G0: data identity and label agreement.
    data_rows = []
    for record in records.values():
        for line in record["stdout"].splitlines():
            if line.startswith('{"row":"data"'):
                data_rows.append(json.loads(line))
    fnv_ok = all(r["data_fnv64"] == lock["data_fnv1a64"] for r in data_rows)
    labels_ok = all(
        r[f"label_fnv64_{split}"] == info["label_fnv1a64"]
        for r in data_rows for split, info in lock["splits"].items()
    )
    agreement, disagreements = score.agree(DATA_DIR, verbose=False)
    gates["G0"] = {"pass": fnv_ok and labels_ok and not disagreements and references["all_bands_pass"],
                   "detail": {"data_fnv_matches": fnv_ok, "label_fnvs_match": labels_ok,
                              "score_generator_disagreements": len(disagreements),
                              "bands_pass": references["all_bands_pass"]}}

    # G2: the rerun reproduces the full arm's rows and PRED lines byte for byte.
    rerun = eval_records("M001", "a0_rerun_entrypoint")
    original = [p for p in paths["M001"] if records[p]["seed"] == 17 and records[p]["status"] == "completed"]
    def full_lines(stdout: str) -> list[str]:
        return [l for l in stdout.splitlines() if '"arm":"full"' in l or l.startswith("PRED full ")]
    g2 = bool(rerun) and len(original) == 1 and full_lines(json.loads(rerun[-1].read_text())["stdout"]) == full_lines(records[original[0]]["stdout"])
    gates["G2"] = {"pass": g2, "detail": {"rerun_records": len(rerun)}}

    # G3: every planned process completed with finite losses.
    incomplete = []
    for experiment in EXPERIMENTS:
        for seed in seeds:
            done = [p for p in paths[experiment] if records[p]["seed"] == seed and records[p]["status"] == "completed"]
            if not done:
                incomplete.append(f"{experiment}/{seed}")
            elif any('"nan":1' in l for l in records[done[-1]]["stdout"].splitlines()):
                incomplete.append(f"{experiment}/{seed} (nan)")
    gates["G3"] = {"pass": not incomplete, "detail": incomplete}

    # G4: every record clean, one commit, descended from the tag, nothing else changed.
    shas = {r.get("git_sha") for r in records.values()}
    dirty = [str(p) for p, r in records.items() if r.get("git_dirty") is not False]
    sha = next(iter(shas)) if len(shas) == 1 else None
    ancestor = sha is not None and subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_TAG, sha], cwd=ROOT).returncode == 0
    prereg_unchanged = at_tag("research/falsification/A0-ablations-v1/PREREGISTRATION.md") == (STUDY_DIR / "PREREGISTRATION.md").read_bytes()
    gates["G4"] = {"pass": len(shas) == 1 and not dirty and ancestor and prereg_unchanged,
                   "detail": {"shas": sorted(str(s) for s in shas), "dirty": dirty, "tag_is_ancestor": ancestor,
                              "preregistration_unchanged": prereg_unchanged}}

    # G5: the binary's counts equal score.py's.
    disagree = [f"{e['arm']}/{e['seed']}/{e['split']}" for e in scored if e.get("rust_count_agrees") is not True]
    gates["G5"] = {"pass": not disagree and not problems, "detail": {"disagreements": disagree, "problems": problems}}

    # G6: the correctness logs the driver wrote at the evaluation commit.
    g6_path = STUDY_DIR / "logs/g6.json"
    g6 = json.loads(g6_path.read_text()) if g6_path.exists() else {}
    gates["G6"] = {"pass": bool(g6) and all(v == "pass" for v in g6.values()), "detail": g6}

    contingency_paths = [p for e in EXPERIMENTS for p in eval_records(e, "a0_contingency_entrypoint")]
    contingency = build_table(contingency_paths, score)[0] if contingency_paths else None
    result = decide(table, references, criteria, gates, contingency)

    # Cross-check against the stock runner aggregate: per-arm mean accuracy per split.
    stock_mismatch = []
    for experiment in EXPERIMENTS:
        out = subprocess.run([sys.executable, "scripts/run_experiment.py", "aggregate", experiment,
                              "--entrypoint", "a0_ablation_entrypoint"], cwd=ROOT, text=True, capture_output=True)
        agg_path = ROOT / out.stdout.strip() if out.stdout.strip() else None
        if agg_path is None or not agg_path.exists():
            stock_mismatch.append(f"{experiment}: stock aggregate failed: {out.stderr.strip()[:200]}")
            continue
        stock = json.loads(agg_path.read_text())
        for group in stock["groups"]:
            key = group["key"]
            if key.get("row") != "final":
                continue
            ours = statistics.fmean(table[key["arm"]][s][key["split"]]["accuracy"] for s in seeds)
            if abs(group["metrics"]["accuracy"]["mean"] - ours) > 1e-12:
                stock_mismatch.append(f"{experiment}/{key['arm']}/{key['split']}")
    result["stock_aggregate_cross_check"] = {"pass": not stock_mismatch, "mismatches": stock_mismatch}

    for experiment in EXPERIMENTS:
        arms = sorted({json.loads(l)["arm"] for p in paths[experiment] for l in records[p]["stdout"].splitlines() if l.startswith('{"row":"meta"')})
        metrics = {arm: {str(s): {split: {k: table[arm][s][split][k] for k in ("n", "correct", "route_accuracy", "cost_adjusted_regret", "task_success", "subsets")}
                                   for split in TEST_SPLITS}
                         for s in seeds if s in table.get(arm, {})}
                   for arm in arms}
        out_path = ROOT / "experiments" / EXPERIMENTS[experiment] / "results/a0_internal_metrics.json"
        out_path.write_text(json.dumps({"study": "A0-ablations-v1", "scope": "A0-internal ablation on synthetic operator-routing v1; not this manifest's baseline comparison", "arms": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (STUDY_DIR / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({cid: v.get("verdict") for cid, v in result["verdicts"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
