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
import re
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
    """Return SUPPORTS, FALSIFIES, HARMFUL, or INCONCLUSIVE with its reason.

    A nonempty blocked reason forces INCONCLUSIVE before evaluating statistics.
    """
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
    """Return an arm's mean split accuracy over the specified seeds."""
    return statistics.fmean(table[arm][s][split]["accuracy"] for s in seeds)


def learnability(table: dict, references: dict, criteria: dict, seeds: list[int]) -> dict:
    """Per configured arm: whether mean test_iid over the supplied seeds reaches its bar."""
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
            f"A0 (one block, at the calibrated width) does not learn operator-routing v1: the full arm's mean "
            f"test_iid {full_mean:.4f} is below {competence_bar:.4f}. No mechanism verdict is issued."
        )
        for contrast in criteria["contrast"]:
            if contrast["kind"] == "not-tested":
                result["verdicts"][contrast["id"]] = {"verdict": "NOT TESTED", "reason": contrast["note"]}
            else:
                result["verdicts"][contrast["id"]] = {"verdict": "NOT ISSUED", "reason": "G1 (competence) failed"}
        return result

    failed_gates = [name for name, gate in result["gates"].items() if not gate["pass"]]
    learn = learnability(table, references, criteria, seeds)
    result["learnability"] = learn

    def blocked(arms: list[str]) -> str | None:
        """Explain any leak, gate, or learnability condition that blocks these arms."""
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
                # DESIGN.md (learnability, "Inconclusive if"): "If the arm is still
                # below the bar, the contrast is INCONCLUSIVE: an optimisation
                # failure, not evidence that the mechanism is needed." criteria.toml
                # sets contingency_steps; the bar is the same, only the steps differ.
                rerun = mean_over_seeds(contingency, arm, "test_iid", seeds)
                if rerun < status["bar"]:
                    return (
                        f"{arm} is still below its learnability bar at 4000 steps "
                        f"({rerun:.4f} < {status['bar']:.4f}): an optimisation failure, "
                        "not evidence that the mechanism is needed"
                    )
        return None

    def source(arms: list[str]) -> tuple[dict, str]:
        """The contingency table decides when a complete-path arm failed learnability."""
        needs = [a for a in arms if learn.get(a) and not learn[a]["passes"] and learn[a]["kind"] == "complete-path"]
        if needs and contingency and all(a in contingency for a in arms):
            return contingency, "decided at 4000 steps"
        return table, ""

    def contrast_stats(c: dict, comparator: str, ablated: str, data: dict) -> dict:
        """Compute paired endpoint differences and uncertainty for the configured contrast."""
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
            elif upper < 0:
                # DESIGN.md: "HARMFUL (the frozen arm is better) is recorded if the CI
                # upper bound is < 0", whether or not the CI is also inside the band.
                inside = " (the CI is also inside the equivalence band)" if -eq < lower else ""
                entry.update(verdict="HARMFUL", reason=f"CI upper < 0: the frozen arm is better{inside}")
            elif -eq < lower and upper < eq:
                entry.update(verdict="EQUIVALENT", reason=f"the CI lies inside (-{eq}, {eq})")
            else:
                verdict, reason = common_rule(stats, c["delta_min"], None)
                if verdict == "SUPPORTS":
                    entry.update(verdict="SUPPORTS", reason=c["supports"])
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
            if c.get("report_mask_share"):
                # The share compares the two content-free arms at one budget. The
                # contingency re-runs only the arm below its bar, so when it decides
                # and lacks the masked arm, the share is the one at S*, labelled so.
                pair = ("no-semantic-slots", "no-semantic-slots-masked")
                share_data = data if all(a in data for a in pair) else table
                if all(a in share_data for a in pair):
                    entry["mask_share"] = statistics.fmean(
                        endpoint(share_data, "no-semantic-slots-masked", s, c["endpoint"], criteria)
                        - endpoint(share_data, "no-semantic-slots", s, c["endpoint"], criteria)
                        for s in seeds
                    )
                    entry["mask_share_budget"] = (
                        f"{criteria['learnability']['contingency_steps']} steps" if share_data is contingency else "S*"
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
        # Reported, never a verdict. An arm below its learnability bar makes the
        # difference partly a training-budget effect, which the report must say.
        result["reported"]["mask_effect"] = {
            "below_learnability_bar": sorted(
                arm for arm in ("no-semantic-slots", "no-semantic-slots-masked")
                if learn.get(arm) and not learn[arm]["passes"]
            ),
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
    if contingency:
        # What the contingency runs reached, beside the bar they were held to.
        result["reported"]["contingency"] = {
            "steps": criteria["learnability"]["contingency_steps"],
            "arms": {
                arm: {
                    "mean_test_iid": mean_over_seeds(contingency, arm, "test_iid", seeds),
                    "bar": learn[arm]["bar"],
                    "passes": mean_over_seeds(contingency, arm, "test_iid", seeds) >= learn[arm]["bar"],
                    "test_iid": [contingency[arm][s]["test_iid"]["accuracy"] for s in seeds],
                    "composite_A": [endpoint(contingency, arm, s, "composite:A", criteria) for s in seeds],
                }
                for arm in sorted(contingency)
                if arm in learn
            },
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
    """Import the independent operator-routing scorer from its repository path."""
    path = ROOT / "benchmarks/operator-routing/score.py"
    spec = importlib.util.spec_from_file_location("operator_routing_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(experiment: str) -> dict[Path, dict]:
    """Read each run once, retaining study records in the order the runner wrote them."""
    results = ROOT / "experiments" / EXPERIMENTS[experiment] / "results"
    records = {}
    for path in sorted(results.glob("run-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("entrypoint") in STUDY_KINDS.values():
            records[path] = record
    return records


def git(*args: str) -> str:
    """Run Git in the repository and return stdout, raising on a nonzero exit."""
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def sha256(path: Path) -> str:
    """Return the SHA-256 hexadecimal digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def at_tag(relative: str) -> bytes | None:
    """Read a file at the preregistration tag, returning None if Git cannot find it."""
    try:
        return subprocess.run(["git", "show", f"{PREREG_TAG}:{relative}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None


def build_table(paths: list[Path], score) -> tuple[dict, list[str], list[dict]]:
    """Index independently scored records by arm, seed, and split, reporting duplicates."""
    document, problems = score.score_records(paths, DATA_DIR)
    table: dict = {}
    for entry in document["results"]:
        # score.py names a split's accuracy route_accuracy; the core reads `accuracy`.
        entry["accuracy"] = entry["route_accuracy"]
        cell = table.setdefault(entry["arm"], {}).setdefault(entry["seed"], {})
        if entry["split"] in cell:
            problems.append(f"{entry['record']}: a second score for {entry['arm']}/{entry['seed']}/{entry['split']}")
        cell[entry["split"]] = entry
    return table, problems, document["results"]


# ------------------------------------------------------------------------ record gates
#
# Pure functions over run records (dicts as scripts/run_experiment.py writes them),
# so each gate condition can be tested without a study on disk. They cover every
# record a verdict can rest on: the evaluation, the G2 rerun and the contingency.

STUDY_KINDS = {
    "eval": "a0_ablation_entrypoint",
    "rerun": "a0_rerun_entrypoint",
    "contingency": "a0_contingency_entrypoint",
}
FREEZE_FILES = frozenset({
    "research/falsification/A0-ablations-v1/lr_selection.tsv",
    "research/falsification/A0-ablations-v1/lr_selection.json",
})
RUN_RECORD_PATH = re.compile(r"^experiments/model/M00[1-4]-[a-z-]+/results/run-[^/]+\.json$")


def has_nan(record: dict) -> bool:
    """Detect the study binary's compact JSON divergence marker in recorded stdout."""
    return any('"nan":1' in line for line in record.get("stdout", "").splitlines())


def chosen_per_seed(records: list[dict]) -> dict[int, dict]:
    """Per seed, the last completed record. DESIGN.md G3 gives a failed or NaN
    process one full retry, so an earlier failure beside a later success is
    expected; a record that did not complete never enters a table."""
    chosen: dict[int, dict] = {}
    for record in records:
        if record.get("status") == "completed":
            chosen[record["seed"]] = record
    return chosen


def completeness(chosen: dict[str, dict[str, dict[int, dict]]], seeds: list[int],
                 planned: dict[str, list[str]], table: dict) -> list[str]:
    """G3: every planned process completed with finite losses, and every arm the
    budget rule kept has scores for every seed. `chosen` is {kind: {experiment:
    {seed: record}}}; the evaluation must cover every experiment in `planned`, a
    contingency every seed of each experiment it ran for, and the rerun its seed."""
    problems = []
    for kind, by_experiment in chosen.items():
        for experiment, by_seed in by_experiment.items():
            wanted = seeds if kind != "rerun" else sorted(by_seed) or [seeds[0]]
            for seed in wanted:
                record = by_seed.get(seed)
                if record is None:
                    problems.append(f"{kind} {experiment}/{seed}: no completed process")
                elif has_nan(record):
                    problems.append(f"{kind} {experiment}/{seed}: a non-finite loss")
    for experiment in planned:
        if experiment not in chosen.get("eval", {}):
            problems.append(f"eval {experiment}: no records")
        for arm in planned[experiment]:
            missing = [s for s in seeds if s not in table.get(arm, {})]
            if missing:
                problems.append(f"{experiment}: the budget kept {arm}, which has no scores for seeds {missing}")
    return problems


def freeze_violations(changed: list[str], entrypoint_of) -> list[str]:
    """G4's diff condition (DESIGN.md, G4): between the preregistration tag and the
    evaluation commit, only the learning-rate selection and the sweep's run records
    may change. `entrypoint_of(path)` reads a changed run record's entrypoint."""
    return [
        path for path in changed
        if path not in FREEZE_FILES
        and not (RUN_RECORD_PATH.match(path) and entrypoint_of(path) == "a0_sweep_entrypoint")
    ]


def provenance(records: dict[str, dict]) -> dict:
    """G4's record condition: one commit, and a clean worktree for every record."""
    shas = sorted({str(r.get("git_sha")) for r in records.values()})
    dirty = sorted(name for name, r in records.items() if r.get("git_dirty") is not False)
    return {"shas": shas, "dirty": dirty, "pass": len(shas) == 1 and not dirty}


def data_identity(records: dict[str, dict], lock: dict) -> dict:
    """G0's record condition: every process read the locked data. A record with
    no data row fails, so an empty or truncated stdout cannot pass vacuously."""
    missing, wrong = [], []
    for name, record in records.items():
        rows = [json.loads(line) for line in record.get("stdout", "").splitlines() if line.startswith('{"row":"data"')]
        if not rows:
            missing.append(name)
            continue
        for row in rows:
            if row.get("data_fnv64") != lock["data_fnv1a64"] or any(
                row.get(f"label_fnv64_{split}") != info["label_fnv1a64"] for split, info in lock["splits"].items()
            ):
                wrong.append(name)
    return {"records_without_data_row": missing, "records_with_other_data": sorted(set(wrong)),
            "pass": bool(records) and not missing and not wrong}


def rerun_reproduces(original: dict | None, rerun: dict | None) -> dict:
    """G2: the rerun reproduces the full arm's JSON rows and PRED lines byte for
    byte. Both must exist and carry full-arm lines; two empty lists are not a match."""
    def full_lines(record: dict | None) -> list[str]:
        """Extract full-arm JSON and PRED lines used for exact rerun comparison."""
        if record is None:
            return []
        return [l for l in record.get("stdout", "").splitlines() if '"arm":"full"' in l or l.startswith("PRED full ")]
    ours, theirs = full_lines(original), full_lines(rerun)
    return {"full_lines": len(ours), "pass": bool(ours) and ours == theirs}


def main(argv: list[str] | None = None) -> int:
    """Verify frozen criteria, score study records, and write gates, verdicts, and metrics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    frozen = at_tag("research/falsification/A0-ablations-v1/criteria.toml")
    if frozen is None:
        raise SystemExit(f"no {PREREG_TAG} tag: there is nothing preregistered to apply")
    if hashlib.sha256(frozen).hexdigest() != sha256(CRITERIA):
        raise SystemExit("criteria.toml differs from the preregistered file; refusing")
    criteria = tomllib.loads(CRITERIA.read_text(encoding="utf-8"))
    references = json.loads((STUDY_DIR / "references.json").read_text(encoding="utf-8"))
    budget = json.loads((STUDY_DIR / "budget.json").read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    score = load_score_module()
    seeds = criteria["seeds"]

    # Every study record, by kind and experiment; the tables use one completed
    # record per (experiment, seed), and the gates see all of them.
    loaded = {e: load_records(e) for e in EXPERIMENTS}
    paths = {
        kind: {e: [p for p, r in loaded[e].items() if r.get("entrypoint") == entrypoint]
               for e in EXPERIMENTS}
        for kind, entrypoint in STUDY_KINDS.items()
    }
    records = {str(p.relative_to(ROOT)): loaded[e][p]
               for by_experiment in paths.values() for e, ps in by_experiment.items() for p in ps}
    by_name = {id(r): name for name, r in records.items()}

    def chosen_of(kind: str) -> dict[str, dict[int, dict]]:
        """Select one completed record per experiment and seed for a study phase."""
        out = {}
        for experiment, ps in paths[kind].items():
            if ps:
                out[experiment] = chosen_per_seed([records[str(p.relative_to(ROOT))] for p in ps])
        return out

    chosen = {kind: chosen_of(kind) for kind in STUDY_KINDS}
    chosen_paths = {kind: [ROOT / by_name[id(r)] for by_seed in chosen[kind].values() for r in by_seed.values()]
                    for kind in STUDY_KINDS}
    chosen_records = {by_name[id(r)]: r for kind in STUDY_KINDS for by_seed in chosen[kind].values() for r in by_seed.values()}

    tables = {}
    scored = {}
    all_problems = []
    for kind in ("eval", "contingency", "rerun"):
        tables[kind], problems, scored[kind] = (
            build_table(chosen_paths[kind], score)
            if kind == "eval" or chosen_paths[kind] else (None, [], [])
        )
        all_problems.extend(problems)
    table = tables["eval"]
    contingency = tables["contingency"]
    gates: dict = {}

    # G0: data identity and label agreement, for every process a verdict rests on.
    identity = data_identity(chosen_records, lock)
    agreement, disagreements = score.agree(DATA_DIR, verbose=False)
    gates["G0"] = {"pass": identity["pass"] and not disagreements and references["all_bands_pass"],
                   "detail": {"records": len(chosen_records),
                              "records_without_data_row": identity["records_without_data_row"],
                              "records_with_other_data": identity["records_with_other_data"],
                              "score_generator_disagreements": len(disagreements),
                              "bands_pass": references["all_bands_pass"]}}

    # G2: the rerun reproduces the full arm's rows and PRED lines byte for byte.
    rerun_seed = next(iter(chosen["rerun"].get("M001", {})), None)
    reproduced = rerun_reproduces(chosen["eval"].get("M001", {}).get(rerun_seed),
                                  chosen["rerun"].get("M001", {}).get(rerun_seed))
    gates["G2"] = {"pass": reproduced["pass"],
                   "detail": {"rerun_records": len(paths["rerun"]["M001"]), "seed": rerun_seed,
                              "full_lines_compared": reproduced["full_lines"]}}

    # G3: every planned process completed with finite losses, every kept arm scored.
    incomplete = completeness(chosen, seeds, budget["arms"], table)
    gates["G3"] = {"pass": not incomplete, "detail": incomplete}

    # G4: every record clean, one commit, descended from the tag, and between the
    # tag and that commit only the lr selection and the sweep records changed.
    origin = provenance(records)
    sha = origin["shas"][0] if len(origin["shas"]) == 1 else None
    ancestor = sha is not None and subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_TAG, sha], cwd=ROOT).returncode == 0
    stray = ["(no single commit)"]
    if ancestor:
        changed = git("diff", "--name-only", f"{PREREG_TAG}..{sha}").split()
        def entrypoint_at(path: str) -> str | None:
            """Read a record's entrypoint at the evaluation commit, or None on lookup failure."""
            try:
                return json.loads(git("show", f"{sha}:{path}")).get("entrypoint")
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                return None  # deleted, or not a record: not a sweep record either
        stray = freeze_violations(changed, entrypoint_at)
    prereg_unchanged = at_tag("research/falsification/A0-ablations-v1/PREREGISTRATION.md") == (STUDY_DIR / "PREREGISTRATION.md").read_bytes()
    gates["G4"] = {"pass": origin["pass"] and ancestor and not stray and prereg_unchanged,
                   "detail": {"records": len(records), "shas": origin["shas"], "dirty": origin["dirty"],
                              "tag_is_ancestor": ancestor, "changed_beyond_lr_selection_and_sweep": stray,
                              "preregistration_unchanged": prereg_unchanged}}

    # G5: the binary's counts equal score.py's, on every scored process.
    disagree = [f"{e['arm']}/{e['seed']}/{e['split']}" + (f" ({kind})" if kind != "eval" else "")
                for kind, entries in scored.items()
                for e in entries if e.get("rust_count_agrees") is not True]
    gates["G5"] = {"pass": not disagree and not all_problems, "detail": {"disagreements": disagree, "problems": all_problems}}

    # G6: the correctness logs the driver wrote at the evaluation commit.
    g6_path = STUDY_DIR / "logs/g6.json"
    g6 = json.loads(g6_path.read_text()) if g6_path.exists() else {}
    gates["G6"] = {"pass": bool(g6) and all(v == "pass" for v in g6.values()), "detail": g6}

    result = decide(table, references, criteria, gates, contingency)

    # Cross-check against the stock runner aggregate: per-arm mean accuracy per split,
    # for the evaluation and, where it ran, the contingency.
    stock_mismatch = []
    for kind, scores_table in (("eval", table), ("contingency", contingency)):
        entrypoint = STUDY_KINDS[kind]
        for experiment in EXPERIMENTS:
            if not paths[kind][experiment]:
                continue
            out = subprocess.run([sys.executable, "scripts/run_experiment.py", "aggregate", experiment,
                                  "--entrypoint", entrypoint], cwd=ROOT, text=True, capture_output=True)
            agg_path = ROOT / out.stdout.strip() if out.stdout.strip() else None
            if agg_path is None or not agg_path.exists():
                stock_mismatch.append(f"{experiment}/{entrypoint}: stock aggregate failed: {out.stderr.strip()[:200]}")
                continue
            stock = json.loads(agg_path.read_text())
            for group in stock["groups"]:
                key = group["key"]
                if key.get("row") != "final":
                    continue
                arm_scores = (scores_table or {}).get(key["arm"], {})
                if any(key["split"] not in arm_scores.get(s, {}) for s in seeds):
                    stock_mismatch.append(f"{experiment}/{entrypoint}/{key['arm']}/{key['split']}: missing scores")
                    continue
                ours = statistics.fmean(arm_scores[s][key["split"]]["accuracy"] for s in seeds)
                if abs(group["metrics"]["accuracy"]["mean"] - ours) > 1e-12:
                    stock_mismatch.append(f"{experiment}/{entrypoint}/{key['arm']}/{key['split']}")
    result["stock_aggregate_cross_check"] = {"pass": not stock_mismatch, "mismatches": stock_mismatch}

    def per_split(scores_table: dict, arm: str) -> dict:
        """Export per-seed test metrics for an arm, omitting absent seeds."""
        return {str(s): {split: {k: scores_table[arm][s][split][k] for k in ("n", "correct", "route_accuracy", "cost_adjusted_regret", "task_success", "subsets")}
                         for split in TEST_SPLITS}
                for s in seeds if s in scores_table.get(arm, {})}

    def arms_run(experiment: str, kind: str) -> list[str]:
        """List arm names reported by selected records for an experiment and phase."""
        return sorted({json.loads(line)["arm"] for r in chosen[kind].get(experiment, {}).values()
                       for line in r["stdout"].splitlines() if line.startswith('{"row":"meta"')})

    for experiment in EXPERIMENTS:
        document = {"study": "A0-ablations-v1", "scope": "A0-internal ablation on synthetic operator-routing v1; not this manifest's baseline comparison",
                    "arms": {arm: per_split(table, arm) for arm in arms_run(experiment, "eval")}}
        if experiment in chosen["contingency"]:
            document["contingency_steps"] = criteria["learnability"]["contingency_steps"]
            document["contingency_arms"] = {arm: per_split(contingency, arm) for arm in arms_run(experiment, "contingency")}
        out_path = ROOT / "experiments" / EXPERIMENTS[experiment] / "results/a0_internal_metrics.json"
        out_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (STUDY_DIR / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({cid: v.get("verdict") for cid, v in result["verdicts"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
