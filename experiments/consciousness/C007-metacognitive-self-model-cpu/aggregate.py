"""Aggregate the C007 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run C007 --seed <s>` writes (results/run-<timestamp>-seed-<s>.json). Nothing of the experiment
is run again. Every declared seed must have a record, the records must be of one configuration and have run the checkout's code, and the
preregistration digests of the manifest, the table, the text and `criteria.toml` must be the frozen ones.

The decision is mechanical (`criteria.toml`), three outcomes plus invalid, INVALID taking precedence:

- a seed is judged by applying every coverage rule and every STRUCTURAL falsifier rule to the one JSON object its harness printed; a failure is a
  defect of the self-model's isolation (it changed the classifier, its advice reached what the verifier or a gate reads, nondeterminism): FALSIFIED
  (kind structural);
- INTEGRITY counters (matching of the self-model and the probe, splits, audits, loss and checkpoint bookkeeping): the harness exits 2 and the
  experiment is INVALID;
- guards: the traces carry no information about the classifier's errors on the shifted split, or the classifier's error rate there is outside the
  informative band; more than the allowed number of seeds on which a judged head hit the step cap; a sensitivity control, a removal control or the
  null world that the pipeline does not resolve: INVALID;
- the paired comparisons (Brier score of the predicted failure probability against the verifier's outcome on the shifted split; comparator minus
  self-model against the information-matched probe and the model's own calibrated confidence, one-sided 95% t bounds): COMPLETED iff both lower bounds
  reach the margin; FALSIFIED (refuted) iff the guards and controls hold and the Bonferroni upper bound against some comparator is below the margin
  (the advantage is below the margin, not absent); NOT_SHOWN otherwise;
- INVALID also when a seed cannot be judged, a declared seed is missing, or the mutation evidence is missing, incomplete, not the plan's or older
  than the code.

Exit status: 0 COMPLETED, 1 FALSIFIED, 2 INVALID or a refusal to aggregate, 3 NOT_SHOWN. `--report PATH` also writes the Markdown reading.
The harness is shown able to fail by `results/mutations.json` (`research/cseries/lib/run_mutations.py C007`)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/cseries/lib"))
import agg  # noqa: E402
import aggregate_run  # noqa: E402
from agg import COMPLETED, FALSIFIED, INVALID, NOT_SHOWN, PairedError  # noqa: E402

BENCHMARK = "c007"
EXPERIMENT = "C007"
CONTROLS = ("split_overlap", "capacity", "input_hash", "outcome_audit", "metric_audit", "write_probe", "advice_probe", "shuffle_probe", "treatment_probe", "fit_probe")
COUNT_KEYS = ("unconverged_judged_arms",)
GUARDED_PATHS = ("research/cseries", "research/consciousness/uv.lock", "research/consciousness/pyproject.toml", "experiments/consciousness/C007-metacognitive-self-model-cpu")
RESULTS_EXCLUDE = ":(exclude)experiments/consciousness/C007-metacognitive-self-model-cpu/results"
METRIC_NAMES = (
    "arms", "id_error_permille", "shifted_error_permille", "null_error_permille", "signal_permille", "unconverged_judged_arms", "environment",
    "head_gap_id_permille", "treatment_divergence_micro", "inert_treatment_arms", "fit_split_errors", "advice_probe_hits", "model_write_paths", "advice_leaks", "nondeterminism", "capacity_mismatch_arms", "input_mismatches", "aux_weight_errors",
    "split_overlap", "error_rate_errors", "outcome_audit_mismatches", "metric_audit_mismatches", "permutation_leaks", "shuffle_errors", "noise_errors",
    "nonfinite_arms", "checkpoint_mismatch_arms", "convergence_mismatch_arms", "hard_failures", "guards_failed",
)
SCOPE = (
    "a functional test, on a CPU toy mixture-of-experts classifier, of indicators HOT-2/HOT-3 as a functional property: does a self-model that sees the "
    "classifier's own routing traces and internal state predict the classifier's errors on a distribution-shifted split (scored by Brier against the "
    "verifier's deterministic outcomes) better than an information-matched probe (the same network on the same trace, the same optimiser and data, "
    "differing only in the weight of an auxiliary self-modelling loss) and than the classifier's own calibrated confidence? The self-model only observes "
    "and proposes: it has no write path to the classifier, the verifier or a gate (structural counters). No verbal self-report is read, trained or "
    "scored. It makes no claim about consciousness and none about language models or OLMoE (the GPU stage B)"
)
LIMITATIONS = [
    "this is a calibration (known-answer) run: the likeliest result is the registered refutation, a negative result about this head family on this toy; it shows that the pipeline resolves a null, not that it would detect a real gain (there is no positive-effect arm), and a COMPLETED would not move any indicator status (that needs stage B); the comparator that drives a verdict is named in its headline; the inert-treatment check shows that the auxiliary loss ran, not that it mattered",
    "the classifier is a two-layer, four-expert toy on Gaussian class clouds and its shift is a displacement with more noise; a result applies to this family, to a trace of 89 features and to a self-model that differs from its probe in one auxiliary loss (a masked prediction of the classifier's own layer-two expert choice and output entropy); a different self-modelling objective, or a self-model that sees more than the probe, is another experiment",
    "information-matched means the same features, capacity, optimiser, data and initial weights; the self-model and the probe therefore differ only in what they were asked to predict besides the error, which makes a COMPLETED outcome hard to expect and a refutation the likelier result; this is stated in the preregistration",
    "the model's own confidence is calibrated by a two-parameter Platt scaling fitted on the calibration split and the validation split of the head's world; both judged heads contain that calibrated confidence as a base term (the error logit is the Platt logit of the confidence plus the network's output) and are recalibrated post hoc on the validation split, so the question is what the trace adds to the model's own confidence; a better calibrator, or one fitted on the shifted split (which no deployed monitor has), would be a stronger comparator",
    "the shifted split is one displacement, one noise scale; calibration under other shifts, selective prediction and routing-choice prediction are reported (AUROC, ECE, risk-coverage area, expert accuracy) and not judged",
    "the removal controls (shuffled traces, a self-model trained on permuted outcomes) and the noise control are controls of the pipeline: they show that the effect measured depends on the pairing of traces and outcomes and that a deficit of about the margin's size is resolved, not that the main effect has any size",
    "the t bounds assume roughly normal paired differences, which the declared seeds cannot check; per-seed differences, medians and both bounds are reported",
    "an indicator property here is a functional property: the experiment says nothing about whether any system is conscious",
]

CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


def judge(result, exit_code, criteria, seed):
    entry = agg.judge_seed(result, exit_code, criteria, seed, BENCHMARK, EXPERIMENT, CONTROLS, COUNT_KEYS)
    entry["unconverged_judged_arms"] = result.get("unconverged_judged_arms") if isinstance(result, dict) else None
    return entry


def paired_comparisons(results: dict, paired: dict) -> dict:
    """The pre-specified comparisons over the declared seeds (Brier on the shifted split; comparator minus self-model, so positive is better):

    - judged: both lower one-sided 95% bounds (`t_support`) at or above the margin supports; the Bonferroni upper bound (`t_refute`, alpha / 2) below
      the margin against either comparator refutes (the advantage is below the margin, not absent);
    - sensitivity controls: gross (the constant predictor minus the self-model must reach the margin: the traces carry usable information), graded
      (the noised self-model minus the self-model must exceed `min_lower_permille`);
    - removal controls: shuffled traces and permuted outcomes: the improvement over the constant predictor must be below the margin at its upper bound;
      shuffled traces again, this time as the self-model's advantage over the probe given the same shuffled traces: below the margin at its upper bound;
    - null world: the self-model and the probe may differ by the margin in neither direction;
    - descriptive comparisons carry the same bounds and are never judged. Raises PairedError."""
    seeds = sorted(results)
    if len(seeds) != int(paired["seeds_required"]):
        raise PairedError(f"{len(seeds)} seeds, the rule is preregistered for {paired['seeds_required']}")
    t_support, t_refute, margin, metric = paired["t_support"], paired["t_refute"], paired["margin_permille"], paired["metric"]
    judged = []
    for other in paired["against"]:
        bound = agg.bound_of(results, seeds, metric, other, paired["arm"], t_support)
        adjusted = agg.bound_of(results, seeds, metric, other, paired["arm"], t_refute)
        judged.append({"against": other, "supports": bound["lower_bound_permille"] >= margin, "refutes": adjusted["upper_bound_permille"] < margin,
                       "upper_bound_adjusted_permille": adjusted["upper_bound_permille"], "margin_permille": margin, **bound})
    gross, graded, removal, null = paired["sensitivity_gross"], paired["sensitivity_graded"], paired["removal"], paired["null_world"]
    gb = agg.bound_of(results, seeds, metric, gross["floor"], gross["arm"], t_support)
    rb = agg.bound_of(results, seeds, metric, graded["degraded"], graded["arm"], t_support)
    removals = {}
    for arm in removal["arms"]:
        bound = agg.bound_of(results, seeds, metric, removal["reference"], arm, t_support)
        removals[arm] = {**bound, "removes_the_effect": bound["upper_bound_permille"] < paired["removal_margin_permille"]}
    rvp = paired["removal_vs_probe"]
    rvb = agg.bound_of(results, seeds, metric, rvp["against"], rvp["arm"], t_support)
    forward = agg.bound_of(results, seeds, null["metric"], null["against"], null["arm"], t_support)
    backward = agg.bound_of(results, seeds, null["metric"], null["arm"], null["against"], t_support)
    descriptive = [{"against": other, **agg.bound_of(results, seeds, metric, other, paired["arm"], t_support)} for other in paired["descriptive_against"]]
    return {
        "supported": all(item["supports"] for item in judged),
        "refuted_against": [item["against"] for item in judged if item["refutes"]],
        "judged": judged,
        "sensitivity": {
            "gross": {**gross, "resolves": gb["lower_bound_permille"] >= gross["margin_permille"], **gb},
            "graded": {**graded, "resolves": rb["lower_bound_permille"] > graded["min_lower_permille"], **rb},
        },
        "removal": removals,
        "removal_vs_probe": {**rvp, **rvb, "removes_the_effect": rvb["upper_bound_permille"] < paired["removal_margin_permille"]},
        "binding_comparator": min(judged, key=lambda item: item["lower_bound_permille"])["against"],
        "null_world": {**null, "forward": forward, "backward": backward,
                       "no_advantage": forward["lower_bound_permille"] < null["margin_permille"] and backward["lower_bound_permille"] < null["margin_permille"]},
        "descriptive": descriptive,
    }


def decide(seeds, static, declared, mutations, paired, counts, allowed, plan_size) -> dict:
    """The verdict from the seeds' judgements (`judge`, in any order), the static evidence, the mutation evidence, the paired comparisons and the count
    of seeds with a judged head at the step cap. Pure. INVALID first when a seed trips a harness-integrity counter."""
    present = sorted(entry["seed"] for entry in seeds)
    problems = []
    missing = sorted(set(declared) - set(present))
    duplicate = sorted({seed for seed in present if present.count(seed) > 1})
    extra = sorted(set(present) - set(declared))
    if missing:
        problems.append(f"no judgement of seeds {missing}")
    if duplicate or extra:
        problems.append(f"judgements of seeds {duplicate + extra} that are duplicated or not declared")
    if static is None:
        problems.append("the static evidence was not run")
    if mutations is None:
        problems.append("no mutation evidence (results/mutations.json)")
    elif mutations.get("killed") != mutations.get("total"):
        problems.append(f"the mutation check killed {mutations.get('killed')} of {mutations.get('total')} mutations")
    elif plan_size is not None and mutations.get("total") != plan_size:
        problems.append(f"the mutation evidence counts {mutations.get('total')} mutations, the plan lists {plan_size}")
    unmet = [e["seed"] for e in seeds if e["valid"] and e["coverage_failed"]]
    tripped = [e["seed"] for e in seeds if e["valid"] and e["falsifiers_failed"]]
    unjudged = [e["seed"] for e in seeds if not e["valid"]]
    uninformative = [e["seed"] for e in seeds if e["valid"] and e.get("guards_failed")]
    broken = [e["seed"] for e in seeds if e.get("integrity_failed")]
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    kind, reasons = None, []
    if broken:
        verdict = INVALID
        problems.append(f"harness integrity counters above 0 on seeds {broken}: a defect of the world or the harness, repaired in a successor")
    elif problems or unjudged:
        verdict = INVALID  # missing seeds, missing or partial mutation evidence, a seed that cannot be judged: INVALID before any other verdict
    elif unmet or tripped or static_failed:
        verdict, kind = FALSIFIED, "structural"
    elif uninformative:
        verdict = INVALID
        problems.append(f"no signal in the traces, or an uninformative error rate, on seeds {uninformative}: nothing for a self-model to do better")
    elif counts["unconverged"] > allowed["unconverged"]:
        verdict = INVALID
        problems.append(f"a judged head hit the step cap on {counts['unconverged']} seeds (more than {allowed['unconverged']}): the comparison would be one of optimisation speed")
    elif paired is None:
        verdict = INVALID
        problems.append("the paired comparison could not be computed")
    else:
        if not paired["sensitivity"]["gross"]["resolves"]:
            reasons.append("the traces carry no usable information on the shifted split (the constant predictor minus the self-model is below the margin)")
        if not paired["sensitivity"]["graded"]["resolves"]:
            reasons.append("the noise control does not resolve (a deficit of about the margin's size is not seen)")
        for arm, item in paired["removal"].items():
            if not item["removes_the_effect"]:
                reasons.append(f"the removal control {arm} still improves on the constant predictor by the margin: the pipeline reports an effect the pairing of traces and outcomes cannot explain")
        if not paired["removal_vs_probe"]["removes_the_effect"]:
            reasons.append("with the traces shuffled the self-model still beats the probe by the margin: an advantage the pairing of traces and outcomes cannot explain")
        if not paired["null_world"]["no_advantage"]:
            reasons.append("the null world shows an advantage of the margin's size where none can exist: the pipeline invents effects")
        if reasons:
            verdict = INVALID
            problems.extend(reasons)
        elif paired["refuted_against"]:
            verdict, kind = FALSIFIED, "refuted"
        elif paired["supported"]:
            verdict = COMPLETED
        else:
            verdict = NOT_SHOWN
    return {
        "verdict": verdict, "kind": kind, "vacuous_seeds": unmet, "falsified_seeds": tripped, "invalid_seeds": unjudged,
        "integrity_seeds": broken, "uninformative_seeds": uninformative, "static_evidence_failed": static_failed,
        "unconverged_seeds": counts["unconverged"], "refuted_against": (paired or {}).get("refuted_against", []),
        "binding_comparator": (paired or {}).get("binding_comparator"),
        "problems": problems, "recommended_status": {COMPLETED: "completed", FALSIFIED: "failed"}.get(verdict),
    }


def counts_of(judgements: list) -> dict:
    return {"unconverged": sum(1 for e in judgements if e.get("unconverged_judged_arms"))}


def verdict_of(judgements, static, declared, mutations, paired, criteria_paired, plan_size) -> dict:
    allowed = {"unconverged": int(criteria_paired["max_unconverged_seeds"])}
    return decide(judgements, static, declared, mutations, paired, counts_of(judgements), allowed, plan_size)


def report(verdict, seeds, static, mutations=None, paired=None) -> str:
    lines = [f"# C007 metacognitive self-model (CPU stage A): {verdict['verdict']}", ""]
    lines.append(f"Seeds: {', '.join(str(e['seed']) for e in sorted(seeds, key=lambda e: e['seed']))}.")
    lines.append("")
    for entry in sorted(seeds, key=lambda e: e["seed"]):
        state = "valid" if entry["valid"] else "INVALID"
        lines.append(f"- seed {entry['seed']}: {state}; {len(entry['coverage_failed'])} coverage rules unmet, {len(entry['falsifiers_failed'])} falsifier rules tripped, "
                     f"{len(entry.get('integrity_failed', []))} integrity counters above 0")
        for key in ("invalid", "coverage_failed", "falsifiers_failed", "integrity_failed"):
            lines.extend(f"    - {key}: {item}" for item in entry.get(key, []))
    lines.append("")
    for item in static or []:
        lines.append(f"- static evidence `{item['command']}`: exit {item['exit_code']}")
    if paired is not None:
        lines.append("")
        lines.append("Paired comparisons (Brier on the shifted split, per mille; comparator minus self-model, positive is better; one-sided 95% t bounds against the margin):")
        for item in paired["judged"]:
            state = "supports" if item["supports"] else ("REFUTES" if item["refutes"] else "straddles")
            lines.append(f"- vs {item['against']}: mean {item['mean_permille']:.2f}, sd {item['sd_permille']:.2f}, median {item['median_permille']:.2f}, "
                         f"{item['positive_seeds']} of {item['n']} seeds positive, lower {item['lower_bound_permille']:.2f}, upper {item['upper_bound_permille']:.2f} "
                         f"(adjusted upper {item['upper_bound_adjusted_permille']:.2f}), margin {item['margin_permille']}: {state}")
        for kind in ("gross", "graded"):
            sens = paired["sensitivity"][kind]
            other = sens.get("floor", sens.get("degraded"))
            lines.append(f"- {kind} sensitivity control ({other} minus {sens['arm']}): mean {sens['mean_permille']:.2f}, lower {sens['lower_bound_permille']:.2f}: "
                         f"{'resolves' if sens['resolves'] else 'DOES NOT RESOLVE'}")
        for arm, item in paired["removal"].items():
            lines.append(f"- removal control {arm} (improvement over the constant predictor): mean {item['mean_permille']:.2f}, upper {item['upper_bound_permille']:.2f}: "
                         f"{'removes the effect' if item['removes_the_effect'] else 'DOES NOT REMOVE THE EFFECT'}")
        null = paired["null_world"]
        item = paired["removal_vs_probe"]
        lines.append(f"- removal control {item['arm']} against {item['against']} (self-model advantage with the same shuffled traces): mean {item['mean_permille']:.2f}, upper {item['upper_bound_permille']:.2f}: "
                     f"{'removes the effect' if item['removes_the_effect'] else 'DOES NOT REMOVE THE EFFECT'}")
        lines.append(f"- null world ({null['arm']} vs {null['against']}): lower {null['forward']['lower_bound_permille']:.2f}, reverse lower {null['backward']['lower_bound_permille']:.2f}: "
                     f"{'no advantage' if null['no_advantage'] else 'ADVANTAGE WHERE NONE CAN EXIST'}")
        for item in paired["descriptive"]:
            lines.append(f"- (descriptive) {item['against']} minus self_model: mean {item['mean_permille']:.2f}, lower {item['lower_bound_permille']:.2f}, upper {item['upper_bound_permille']:.2f}")
    if mutations is not None:
        lines.append(f"- mutation check: {mutations.get('killed')} of {mutations.get('total')} killed")
    for problem in verdict["problems"]:
        lines.append(f"- problem: {problem}")
    if paired is not None:
        lines.append(f"- headline: the verdict rests on the comparator {verdict['binding_comparator']} (the one with the smaller lower bound); refuted against: {verdict['refuted_against'] or 'none'}")
    lines.append("")
    lines.append("Limitations: " + "; ".join(LIMITATIONS) + ".")
    return "\n".join(lines) + "\n"


def publish(results_dir: Path, metrics: dict, run: dict) -> None:
    """Write metrics.json and run.json as one aggregate bound by the metrics' SHA-256 (the repository's single publisher)."""
    import experiment_records

    experiment_records.publish_aggregate(results_dir, metrics, run)


if __name__ == "__main__":
    aggregate_run.main(sys.modules[__name__])
