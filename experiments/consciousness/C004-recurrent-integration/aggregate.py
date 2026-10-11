"""Aggregate the C004 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run C004 --seed <s>` writes (results/run-<timestamp>-seed-<s>.json). Nothing of
the experiment is run again. Every declared seed must have a record, the records must be of one configuration and have run the
checkout's code, and the preregistration digests of the manifest, the table, the text and `criteria.toml` must be the frozen ones.

The decision is mechanical (`criteria.toml`), three outcomes plus invalid, INVALID taking precedence:

- a seed is judged by applying every coverage rule and every STRUCTURAL falsifier rule to the one JSON object its harness printed; a
  failure is a defect of the recurrent arm, the feed-forward arm or the isolated reset-state ablation: FALSIFIED (kind structural);
- INTEGRITY counters (parameter or FLOP matching, sequence-length leak, overlap of training, validation and evaluation data, labels,
  loss and checkpoint bookkeeping): the harness exits 2 and the experiment is INVALID;
- guards: the feed-forward comparator saturated on a seed; more than the allowed number of seeds on which the recurrent arm, or the
  feed-forward comparator, did not learn the training distribution (a strawman comparator is not a comparison) or a judged arm hit the
  step cap; a sensitivity control or a null world that the pipeline does not resolve: INVALID; the extrapolation-removed control is read
  only for a SUPPORTED contrast (a contrast below the margin is a refutation whatever the control says);
- the paired comparisons (recurrent minus feed-forward, recurrent minus reset-state, accuracy at the primary length, one-sided 95% t
  bounds): COMPLETED iff both lower bounds reach the margin; FALSIFIED (refuted) iff the guards hold and the Bonferroni upper bound
  against some comparator is below the margin (the advantage is below the margin, not absent); NOT_SHOWN otherwise;
- INVALID also when a seed cannot be judged, a declared seed is missing, or the mutation evidence is missing, incomplete, not the plan's
  or older than the code.

Exit status: 0 COMPLETED, 1 FALSIFIED, 2 INVALID or a refusal to aggregate, 3 NOT_SHOWN. `--report PATH` also writes the Markdown
reading. The harness is shown able to fail by `results/mutations.json` (`research/cseries/lib/run_mutations.py C004`)."""

from __future__ import annotations

import sys
import tomllib
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/cseries/lib"))
import agg  # noqa: E402
import aggregate_run  # noqa: E402
from agg import COMPLETED, FALSIFIED, INVALID, NOT_SHOWN, PairedError  # noqa: E402

BENCHMARK = "c004"
EXPERIMENT = "C004"
CONTROLS = ("length_leak", "inlength_probe", "overlap", "param_mismatch", "flop_mismatch", "label_check", "isolation_probe", "causal_probe")
COUNT_KEYS = ("recurrent_trained", "feedforward_trained", "unconverged_judged_arms")
GUARDED_PATHS = ("research/cseries", "research/consciousness/uv.lock", "research/consciousness/pyproject.toml", "experiments/consciousness/C004-recurrent-integration")
RESULTS_EXCLUDE = ":(exclude)experiments/consciousness/C004-recurrent-integration/results"
METRIC_NAMES = (
    "arms", "comparator_primary_permille", "recurrent_trained", "feedforward_trained", "unconverged_judged_arms", "environment",
    "feedforward_shape", "recurrent_carry_hits", "recurrent_carry_flip_permille", "feedforward_carry_flip_permille", "label_flip_permille",
    "reset_carry_leaks", "ff_causal_leaks", "nondeterminism", "param_mismatch_arms", "flop_mismatch_arms", "length_leaks", "inlength_control_errors",
    "train_eval_overlap", "val_eval_overlap", "nonfinite_arms", "checkpoint_mismatch_arms", "convergence_mismatch_arms", "label_mismatches",
    "reset_above_chance", "perm_label_leaks", "corruption_rate_errors", "hard_failures", "guards_failed",
)
SCOPE = (
    "a functional test, on a CPU toy task (a modular counter: the running sum modulo 6 of a sequence of symbols, a state that must be "
    "carried over every step), of algorithmic recurrence with a carried state (indicator RPT-1; RPT-2 is not targeted): a gated recurrent "
    "net against a parameter- and FLOP-matched causal feed-forward stack with unshared layers and against the same recurrent net with the "
    "carried state reset at every step, trained on 2 to 6 steps and judged by accuracy at 12 steps (twice the longest trained sequence), "
    "by one-sided 95% t bounds of the paired difference over the declared seeds. It makes no claim about consciousness and none about "
    "language models"
)
LIMITATIONS = [
    "the task is a modular counter (a state-tracking problem in the cyclic group of order six), chosen because the answer needs a state carried over every step and because a small attention stack can learn it in distribution; a harder group (S3) was tried first and a feed-forward stack of the matched size did not learn it in the CPU budget, so it could not serve as a comparator; a result applies to this task family, to training on 2 to 6 steps and to evaluation at 12 steps (secondary 18 and 24), and says nothing about whether a carried state helps on language or on the experts of a Mixture of Experts (that is the GPU stage B)",
    "the feed-forward comparator is a small causal transformer (width fixed by parameter matching, standard ALiBi relative positions so that it accepts longer sequences, five layers at positional slope scale 4, pilot-selected on one pilot seed: the shallowest depth at which the stack trained on the long length reaches 0.80, which is tuning toward the extrapolation-removed control; its learning rate came from a grid of three values against two for the recurrent net, with the same warmup, optimiser and plateau rule); the stack remains optimisation-limited, 910 to 936 per mille in distribution (against 1000 for the recurrent arm) and 772 to 911 at 12 steps when trained on that length; a deeper or wider stack, a different positional scheme or a curriculum might extrapolate further; parameter and FLOP counts are matched at 6 steps, and at 12 and more steps the feed-forward stack costs more FLOPs than the recurrent net, which is reported",
    "the saturation ceiling (980 per mille) and the share of one half in the extrapolation-removed control are not derived from the planning model; the share was loosened after exploration on pilot seed 1002 (the first criterion, that the in-length stack come within the margin of the recurrent arm, could not be met by a stack that is optimisation-limited, an empirical finding), the one-half threshold is a verbal a-priori argument, and the control is read only when the contrast is supported",
    "the comparator class is a stack of attention layers of a fixed depth: whether it extrapolates is a known property of the class, and a COMPLETED outcome is close to a known answer (a fixed-depth stack does not count beyond the depth it was trained for); the experiment is a validated harness with a control that can fail (the stack trained on the long length), not a discovery",
    "the reset-state ablation sees one step and so is a floor by construction on this task; beating it shows that the carried state is what the recurrent arm uses, not that recurrence beats a strong alternative (that is the feed-forward comparison)",
    "the t bounds assume roughly normal paired differences, which the declared seeds cannot check; per-seed differences, medians and both bounds are reported",
    "the window null world (the sum of the last two steps) is easy and saturates; it guards the pipeline against an artefact that favours one family uniformly, not against a training asymmetry on hard tasks",
    "the label-permutation control and the corrupted-input control are sensitivity controls of the pipeline: they show that a collapse to chance and a deficit of about the margin's size are resolved, not that the main effect has the size of either",
    "an indicator property here is a functional property: the experiment says nothing about whether any system is conscious",
]

CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


def judge(result, exit_code, criteria, seed):
    entry = agg.judge_seed(result, exit_code, criteria, seed, BENCHMARK, EXPERIMENT, CONTROLS, COUNT_KEYS)
    if isinstance(result, dict):
        entry["recurrent_trained"] = result.get("recurrent_trained") if result.get("recurrent_trained") in (0, 1) else None
        entry["feedforward_trained"] = result.get("feedforward_trained") if result.get("feedforward_trained") in (0, 1) else None
        entry["unconverged_judged_arms"] = result.get("unconverged_judged_arms")
    else:
        entry.update({"recurrent_trained": None, "feedforward_trained": None, "unconverged_judged_arms": None})
    return entry


def extrapolation_share_bound(results: dict, seeds: list, spec: dict, t_support: str) -> dict:
    """The one-sided lower bound of the stack trained on the primary length, minus the stack, minus `share` of the recurrent arm's advantage over the stack, per seed
    (positive: the stack recovers more than `share` of the contrast when length extrapolation is removed). With share one half: more than half of the contrast is due to
    length extrapolation, not to capacity."""
    from decimal import Decimal
    from fractions import Fraction

    share = Fraction(int(spec["share_permille"]), 1000)
    recovered = agg.differences(results, seeds, spec["metric"], spec["trained_on_length"], spec["stack"])
    advantage = agg.differences(results, seeds, spec["metric"], spec["arm"], spec["stack"])
    diffs = [r - share * a for r, a in zip(recovered, advantage)]
    return {"arms": f"({spec['trained_on_length']} - {spec['stack']}) - {spec['share_permille']}/1000 ({spec['arm']} - {spec['stack']})", "metric": spec["metric"],
            "per_seed_permille": [float(d * 1000) for d in diffs], "median_permille": float(agg.median(diffs) * 1000), "positive_seeds": sum(1 for d in diffs if d > 0),
            "recovered_mean_permille": float(sum(recovered, Fraction(0)) / len(recovered) * 1000), "advantage_mean_permille": float(sum(advantage, Fraction(0)) / len(advantage) * 1000),
            **agg.paired_bound(diffs, Decimal(t_support))}


def paired_comparisons(results: dict, paired: dict) -> dict:
    """The pre-specified comparisons over the declared seeds (recurrent minus comparator, accuracy at the primary length):

    - judged: both lower one-sided 95% bounds (`t_support`) at or above the margin supports; the Bonferroni upper bound (`t_refute`,
      alpha / 2) below the margin against either comparator refutes (the advantage is below the margin, not absent);
    - sensitivity controls: gross (recurrent minus the same net trained on permuted labels must reach the margin), graded (recurrent
      minus the same net on corrupted inputs, a known deficit of the margin's order, must exceed `min_lower_permille`);
    - extrapolation-removed control: the same stack trained with the primary length in its training set must recover more than `share_permille`/1000 of the
      recurrent arm's advantage over the stack (lower one-sided bound of the paired difference at least zero), else the main contrast is one of capacity;
    - null worlds: with permuted labels, and on the memoryless task, the recurrent-minus-feed-forward difference must show no advantage
      of the margin's size in either direction;
    - descriptive comparisons carry the same bounds and are never judged. Raises PairedError."""
    seeds = sorted(results)
    if len(seeds) != int(paired["seeds_required"]):
        raise PairedError(f"{len(seeds)} seeds, the rule is preregistered for {paired['seeds_required']}")
    t_support, t_refute, margin, metric = paired["t_support"], paired["t_refute"], paired["margin_permille"], paired["metric"]
    judged = []
    for other in paired["against"]:
        bound = agg.bound_of(results, seeds, metric, paired["arm"], other, t_support)
        adjusted = agg.bound_of(results, seeds, metric, paired["arm"], other, t_refute)
        judged.append({"against": other, "supports": bound["lower_bound_permille"] >= margin, "refutes": adjusted["upper_bound_permille"] < margin,
                       "upper_bound_adjusted_permille": adjusted["upper_bound_permille"], "margin_permille": margin, **bound})
    gross, graded = paired["sensitivity_gross"], paired["sensitivity_graded"]
    gb = agg.bound_of(results, seeds, metric, gross["arm"], gross["permuted"], t_support)
    rb = agg.bound_of(results, seeds, metric, graded["arm"], graded["corrupted"], t_support)
    nulls = {}
    for name in ("null_permuted", "null_window"):
        spec = paired[name]
        forward = agg.bound_of(results, seeds, spec["metric"], spec["arm"], spec["against"], t_support)
        backward = agg.bound_of(results, seeds, spec["metric"], spec["against"], spec["arm"], t_support)
        nulls[name] = {**spec, "forward": forward, "backward": backward,
                       "no_advantage": forward["lower_bound_permille"] < spec["margin_permille"] and backward["lower_bound_permille"] < spec["margin_permille"]}
    removed = paired["extrapolation_removed"]
    eb = extrapolation_share_bound(results, seeds, removed, t_support)
    descriptive = []
    for descriptive_metric in paired.get("descriptive_metrics", []):
        for other in paired["against"]:
            descriptive.append({"metric": descriptive_metric, **agg.bound_of(results, seeds, descriptive_metric, paired["arm"], other, t_support), "against": other})
    return {
        "supported": all(item["supports"] for item in judged),
        "refuted_against": [item["against"] for item in judged if item["refutes"]],
        "judged": judged,
        "sensitivity": {
            "gross": {**gross, "resolves": gb["lower_bound_permille"] >= gross["margin_permille"], **gb},
            "graded": {**graded, "resolves": rb["lower_bound_permille"] > graded["min_lower_permille"], **rb},
        },
        "extrapolation_removed": {**removed, **eb, "collapses": eb["lower_bound_permille"] >= 0},
        "nulls": nulls,
        "descriptive": descriptive,
    }


def decide(seeds, static, declared, mutations, paired, counts, allowed, plan_size) -> dict:
    """The verdict from the seeds' judgements (`judge`, in any order), the static-evidence results, the mutation evidence (`killed`,
    `total`), the paired comparisons and the counts across seeds (`counts`: untrained recurrent, untrained comparator, unconverged
    seeds; `allowed`: their allowances). Pure. INVALID first when a seed trips a harness-integrity counter."""
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
    kind = None
    reasons = []
    if broken:
        verdict = INVALID
        problems.append(f"harness integrity counters above 0 on seeds {broken}: a defect of the task or the harness, repaired in a successor")
    elif problems or unjudged:
        verdict = INVALID  # missing seeds, missing or partial mutation evidence, a seed that cannot be judged: INVALID before any other verdict
    elif unmet or tripped or static_failed:
        verdict, kind = FALSIFIED, "structural"
    elif uninformative:
        verdict = INVALID
        problems.append(f"the feed-forward comparator is saturated (at or above the ceiling) on seeds {uninformative}: the margin cannot be seen")
    elif counts["untrained_recurrent"] > allowed["untrained_recurrent"]:
        verdict = INVALID
        problems.append(f"the recurrent arm did not learn the training distribution on {counts['untrained_recurrent']} seeds (more than {allowed['untrained_recurrent']})")
    elif counts["untrained_comparator"] > allowed["untrained_comparator"]:
        verdict = INVALID
        problems.append(f"the feed-forward comparator did not learn the training distribution on {counts['untrained_comparator']} seeds (more than {allowed['untrained_comparator']}): an untrained comparator is a strawman")
    elif counts["unconverged"] > allowed["unconverged"]:
        verdict = INVALID
        problems.append(f"a judged arm hit the step cap on {counts['unconverged']} seeds (more than {allowed['unconverged']}): the comparison would be one of optimisation speed")
    elif paired is None:
        verdict = INVALID
        problems.append("the paired comparison could not be computed")
    else:
        if not paired["sensitivity"]["gross"]["resolves"]:
            reasons.append("the label-permutation control does not resolve (recurrent minus the same net trained on permuted labels is below the margin)")
        if not paired["sensitivity"]["graded"]["resolves"]:
            reasons.append("the corrupted-input control does not resolve (a deficit of about the margin's size is not seen)")
        for name, null in paired["nulls"].items():
            if not null["no_advantage"]:
                reasons.append(f"the null world {name} shows an advantage of the margin's size where none can exist: the pipeline invents effects")
        if reasons:
            verdict = INVALID
            problems.extend(reasons)
        elif paired["refuted_against"]:
            verdict, kind = FALSIFIED, "refuted"
        elif paired["supported"] and not paired["extrapolation_removed"]["collapses"]:
            # the extrapolation-removed control attributes a SUPPORTED contrast to extrapolation; it is not read for a contrast below the margin (there a
            # competent stack, which gains little from seeing the length, would fail it and the refutation path would close)
            verdict = INVALID
            problems.append("the stack trained on the primary length does not recover more than the preregistered share of the contrast (its lower bound is below zero): "
                            "the supported contrast would be one of capacity and not of extrapolation")
        elif paired["supported"]:
            verdict = COMPLETED
        else:
            verdict = NOT_SHOWN
    return {
        "verdict": verdict, "kind": kind, "vacuous_seeds": unmet, "falsified_seeds": tripped, "invalid_seeds": unjudged,
        "integrity_seeds": broken, "uninformative_seeds": uninformative, "static_evidence_failed": static_failed,
        "untrained_recurrent_seeds": counts["untrained_recurrent"], "untrained_comparator_seeds": counts["untrained_comparator"],
        "unconverged_seeds": counts["unconverged"], "refuted_against": (paired or {}).get("refuted_against", []),
        "problems": problems, "recommended_status": {COMPLETED: "completed", FALSIFIED: "failed"}.get(verdict),
    }


def counts_of(judgements: list) -> dict:
    return {
        "untrained_recurrent": sum(1 for e in judgements if e.get("recurrent_trained") == 0),
        "untrained_comparator": sum(1 for e in judgements if e.get("feedforward_trained") == 0),
        "unconverged": sum(1 for e in judgements if e.get("unconverged_judged_arms")),
    }


def verdict_of(judgements, static, declared, mutations, paired, criteria_paired, plan_size) -> dict:
    allowed = {
        "untrained_recurrent": int(criteria_paired["max_untrained_recurrent_seeds"]),
        "untrained_comparator": int(criteria_paired["max_untrained_comparator_seeds"]),
        "unconverged": int(criteria_paired["max_unconverged_seeds"]),
    }
    return decide(judgements, static, declared, mutations, paired, counts_of(judgements), allowed, plan_size)


def report(verdict, seeds, static, mutations=None, paired=None) -> str:
    lines = [f"# C004 recurrent integration: {verdict['verdict']}", ""]
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
        lines.append("Paired comparisons (recurrent minus comparator, per mille of accuracy at the primary length; one-sided 95% t bounds against the margin):")
        for item in paired["judged"]:
            state = "supports" if item["supports"] else ("REFUTES" if item["refutes"] else "straddles")
            lines.append(f"- vs {item['against']}: mean {item['mean_permille']:.1f}, sd {item['sd_permille']:.1f}, median {item['median_permille']:.1f}, "
                         f"{item['positive_seeds']} of {item['n']} seeds positive, lower {item['lower_bound_permille']:.1f}, upper {item['upper_bound_permille']:.1f} "
                         f"(adjusted upper {item['upper_bound_adjusted_permille']:.1f}), margin {item['margin_permille']}: {state}")
        for kind in ("gross", "graded"):
            sens = paired["sensitivity"][kind]
            other = sens.get("permuted", sens.get("corrupted"))
            lines.append(f"- {kind} sensitivity control ({sens['arm']} minus {other}): mean {sens['mean_permille']:.1f}, lower {sens['lower_bound_permille']:.1f}: "
                         f"{'resolves' if sens['resolves'] else 'DOES NOT RESOLVE'}")
        removed = paired["extrapolation_removed"]
        lines.append(f"- extrapolation-removed control ({removed['trained_on_length']} trained with the primary length recovers {removed['recovered_mean_permille']:.1f} of the {removed['advantage_mean_permille']:.1f} "
                     f"advantage of {removed['arm']} over {removed['stack']}; excess over the share {removed['share_permille']}/1000: mean {removed['mean_permille']:.1f}, lower {removed['lower_bound_permille']:.1f}): "
                     f"{'not read: the contrast is not supported' if not paired['supported'] else 'the contrast is due to extrapolation' if removed['collapses'] else 'MORE THAN HALF OF THE CONTRAST IS NOT RECOVERED (capacity, not extrapolation)'}")
        for name, null in paired["nulls"].items():
            lines.append(f"- null world {name} ({null['arm']} minus {null['against']}): lower {null['forward']['lower_bound_permille']:.1f}, "
                         f"reverse lower {null['backward']['lower_bound_permille']:.1f}: {'no advantage' if null['no_advantage'] else 'ADVANTAGE WHERE NONE CAN EXIST'}")
        for item in paired["descriptive"]:
            lines.append(f"- (descriptive) {item['metric']} vs {item['against']}: mean {item['mean_permille']:.1f}, lower {item['lower_bound_permille']:.1f}, upper {item['upper_bound_permille']:.1f}")
    if mutations is not None:
        lines.append(f"- mutation check: {mutations.get('killed')} of {mutations.get('total')} killed")
    for problem in verdict["problems"]:
        lines.append(f"- problem: {problem}")
    lines.append("")
    lines.append("Limitations: " + "; ".join(LIMITATIONS) + ".")
    return "\n".join(lines) + "\n"


def publish(results_dir: Path, metrics: dict, run: dict) -> None:
    """Write metrics.json and run.json as one aggregate bound by the metrics' SHA-256 (the repository's single publisher)."""
    import experiment_records

    experiment_records.publish_aggregate(results_dir, metrics, run)


if __name__ == "__main__":
    aggregate_run.main(sys.modules[__name__])
