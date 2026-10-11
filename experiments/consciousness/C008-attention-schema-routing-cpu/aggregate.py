"""Aggregate the C008 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run C008 --seed <s>` writes (results/run-<timestamp>-seed-<s>.json). Nothing of the
experiment is run again. Every declared seed must have a record, the records must be of one configuration and have run the checkout's code,
and the preregistration digests of the manifest, the table, the text and `criteria.toml` must be the frozen ones.

The decision is mechanical (`criteria.toml`), three outcomes plus invalid, INVALID taking precedence:

- a seed is judged by applying every coverage rule and every STRUCTURAL falsifier rule to the one JSON object its harness printed; a failure
  is a defect of an arm (the describe-only schema reaches the router, the schema reaches the environment, nondeterminism): FALSIFIED (kind
  structural);
- INTEGRITY counters (matching, streams, the verifier, the demand schedule, loss and checkpoint bookkeeping): the harness exits 2 and the
  experiment is INVALID;
- guards: no headroom between the best comparator and the oracle, comparators that did not train, more than the allowed number of seeds on
  which the schema did not learn to predict the router's own load or a judged arm hit the step cap, a sensitivity control or the null world
  that the pipeline does not resolve: INVALID;
- the paired comparisons (the schema-controlled router minus each of the schema-free router, the parameter-matched schema-free router, the
  heuristic balancer, the describe-only schema and the history-fed router, verified-token rate on the evaluation stream, one-sided 95% t bounds): COMPLETED iff all
  five lower bounds reach the margin and the shuffled-history attribution is ATTRIBUTED (a supported contrast without it is NOT_SHOWN); FALSIFIED (refuted) iff the guards and controls hold and the Bonferroni upper bound against some
  comparator is below the margin (the advantage is below the margin, not absent); NOT_SHOWN otherwise;
- INVALID also when a seed cannot be judged, a declared seed is missing, or the mutation evidence is missing, incomplete, not the plan's or
  older than the code.

Exit status: 0 COMPLETED, 1 FALSIFIED, 2 INVALID or a refusal to aggregate, 3 NOT_SHOWN. `--report PATH` also writes the Markdown reading.
The harness is shown able to fail by `results/mutations.json` (`research/cseries/lib/run_mutations.py C008`)."""

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

BENCHMARK = "c008"
EXPERIMENT = "C008"
CONTROLS = ("stream_overlap", "param_mismatch", "flop_mismatch", "reward_audit", "authority_probe", "influence_probe", "describe_probe", "history_probe")
COUNT_KEYS = ("schema_trained", "unconverged_judged_arms")
GUARDED_PATHS = ("research/cseries", "research/consciousness/uv.lock", "research/consciousness/pyproject.toml", "experiments/consciousness/C008-attention-schema-routing-cpu")
RESULTS_EXCLUDE = ":(exclude)experiments/consciousness/C008-attention-schema-routing-cpu/results"
METRIC_NAMES = (
    "arms", "oracle_permille", "comparator_best_permille", "headroom_permille", "comparator_above_random_permille", "schema_top1_permille",
    "schema_gain_permille", "blind_reference_permille", "history_router_blind", "schema_trained", "unconverged_judged_arms", "environment", "control_influence_hits", "describe_only_influence", "authority_leaks",
    "nondeterminism", "param_mismatch_arms", "flop_mismatch_arms", "stream_overlap", "reward_audit_mismatches", "oracle_violations",
    "schedule_errors", "nonfinite_arms", "checkpoint_mismatch_arms", "convergence_mismatch_arms", "epsilon_rate_errors", "shuffle_errors",
    "hard_failures", "guards_failed",
)
ATTRIBUTED, NOT_ATTRIBUTED, UNRESOLVED = "attributed", "not attributed", "unresolved by noise"
SCOPE = (
    "a functional test, on a CPU toy router, of indicator AST-1: an attention schema (a learned predictor of the router's OWN next routing load, read "
    "from its own recent history) that is also used to control routing, against the same router without a schema (FLOP-matched and "
    "parameter-matched), an auxiliary-loss-free heuristic balancer, a describe-only schema (trained identically, not used for control) and a router fed the recent load history directly without a separate predictor (the history router). The world "
    "is bursty cyclic demand over eight experts of limited capacity with a deterministic verifier; the judged endpoint is the fraction of "
    "tokens verified on a fresh evaluation stream, by one-sided 95% t bounds of the paired difference over the declared seeds. The schema has "
    "no authority over expert capacity or competence (a structural counter). It makes no claim about consciousness and none about "
    "language models or OLMoE routing (the GPU stage B)"
)
LIMITATIONS = [
    "a result applies to this toy world: eight experts, a cyclic demand of eight regimes of three batches, 256 tokens per batch, capacity 64, a history of four batches and a router of one hidden layer; the schema can only help where the demand is predictable from the router's own load history, which this world builds in on purpose (a stationary world, the null, is run beside it); it says nothing about real routing traces",
    "every tuned configuration (learning rates, entropy coefficients, control-law forms, the balancer step, the step caps) was selected from single draws of the grids on pilot seeds 1001 to 1003, which predate the fix of the initial-weight seeding and are therefore unseeded draws; selection differences of a few per mille (control 901.3 against 899.7) are inside that noise",
    "the heuristic balancer is the schema-free router with an auxiliary-loss-free bias moved by the previous batch's load; its learning rate and step were tuned on pilot seeds on an equal grid; a better-tuned or differently shaped balancer (a predictor-free forecast, an integral term) might close the gap",
    "the schema-free router is matched to the control in FLOPs per batch (the schema's included) and a second one in parameters; neither has the schema's history input, which is the information the schema adds; the history router, which is given the history directly without a separate predictor, IS an arm of this experiment and the one against which the schema's added value is judged (a refutation against it is the expected result, see PREREGISTRATION.md)",
    "the schema is trained by its prediction loss alone, detached from the policy: a joint training, or a schema that sees more than the router's own load history, is another experiment",
    "recovery after an expert is revoked or overloaded, and routing churn between policy versions beyond the descriptive probe, are not judged at stage A",
    "the t bounds assume roughly normal paired differences, which the declared seeds cannot check; per-seed differences, medians and both bounds are reported",
    "the shuffled-history attribution reads whether the gain depends on the router's own recent history; it is judged: a supported contrast whose shuffled-history control is not ATTRIBUTED is NOT_SHOWN",
    "an indicator property here is a functional property: the experiment says nothing about whether any system is conscious",
]

CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


def judge(result, exit_code, criteria, seed):
    entry = agg.judge_seed(result, exit_code, criteria, seed, BENCHMARK, EXPERIMENT, CONTROLS, COUNT_KEYS)
    if isinstance(result, dict):
        entry["schema_trained"] = result.get("schema_trained") if result.get("schema_trained") in (0, 1) else None
        entry["unconverged_judged_arms"] = result.get("unconverged_judged_arms")
    else:
        entry.update({"schema_trained": None, "unconverged_judged_arms": None})
    return entry


def attribution_state(bound: dict, margin: int) -> str:
    if bound["lower_bound_permille"] >= margin:
        return ATTRIBUTED
    if bound["upper_bound_permille"] < margin:
        return NOT_ATTRIBUTED
    return UNRESOLVED


def paired_comparisons(results: dict, paired: dict) -> dict:
    """The pre-specified comparisons over the declared seeds (control minus comparator, verified-token rate):

    - judged: every lower one-sided 95% bound (`t_support`) at or above the margin supports; the Bonferroni upper bound (`t_refute`, alpha / 5)
      below the margin against any comparator refutes (the advantage is below the margin, not absent);
    - sensitivity controls: gross (schema-free minus a uniform router must reach the margin), graded (schema-free minus itself with a
      fraction of its tokens re-routed at random, a known deficit of the margin's order, must exceed `min_lower_permille`);
    - null world: with no regime changes the schema-controlled arm may show no advantage of the margin's size (lower bound below it) and may not cost more
      than the cost band (upper bound of the reverse contrast below it);
    - attribution (judged for COMPLETED): control minus control with a shuffled history, three-valued; COMPLETED needs ATTRIBUTED;
    - the hand-set regime-blind reference minus each arm (descriptive: how far below a policy that knows nothing of the regimes the trained arms sit).
      Raises PairedError."""
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
    gross, graded, null = paired["sensitivity_gross"], paired["sensitivity_graded"], paired["null_world"]
    gb = agg.bound_of(results, seeds, metric, gross["arm"], gross["floor"], t_support)
    rb = agg.bound_of(results, seeds, metric, graded["arm"], graded["degraded"], t_support)
    forward = agg.bound_of(results, seeds, null["metric"], null["arm"], null["against"], t_support)
    backward = agg.bound_of(results, seeds, null["metric"], null["against"], null["arm"], t_support)
    shuffled = agg.bound_of(results, seeds, metric, paired["arm"], paired["attribution_arm"], t_support)
    blind = [{"against": other, **agg.bound_of(results, seeds, metric, paired["blind_reference_arm"], other, t_support)} for other in [paired["arm"], *paired["against"]]]
    return {
        "supported": all(item["supports"] for item in judged),
        "refuted_against": [item["against"] for item in judged if item["refutes"]],
        "judged": judged,
        "sensitivity": {
            "gross": {**gross, "resolves": gb["lower_bound_permille"] >= gross["margin_permille"], **gb},
            "graded": {**graded, "resolves": rb["lower_bound_permille"] > graded["min_lower_permille"], **rb},
        },
        "null_world": {**null, "forward": forward, "backward": backward,
                       "no_advantage": forward["lower_bound_permille"] < null["margin_permille"],
                       "no_cost": backward["upper_bound_permille"] < null["cost_band_permille"]},
        "blind_reference": blind,
        "attribution": {"against": paired["attribution_arm"], "state": attribution_state(shuffled, margin), "margin_permille": margin,
                        "half_width_permille": (shuffled["upper_bound_permille"] - shuffled["lower_bound_permille"]) / 2, **shuffled},
    }


def decide(seeds, static, declared, mutations, paired, counts, allowed, plan_size) -> dict:
    """The verdict from the seeds' judgements (`judge`, in any order), the static evidence, the mutation evidence, the paired comparisons and
    the counts across seeds. Pure. INVALID first when a seed trips a harness-integrity counter."""
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
        problems.append(f"no headroom below the oracle, or untrained comparators, on seeds {uninformative}: the margin cannot be seen")
    elif counts["untrained_schema"] > allowed["untrained_schema"]:
        verdict = INVALID
        problems.append(f"the schema did not learn to predict the router's own load on {counts['untrained_schema']} seeds (more than {allowed['untrained_schema']}): a schema that did not learn cannot control")
    elif counts["unconverged"] > allowed["unconverged"]:
        verdict = INVALID
        problems.append(f"a judged arm hit the step cap on {counts['unconverged']} seeds (more than {allowed['unconverged']}): the comparison would be one of optimisation speed")
    elif paired is None:
        verdict = INVALID
        problems.append("the paired comparison could not be computed")
    else:
        if not paired["sensitivity"]["gross"]["resolves"]:
            reasons.append("the uniform-router control does not resolve (schema-free minus a uniform router is below the margin)")
        if not paired["sensitivity"]["graded"]["resolves"]:
            reasons.append("the epsilon-routing control does not resolve (a deficit of about the margin's size is not seen)")
        if not paired["null_world"]["no_advantage"]:
            reasons.append("the stationary null world shows an advantage of the margin's size where none can exist: the pipeline invents effects")
        if not paired["null_world"]["no_cost"]:
            reasons.append("in the stationary null world the schema-controlled arm is worse than the schema-free one by more than the cost band: not a tie")
        if reasons:
            verdict = INVALID
            problems.extend(reasons)
        elif paired["refuted_against"]:
            verdict, kind = FALSIFIED, "refuted"
        elif paired["supported"] and paired["attribution"]["state"] == ATTRIBUTED:
            verdict = COMPLETED
        else:
            verdict = NOT_SHOWN
            if paired["supported"]:
                problems.append(f"the control beats every comparator by the margin but the shuffled-history control is {paired['attribution']['state']}: the gain is not attributed to the router's own history")
    return {
        "verdict": verdict, "kind": kind, "vacuous_seeds": unmet, "falsified_seeds": tripped, "invalid_seeds": unjudged,
        "integrity_seeds": broken, "uninformative_seeds": uninformative, "static_evidence_failed": static_failed,
        "untrained_schema_seeds": counts["untrained_schema"], "unconverged_seeds": counts["unconverged"],
        "refuted_against": (paired or {}).get("refuted_against", []),
        "control_role_not_shown": bool(paired and not [i for i in paired["judged"] if i["against"] == "describe_only" and i["supports"]]),  # a refutation, a straddle or no support: not shown
        "attribution": (paired or {}).get("attribution", {}).get("state"),
        "problems": problems, "recommended_status": {COMPLETED: "completed", FALSIFIED: "failed"}.get(verdict),
    }


def counts_of(judgements: list) -> dict:
    return {
        "untrained_schema": sum(1 for e in judgements if e.get("schema_trained") == 0),
        "unconverged": sum(1 for e in judgements if e.get("unconverged_judged_arms")),
    }


def verdict_of(judgements, static, declared, mutations, paired, criteria_paired, plan_size) -> dict:
    allowed = {"untrained_schema": int(criteria_paired["max_untrained_schema_seeds"]), "unconverged": int(criteria_paired["max_unconverged_seeds"])}
    return decide(judgements, static, declared, mutations, paired, counts_of(judgements), allowed, plan_size)


def report(verdict, seeds, static, mutations=None, paired=None) -> str:
    lines = [f"# C008 attention-schema routing (CPU stage A): {verdict['verdict']}", ""]
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
        lines.append("Paired comparisons (control minus comparator, per mille of verified tokens; one-sided 95% t bounds against the margin):")
        for item in paired["judged"]:
            state = "supports" if item["supports"] else ("REFUTES" if item["refutes"] else "straddles")
            lines.append(f"- vs {item['against']}: mean {item['mean_permille']:.1f}, sd {item['sd_permille']:.1f}, median {item['median_permille']:.1f}, "
                         f"{item['positive_seeds']} of {item['n']} seeds positive, lower {item['lower_bound_permille']:.1f}, upper {item['upper_bound_permille']:.1f} "
                         f"(adjusted upper {item['upper_bound_adjusted_permille']:.1f}), margin {item['margin_permille']}: {state}")
        for kind in ("gross", "graded"):
            sens = paired["sensitivity"][kind]
            other = sens.get("floor", sens.get("degraded"))
            lines.append(f"- {kind} sensitivity control ({sens['arm']} minus {other}): mean {sens['mean_permille']:.1f}, lower {sens['lower_bound_permille']:.1f}: "
                         f"{'resolves' if sens['resolves'] else 'DOES NOT RESOLVE'}")
        null = paired["null_world"]
        lines.append(f"- null world ({null['arm']} minus {null['against']}): lower {null['forward']['lower_bound_permille']:.1f}, reverse lower "
                     f"{null['backward']['lower_bound_permille']:.1f}, reverse upper {null['backward']['upper_bound_permille']:.1f}: "
                     f"{'no advantage' if null['no_advantage'] else 'ADVANTAGE WHERE NONE CAN EXIST'}; {'no cost beyond the band' if null['no_cost'] else 'COST BEYOND THE BAND'} ({null['cost_band_permille']})")
        for item in paired["blind_reference"]:
            lines.append(f"- (descriptive) hand-set regime-blind reference minus {item['against']}: mean {item['mean_permille']:.1f}, sd {item['sd_permille']:.1f}")
        a = paired["attribution"]
        lines.append(f"- attribution (control minus {a['against']}, margin {a['margin_permille']}): mean {a['mean_permille']:.1f}, median {a['median_permille']:.1f}, "
                     f"half-width {a['half_width_permille']:.1f}, lower {a['lower_bound_permille']:.1f}, upper {a['upper_bound_permille']:.1f}: {a['state'].upper()}")
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
