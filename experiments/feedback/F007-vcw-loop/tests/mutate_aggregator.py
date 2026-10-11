"""Mutation check of the aggregator's statistical rules: plant one defect at a time in a copy of
`aggregate.py` and require `scripts/tests/test_f007_aggregate.py` to fail on it.

The harness's counters are mutation-checked by `scripts/mutation_check.py F007`; the rules that live only in
the aggregator (the margin against each baseline, the verdict order, the equivalence interval, the secondary
hypotheses, the agreement with the planning model, ...) are not harness counters, so they are checked here.

    python3 experiments/feedback/F007-vcw-loop/tests/mutate_aggregator.py --check   # anchors only
    python3 experiments/feedback/F007-vcw-loop/tests/mutate_aggregator.py           # writes aggregator-mutations.json
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[2]
SOURCE = (EXPERIMENT / "aggregate.py").read_text(encoding="utf-8")

MUTATIONS = [
    ("verdict-bands-swap-bounds", 'entry["verdict"] = "above" if entry["lcb"] > margin else "below" if entry["ucb"] < margin else "straddles"',
     'entry["verdict"] = "above" if entry["ucb"] > margin else "below" if entry["lcb"] < margin else "straddles"'),
    ("completed-needs-only-one-baseline", 'elif all(entry["verdict"] == "above" for entry in baselines.values()):',
     'elif any(entry["verdict"] == "above" for entry in baselines.values()):'),
    ("falsified-by-the-margin-removed", 'elif any(entry["verdict"] == "below" for entry in baselines.values()):',
     'elif False and any(entry["verdict"] == "below" for entry in baselines.values()):'),
    ("invalid-no-longer-first", "    if problems or analysis is None:\n        verdict = INVALID",
     "    if False:\n        verdict = INVALID"),
    ("interval-uses-the-wrong-variance", "for d in diffs) / (k - 1))", "for d in diffs) / k)"),
    ("t-quantile-is-wrong", "4: 2.131847", "4: 1.0"),
    ("equivalence-ignores-the-upper-bound", 'entry["passes"] = entry["lcb"] >= -band and entry["ucb"] <= band',
     'entry["passes"] = entry["lcb"] >= -band'),
    ("shuffled-control-is-inverted", 'shuffled["passes"] = shuffled["lcb"] >= decision["shuffled_min_loss_permille"] / 1000',
     'shuffled["passes"] = shuffled["lcb"] <= decision["shuffled_min_loss_permille"] / 1000'),
    ("best-of-n-control-is-dropped", 'single["passes"] = single["lcb"] >= decision["bon_min_gain_permille"] / 1000',
     'single["passes"] = True'),
    ("oracle-control-is-dropped", '"passes": oracle["lcb"] >= -decision["oracle_tolerance_permille"] / 1000}',
     '"passes": True}'),
    ("model-agreement-is-ignored", '"passes": all(v["passes"] for v in agreement.values())}', '"passes": True}'),
    ("ceiling-and-floor-are-ignored", '"passes": all(model["floor_permille"] / 1000 <= v <= model["ceiling_permille"] / 1000 for v in saturated.values()),',
     '"passes": True,'),
    ("a-missing-key-crashes-instead-of-invalid", "except (KeyError, TypeError, ZeroDivisionError, IndexError) as error:",
     "except ZeroDivisionError as error:"),
    ("workspace-secondary-reads-the-primary-world", 'entry = interval(seed_differences(results, spec["world"], "full", "no_workspace"))',
     'entry = interval(seed_differences(results, primary, "full", "no_workspace"))'),
    ("l3-secondary-ignores-the-population", 'pops = [population_difference(results[s], spec["world"], "l3_population") for s in sorted(results)]',
     'pops = [(pair_difference(results[s], spec["world"], "full", "no_l3")[0], 1) for s in sorted(results)]'),
    ("untested-contrasts-are-labelled-no-effect", "    if arm in UNTESTED_IN_PRIMARY:\n        return NON_DISCRIMINATING", "    if False:\n        return NON_DISCRIMINATING"),
    ("secondary-verdict-ignores-the-upper-bound", "    if entry[\"ucb\"] < margin:\n        return \"REFUTED\"", "    if False:\n        return \"REFUTED\""),
    ("frozen-confidence-is-unchecked", '        "confidence_one_sided_permille": 950,\n', ""),
    ("frozen-margin-is-unchecked", '        "primary_margin_permille": decision["margin_permille"],\n', ""),
    ("source-digests-are-not-compared", 'controls["one_harness_for_every_seed"] = {"digests": digests, "passes": len(digests) == 1}',
     'controls["one_harness_for_every_seed"] = {"digests": digests, "passes": True}'),
    ("source-digest-is-not-required", '    elif not re.fullmatch(r"[0-9a-f]{64}", str(procedure.get("source_digest"))):',
     '    elif False:'),
    ("first-draft-differences-are-ignored", '        "passes": differing == 0,',
     '        "passes": True,'),
    ("a-tripped-falsifier-no-longer-falsifies", "    elif tripped:\n        verdict = FALSIFIED", "    elif False:\n        verdict = FALSIFIED"),
]


def anchors() -> list[str]:
    return [f"{name}: the anchor occurs {SOURCE.count(find)} times" for name, find, _ in MUTATIONS if SOURCE.count(find) != 1]


def main() -> int:
    problems = anchors()
    if problems:
        print("\n".join("ERROR: " + p for p in problems))
        return 1
    if "--check" in sys.argv:
        print(f"OK: {len(MUTATIONS)} aggregator mutation anchors")
        return 0
    outcomes = []
    for name, find, replace in MUTATIONS:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.py"
            path.write_text(SOURCE.replace(find, replace, 1), encoding="utf-8")
            environment = dict(os.environ, F007_AGGREGATE_PATH=str(path), PYTHONDONTWRITEBYTECODE="1")
            ran = subprocess.run(
                [sys.executable, "-m", "unittest", "scripts.tests.test_f007_aggregate"],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
        result = "killed" if ran.returncode != 0 else "survived"
        outcomes.append({"name": name, "result": result})
        print(f"{name}: {result}", flush=True)
    killed = sum(o["result"] == "killed" for o in outcomes)
    (HERE / "aggregator-mutations.json").write_text(
        json.dumps({"killed": killed, "total": len(outcomes), "mutations": outcomes}, indent=1) + "\n", encoding="utf-8")
    print(f"{killed} of {len(outcomes)} killed")
    return 0 if killed == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
