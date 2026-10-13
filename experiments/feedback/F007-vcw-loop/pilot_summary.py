"""Summarise the pilot runs (seeds >= 1001) in pilots/ with the aggregator's own statistics.

Pilots are NOT evidence: they are runs of the harness on seeds that are never confirmatory, kept to show the
pipeline works end to end, to measure variances for the power analysis and to expose defects of the design
before the freeze. This applies `aggregate.analyse` (one-sided 95% t bounds over the pilot seeds; with three
seeds df = 2, so the intervals are wider than the confirmatory ones) and prints the integrity counters.

    python3 experiments/feedback/F007-vcw-loop/pilot_summary.py > experiments/feedback/F007-vcw-loop/pilots/summary.md
"""

import importlib.util
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("f007_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)
CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))


def main() -> None:
    results = {}
    for path in sorted((HERE / "pilots").glob("pilot-*.json")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            result = json.loads(text.splitlines()[-1])
            results[result["seed"]] = result
    print(f"# F007 pilots (seeds {sorted(results)}), not evidence\n")
    for seed, result in sorted(results.items()):
        counters = {name: result[name] for name in
                    [aggregate.parse_rule(r)[0] for r in CRITERIA["falsifiers"] + CRITERIA["invalid_if"]]}
        bad = {k: v for k, v in counters.items() if v}
        print(f"- seed {seed}: scale {result['scale']}, hard_failures {result['hard_failures']}, "
              f"counters not zero: {bad or 'none'}, coverage {result['coverage']}, "
              f"elapsed {result['procedure']['elapsed_ns'] / 1e9:.0f} s")
    if len(results) >= 2:
        analysis = aggregate.analyse(results, CRITERIA["decision"])
        print()
        verdict = aggregate.decide(
            [aggregate.judge_seed(r, 0, CRITERIA, s) | {"valid": True, "invalid": []} for s, r in results.items()],
            analysis, sorted(results), CRITERIA["decision"], {"killed": 1, "total": 1})
        print(aggregate.report(verdict, [{"seed": s, "valid": True, "invalid": [], "coverage_failed": [],
                                          "falsifiers_failed": [], "controls_failed": []} for s in results],
                               analysis, None))
        print("\n(The verdict line above applies the confirmatory rule to pilot data for illustration only.)")


if __name__ == "__main__":
    main()
