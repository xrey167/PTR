# Tests for experiments/feedback/F007-vcw-loop

The harness is `ptr-bench vcw-loop` ([`bins/ptr-bench/src/experiments/f007/`](../../../../bins/ptr-bench/src/experiments/f007/mod.rs)).
Its unit tests (`cargo test -p ptr-bench f007`) cover the tasks against the real verifiers (a clean candidate passes,
each planted defect is reported as exactly that code, reported codes are a subset of planted ones), the worlds
(skill tables, registries, the shuffled map), the lanes (L3 Birth only through admission, withheld broadcast changes
nothing unless a Pod reads it, true map beats shuffled map), the audit (each integrity counter fires on a doctored
record), training (pilot seeds only, equal tuning budgets), determinism, the rule lists pinned to `criteria.toml`
(every rule resolves and can fail), and the compiled constants against `config.toml`.
`scripts/tests/test_f007_aggregate.py` holds the aggregator's verdicts.

[`mutations.toml`](mutations.toml) plants one defect at a time (product and harness) and requires the named
counter to fire; `mutate_aggregator.py` does the same for the aggregator's statistical rules:

```sh
python3 scripts/mutation_check.py F007          # writes results/mutations.json
python3 scripts/mutation_check.py F007 --check  # anchors only
python3 scripts/mutation_check.py F007 --only <name>
```
