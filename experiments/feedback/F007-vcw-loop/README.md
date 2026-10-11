# F007-vcw-loop

`planned`, not frozen, not run on a confirmatory seed. The whole verifier-closed workspace loop
(`VcwLoop` with finding router, learned router promoted through the off-policy gate, workspace, self-model,
L3 through the ledger) against a trained softmax gate and best-of-N at a matched verifier-call budget, on
**simulated** expert Pods with declared per-(Pod, finding code) skill tables, over arithmetic, JSON-schema and
code tasks judged by the real `ptr-verifier` verifiers. A mechanism check: a gain is a consequence of the
declared tables, and a homogeneous-skill world must tie.

* Design, margins and the freeze procedure: [`PREREGISTRATION.md`](PREREGISTRATION.md)
* Rules: [`criteria.toml`](criteria.toml); constants: [`config.toml`](config.toml)
* Harness: `ptr-bench vcw-loop <seed>` (`bins/ptr-bench/src/experiments/f007/`); `--preflight`
* Aggregator: [`aggregate.py`](aggregate.py) (tests: `scripts/tests/test_f007_aggregate.py`)
* Planning model and power: [`power.py`](power.py); pilots (seeds >= 1001, not evidence): [`pilots/`](pilots/README.md)
* Mutation plan: [`tests/mutations.toml`](tests/mutations.toml)

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
