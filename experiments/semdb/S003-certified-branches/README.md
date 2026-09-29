# S003-certified-branches

## Hypothesis
Do dependency-certified agent branches avoid lost updates and phantoms while merging more concurrent work than serial execution?

## Primary metrics
lost updates; undetected phantoms; certified merge rate; conflict rate; merge latency

## Baseline
serial execution; last-writer-wins branches; key-level optimistic locking without range digests

## Falsification
Any lost update or undetected phantom is a hard failure; no throughput gain over serial execution at conflict rates below 10% rejects the default.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

`experiments/preregistration.toml` lists every parameter S003 must preregister in the `[preregistration]` table of `config.toml` before it leaves `planned`; `scripts/check_research_gates.py` fails CI until each is pinned and `experiment.toml` names the table's digest.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
