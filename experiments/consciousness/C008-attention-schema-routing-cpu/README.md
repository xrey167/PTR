# C008-attention-schema-routing-cpu

The CPU stage A of C003, registered as its own experiment (the scaffolder accepts only a capital letter and three digits). **Status: planned, built to freeze-ready (version 2, after the pre-freeze review); no declared seed has been run.** A construction-validity check of a world built so that a schema can help; it leaves AST-L2 unchanged whatever its outcome. Indicator AST-1 as a computation
([doc 38](../../../docs/architecture/38-consciousness-indicators.md)); indicator properties are not consciousness and nothing here claims that any system is conscious. The GPU stage B (OLMoE L1 routing) is
`C003-attention-schema-routing`, which remains planned.

## What is tested, and what is not

A toy router sends tokens of eight classes to eight experts under a bursty, cyclic demand (eight regimes of three batches; every class is the "first" hot class in one regime and the "second" in the next, so its best split over its two competent
experts flips); an expert serves its first 64 tokens per batch and drops the rest; a deterministic verifier rewards a token routed to a competent expert with room. An **attention schema** reads the router's own recent propensity and predicts its next
propensity; **control** adds a learned law of that prediction to the router's logits. Does the controlled router verify more tokens, by the margin, than (a) the same router without a schema (FLOP-matched), (b) the same, parameter-matched,
(c) an auxiliary-loss-free heuristic balancer, (d) a **describe-only** schema trained identically but not used for control, and (e) a **router that reads the same history directly with no schema** (parameter-matched)? The schema has no authority over expert capacity or competence (a structural counter); a describe-only schema must
not reach the router (a structural counter). Without learning, a router that knew the regime would gain about 60 per mille over a regime-blind one with independent token sampling (the margin, 25, asks for about 40 per cent of it); the earlier
designs on pilot seeds and why they changed are in the preregistration (section 6).

## Controls

A uniform router (gross) and an epsilon-routed schema-free router (graded) must be resolved as deficits; a stationary null world (no regime changes) must show no difference between the controlled and the schema-free router in either direction; the
comparators must leave headroom below the oracle (a maximum flow) and beat a uniform router; the schema must beat persistence ("the next batch is like the last"); the controlled router reading a shuffled history is the reported attribution (does the gain depend on
the router's own recent history) is now **judged**: COMPLETED needs it attributed. In the stationary world the schema may cost something, not more than 50 per mille. A hand-set regime-blind reference is reported. Any failing guard or control makes the experiment INVALID.

## Decision (three outcomes plus invalid)

| Outcome | Condition |
|---|---|
| COMPLETED (supported) | guards and controls hold; all five lower one-sided 95% bounds over the 20 seeds of verified-token rate (control minus schema-free, minus parameter-matched schema-free, minus heuristic, minus describe-only, minus the history router) >= 25 per mille and the shuffled-history control attributed; a construction check, AST-L2 unchanged |
| FALSIFIED (refuted) | guards and controls hold and, for some comparator, the Bonferroni upper bound < 25 per mille (against describe-only alone: the control role is not shown; against the history router: access to the history, not a separate predictor with a control law, accounts for the gain) |
| NOT_SHOWN | bounds straddle the margin, or the gain is not attributed to the router's own history |
| INVALID | an integrity counter, seeds or mutation evidence missing or partial (these precede every other verdict), no headroom, untrained comparators or schema, a judged arm at the step cap, an unresolved control, a null world that shows an effect or a cost beyond the band |

A structural defect on any seed (a describe-only schema that reaches the router, a schema that changes what the verifier reads, nondeterminism, a coverage rule) is FALSIFIED, kind structural. Exit status of `aggregate.py`: 0 supported, 1 falsified, 3 not shown, 2 invalid or refusal.

## Expected outcome (pilots, not a result)

On pilot seeds the control arm verifies about 60 per mille more tokens than the schema-free router and the describe-only schema (the margin is 25), a shuffled history removes the gain, **but a router that reads the same history directly is within about 6 per mille of the control arm**:
the registered outcome is expected to be FALSIFIED (refuted) against the history router (about 0.85). The world was designed, after earlier designs failed, to contain a reachable gain; see the preregistration, sections 2, 4, 6 and 7.

## Files

| File | Content |
|---|---|
| `PREREGISTRATION.md` | question, design, decision rule, pilot disclosure and tuning, limits, entry point and the freeze procedure (commands in the repository: `research/cseries/lib/pin_freeze.py`) |
| `config.toml` | `[pinned_thresholds]` read by the harness; `[preregistration]` with two digests pinned at the freeze |
| `criteria.toml` | coverage, guards, falsifiers and the paired rule |
| `aggregate.py` | judges the seed records, writes `results/metrics.json` and `run.json`; refuses on stale mutation evidence |
| `tests/mutations.toml` | planted defects of the harness and the counter, control or unit test each must trip |
| `tests/aggregator_mutations.toml` | planted defects of the aggregator and of `criteria.toml`, run with the shared library's by `research/cseries/lib/run_agg_mutations.py C008` |
| `../../../research/cseries/` | harness (`c003a/harness.py`), world, models, planning model, reachable-gain script, pilot scripts and raw outputs (`c003a/pilots/`), shared library, mutation runner, tests |
| `../../../scripts/tests/test_c008_aggregate.py` | tests of the aggregator and of the statistics |

## Running (after the freeze)

```bash
UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project research/consciousness   # once; the environment is C001's, outside the repository
python3 scripts/run_experiment.py prepare C008
python3 scripts/run_experiment.py run C008 --seed 7001       # then 7002 ... 7020; each seed runs once
OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_mutations.py C008   # clean committed tree; writes results/mutations.json
python3 experiments/consciousness/C008-attention-schema-routing-cpu/aggregate.py
```

Tooling gaps (mutation runner for Python entrypoints; the Python harness is not in `experiment_records.CODE_PATHS`; the stripped runner environment) are the same as C004's and are handled the same way (see its README).
