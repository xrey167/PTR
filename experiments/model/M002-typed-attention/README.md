# M002-typed-attention

## Hypothesis
Do type/epistemic/validity biases improve constraint retention?

## Primary metrics
hard violations; calibration; latency

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## A0-internal ablation (research/falsification/A0-ablations-v1)

The `a0_*` entrypoints in `experiment.toml` run the A0 mechanism ablation study on
synthetic operator-routing v1: no-typed-attention (necessity given QK), the raw-blind manipulation check, and the blind-query pair (sufficiency). It is an A0-internal ablation on synthetic
data, **not** this manifest's baseline comparison. That comparison needs the plain
model baseline, which is not pinned yet (owner decision O2), so `status` stays
`planned` and `metrics.json` stays reserved for it; the study writes
`results/a0_internal_metrics.json` instead. Design, criteria and results:
`research/falsification/A0-ablations-v1/`.

**Result, 2026-09-25 (A0-internal evidence only; [RESULTS.md](../../../research/falsification/A0-ablations-v1/RESULTS.md)).** The raw-blind
manipulation check PASSES (+0.123 on test_iid, 5 of 5 seeds), so the raw path is
used. M002-necessity FALSIFIES: a benefit of 2 points or more from the rank-1
typed bias on composite B is excluded in A0 (mean -0.005, interval [-0.021,
+0.011]); see `FALSIFIED-M002-necessity.md`. M002-sufficiency was not run: the
preregistered budget rule dropped the blind-query pair.

