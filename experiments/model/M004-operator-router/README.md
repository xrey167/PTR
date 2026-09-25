# M004-operator-router

## Hypothesis
Does learned reasoning-method routing outperform LLM-only reasoning?

## Primary metrics
regret; quality-cost utility; route accuracy

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## A0-internal ablation (research/falsification/A0-ablations-v1)

The `a0_*` entrypoints in `experiment.toml` run the A0 mechanism ablation study on
synthetic operator-routing v1: frozen-router, a negative control: A0's router is its only output head and cannot be removed. It is an A0-internal ablation on synthetic
data, **not** this manifest's baseline comparison. That comparison needs the plain
model baseline, which is not pinned yet (owner decision O2), so `status` stays
`planned` and `metrics.json` stays reserved for it; the study writes
`results/a0_internal_metrics.json` instead. Design, criteria and results:
`research/falsification/A0-ablations-v1/`.

**Result, 2026-09-25 (A0-internal evidence only; [RESULTS.md](../../../research/falsification/A0-ablations-v1/RESULTS.md)).** The negative control
is INCONCLUSIVE: frozen-router against the full arm on test_iid gives mean +0.003,
interval [-0.014, +0.020], just outside the preregistered equivalence band of
±0.02. The learned router beats the designer's hand-weighted router (0.806 on
test_iid) by at least 5 points on every seed (lowest full seed 0.863). Neither
statement is evidence about routing beyond A0 on this synthetic task.

