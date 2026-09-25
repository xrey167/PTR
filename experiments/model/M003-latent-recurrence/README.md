# M003-latent-recurrence

## Hypothesis
Can latent recurrent steps reduce explicit reasoning tokens without quality loss?

## Primary metrics
task success; tokens; wall time; latent steps

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## A0-internal ablation (research/falsification/A0-ablations-v1)

The `a0_*` entrypoints in `experiment.toml` run the A0 mechanism ablation study on
synthetic operator-routing v1: latent-0 against the full arm's two steps, latent-linear for attribution, and latent-1 and latent-4 at equal parameters. It is an A0-internal ablation on synthetic
data, **not** this manifest's baseline comparison. That comparison needs the plain
model baseline, which is not pinned yet (owner decision O2), so `status` stays
`planned` and `metrics.json` stays reserved for it; the study writes
`results/a0_internal_metrics.json` instead. Design, criteria and results:
`research/falsification/A0-ablations-v1/`.

**Result, 2026-09-25 (A0-internal evidence only; [RESULTS.md](../../../research/falsification/A0-ablations-v1/RESULTS.md)).** M003-nonlinearity
SUPPORTS-NONLINEARITY: removing the latent step costs 8.0 points on test_iid
(interval [+0.069, +0.091]) and removing only its gelu costs 6.5 (interval
[+0.050, +0.079]), 5 of 5 seeds each. M003-depth FALSIFIES: a second tied
refinement step adds nothing measurable on composite C (mean +0.001, interval
[-0.017, +0.019]); see `FALSIFIED-M003-depth.md`. latent-4 was not run (budget
rule), so there is no dose-response statement.

