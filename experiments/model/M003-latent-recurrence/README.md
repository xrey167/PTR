# M003-latent-recurrence

## Hypothesis
Can latent recurrent steps reduce explicit reasoning tokens without quality loss?

## Primary metrics
task success; tokens; wall time; latent steps

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## A0-internal ablation (research/falsification/A0-ablations-v1)

The `a0_*` entrypoints in `experiment.toml` run the A0 mechanism ablation study on
synthetic operator-routing v1: latent-0 against the full arm's two steps, latent-linear for attribution, and latent-1 at equal parameters. The design's latent-4 was dropped by the preregistered budget rule, so no entrypoint runs it. It is an A0-internal ablation on synthetic
data, **not** this manifest's baseline comparison. That comparison needs the plain
model baseline, which is not pinned yet (owner decision O2), so `status` stays
`planned` and `metrics.json` stays reserved for it; the study writes
`results/a0_internal_metrics.json` instead. Design, criteria and results:
`research/falsification/A0-ablations-v1/`. `hardware_profile` names the
study's measured host (`hardware/a0-cpu-4core.toml`); when the baseline
comparison gets its `entrypoint`, the manifest must name the profile of the host
that runs it.

**Corrected status, 2026-09-29.** Gated mechanism hypotheses are **INCONCLUSIVE**:
archived G6 logs lack the required binding to the evaluated commit. Raw measurements
remain descriptive only; see [RESULTS.md](../../../research/falsification/A0-ablations-v1/RESULTS.md).
The historical implementation also left non-live attributes visible through raw
tokens. Fixing admission requires a new frozen study and new training; it does not
validate the archived runs. The matched baseline comparison remains planned.
