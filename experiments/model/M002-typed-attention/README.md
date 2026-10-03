# M002-typed-attention

## Hypothesis
Do type/epistemic/validity biases improve constraint retention?

## Primary metrics
hard violations; calibration; latency

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## A0-internal ablation (research/falsification/A0-ablations-v1)

The `a0_*` entrypoints in `experiment.toml` run the A0 mechanism ablation study on
synthetic operator-routing v1: no-typed-attention (necessity given QK) and the raw-blind manipulation check. The design's blind-query pair (sufficiency) was dropped by the preregistered budget rule, so no entrypoint runs it. It is an A0-internal ablation on synthetic
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

## Current evidence gate audit

On 2026-10-01 the frozen operator-routing dataset reproduced with data FNV-1a-64
`0ad71688f09b0d0d`; `score.py agree` matched all 36,000 records, and the A0
T7/T8 self-test passed. The Burn harness now also has a `paired` phase that
trains one A0 arm and the matched plain baseline in one atomic process, with the
same seed, optimizer, steps and learning rate. A two-step M002 smoke run passed.
The manifest remains **planned** until a clean freeze commit can bind the new
entrypoint, dataset provenance and baseline digest without relabeling archived
A0 records. M002 remains **INCONCLUSIVE/NO-GO for claims**.
