# M001-semantic-slots

## Hypothesis
Do typed semantic slots improve OOD semantic fidelity under matched compute?

## Primary metrics
semantic-slot accuracy; hard-constraint fidelity; token efficiency

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Mechanism pilot

`model/burn-a0/examples/m001_pilot.rs` is a deliberately trivial sanity check for the experimental pipeline. Raw tokens are identical across four examples; the typed condition receives a label-carrying slot type while the ablation masks that semantic distinction. Both models use the same PTR-A0 architecture, optimizer, seed and number of steps.

A positive delta only shows that the typed path can carry and learn usable signal through the current cross-attention/recurrent/router implementation. It is **not** the M001 OOD result and must not be reported as evidence that PTR beats a matched language-model baseline.

**Superseded.** Rerun on 2026-09-25 at seed 17 against the current A0, the pilot gives typed 1.0 and ablated 1.0 (delta 0), not the recorded 0.75: the recorded ablated arm could reach at most 0.25 by construction, and the model code has moved on since `results/pilot_*.json` were written. The files are kept as history. The A0 mechanism ablation study below replaces it.

## A0-internal ablation (research/falsification/A0-ablations-v1)

The `a0_*` entrypoints in `experiment.toml` run the A0 mechanism ablation study on
synthetic operator-routing v1: no-semantic-slots and no-semantic-slots-masked against the full arm. It is an A0-internal ablation on synthetic
data, **not** this manifest's baseline comparison. That comparison needs the plain
model baseline, which is not pinned yet (owner decision O2), so `status` stays
`planned` and `metrics.json` stays reserved for it; the study writes
`results/a0_internal_metrics.json` instead. Design, criteria and results:
`research/falsification/A0-ablations-v1/`.

**Result, 2026-09-25 (A0-internal evidence only; [RESULTS.md](../../../research/falsification/A0-ablations-v1/RESULTS.md)).** M001-primary
SUPPORTS: at 1500 steps the full arm beat no-semantic-slots-masked on OOD
composite A by +0.118 (95% interval [+0.060, +0.177], 5 of 5 seeds). M001-secondary
is INCONCLUSIVE: no-semantic-slots fell below its learnability bar at 1500 steps,
so the preregistered contingency re-ran it and the full arm at 4000 steps, where
it reached 0.868 on test_iid and came within +0.013 of the full arm on composite A
(interval [-0.026, +0.052]). Typed slot content may mostly buy learning speed in
this task; the masked arm was not re-run at 4000 steps, so that is open.

