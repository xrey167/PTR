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
