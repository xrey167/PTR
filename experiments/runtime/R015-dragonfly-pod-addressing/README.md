# R015-dragonfly-pod-addressing

## Hypothesis
A per-pod learned address (vector z and bias over a frozen encoder, loaded only when its activation gate passed) routes held-out questions to the right registry pod more reliably than description-vector cosine alone, because it can abstain when the best candidate is not clearly the answer.

## Primary metrics
routing_accuracy,abstain_rate,false_route_rate

## Baseline
cosine_only,constant_score

## Falsification
NO-GO if, across the 5 seeds, the learned router does not have a lower false_route_rate than the cosine-only baseline, or its routing_accuracy falls below the baseline by more than the margin pinned at freeze; the representations then stay deactivated and the default PodRouter stays the only router. A learned router that routes to a pod outside registry.candidates, or loads a representation past an unmet activation gate, fails outright.

## Design

Design: [40 — Pod linking](../../../docs/architecture/40-pod-linking.md); decision: [ADR-0022](../../../research/decisions/ADR-0022-learned-pod-addresses-advisory.md); candidates: `evaluations/components/pod-routing`.

One run is one seed, in two steps:

1. Training step: `PYTHONPATH=training/src python3 -m ptr_training.dragonfly.experiment --seed <seed>` draws a synthetic world (six supplier pods with distinct names and a trusted second spelling), trains and exports one address per pod, and writes `target/dragonfly-addressing/<seed>/world.json` (the pods' description vectors and the probe questions with labels). The probes use phrasings the trainer never saw, plus distractors and unregistered suppliers that name no pod.
2. Evaluation step (the manifest's entrypoint): `cargo run --release --locked -p ptr-bench -- dragonfly-addressing <seed>` loads the exported artifacts through `PodRepresentation::from_artifact`, routes every probe through `route_checked` with three arms, and prints one JSON object. It measures; it judges nothing.

Arms: `learned` (`LearnedPodRouter`, may abstain), `cosine_only` (closest description vector, always answers), `constant_score` (the default `PodRouter`). `routing_accuracy` is labelled probes routed to their pod over all labelled probes; `abstain_rate` is probes declined over all probes; `false_route_rate` is probes routed to a wrong pod, or to any pod when none is meant, over all probes.

## Status

`planned`. The harness and the training step exist, and the encoder is a deterministic stand-in (`hashed-bow-64`), which is no evidence about a language model. The frozen encoder and the accuracy margin are `must-be-pinned-before-prepared` in `config.toml`. An unfrozen smoke run of the five seeds (not evidence, recorded to show the harness works) gave `learned` a lower `false_route_rate` than `cosine_only` on all five seeds and a higher or equal `routing_accuracy` on three of them. Nothing was tuned after looking at it.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
