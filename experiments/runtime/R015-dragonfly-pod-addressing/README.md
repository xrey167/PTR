# R015-dragonfly-pod-addressing

## Hypothesis
A per-pod learned address (vector z and bias over a frozen encoder, loaded only when its activation gate passed) routes held-out questions to the right registry pod more reliably than description-vector cosine alone, because it can abstain when the best candidate is not clearly the answer.

## Primary metrics
routing_accuracy,abstain_rate,false_route_rate

## Baseline
cosine_only,constant_score

## Falsification
NO-GO unless, on every one of the 5 confirmatory seeds, the learned router has routing_accuracy at least that of the cosine-only baseline (margin 0 permille) and a false_route_rate strictly below it; both must hold on every seed, or the representations stay deactivated and the default PodRouter stays the only router. A learned router that routes to a pod outside registry.candidates, or loads a representation past an unmet activation gate, fails outright.

## Design

Design: [40 — Pod linking](../../../docs/architecture/40-pod-linking.md); decision: [ADR-0023](../../../research/decisions/ADR-0023-learned-pod-addresses-advisory.md); candidates: `evaluations/components/pod-routing`.

One run is one seed, in two steps:

1. Training step: `PYTHONPATH=training/src python3 -m ptr_training.dragonfly.experiment --seed <seed>` draws a synthetic world (six supplier pods with distinct names and a trusted second spelling), trains and exports one address per pod, and writes `target/dragonfly-addressing/<seed>/world.json` (the pods' description vectors and the probe questions with labels). The probes use phrasings the trainer never saw, plus distractors and unregistered suppliers that name no pod.
2. Evaluation step (the manifest's entrypoint): `cargo run --release --locked -p ptr-bench -- dragonfly-addressing <seed>` loads the exported artifacts through `PodRepresentation::from_artifact`, routes every probe through `route_checked` with three arms, and prints one JSON object. It measures; it judges nothing.

Arms: `learned` (`LearnedPodRouter`, may abstain), `cosine_only` (closest description vector, always answers), `constant_score` (the default `PodRouter`). `routing_accuracy` is labelled probes routed to their pod over all labelled probes; `abstain_rate` is probes declined over all probes; `false_route_rate` is probes routed to a wrong pod, or to any pod when none is meant, over all probes.

## Status

`planned`. The harness and the training step exist, and the encoder is a deterministic stand-in (`hashed-bow-64`), which is no evidence about a language model. The frozen encoder is `must-be-pinned-before-prepared` in `config.toml`; the accuracy margin is pinned at 0 permille and the criterion above comes from the plan, not from data.

Chronology, disclosed: on 2026-10-13, before the criterion was fixed, the harness was run once on the confirmatory seeds 17, 29, 43, 71 and 101 to see that it works. That result is not usable and is not used; tuning and pilots run on seeds >= 1001 only, and the confirmatory seeds run once, after the freeze.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
