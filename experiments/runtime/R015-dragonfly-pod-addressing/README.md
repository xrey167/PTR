# R015-dragonfly-pod-addressing

## Hypothesis
A per-pod learned address (vector z and bias over a frozen encoder, loaded only when its activation gate passed) routes held-out questions to the right registry pod more reliably than description-vector cosine alone, because it can abstain when the best candidate is not clearly the answer.

## Primary metrics
routing_accuracy,abstain_rate,false_route_rate,reflex_failover_coverage,reflex_dispatch_hits

## Baseline
cosine_only,constant_score

## Falsification
NO-GO unless, on every one of the 5 confirmatory seeds, the learned router has routing_accuracy at least that of the cosine-only baseline (margin 0 permille) and a false_route_rate strictly below it; both must hold on every seed, or the representations stay deactivated and the default PodRouter stays the only router. A seed whose validity controls fail (oracle ceiling, constant-score floor, full activation) is INVALID, not a pass. The ablations (no abstention, no bias), the sensitivity sweep and the oracle are reported, not judged. A learned router that routes to a pod outside registry.candidates, or loads a representation past an unmet activation gate, fails outright. Two further claims are judged apart from that one (PREREGISTRATION.md 4a): reflex_failover_coverage is NO-GO unless every miss of the learned router is caught by the default pod on every seed (and INVALID, not a pass, when a seed has no miss), and reflex_dispatch_hits is NO-GO unless the learned router's own dispatches are more often to the named pod than description cosine's on every seed (INVALID when it dispatches nothing). Criteria and stop rules: PREREGISTRATION.md and criteria.toml.

## Design

Design: [43 — Pod linking](../../../docs/architecture/43-pod-linking.md); decision: [ADR-0023](../../../research/decisions/ADR-0023-learned-pod-addresses-advisory.md); candidates: `evaluations/components/pod-routing`; preregistration: [PREREGISTRATION.md](PREREGISTRATION.md); decision rules: [criteria.toml](criteria.toml).

One run is one seed and one provenance-tracked command, the manifest's entrypoint: `cargo run --release --locked -p ptr-bench -- dragonfly-addressing <seed>`. The harness generates the seed's world itself and consumes it: it starts the training step (`python3 -m ptr_training.dragonfly.experiment`, with `training/src` on `PYTHONPATH`) as a child process in a scratch directory of its own under the runner's temporary directory, loads the exported artifacts through `PodRepresentation::from_artifact`, routes every probe through `route_checked`, prints one JSON object and removes the scratch directory. A clean checkout needs no prior step. The world is six supplier pods with distinct names and a trusted second spelling; the probes use phrasings the trainer never saw, plus distractors and unregistered suppliers that name no pod.

The environment is part of the experiment. A confirmatory run starts the training step through `uv run --locked --project training --extra dragonfly` (virtual environment in the scratch directory), with the exact interpreter the preregistration pins (`python_version`, passed as `uv run --python` and compared with `sys.version_info` by the step), so the installed packages are the ones `training/uv.lock` names; `--locked` refuses an out-of-date lock. The training step checks again: without `--smoke` it exits 2 when the installed `sentence-transformers`, `torch`, `transformers` or `numpy` differ from the lock or are missing. Every run, smoke or not, records the interpreter version, those package versions, the locked versions and the SHA-256 of `training/uv.lock` in the printed object (`environment`); a smoke run (plain `python3`, no network) only records a mismatch. `uv` and network access to the package index are therefore prerequisites of a confirmatory run; without them it exits 2. The manifest declares a `preflight` (`... dragonfly-addressing --preflight`), which `scripts/run_experiment.py` runs before it reserves a seed: the training step in the same confirmatory environment checks the pinned encoder (it loads and encodes a probe), the interpreter and the locked packages, builds no world and exits 2 when any of them is missing, so an environment failure never spends a one-shot confirmatory seed.

The training step builds the encoder that `config.toml` pins (`encoder_model`, `encoder_revision`). While the pin is still a `must-be-pinned-…` placeholder, or the pinned encoder cannot be built, the run exits 2 (harness failure): it never falls back to another encoder. `... dragonfly-addressing <seed> --smoke` selects the deterministic stand-in encoder (`hashed-bow-64`) and marks the output `"evidence": "none (stand-in encoder)"`; it checks the machinery on a pilot seed and is no result.

The `reflex` block of the printed object measures two further claims, each with its own verdict (PREREGISTRATION.md 4a, `[claims.*]` of `criteria.toml`): the `learned` arm is put behind the runtime's failover path (a fallback plan naming the lexically smallest pod id as the default, whose lifecycle generation is live), every probe is dispatched as a Pod request, and the runtime's reflex counters are read. The harness host admits the activated representations explicitly (project, pod and digest of what the router's store holds); a sampler that names none is routed against an empty set, so nothing would be scored. `reflex_failover_coverage` is the mechanism (failovers over misses), `reflex_dispatch_hits` the benefit (the precision of the router's own dispatches against description cosine). A seed with no miss or no hit is INVALID for the claim, not a pass.

Arms: `learned` (`LearnedPodRouter`, may abstain), `learned_no_abstain` and `learned_no_bias` (ablations), `sweep` (four pre-fixed threshold pairs), `cosine_only` (closest description vector, always answers), `constant_score` (the default `PodRouter`, the floor) and `oracle` (the ceiling). `routing_accuracy` is labelled probes routed to their pod over all labelled probes; `abstain_rate` is probes declined over all probes; `false_route_rate` is probes routed to a wrong pod, or to any pod when none is meant, over all probes.

## Status

`planned`. The harness, the training step, the preregistration text and the decision criteria exist. Still open before the freeze: the frozen encoder (`must-be-pinned-before-prepared` in `config.toml`), the two content digests of the preregistration text and criteria, and the aggregator that reads `criteria.toml`. Until the encoder is pinned a non-smoke run exits 2 by design.

Chronology, disclosed: on 2026-10-13, before the criterion was fixed, the harness was run once on the seeds 17, 29, 43, 71 and 101 to see that it works. Those seeds are burned: they are listed as `explorative_seeds`, their numbers are not used, and they are no confirmatory seeds. The confirmatory seeds are 2017, 2029, 2043, 2071 and 2101, never run; tuning, pilots and smoke runs use seeds >= 1001 only (pilot seeds 1001 and 1002), and each confirmatory seed runs once, after the freeze.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
