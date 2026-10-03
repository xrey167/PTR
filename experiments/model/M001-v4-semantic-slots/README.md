# M001-v4-semantic-slots

Clean same-freeze evidence run for semantic-slot fidelity against the matched
plain cross-attention baseline. M001-v2 remains historical and is not counted
as v4 evidence.

The `paired` Burn entrypoint trains the full A0 arm and the plain baseline in
one process with identical seed, optimizer, batch, learning rate and steps.

## Decision

This study is **INCONCLUSIVE/NO-GO** for mechanism and model-superiority
claims. The completed records remain immutable reproducibility evidence, but
they do not satisfy every required evidence gate. On `ood_compose_regime`, the
paired accuracy delta is positive (`+0.0868`, 95% CI `[0.0655, 0.1081]`), while
the lower-is-better ECE15 delta (`+0.2610`, CI `[0.1956, 0.3265]`) and NLL delta
(`+7.1180`, CI `[6.4008, 7.8352]`) establish a calibration regression. Accuracy
is descriptive only. `DECISION.toml` binds this interpretation to the immutable
CI artifact.
