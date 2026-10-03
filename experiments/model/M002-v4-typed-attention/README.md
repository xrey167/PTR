# M002-v4-typed-attention

Clean same-freeze evidence run for the M002 typed-attention comparison.
Each declared seed is executed from the same prepared commit in an isolated
worktree; the resulting records are aggregated only after all five runs.

M002-v2 and M002-v3 remain historical and are not counted as v4 evidence.

## Decision

This study is **INCONCLUSIVE/NO-GO** for typed-attention and model-superiority
claims. Its recorded contrast is `no-typed-attention` versus
`plain-transformer`; because those arms differ in more than the typed-attention
factor, M002-v4 does not isolate Typed Attention. The completed records remain
immutable reproducibility evidence. On `ood_compose_regime`, the paired
accuracy delta is positive (`+0.0941`, 95% CI `[0.0596, 0.1287]`), while the
lower-is-better ECE15 delta (`+0.2492`, CI `[0.1818, 0.3166]`) and NLL delta
(`+6.7761`, CI `[6.0359, 7.5162]`) establish a calibration regression. Accuracy
is descriptive only. `DECISION.toml` binds this interpretation to the immutable
CI artifact.
