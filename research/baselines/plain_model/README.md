# Plain Model Baseline

Matched baseline contract for PTR architecture experiments. A reported run must pin the exact backbone revision, inference backend, context policy, sampling configuration and compute budget.

The baseline deliberately disables PTR semantic slots, latent recurrence, operator routing and PTR memory so architecture gains can be attributed through controlled ablation.

## Current implementation

`model/burn-a0` contains a token-only six-latent cross-attention baseline with
the same frozen operator-routing loader, seed stream, batch size, Adam schedule
and output format as the A0 study. It has no typed metadata: the six latent
tokens are initialized by pooling 26 ordinary learned rows (5, 5, 4, 4, 4, 4),
so every parameter is used. At `d_model = 48` it has 33,323 parameters versus
33,324 for the PTR-A0 full arm (−0.003%). The baseline emits
`estimated_flops_per_example` in its meta row.

The checked compute proxy is 1,665,216 FLOPs/example versus the frozen A0
reference 1,665,792 (−0.035%), inside the preregistered ±1% gate. This clears
the budget contract; M001/M002 still require fresh five-seed runs and their
evidence gates before any scientific claim.
