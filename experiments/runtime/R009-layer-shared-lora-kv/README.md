# R009-layer-shared-lora-kv

## Hypothesis
with per-layer chain keys, layers below a LoRA's first adapted layer share base KV pages bit-identically, and aLoRA prefixes reuse base KV exactly, evaluated on at least one configuration with 0 < f < L (pinned f = L/2); configurations with f = 0 alone cannot pass.

## Primary metrics
shared_page_fraction, predicted_shared_fraction, bitwise_mismatch_count, base_pages_dropped_on_revoke

## Baseline
whole-cache adapter binding (no sharing)

## Falsification
shared pages differ from base bitwise, measured sharing deviates from the predicted f/L fraction, revoking the adapter drops any base-keyed page, or the aLoRA prefix is not bit-identical; the criteria must be evaluated on the pinned 0 < f < L configuration (f = L/2), and a run covering only f = 0 configurations fails.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
