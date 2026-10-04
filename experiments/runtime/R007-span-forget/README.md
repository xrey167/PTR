# R007-span-forget

## Hypothesis
evicting a span and re-prefilling everything after it yields a state bit-identical to a fresh prefill without the span and leaves no probeable trace of the span (canary_logit_shift is exactly 0); the experiment is bitwise only and uses seeded random weights with a pinned seed and shape.

## Primary metrics
bitwise_mismatch_count, canary_logit_shift, recompute_tokens, span_insertions, span_evictions, downstream_recomputations, canary_probes, stale_downstream_rows_reused, tokens_after_span

## Baseline
fresh prefill of the prompt without the span

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: span_insertions < 1, span_evictions < 1, downstream_recomputations < 1 or canary_probes < 1 on the pinned canary-bearing span case (canary_probes must be made after the span eviction and the downstream recomputation). Falsified if any of: bitwise_mismatch_count > 0 (any state difference against a fresh prefill without the span); canary_logit_shift != 0 (the absolute difference of the canary token's logit versus the fresh prefill, max over probes; there is no noise threshold); stale_downstream_rows_reused > 0 or recompute_tokens != tokens_after_span (every token after the span must be recomputed); any one of the 5 listed seeds misses any criterion above (every seed must pass).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
