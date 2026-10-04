# R007-span-forget

## Hypothesis
evicting a span and re-prefilling everything after it yields a state identical to a fresh prefill without the span and leaves no probeable trace of the span; canary_logit_shift is the absolute difference of the canary token's logit versus the fresh prefill, maximised over probes, and with zero bitwise mismatch it must be exactly zero.

## Primary metrics
bitwise_mismatch_count, canary_logit_shift, recompute_tokens, span_insertions, span_evictions, downstream_recomputations, canary_probes, stale_downstream_rows_reused, tokens_after_span

## Baseline
fresh prefill of the prompt without the span

## Falsification
any state difference against a fresh prefill without the span, or a canary logit shift above the noise threshold, where the threshold is pinned before running as the 99th percentile of the canary logit shift over 100 seeded control prompts that never contained the canary; zero difference is accepted only if the pinned canary-bearing span case was exercised with span_insertions, span_evictions, downstream_recomputations and canary_probes each >= 1, otherwise the run is vacuous and falsified; canary_probes must be performed after the span eviction and the downstream recomputation; canary_logit_shift is the absolute difference of the canary token's logit, max over probes; if bitwise_mismatch_count == 0 the shift must be exactly 0 and the 99th-percentile noise threshold applies only to the non-bitwise path; stale_downstream_rows_reused must be 0 and recompute_tokens must equal tokens_after_span (every token after the span recomputed); every seed must meet every criterion.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
