# R007-span-forget

## Hypothesis
evicting a span and re-prefilling everything after it yields a state identical to a fresh prefill without the span and leaves no probeable trace of the span.

## Primary metrics
bitwise_mismatch_count, canary_logit_shift, recompute_tokens, span_insertions, span_evictions, downstream_recomputations, canary_probes

## Baseline
fresh prefill of the prompt without the span

## Falsification
any state difference against a fresh prefill without the span, or a canary logit shift above the noise threshold, where the threshold is pinned before running as the 99th percentile of the canary logit shift over 100 seeded control prompts that never contained the canary; zero difference is accepted only if the pinned canary-bearing span case was exercised with span_insertions, span_evictions, downstream_recomputations and canary_probes each >= 1, otherwise the run is vacuous and falsified.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
