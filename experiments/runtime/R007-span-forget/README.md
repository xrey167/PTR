# R007-span-forget

## Hypothesis
evicting a span and re-prefilling everything after it yields a state identical to a fresh prefill without the span and leaves no probeable trace of the span.

## Primary metrics
bitwise_mismatch_count, canary_logit_shift, recompute_tokens

## Baseline
fresh prefill of the prompt without the span

## Falsification
any state difference or a canary logit shift above the noise floor.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
