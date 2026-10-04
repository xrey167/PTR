# E006-engine-prefix-reuse

## Hypothesis
binding an engine's prefix cache to a PTR-derived salt from the currently admitted binding yields at least 1.5x TTFT gain at 50% shared prefix while never hitting pre-revocation blocks.

## Primary metrics
ttft_ms, hit_rate, post_revocation_hits

## Baseline
same engine without prefix caching / without salt

## Falsification
TTFT gain below 1.5x at 50% shared prefix, or any hit on a pre-revocation block.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
