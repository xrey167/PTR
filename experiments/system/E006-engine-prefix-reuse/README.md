# E006-engine-prefix-reuse

## Hypothesis
binding an engine's prefix cache to a PTR-derived salt from the currently admitted binding yields at least 1.5x TTFT gain at 50% shared prefix while never hitting pre-revocation blocks.

## Primary metrics
ttft_ms, hit_rate, post_revocation_hits

## Baseline
primary comparator: same engine with prefix caching DISABLED (the denominator of the TTFT gain); secondary control: same engine with caching enabled WITHOUT salt (used only for the post-revocation-hit check, where hits after revocation are expected and demonstrate that the salt matters)

## Falsification
TTFT gain against the primary comparator (caching disabled) below 1.5x at 50% shared prefix, or any hit on a pre-revocation block with the salted cache.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
