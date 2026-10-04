# R010-moe-expert-kv-revocation

## Hypothesis
revoking expert (j,i) invalidates exactly the KV pages of layers above j and nondeterministic routing is never tagged Exact.

## Primary metrics
stale_pages_served, wrongly_dropped_pages, nondeterministic_exact_tags

## Baseline
invalidate the whole cache on any expert change

## Falsification
any page above j served after revocation, any page at or below j dropped, or any nondeterministic-routing run tagged Exact.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
