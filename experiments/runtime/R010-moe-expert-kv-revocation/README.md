# R010-moe-expert-kv-revocation

## Hypothesis
revoking expert (j,i) invalidates exactly the KV pages of layers above j and nondeterministic routing is never tagged Exact.

## Primary metrics
stale_pages_served, wrongly_dropped_pages, nondeterministic_exact_tags

## Baseline
invalidate the whole cache on any expert change

## Falsification
any page above j that was created before the revocation is served after it (stale_pages_served > 0), or any page at or below j is dropped because of the revocation, or any nondeterministic-routing run is tagged Exact. Pages recomputed after the revocation are allowed.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
