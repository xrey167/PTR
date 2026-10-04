# R010-moe-expert-kv-revocation

## Hypothesis
revoking expert (j,i) invalidates exactly the KV pages of layers above j and nondeterministic routing is never tagged Exact; with deterministic (dropless, batch-invariant) routing, reused KV and outputs are bit-identical to a full prefill of the same prompt under the same expert generations; a nondeterministic-routing negative-control arm that actually attempts reuse is never tagged Exact.

## Primary metrics
stale_pages_served, wrongly_dropped_pages, nondeterministic_exact_tags, bitwise_mismatch_count, nondeterministic_reuse_attempts

## Baseline
invalidate the whole cache on any expert change

## Falsification
any page above j that was created before the revocation is served after it (stale_pages_served > 0), or any page at or below j is dropped because of the revocation, or any nondeterministic-routing run is tagged Exact, or, with deterministic (dropless, batch-invariant) routing, any bitwise mismatch (bitwise_mismatch_count > 0) between reused KV/outputs and a full prefill of the same prompt under the same expert generations. Pages recomputed after the revocation are allowed; the nondeterministic-routing negative-control arm must attempt reuse (nondeterministic_reuse_attempts > 0, else the run is vacuous and falsified) and must report nondeterministic_exact_tags == 0.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
