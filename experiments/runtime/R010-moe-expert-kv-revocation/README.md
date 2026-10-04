# R010-moe-expert-kv-revocation

## Hypothesis
revoking expert (j,i) invalidates exactly the KV pages of layers above j and nondeterministic routing is never tagged Exact; with deterministic (dropless, batch-invariant) routing, reused KV and outputs are bit-identical to a full prefill of the same prompt under the same expert generations; a nondeterministic-routing negative-control arm that actually attempts reuse is never tagged Exact; the revocation scenario is deterministic and non-vacuous: pages are already cached on both sides of layer j and an expert revocation is actually performed.

## Primary metrics
stale_pages_served, wrongly_dropped_pages, nondeterministic_exact_tags, bitwise_mismatch_count, nondeterministic_reuse_attempts, deterministic_revocation_attempts, pages_above_j_before_revocation, pages_at_or_below_j_before_revocation, deterministic_reuse_events, post_revocation_reuse_probes, retained_pages_reused_after_revocation, recomputed_pages_above_j_served

## Baseline
invalidate the whole cache on any expert change

## Falsification
any page above j that was created before the revocation is served after it (stale_pages_served > 0), or any page at or below j is dropped because of the revocation, or any nondeterministic-routing run is tagged Exact, or, with deterministic (dropless, batch-invariant) routing, any bitwise mismatch (bitwise_mismatch_count > 0) between reused KV/outputs and a full prefill of the same prompt under the same expert generations. Pages recomputed after the revocation are allowed; the nondeterministic-routing negative-control arm must attempt reuse (nondeterministic_reuse_attempts > 0, else the run is vacuous and falsified) and must report nondeterministic_exact_tags == 0; the run is vacuous and falsified if deterministic_revocation_attempts < 1, pages_above_j_before_revocation < 1, pages_at_or_below_j_before_revocation < 1, or deterministic_reuse_events < 1 (no reuse was exercised for the bit-identity check); post-revocation probes are required: post_revocation_reuse_probes >= 1 that reuse retained pages at or below j (retained_pages_reused_after_revocation >= 1) and serve pages above j recomputed after the revocation (recomputed_pages_above_j_served >= 1), each bit-identical to a full prefill, otherwise vacuous and falsified; the pinned model has 4 layers with j = 1, so layers 2 and 3 are above j and layers 0 and 1 are at or below j.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
