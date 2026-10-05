# R006-positional-reuse-oracle

## Hypothesis
with pre-RoPE key storage, reusing a chunk at a shifted position in a one-layer decoder is bit-identical to recomputation; reuse is exercised through the catalog and its secondary index for every pinned shift, and a revoked upstream segment leaves its content unreachable through the secondary index; a secondary-index shift hit is admitted only if the page's own source-segment lineage (identity, generation, digest) equals the request's segment exactly and every upstream input of the page's original prefix is live (committed at the ledger anchor, not revoked), while the request's ordered prefix lineage may differ.

## Primary metrics
bitwise_mismatch_count, reuse_latency_ms, reuse_attempts_per_shift, secondary_index_lookups, nonaligned_insertion_cases, upstream_revocation_cases, revoked_upstream_page_reachable_count, fresh_prefill_after_revocation, reused_tokens_per_shift, secondary_index_hits_per_shift, post_revocation_secondary_lookups_matching_content, old_upstream_pages_seeded_before_revocation, dead_original_upstream_shift_attempts, shift_hits_with_dead_original_upstream

## Baseline
PreRope full prefill at the new position

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: reuse_attempts_per_shift records fewer than 1 reuse result for any of the 5 pinned shifts, or reused_tokens_per_shift == 0 or secondary_index_hits_per_shift < 1 for any pinned shift; secondary_index_lookups < 1 (reuse must go through the catalog and its position-independent secondary index, not pages handed to the backend); nonaligned_insertion_cases < 1 for the 1-token insertion or for the 7-token insertion; upstream_revocation_cases < 1, old_upstream_pages_seeded_before_revocation < 1 (at least one page derived from the old upstream segment must be seeded in the secondary index before the revocation, so the post-revocation lookup has something to find), or fresh_prefill_after_revocation < 1 (the one-layer oracle has no layers >= 1); post_revocation_secondary_lookups_matching_content < 1; dead_original_upstream_shift_attempts < 1 (a shift hit on a page whose original upstream input is revoked or uncommitted must be attempted). Falsified if any of: bitwise_mismatch_count > 0 at any pinned shift distance (position_shift_tokens = [1, 7, 64, 513, 4096]); revoked_upstream_page_reachable_count > 0 after the upstream revocation, or any post-revocation secondary lookup returning the revoked page; shift_hits_with_dead_original_upstream > 0 (a shift hit admitted although an upstream input of the page's original prefix is dead); any one of the 5 listed seeds misses any criterion above (every seed must pass). The comparison baseline is a PreRope full prefill at the new position.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
