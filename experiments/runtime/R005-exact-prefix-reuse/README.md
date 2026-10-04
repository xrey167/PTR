# R005-exact-prefix-reuse

## Hypothesis
a token-keyed, salt-bound radix prefix catalog over 16-token KV pages reuses shared prefixes bit-identically to full prefill, never forks a revoked generation, and achieves a hit rate of at least 40% on a 50%-shared-prefix workload, with at least one revoked-generation fork attempt exercised so the revoked-fork property is non-vacuous; an Approximate run never yields an admitted action or an authoritative SemDB promotion, and an Approximate result persisted as a candidate stays denied in a later otherwise-Exact run until a fresh Exact recomputation; the catalog is bounded (max_trie_nodes and max_kv_page_bytes pinned in config), evicts under a workload of unique prompts exceeding that capacity, and never evicts a page referenced by an active fork.

## Primary metrics
prefill_tokens_saved, wall_time_ms, hit_rate, bitwise_mismatch_count, revoked_fork_count, revoked_fork_attempts, approximate_run_actions_admitted, approximate_action_admission_attempts, approximate_semdb_promotions, approximate_semdb_promotion_attempts, approximate_candidate_path_attempts, approximate_candidate_exact_admissions, peak_trie_nodes, peak_kv_page_bytes, evictions, active_fork_page_evictions, evictions_with_active_fork_pinned

## Baseline
full prefill on the CPU reference decoder

## Falsification
any bitwise mismatch vs full prefill, any successful fork of a revoked generation, or hit rate below 40% on a 50%-shared-prefix workload, or revoked_fork_attempts == 0 (vacuous run), or revoked_fork_count > 0; also falsified by any action admitted from a run whose reuse mode was Approximate (approximate_run_actions_admitted > 0), any SemDB promotion from an Approximate run (approximate_semdb_promotions > 0), or any admission along the path Approximate result, persisted candidate, later otherwise-Exact run (approximate_candidate_exact_admissions > 0; it must stay denied until a fresh Exact recomputation), and the run is vacuous and falsified if approximate_action_admission_attempts < 1, approximate_semdb_promotion_attempts < 1, or approximate_candidate_path_attempts < 1; also falsified if peak_trie_nodes exceeds max_trie_nodes (4096) or peak_kv_page_bytes exceeds max_kv_page_bytes (64 MiB) at any time, if evictions == 0 under the pinned over-capacity workload of unique prompts (vacuous), or if a page referenced by an active fork is evicted (active_fork_page_evictions > 0) or no eviction was attempted while an active fork pinned a page (evictions_with_active_fork_pinned < 1, vacuous); revoked-generation fork attempts must be performed after the revocation; hit_rate = reused_prefix_tokens / total_prompt_tokens.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
