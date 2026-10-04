# R005-exact-prefix-reuse

## Hypothesis
a token-keyed, salt-bound radix prefix catalog over 16-token KV pages reuses shared prefixes bit-identically to full prefill, never forks a revoked generation, and achieves a hit rate of at least 40% on a 50%-shared-prefix workload, with at least one revoked-generation fork attempt exercised so the revoked-fork property is non-vacuous.

## Primary metrics
prefill_tokens_saved, wall_time_ms, hit_rate, bitwise_mismatch_count, revoked_fork_count, revoked_fork_attempts

## Baseline
full prefill on the CPU reference decoder

## Falsification
any bitwise mismatch vs full prefill, any successful fork of a revoked generation, or hit rate below 40% on a 50%-shared-prefix workload, or revoked_fork_attempts == 0 (vacuous run), or revoked_fork_count > 0.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
