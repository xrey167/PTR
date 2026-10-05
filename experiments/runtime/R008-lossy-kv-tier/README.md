# R008-lossy-kv-tier

## Hypothesis
a ShadowKV/TailorKV-style lossy tier object reduces KV memory by at least 2x at top-1 next-token agreement of at least 98%, and is always tagged Approximate; lossy bytes count all tiers plus index and metadata, a lossy object is never served when allow_approximate is false, and an object whose dependency was revoked is not served after the revocation or after a restart; agreement is measured on pinned pretrained Qwen/Qwen2.5-0.5B.

## Primary metrics
kv_memory_reduction, top1_agreement, approximate_tag_violations, lossy_reuse_events, lossy_hits_with_allow_approximate_false, allow_approximate_false_attempts, lossy_revocation_attempts, lossy_hits_after_revocation, lossy_restart_attempts, lossy_hits_after_restart, exact_kv_bytes, lossy_kv_bytes, valid_objects_surviving_restart, revoked_lossy_objects_present_before_revocation, revoked_lossy_objects_present_before_restart

## Baseline
exact KvSnapshot tier

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: lossy_reuse_events < 1; allow_approximate_false_attempts < 1; revoked_lossy_objects_present_before_revocation < 1 or revoked_lossy_objects_present_before_restart < 1 (the restart arm must persist an object whose dependency is revoked), lossy_revocation_attempts < 1 or lossy_restart_attempts < 1; results missing for any pinned candidate keep fraction (0.5, 0.25, 0.125) in top1_agreement or kv_memory_reduction. Falsified if any of: no complete candidate row reaches top1_agreement >= 0.98 (the best configuration is the one with the highest kv_memory_reduction among complete rows reaching the agreement threshold); the best configuration's kv_memory_reduction < 2.0 (kv_memory_reduction = exact_kv_bytes / lossy_kv_bytes, with lossy_kv_bytes counting all tiers plus index and metadata); approximate_tag_violations > 0 (any lossy reuse not tagged Approximate); lossy_hits_with_allow_approximate_false > 0; lossy_hits_after_revocation > 0 or lossy_hits_after_restart > 0 for an object whose dependency was revoked (objects that are still valid may survive a restart; that is not a criterion); any one of the 5 listed seeds misses any criterion above (every seed must pass). top1_agreement is the fraction of the pinned positions whose argmax next token equals the exact tier's, measured on pinned pretrained Qwen/Qwen2.5-0.5B (revision 060db6499f32faf8b98477b0a26969ef7d8b9987, 24 layers, vocabulary 151936, hidden size 896). valid_objects_surviving_restart is reported as information only and is not a criterion.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
