# R008-lossy-kv-tier

## Hypothesis
a ShadowKV/TailorKV-style lossy tier object reduces KV memory by at least 2x at top-1 next-token agreement of at least 98%, and is always tagged Approximate.

## Primary metrics
kv_bytes_ratio, top1_agreement, approximate_tag_violations

## Baseline
exact KvSnapshot tier

## Falsification
memory reduction below 2x at >=98% top-1 agreement, or any lossy reuse not tagged Approximate.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
