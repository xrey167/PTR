# R008-lossy-kv-tier

## Hypothesis
a ShadowKV/TailorKV-style lossy tier object reduces KV memory by at least 2x at top-1 next-token agreement of at least 98%, and is always tagged Approximate.

## Primary metrics
kv_bytes_ratio, top1_agreement, approximate_tag_violations

## Baseline
exact KvSnapshot tier

## Falsification
no lossy configuration reaches >=98% top-1 agreement, or the best configuration reaching >=98% top-1 agreement reduces KV memory by less than 2x, or any lossy reuse is not tagged Approximate.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
