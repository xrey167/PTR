# R006-positional-reuse-oracle

## Hypothesis
with pre-RoPE key storage, reusing a chunk at a shifted position in a one-layer decoder is bit-identical to recomputation.

## Primary metrics
bitwise_mismatch_count, reuse_latency_ms

## Baseline
recompute at the new position

## Falsification
any bitwise mismatch.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
