# R006-positional-reuse-oracle

## Hypothesis
with pre-RoPE key storage, reusing a chunk at a shifted position in a one-layer decoder is bit-identical to recomputation.

## Primary metrics
bitwise_mismatch_count, reuse_latency_ms

## Baseline
recompute at the new position

## Falsification
any bitwise mismatch at any pinned shift distance (config.toml position_shift_tokens = [1, 7, 64, 513, 4096]); zero bitwise mismatches are required for every pinned distance.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
