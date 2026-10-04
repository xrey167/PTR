# M016-approximate-chunk-reuse

## Hypothesis
on an N-layer decoder, position-independent chunk reuse at nonzero position shifts (pinned shifts 16, 64 and 256 tokens) with selective recompute (first k in {0,16,32} tokens or ~15%) keeps top-1 next-token agreement >=99% while saving >=2x prefill.

## Primary metrics
top1_agreement, kl_vs_full, prefill_speedup, recompute_fraction

## Baseline
full prefill

## Falsification
at the best k, for ANY pinned nonzero position shift (16, 64, 256 per config.toml [pinned_thresholds]), top-1 agreement below 99% or prefill saving below 2x. Same-position (zero-shift) reuse does not count toward the hypothesis.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
