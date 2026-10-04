# M009-approximate-chunk-reuse

## Hypothesis
on an N-layer decoder, position-independent chunk reuse with selective recompute (first k in {0,16,32} tokens or ~15%) keeps top-1 next-token agreement >=99% while saving >=2x prefill.

## Primary metrics
top1_agreement, kl_vs_full, prefill_speedup, recompute_fraction

## Baseline
full prefill

## Falsification
at the best k, top-1 agreement below 99% or prefill saving below 2x.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
