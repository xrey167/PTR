# M005-epistemic-calibration

## Hypothesis
Do explicit epistemic states reduce false certainty?

## Primary metrics
Brier; ECE; unknown recall

## Plain baseline (owner decision O2, 2026-10-06)

Owner decision O2 option (c) ([ADR-0021](../../../research/decisions/ADR-0021-o2-plain-baseline-hybrid.md)) pins the A0-scale plain baseline for M001 to M004 only. M005 has no matching plain arm at the A0 tier, so it is **out of scope for the A0 tier until the pretrained open LM baseline exists**; that baseline is deferred to the GPU phase. Until then no M005 claim is made.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
