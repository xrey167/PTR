# M015-rlvr-vs-distill-passk

## Hypothesis
on the pod-MoE, RLVR raises pass@1 but not pass@256 relative to the base, while distillation from a stronger teacher raises large-k pass@k.

## Primary metrics
pass_at_1, pass_at_k_curve (k up to 256, unbiased estimator)

## Baseline
base model; SFT-distilled control

## Falsification
RLVR raises pass@256 above base and distillation does not, over >=3 seeds.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
