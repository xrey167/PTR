# M015-rlvr-vs-distill-passk

## Hypothesis
on the pod-MoE, RLVR raises pass@1 but not pass@256 relative to the base, while distillation from a stronger teacher raises large-k pass@k.

## Primary metrics
pass_at_1, pass_at_k_curve (k up to 256, unbiased estimator), samples_per_problem

## Baseline
base model; SFT-distilled control

## Falsification
over >=3 seeds, any of: RLVR does not raise pass@1 over the base by at least 2 points; RLVR raises pass@256 over the base by more than 2 points; distillation does not raise pass@256 over the base by at least 2 points; every compared arm (base, RLVR, SFT-distilled) must be evaluated with samples_per_problem = 256 (>= the maximum k); an arm with fewer samples makes the run invalid and falsified.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
