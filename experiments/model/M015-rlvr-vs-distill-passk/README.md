# M015-rlvr-vs-distill-passk

## Hypothesis
on the pod-MoE, RLVR raises pass@1 but not pass@256 relative to the base, while distillation from a stronger teacher raises large-k pass@k; the teacher, sampling temperature and top-p, the held-out problem set (>= 500 problems) and the distillation data and compute matching are pinned.

## Primary metrics
pass_at_1, pass_at_k_curve (k up to 256, unbiased estimator), samples_per_problem

## Baseline
base model; SFT-distilled control

## Falsification
over >=3 seeds, any of: RLVR does not raise pass@1 over the base by at least 2 points; RLVR raises pass@256 over the base by more than 2 points; distillation does not raise pass@256 over the base by at least 2 points; every compared arm (base, RLVR, SFT-distilled) must be evaluated with samples_per_problem = 256 (>= the maximum k); an arm with fewer samples makes the run invalid and falsified; pass@k uses the unbiased estimator with n = samples_per_problem = 256 samples per problem, and pass@1 and pass@256 gains are absolute differences in points; RLVR and distillation are each compared with the base separately, and the pass@k curve must be recorded for every compared arm in every seed; the teacher model is recorded in run.json before the first run, sampling uses the pinned temperature and top-p, the problem set is held out with at least 500 problems, distillation data is disjoint from it and distillation compute is matched to RLVR compute; every arm receives the same tuning budget; every seed must meet every criterion.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
