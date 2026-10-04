# M015-rlvr-vs-distill-passk

## Hypothesis
on the pod-MoE, RLVR raises pass@1 but not pass@256 relative to the base, while distillation from a stronger teacher raises large-k pass@k; pass@k (k up to 256) uses the unbiased estimator with 256 samples per problem on the full GSM8K test split (1319 problems), the teacher is Qwen/Qwen3-30B-A3B at a pinned revision, distillation compute is matched to RLVR training FLOPs, and the criteria are judged on paired bootstrap 95% confidence intervals over problems.

## Primary metrics
pass_at_1, pass_at_k_curve, samples_per_problem, training_flops, pass_at_1_gain_ci_lower, pass_at_256_gain_ci_lower

## Baseline
base model; SFT-distilled control

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: samples_per_problem != 256 for any compared arm (base, RLVR, SFT-distilled); training_flops of the RLVR and distilled arms differ by more than compute_relative_tolerance (declared per arm); the pass@k curve is not recorded for every compared arm in every seed, or the pinned teacher and problem-set revisions are not recorded in run.json. Falsified if any of: the lower bound of the paired bootstrap 95% CI of RLVR's pass@1 gain over the base is below 2 points; the lower bound of the paired bootstrap 95% CI of RLVR's pass@256 gain over the base is above 2 points; the lower bound of the paired bootstrap 95% CI of the distilled arm's pass@256 gain over the base is below 2 points; any one of the 5 listed seeds misses any criterion above (every seed must pass). Gains are absolute differences in points; the CIs use 10000 paired bootstrap resamples over problems.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
