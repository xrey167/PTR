# M010-moe-rl-stability

## Hypothesis
on OLMoE-based pod-MoE, sequence-level (GSPO) and soft-gated (SAPO) policy optimisation are more stable than token-level GRPO without routing replay at 4 mini-steps, and GRPO, GSPO and SAPO coincide at 1 mini-step.

## Primary metrics
pass_at_1, crash_step, expert_churn_per_layer, routing_entropy, load_cv, train_inference_kl, clip_fraction, grad_norm, policy_ratio_variance_per_token, policy_ratio_variance_per_sequence

## Baseline
GRPO with clip-higher and no routing replay; GRPO with routing replay; REINFORCE++

## Falsification
over >=3 seeds at 4 mini-steps, GSPO or SAPO shows no lower policy_ratio_variance_per_token than GRPO without routing replay, or, only in seeds where the GRPO control crashes, the median crash_step of GSPO or SAPO (runs without collapse are right-censored at the final step, survival-style) is not later than GRPO's; if GRPO does not crash in any seed within the step budget the crash-ordering criterion is not evaluated and the decision rests on policy_ratio_variance_per_token and the 1-mini-step null control alone; or, at 1 mini-step, GSPO's or SAPO's pass_at_1 differs from GRPO's by more than 2 points absolute (broken null control; REINFORCE++ is excluded from the null control because its advantage estimator differs). Collapse is pinned in config before running: reward drop >50% from its running maximum or non-finite loss.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
