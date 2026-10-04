# M010-moe-rl-stability

## Hypothesis
on OLMoE-based pod-MoE, sequence-level (GSPO) and soft-gated (SAPO) policy optimisation are more stable than token-level GRPO without routing replay at 4 mini-steps, and GRPO, GSPO and SAPO coincide at 1 mini-step; all arms train for the same pinned budget of max_rl_steps (optimizer steps) with shared hyperparameters (learning rate and batch size, tuned once on mean validation pass@1 over all arms on a disjoint split, identical at 1 and 4 mini-steps) and algorithm-specific parameters pinned to the paper values; policy_ratio_variance_per_token is the variance over all tokens of a rollout batch averaged over steps; rewards are verifiable and lie in [0, 1], and collapse means the reward smoothed over a 20-step window is at most 0.5 times its smoothed running maximum while that maximum is above 0 (while it is 0 only a non-finite loss counts), or a non-finite loss.

## Primary metrics
pass_at_1, crash_step, expert_churn_per_layer, routing_entropy, load_cv, train_inference_kl, clip_fraction, grad_norm, policy_ratio_variance_per_token, policy_ratio_variance_per_sequence, rl_steps_completed, max_rl_steps, collapse_smoothing_window_steps

## Baseline
GRPO with clip-higher and no routing replay; GRPO with routing replay; REINFORCE++

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: any arm trained with a max_rl_steps budget or shared hyperparameters different from the others, or the 1 and 4 mini-step settings not sharing hyperparameters; results missing for GSPO or for SAPO in any seed. Falsified if any of: at 4 mini-steps, for GSPO and for SAPO separately, policy_ratio_variance_per_token not lower than GRPO without routing replay; in a seed where the GRPO control crashes, the crash_step of GSPO or SAPO (runs without collapse right-censored at the final step) not later than GRPO's (if GRPO crashes in no seed this criterion is not evaluated and the decision rests on the other criteria); GSPO or SAPO collapses in a seed in which the matched GRPO control survives the max_rl_steps budget; at 1 mini-step, the pass_at_1 of GSPO or of SAPO differs from GRPO's by more than 2 points absolute (broken null control; REINFORCE++ is excluded because its advantage estimator differs); any one of the 5 listed seeds misses any criterion above (every seed must pass).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
