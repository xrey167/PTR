# M010-moe-rl-stability

## Hypothesis
on OLMoE-based pod-MoE, sequence-level (GSPO) and soft-gated (SAPO) policy optimisation are more stable than token-level GRPO without routing replay at 4 mini-steps, and all arms coincide at 1 mini-step.

## Primary metrics
pass_at_1, crash_step, expert_churn_per_layer, routing_entropy, load_cv, train_inference_kl, clip_fraction, grad_norm

## Baseline
GRPO with clip-higher and no routing replay; GRPO with routing replay; REINFORCE++

## Falsification
GSPO/SAPO show no lower churn-linked ratio variance or no later crash than GRPO at 4 mini-steps over >=3 seeds, or arms differ significantly at 1 mini-step (broken null control).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
