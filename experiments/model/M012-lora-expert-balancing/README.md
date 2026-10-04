# M012-lora-expert-balancing

## Hypothesis
auxiliary-loss-free bias balancing of the LoRA-expert router gives equal or better task quality than an auxiliary balance loss at comparable load balance.

## Primary metrics
task_accuracy, max_violation, load_cv, dead_experts

## Baseline
auxiliary balance loss; phi-balancing

## Falsification
with equivalence margins pinned before running (task accuracy +-1 point absolute, load_cv +-0.05): auxiliary-loss-free bias has lower task quality than the auxiliary loss beyond the margin while its load_cv is within the margin, or worse load_cv beyond the margin while quality is within the margin, over >=3 seeds.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
