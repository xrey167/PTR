# M012-lora-expert-balancing

## Hypothesis
auxiliary-loss-free bias balancing of the LoRA-expert router gives equal or better task quality than an auxiliary balance loss at comparable load balance.

## Primary metrics
task_accuracy, max_violation, load_cv, dead_experts

## Baseline
auxiliary balance loss; phi-balancing

## Falsification
with equivalence margins pinned before running (task accuracy +-1 point absolute, load_cv +-0.05): each out-of-margin regression independently falsifies, regardless of the other metric: auxiliary-loss-free bias has task quality lower than the auxiliary loss by more than 1 point absolute (falsifies regardless of load_cv), or load_cv worse by more than 0.05 (falsifies regardless of task quality); both regressing is the dominated outcome and also falsifies; over >=3 seeds.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
