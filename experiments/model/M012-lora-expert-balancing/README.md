# M012-lora-expert-balancing

## Hypothesis
auxiliary-loss-free bias balancing of the LoRA-expert router gives equal or better task quality than an auxiliary balance loss at comparable load balance.

## Primary metrics
task_accuracy, max_violation, load_cv, dead_experts

## Baseline
auxiliary balance loss; phi-balancing

## Falsification
lower task quality than auxiliary loss at comparable balance, or worse balance at equal quality.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
