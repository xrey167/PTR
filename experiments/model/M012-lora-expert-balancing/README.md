# M012-lora-expert-balancing

## Hypothesis
auxiliary-loss-free bias balancing of the LoRA-expert router gives equal or better task quality than an auxiliary balance loss at comparable load balance; the margins are applied against each baseline arm separately; the auxiliary-loss-free arm has no dead experts, load_cv is the mean over layers of the per-layer population-std coefficient of variation at the final step, and every arm gets the same preregistered tuning budget.

## Primary metrics
task_accuracy, max_violation, load_cv, dead_experts, max_violation_reported_only

## Baseline
auxiliary balance loss; phi-balancing

## Falsification
with equivalence margins pinned before running (task accuracy +-1 point absolute, load_cv +-0.05): each out-of-margin regression independently falsifies, regardless of the other metric: auxiliary-loss-free bias has task quality lower than the auxiliary loss by more than 1 point absolute (falsifies regardless of load_cv), or load_cv worse by more than 0.05 (falsifies regardless of task quality); both regressing is the dominated outcome and also falsifies; over >=3 seeds; both arms must train for the same pinned number of steps; load_cv is the standard deviation divided by the mean of per-expert token load (lower is better) and task accuracy is in points on the pinned evaluation suite; all margins and independent-falsification rules apply separately against the auxiliary-loss arm AND against the phi-balancing arm (task_accuracy and load_cv are reported per arm; each out-of-margin regression versus either baseline falsifies independently); load_cv is computed per layer as population std / mean of per-expert token load and then averaged over layers at the final step; dead_experts must be 0 for the auxiliary-loss-free arm (dead_experts > 0 falsifies); max_violation is reported only and is not a criterion; every arm receives the same tuning budget over the pinned search space (learning rate, batch size, aux coefficient, bias update rate, LoRA rank and alpha); every seed must meet every criterion.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
