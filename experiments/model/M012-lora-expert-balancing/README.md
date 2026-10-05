# M012-lora-expert-balancing

## Hypothesis
auxiliary-loss-free bias balancing of the LoRA-expert router gives equal or better task quality than an auxiliary balance loss at comparable load balance; the margins are applied against each baseline arm separately; the auxiliary-loss-free arm must meet the pinned balance criteria (final load_cv <= 0.5 per layer, maximum expert token fraction <= 2/E per layer for E experts in that layer, dead_experts == 0; each arm's pass or fail is reported), may rank worse than a baseline by at most 0.05 load_cv, and load_cv is the mean over layers of the per-layer population-std coefficient of variation at the final step, and every arm gets the same preregistered tuning budget; dead_experts means zero routed tokens on the pinned evaluation set at the final step, and load_cv is computed on that set.

## Primary metrics
task_accuracy, max_violation, load_cv, dead_experts, max_violation_reported_only, max_expert_token_fraction_per_layer, balance_criteria_failed_arms, balance_criteria_failed_baseline_arms, tuning_trials_used, tuning_budget_mismatches

## Baseline
auxiliary balance loss; phi-balancing

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: arms trained for different numbers of steps; results missing for the auxiliary-loss arm or the phi-balancing arm in any seed; tuning_budget_mismatches > 0 (tuning_trials_used per arm differs from the pinned 8). Falsified if any of: task_accuracy of the auxiliary-loss-free bias arm lower than the auxiliary-loss arm's by more than 1 point absolute, or lower than the phi-balancing arm's by more than 1 point (each regression falsifies independently of load_cv); load_cv of the auxiliary-loss-free bias arm worse than the auxiliary-loss arm's by more than 0.05, or worse than the phi-balancing arm's by more than 0.05 (each regression falsifies independently of quality); dead_experts > 0 for the auxiliary-loss-free arm (zero routed tokens on the pinned evaluation set at the final step); the auxiliary-loss-free arm fails any absolute balance criterion on any layer (final load_cv > 0.5, maximum expert token fraction > 2/E, or dead_experts > 0), recorded as balance_criteria_failed_arms > 0; any one of the 5 listed seeds misses any criterion above (every seed must pass). load_cv is the population std divided by the mean of per-expert token load, per layer on the pinned evaluation set, then averaged over layers at the final step; max_violation is reported only and is not a criterion. Each arm's balance pass or fail is reported (balance_criteria_failed_arms lists the failing arms); a baseline arm failing the balance criteria is a reported result, not a prerequisite failure, and does not invalidate the comparison (balance_criteria_failed_baseline_arms is reported as information only).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
