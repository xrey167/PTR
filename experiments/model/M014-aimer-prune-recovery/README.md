# M014-aimer-prune-recovery

## Hypothesis
AIMER weights-only expert ranking with pruning at 25%/50% followed by short SFT and GSPO/SAPO recovers most quality (quality_retention >= 0.90 of the unpruned model with the identical recovery budget (control arm) at 25% pruning and >= 0.80 at 50% pruning) at lower VRAM, and passes the ForgettingGate on the public suite under one identical pinned recovery budget (2000 SFT steps, 1000 RL steps) for every pruning arm; AIMER ranking beats random and usage-frequency pruning at each fraction after the same recovery, recovery runs SFT then GSPO and, as a second arm, SFT then SAPO (both must meet every criterion), pruned experts are physically removed, and quality is the pinned macro-average task accuracy on the pinned suite.

## Primary metrics
quality_retention, vram_bytes, ranking_time_s, gate_pass, public_regression, task_forgetting, backward_transfer, quality_unpruned_control, average_forgetting

## Baseline
unpruned model; random expert pruning; usage-frequency pruning

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: the unpruned control with the identical recovery budget is missing; any cell of the full matrix of method (AIMER, random, usage-frequency) x fraction (25%, 50%) x optimizer (SFT then GSPO, SFT then SAPO) x seed is missing; any arm ran with a recovery budget different from the pinned 2000 SFT steps and 1000 RL steps, or the ForgettingGate was evaluated with anything other than the four pinned thresholds. Falsified if any of: at either fraction and for either optimizer, AIMER's quality_retention is not better than random pruning's or not better than usage-frequency pruning's after the same recovery budget; quality_retention against the unpruned control is below 0.90 at 25% pruning or below 0.80 at 50% pruning, for either optimizer; the pruned and recovered model fails the ForgettingGate on the public suite (gate_pass false: average_forgetting > gate_max_average_forgetting, any task's forgetting > gate_max_task_forgetting, backward_transfer < gate_min_backward_transfer, or public_regression > gate_max_public_regression; equality passes each threshold); vram_bytes of the pruned and recovered model (measured after physical expert removal at the pinned vram_batch_size and vram_sequence_length) not lower than the unpruned model's; any one of the 5 listed seeds misses any criterion above (every seed must pass). quality_retention = quality of the pruned and recovered model divided by quality of the unpruned model recovered with the identical budget (higher is better), as the pinned macro-average task accuracy on the pinned suite.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
