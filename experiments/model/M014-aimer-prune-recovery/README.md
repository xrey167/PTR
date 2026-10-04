# M014-aimer-prune-recovery

## Hypothesis
AIMER weights-only expert ranking with pruning at 25%/50% followed by short SFT and GSPO/SAPO recovers most quality (quality_retention >= 0.90 of the unpruned model at 25% pruning and >= 0.80 at 50% pruning) at lower VRAM, and passes the ForgettingGate on the public suite under one identical pinned recovery budget (2000 SFT steps, 1000 RL steps) for every pruning arm.

## Primary metrics
quality_retention, vram_bytes, ranking_time_s, gate_pass, public_regression, task_forgetting, backward_transfer

## Baseline
unpruned model; random expert pruning; usage-frequency pruning

## Falsification
at 25% and 50% pruning over >=3 seeds: AIMER is not better than random pruning or not better than usage-frequency pruning in quality_retention after the same recovery budget, or quality_retention against the unpruned model is below 90% at 25% pruning or below 80% at 50% pruning, or the pruned and recovered model fails the ForgettingGate on the public suite (gate_pass false: average_forgetting > gate_max_average_forgetting, any task's forgetting > gate_max_task_forgetting, backward_transfer < gate_min_backward_transfer, or public_regression (serving public score minus candidate public score) > gate_max_public_regression (equality passes each threshold; any single violation fails the gate)), or the pruned and recovered model's vram_bytes is not lower than the unpruned model's at the same batch and sequence length; also falsified if any pruning arm (AIMER, random, usage-frequency, at either fraction) ran with a recovery budget different from the pinned budget of 2000 SFT steps and 1000 RL steps (GSPO/SAPO), or if the ForgettingGate is evaluated with anything other than the four pinned thresholds in config; quality_retention = task quality of the pruned and recovered model divided by task quality of the unpruned model (higher is better); vram_bytes is measured at the pinned vram_batch_size and vram_sequence_length.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
