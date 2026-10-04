# M014-aimer-prune-recovery

## Hypothesis
AIMER weights-only expert ranking with pruning at 25%/50% followed by short SFT and GSPO/SAPO recovers most quality at lower VRAM, and passes the ForgettingGate on the public suite.

## Primary metrics
quality_retention, vram_bytes, ranking_time_s, gate_pass

## Baseline
unpruned model; random expert pruning; usage-frequency pruning

## Falsification
at 25% and 50% pruning over >=3 seeds: AIMER is not better than random pruning or not better than usage-frequency pruning in quality_retention after the same recovery budget, or quality_retention against the unpruned model is below 90% at 25% pruning, or the pruned and recovered model fails the ForgettingGate on the public suite, or the pruned and recovered model's vram_bytes is not lower than the unpruned model's at the same batch and sequence length.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
