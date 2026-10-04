# M014-aimer-prune-recovery

## Hypothesis
AIMER weights-only expert ranking with pruning at 25%/50% followed by short SFT and GSPO/SAPO recovers most quality at lower VRAM, and passes the ForgettingGate on the public suite.

## Primary metrics
quality_retention, vram_bytes, ranking_time_s, gate_pass

## Baseline
unpruned model; random expert pruning; usage-frequency pruning

## Falsification
AIMER not better than random pruning at equal ratio, or pruned model fails the ForgettingGate after recovery.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
