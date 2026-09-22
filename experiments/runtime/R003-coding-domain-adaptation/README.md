# R003-coding-domain-adaptation

## Hypothesis
Full-parameter continued pretraining on the user's own code corpus measurably improves code-task performance over the unmodified base coding LLM.

## Primary metrics
held-out own-repo completion accuracy/exact-match; held-out own-repo perplexity delta vs. base; public code-eval regression (no degradation)

## Rule
Record matched baselines (unmodified base checkpoint, same decoding config), hardware (dual RTX 3090, PCIe-only, no NVLink), seeds and negative results. Do not change success criteria after observing results.
