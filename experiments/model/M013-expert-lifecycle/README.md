# M013-expert-lifecycle

## Hypothesis
a gated expert lifecycle (birth, merge, split, freeze, prune, reactivate) under domain shift forgets less and matches a static expert set at equal compute; reactivating a retired expert recovers a returning domain faster than a new expert.

## Primary metrics
average_forgetting, backward_transfer, accuracy_matrix, expert_count, steps_to_recover

## Baseline
static expert set at equal compute; naive adapter chain

## Falsification
forgetting above the static baseline, or reactivation not faster than birth of a new expert.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
