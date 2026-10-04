# M013-expert-lifecycle

## Hypothesis
a gated expert lifecycle (birth, merge, split, freeze, prune, reactivate) under domain shift forgets less and matches a static expert set at equal compute; reactivating a retired expert recovers a returning domain faster than a new expert.

## Primary metrics
average_forgetting, backward_transfer, accuracy_matrix, expert_count, steps_to_recover

## Baseline
static expert set at equal compute; naive adapter chain

## Falsification
over >=3 seeds at equal compute: average_forgetting of the gated lifecycle is not strictly below the static expert set, or its final mean accuracy (accuracy_matrix) is more than 1 point below the static set, or reactivation of a retired expert does not reach the pre-retirement accuracy in fewer steps than birth of a new expert (steps_to_recover).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
