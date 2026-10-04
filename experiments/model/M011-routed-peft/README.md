# M011-routed-peft

## Hypothesis
routed LoRA experts (PERFT-R) on OLMoE beat a single LoRA and MixLoRA-style experts at equal trainable parameters, and emergent specialisation is at least as good as prescribed code/math/tool specialisation; emergent specialisation must be non-inferior to prescribed specialisation within 1 point, mean task accuracy being the macro average over domains, and every arm gets the same preregistered tuning budget; the PERFT-R arm used for the 1-point criteria is the emergent one, and LoRA rank is pinned per arm to match trainable parameters within 1%, not tuned.

## Primary metrics
task_accuracy_per_domain, expert_specialisation_index, trainable_params, latency_per_token

## Baseline
single LoRA at equal parameters; MixLoRA-style top-k LoRA experts

## Falsification
The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet: trainable_params differ by more than the pinned relative tolerance (1%) between arms, or arms trained with different budgets; task_accuracy_per_domain missing for any pinned domain (code, math, tool) in any arm or seed. Falsified if any of: the emergent PERFT-R arm's macro-average task accuracy is not better than the single LoRA's by at least 1 point; the emergent PERFT-R arm's macro-average task accuracy is not better than MixLoRA-style experts' by at least 1 point; prescribed specialisation beats emergent by more than 1 point (emergent specialisation must be non-inferior within 1 point); any one of the 5 listed seeds misses any criterion above (every seed must pass). Mean task accuracy is the macro average over the pinned domains; LoRA rank is fixed per arm from the trainable-parameter count before any run and is not tuned.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
