# M011-routed-peft

## Hypothesis
routed LoRA experts (PERFT-R) on OLMoE beat a single LoRA and MixLoRA-style experts at equal trainable parameters, and emergent specialisation is at least as good as prescribed code/math/tool specialisation.

## Primary metrics
task_accuracy_per_domain, expert_specialisation_index, trainable_params, latency_per_token

## Baseline
single LoRA at equal parameters; MixLoRA-style top-k LoRA experts

## Falsification
PERFT-R not better than single LoRA at equal parameters over >=3 seeds.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
