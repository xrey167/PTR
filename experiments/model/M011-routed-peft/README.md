# M011-routed-peft

## Hypothesis
routed LoRA experts (PERFT-R) on OLMoE beat a single LoRA and MixLoRA-style experts at equal trainable parameters, and emergent specialisation is at least as good as prescribed code/math/tool specialisation; emergent specialisation must be non-inferior to prescribed specialisation within 1 point, mean task accuracy being the macro average over domains, and every arm gets the same preregistered tuning budget.

## Primary metrics
task_accuracy_per_domain, expert_specialisation_index, trainable_params, latency_per_token

## Baseline
single LoRA at equal parameters; MixLoRA-style top-k LoRA experts

## Falsification
over >=3 seeds at equal trainable parameters, PERFT-R is not better than the single LoRA by at least 1 point mean task accuracy, or not better than MixLoRA-style experts by at least 1 point, or prescribed specialisation beats emergent specialisation by more than 1 point; all arms must train with the same pinned budget and trainable_params within the pinned tolerance of each other, otherwise the run is invalid and falsified; task accuracy is recorded per domain (code, math, tool) for every arm, a missing domain cell falsifies, and PERFT-R must meet the 1-point criteria against the single LoRA and against MixLoRA-style experts separately; emergent specialisation is judged by non-inferiority within 1 point of prescribed specialisation; mean task accuracy is the macro average over the pinned domains; every arm receives the same tuning budget over the pinned search space (learning rate, batch size, LoRA rank and alpha); every seed must meet every criterion.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
