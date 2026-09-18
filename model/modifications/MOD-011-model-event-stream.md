# MOD-011-model-event-stream

## Hypothesis
Emit structured intermediate model events so the runtime can interleave Pods and semantic commits.

## Implementation target
`crates/ptr-core/` plus an isolated experiment under `experiments/model/`.

## Required training signal
Create explicit supervised or verifier-derived targets. Do not rely on hidden chain-of-thought text.

## Required ablations
- baseline without this modification;
- modification alone;
- modification combined with the current best stack;
- matched compute and parameter budget where possible.

## Primary metrics
interleaving correctness, latency, resumability

## Falsification
The modification is removed or redesigned if it does not improve its target metric under matched conditions, or if gains disappear under OOD / long-horizon tests.
