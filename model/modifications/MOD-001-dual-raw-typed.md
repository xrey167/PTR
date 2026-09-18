# MOD-001-dual-raw-typed

## Hypothesis
Maintain raw token states and typed semantic slots in parallel; neither may silently overwrite the other.

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
raw↔typed consistency, semantic preservation, OOD ambiguity

## Falsification
The modification is removed or redesigned if it does not improve its target metric under matched conditions, or if gains disappear under OOD / long-horizon tests.
