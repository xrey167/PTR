# MOD-004-epistemic-types

## Hypothesis
Represent epistemic state explicitly (unknown, assumed, hypothesis, observed, inferred, verified) while keeping uncertainty representation (point/interval/distribution) as a separate model axis.

## Implementation target
`ptr-types` defines the shared EpistemicState/UncertaintyKind semantics; `crates/ptr-core/` encodes those axes neurally, with an isolated experiment under `experiments/model/`.

## Required training signal
Create explicit supervised or verifier-derived targets. Do not rely on hidden chain-of-thought text.

## Required ablations
- baseline without this modification;
- modification alone;
- modification combined with the current best stack;
- matched compute and parameter budget where possible.

## Primary metrics
Brier/ECE, false certainty, unknown recognition

## Falsification
The modification is removed or redesigned if it does not improve its target metric under matched conditions, or if gains disappear under OOD / long-horizon tests.
