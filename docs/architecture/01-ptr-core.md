# PTR Core

The model keeps token states and typed semantic slots in parallel. Slots carry type, epistemic state, validity, confidence and provenance. Typed attention can bias or mask access based on those dimensions. A latent recurrent reasoner performs internal steps without requiring natural-language chain-of-thought tokens. An operator router selects semantic, deductive, probabilistic, statistical, search, optimization, simulation, symbolic or external-Pod computation.

Two research families are reserved: `PTR-AR` and `PTR-Diff`. Every modification has a dedicated spec under `model/modifications/` and must be ablated independently.
