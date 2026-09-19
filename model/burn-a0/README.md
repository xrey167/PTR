# PTR Burn A0

This is the first **trainable tensor implementation** of a PTR model component. It is deliberately kept outside the production workspace while the model-framework evaluation remains open.

## Implemented

- token embeddings for the raw path;
- slot-type embeddings for the typed path;
- epistemic, validity and provenance-bucket embeddings;
- confidence projection into the typed latent space;
- learned typed metadata bias on both raw→slot and slot→raw attention;
- bidirectional raw↔slot cross-attention;
- residual projection back into raw and semantic workspaces;
- shared recurrent latent refinement steps;
- operator-router logits over the updated semantic slots;
- upstream Burn device-dispatch support;
- Flex CPU forward, autodiff and tiny supervised-router training tests.

## Alignment with the PTR cognitive type kernel

Burn A0 already keeps slot-type, epistemic, validity, provenance and confidence metadata in separate channels. That is directionally aligned with the target architecture.

The remaining alignment work is explicit:

- replace opaque research-local `slot_type` semantics with the shared `SemanticRole` taxonomy;
- add a separate `UncertaintyKind` channel instead of folding distribution semantics into slot type or epistemic state;
- map `SemanticRole`, `EpistemicState`, `UncertaintyKind` and `ReasoningOperator` through a versioned cognitive type codebook;
- record the codebook version in datasets/checkpoints/runs;
- do not use Rust enum discriminants directly as persistent tensor IDs.

The current integer category counts are therefore research-local and must not be treated as a frozen wire/checkpoint schema.

## Not implemented yet

- pretrained language backbone;
- explicit lifecycle-generation masks inside neural attention;
- branch/distribution latent state and explicit uncertainty-kind embeddings;
- ActionIR/verifier neural heads;
- full language modeling / multi-objective PTR loss;
- checkpoint import/export and Torch numerical parity;
- CUDA/CubeCL benchmark evidence.

This isolated prototype pins **Burn 0.22.0-pre.3**, uses **Rust 1.95**, and keeps its own lockfile. This explicit prerelease/device-API migration removes the old bincode recording dependency. It does not raise the default runtime core's Rust 1.85 gate or establish a production framework choice. Training uses `Device::flex().autodiff()`.

The synthetic router training test proves only that gradients and optimizer updates flow through the typed metadata, cross-attention, recurrence and router path. It is **not** evidence for M001/M002/M003/M004 task superiority.

The attention bias is query/key-dependent: a query-only constant would cancel in softmax. A cancellation control and a gradient test cover that regression. Validity IDs still do not implement a hard revocation mask. See the security/P0 review for remaining gates.
