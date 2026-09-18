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
- generic Burn backend support;
- NdArray forward, autodiff and tiny supervised-router training tests.

## Not implemented yet

- pretrained language backbone;
- explicit lifecycle-generation masks inside neural attention;
- branch/distribution latent state;
- ActionIR/verifier neural heads;
- full language modeling / multi-objective PTR loss;
- checkpoint import/export and Torch numerical parity;
- CUDA/CubeCL benchmark evidence.

This package uses Burn **0.18.0** because that release declares Rust 1.85 compatibility, matching PTR's current MSRV. The separate lockfile pins the dependency graph, including a Rust-1.85-compatible bytemuck version.

The synthetic router training test proves only that gradients and optimizer updates flow through the typed metadata, cross-attention, recurrence and router path. It is **not** evidence for M001/M002/M003/M004 task superiority.
