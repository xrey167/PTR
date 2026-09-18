# PTR Burn A0

This is the first **trainable tensor implementation** of a PTR model component. It is deliberately kept outside the production workspace while the model-framework evaluation remains open.

## Implemented

- token embeddings for the raw path;
- slot-type embeddings for the typed path;
- bidirectional raw↔slot cross-attention;
- residual projection back into raw and semantic workspaces;
- operator-router logits over the updated semantic slots;
- generic Burn backend support;
- NdArray forward and autodiff smoke tests.

## Not implemented yet

- pretrained language backbone;
- epistemic/validity/provenance attention biases;
- recurrent latent reasoning;
- branch/distribution state;
- ActionIR/verifier heads;
- full language modeling loss;
- CUDA/CubeCL benchmark evidence.

This package uses Burn **0.18.0** because that release declares Rust 1.85 compatibility, matching PTR's current MSRV. Newer Burn releases require a newer Rust toolchain and must be evaluated before migration.
