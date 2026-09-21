# PTR Burn A0

This is the first **trainable tensor implementation** of a PTR model component. It is deliberately kept outside the production workspace while the model-framework evaluation remains open.

## Implemented

- token embeddings for the raw path;
- slot-type embeddings for the typed path, one row per `SemanticRole` in the shared codebook;
- epistemic-state embeddings, one row per `EpistemicState` in that codebook;
- research-local provenance-bucket embeddings, the one width still chosen rather than derived;
- confidence projection into the typed latent space;
- learned typed metadata bias on both raw→slot and slot→raw attention;
- bidirectional raw↔slot cross-attention;
- residual projection back into raw and semantic workspaces;
- shared recurrent latent refinement steps;
- operator-router logits over the updated semantic slots, one logit per `ReasoningOperator` in that codebook;
- enforced lifecycle admission: a `ValidityMask` from committed state enters as a negative-infinity attention bias and weights the router's mean, so an excluded slot cannot reach the output at all;
- checkpoints written and read with the shared `CheckpointHeader`, so weights cannot load under an assignment they were not trained under;
- upstream Burn device-dispatch support;
- Flex CPU forward, autodiff and tiny supervised-router training tests.

## Alignment with the PTR cognitive type kernel

Slot types, epistemic states and operator routing now go through the versioned
cognitive codebook in `ptr-types`. Every typed width is that family's cardinality —
nine semantic roles, six epistemic states, eleven reasoning operators — rather than
a number a caller passed, and codes are minted through `Codebook::code_of` inside a
`CodeGrid<T>`, so an index past the end of a table or a code from another family is
unrepresentable rather than unlikely. A code is a position in a versioned table and
never a Rust discriminant.

The remaining alignment work is explicit:

- add a separate `UncertaintyKind` channel instead of folding distribution semantics into slot type or epistemic state;
- `provenance_bucket_count` stays outside the codebook by decision: provenance bucketing is a research-local hashing of sources with no kernel taxonomy to size it from. Its width is no longer unchecked — it is a recorded exception in `ptr_types::EXCEPTIONS`, published in the generated artifact, and resolved into this crate's default by a `const fn`, so removing the record fails the build. What remains true is narrower: a checkpoint written at another width is refused **by tensor shape**, through burn's record validation, and the identity header never consults the width at all because the width is not a code family. Both are asserted in `tests/checkpoint.rs`;
- the codebook version travels with every `CodeGrid` and is compared in `forward`. That comparison is exercised by `codebook_guard_tests` in `src/lib.rs`, which relabels a well-formed grid as version 2 from inside the crate and drives both guards, each asserted on its own full panic message because the two share a phrase. Still weaker than it sounds: it is a test-only constructor, not a second version, so the guard has never met a real foreign artifact.

## Toolchain

This workspace is excluded from the root workspace and requires **1.95**
(`Cargo.toml`), which is why `.github/workflows/burn-a0.yml` runs it on stable and
on 1.95.0 rather than on the repository's 1.85.0 MSRV.

It carries its own `clippy.toml`. Clippy takes its MSRV from the nearest
`clippy.toml` walking upward and that file wins over `rust-version`, so without
one here the crate was linted at the root's `1.85.0` and every lint whose
suggestion needs a newer compiler was suppressed. Clippy does report the
disagreement on every run, but the message is not a named lint, so `-D warnings`
cannot promote it and the job exits 0 either way.
`scripts/check_msrv_alignment.py` is what keeps the two equal.

## Not implemented yet

- pretrained language backbone;
- lifecycle *generation* masks: admission is per-slot validity, and a generation is not yet an input to attention;
- branch/distribution latent state and explicit uncertainty-kind embeddings;
- ActionIR/verifier neural heads;
- full language modeling / multi-objective PTR loss;
- Torch numerical parity, and any checkpoint interchange beyond this workspace's own burnpack payload;
- a training loop that binds a checkpoint as it saves one: the header carries the assignment, and attaching committed facts is something `ptr-runtime` does afterwards;
- CUDA/CubeCL benchmark evidence.

This isolated prototype pins **Burn 0.22.0-pre.3**, uses **Rust 1.95**, and keeps its own lockfile. This explicit prerelease/device-API migration removes the old bincode recording dependency. It does not raise the default runtime core's Rust 1.85 gate or establish a production framework choice. Training uses `Device::flex().autodiff()`.

`examples/write_checkpoint.rs` regenerates `crates/ptr-runtime/tests/fixtures/ptr-a0-v1.ckpt`, the artifact the production runtime's tests bind. The two workspaces do not build together, so a committed artifact is the only thing that would catch them disagreeing about the format.

The synthetic router training test proves only that gradients and optimizer updates flow through the typed metadata, cross-attention, recurrence and router path. It is **not** evidence for M001/M002/M003/M004 task superiority.

The attention bias is query/key-dependent: a query-only constant would cancel in softmax. A cancellation control and a gradient test cover that regression.

Lifecycle validity is no longer a learned id. It was one — an embedding summed into the typed metadata, which made it a hint the rest of the network could outvote, and the fact being outvoted was "this generation was revoked". It is now applied as admission and cannot be weighed. See `docs/architecture/26-cognitive-codebook.md` and the security/P0 review for what remains.
