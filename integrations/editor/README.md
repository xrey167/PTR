# Editor / Developer Integrations

Editor integrations improve contributor ergonomics but are **not** PTR runtime architecture and may never become build/runtime requirements.

## Rust / Emacs candidates

- [rust-mode](rust-mode.md) — lightweight Rust major mode and basic Cargo/rustfmt/Clippy integration.
- [rustic](rustic.md) — maintained, broader Rust development environment based on rust-mode.
- [cargo-mode](cargo-mode.md) — Cargo-focused minor mode that can complement a lighter Rust editing stack.

The duplicate rustic link supplied during research is represented once.

## Composition rule

Evaluate **profiles**, not only packages:

1. minimal: rust-mode + rust-analyzer via Eglot/lsp-mode as selected by the developer;
2. cargo-focused: rust-mode + cargo-mode;
3. integrated: rust-mode + rustic;
4. integrated-plus: rustic + cargo-mode only if measured ergonomics justify overlapping Cargo command surfaces.

No profile is a repository requirement. Project correctness remains defined by CLI/CI commands.
