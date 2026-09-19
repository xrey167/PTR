# rust-lang/rust-mode

Repository: https://github.com/rust-lang/rust-mode

## Role

Lightweight Emacs Rust editing candidate providing Rust syntax/indentation plus rustfmt, Cargo and Clippy integration. It also supports a native tree-sitter-derived mode and documents Eglot/lsp-mode integration with rust-analyzer.

## PTR use

- optional contributor editor mode;
- expose PTR's normal Cargo commands rather than inventing editor-only semantics;
- keep format/check/test behavior identical to CI;
- do not depend on editor state for generated files or architecture metadata.

## Constraints

- developer-only dependency;
- editor support must degrade to normal CLI workflows;
- Rust Analyzer remains language intelligence, not a PTR architectural backend;
- license: MIT OR Apache-2.0.

## Evaluation

See [rust-editor-emacs](../../evaluations/components/rust-editor-emacs/README.md).
