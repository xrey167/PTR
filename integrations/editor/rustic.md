# emacs-rustic/rustic

Repository: https://github.com/emacs-rustic/rustic

## Role

Maintained Rust development environment for Emacs based on rust-mode. Candidate features relevant to PTR include Cargo command UI, test-at-point, Clippy, macro expansion, rustfmt and automatic LSP/Eglot configuration.

## PTR use

- optional integrated Emacs profile;
- useful when a contributor wants one cohesive Rust/Cargo/LSP workflow;
- Cargo commands should map to repository-standard locked/workspace commands where appropriate;
- macro expansion is especially useful while PTR develops macro_rules-based APIs.

## Constraints

- overlaps with rust-mode Cargo helpers and cargo-mode;
- do not auto-enable overlapping Cargo UIs before evaluation;
- editor automation must not bypass repository checks;
- license: MIT OR Apache-2.0.

## Evaluation

See [rust-editor-emacs](../../evaluations/components/rust-editor-emacs/README.md).
