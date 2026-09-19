# rust-editor-emacs

Optional developer-experience evaluation for Rust editing in Emacs.

This component has **no runtime authority** and no selection is required for PTR to build or operate. The purpose is to keep contributor tooling reproducible without coupling project semantics to an editor.

## Candidate profiles

- rust-mode
- rust-mode + cargo-mode
- rust-mode + rustic
- rustic + cargo-mode only as an overlap experiment

Compare command parity, maintenance burden, rust-analyzer integration, Cargo/test ergonomics, macro workflows, remote behavior and conflicts before documenting a recommended profile.
