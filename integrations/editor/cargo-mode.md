# ayrat555/cargo-mode

Repository: https://github.com/ayrat555/cargo-mode

## Role

Cargo-focused Emacs minor-mode candidate. It dynamically exposes Cargo tasks and includes direct commands for build, test, current-buffer/current-test execution and re-running the last command.

## PTR use

- optional addition to a lightweight rust-mode profile;
- useful for test-at-point and Cargo command selection;
- configure commands to preserve PTR's workspace/locked/MSRV expectations rather than creating a separate editor-only workflow.

## Constraints

- overlaps with rustic's Cargo/test command surface;
- no runtime/build dependency on Emacs or cargo-mode;
- license: MIT.

## Evaluation

See [rust-editor-emacs](../../evaluations/components/rust-editor-emacs/README.md).
