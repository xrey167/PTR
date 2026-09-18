# Testing Layout

Every first-class PTR workspace area owns a `tests/` directory. Rust crates use Cargo integration tests; research/config/data areas use the directory for fixtures, validators and executable checks.

A tests directory is organizational structure, not evidence by itself. The component dashboard counts Rust `#[test]` markers, while experiment/evaluation results must still contain reproducible artifacts.
