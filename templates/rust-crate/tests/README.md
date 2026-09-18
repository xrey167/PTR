# Template tests

This reference crate demonstrates the PTR test split:

- private/internal invariants use `#[cfg(test)] mod tests` beside source modules;
- `tests/public_contract.rs` exercises only the public crate contract;
- `tests/validation.rs` covers named fallible validation/error behavior;
- `tests/common/mod.rs` contains fixtures/builders only and must not contain `#[test]` functions.

Add thematic integration files such as `lifecycle.rs`, `failure.rs`, `replay.rs`, `concurrency.rs` or `adapter_<name>.rs` as the component grows. Avoid turning one generic smoke test into an unrelated test collection.
