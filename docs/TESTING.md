# Testing Layout

Every first-class PTR workspace area owns a `tests/` directory. Rust crates use Cargo integration tests; research/config/data areas use the directory for fixtures, validators and executable checks.

A tests directory is organizational structure, not evidence by itself. The component dashboard counts Rust `#[test]` markers, while experiment/evaluation results must still contain reproducible artifacts.

## Rust crate test structure

Prefer this layout when a crate has more than trivial coverage:

```text
src/
  lib.rs
  <module>.rs
  ...
tests/
  common/
    mod.rs
  public_contract.rs
  lifecycle.rs
  failure.rs
  adapters.rs
```

Use the layers for different purposes.

### Unit tests

Unit tests live beside the implementation they exercise:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_generation_reports_expected_and_actual() {
        // ...
    }
}
```

Use unit tests for:

- private helpers and internal invariants;
- exhaustive enum/match behavior;
- constructors and validation helpers;
- typed error variants and expected/actual values;
- boundary conditions that do not need the public crate surface.

Unit tests may access private items through `super::*`. Do not make an item `pub` merely so an external integration test can reach it.

### Integration tests

Cargo integration tests live under `tests/*.rs` and must exercise the crate through its public API.

Use integration tests for:

- public contracts and re-exports;
- cross-module behavior;
- backend/adapter conformance;
- persistence/reopen/replay;
- process/network boundaries;
- feature-gated behavior;
- failure semantics visible to callers.

A file such as `tests/lifecycle.rs` is its own integration-test crate.

### Shared test code

Common fixtures/builders/assertions belong in `tests/common/mod.rs`:

```rust
pub fn request() -> ExecuteRequest<String> {
    // ...
}
```

Each integration-test crate that uses it declares:

```rust
mod common;
```

Rules for `tests/common`:

- no `#[test]` annotations;
- no production semantics;
- builders, fixtures, temporary-path helpers and reusable assertions only;
- helpers return `Result` when setup can fail;
- failure messages remain named and diagnostic;
- keep helpers deterministic unless a test explicitly targets nondeterminism.

If a helper becomes useful to production code, move the underlying behavior into the crate rather than promoting test code into production by accident.

### Test modules

Within a larger unit-test block, group behavior by submodule when useful:

```rust
#[cfg(test)]
mod tests {
    mod validation {
        use super::super::*;

        #[test]
        fn empty_name_is_rejected() {
            // ...
        }
    }

    mod lifecycle {
        use super::super::*;

        #[test]
        fn revoked_state_cannot_transition_to_ready() {
            // ...
        }
    }
}
```

Do not create nested modules solely for visual symmetry; group by meaningful behavior.

## Test function naming

Test function names state the observable behavior, not an implementation detail.

Prefer:

```rust
#[test]
fn revoked_generation_is_rejected_after_reopen() { ... }

#[test]
fn invalid_backend_name_returns_named_validation_error() { ... }
```

Avoid names such as `test_1`, `works`, or `basic_test`.

Use:

- `#[test]` for synchronous tests;
- `#[tokio::test]` only when the behavior is genuinely async;
- feature-gated test files/modules for optional adapters;
- ignored tests only for intentionally expensive/manual cases, with a reason.

## Assertions and failures

Prefer exact, typed assertions:

```rust
assert_eq!(
    result,
    Err(ValidationError::MissingField {
        field: "preferred_backend",
        message: "preferred backend must be selected before execution",
    })
);
```

For fallible setup in tests, `expect("named invariant/setup message")` is acceptable when failure should abort the test. Production/library code follows the stricter rules in `docs/RUST_API_STYLE.md`.

Tests for fallible APIs should cover at least:

1. success;
2. each meaningful error variant;
3. expected/actual mismatch fields;
4. boundary/empty values;
5. retry/cancellation/timeouts where applicable;
6. tracing/error-code behavior where it is part of the contract.

## Common test categories

As a crate matures, use descriptive integration files such as:

- `public_contract.rs`
- `validation.rs`
- `lifecycle.rs`
- `failure.rs`
- `replay.rs`
- `concurrency.rs`
- `serialization.rs`
- `adapter_<name>.rs`

A generic `smoke.rs` is acceptable for an initial scaffold, but should be split once it starts testing multiple unrelated behaviors.
