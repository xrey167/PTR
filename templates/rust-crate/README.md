# PTR Rust Crate Template

Reference source layout for new PTR crates and large-crate refactors.

This template is intentionally not a workspace member. Copy only modules the component needs; do not create empty layers for symmetry.

Key properties:

- thin `lib.rs` facade;
- private modules plus deliberate `pub use`;
- `pub(crate)` for internal cross-module helpers;
- typed request/options values instead of ambiguous long argument lists;
- `derive` based on actual value semantics;
- closed state represented with enums and exhaustive `match`;
- generic `T` only where behavior is genuinely type-independent;
- `Option<T>` for legitimate absence and `Result<T, E>` for failure;
- `HashMap` only where unordered keyed lookup is the intended semantics;
- typed error enums with machine-readable variants/codes and `expected`/`actual` fields for mismatches;
- `Result<T, E>` for expected failure paths, `Option<T>` for legitimate absence, and no production `unwrap()` for recoverable failures;
- a structured `TraceSink` boundary so `tracing`, OpenTelemetry or another backend can be swapped without changing domain code;
- unit tests beside implementation modules, integration tests under `tests/`, and shared fixtures in `tests/common/mod.rs`;
- `check_*`/`validate_*` helpers returning typed `Result` values with named failure fields/messages;
- lazy `impl Iterator` views, `IntoIterator` batch inputs, standard iterator adapters and closure bounds (`Fn`/`FnMut`/`FnOnce`) where sequence semantics fit;
- exhaustive `match` for closed semantic states and match guards for cheap deterministic refinements such as empty/range/threshold conditions;
- private `macro_rules!` for internal compile-time boilerplate and hygienic `$crate`-based `#[macro_export]` only for intentional public macro APIs.

See `docs/RUST_API_STYLE.md`.

## Merge-review regression

The empty BackendRegistry implements Default manually: creating an empty map
does not require a default backend value. A non-Default entry regression and the
public Service tests cover this requirement, including Arc<dyn Backend<...>>.
The template is tested on Rust stable and the project's Rust 1.85 MSRV. Lazy
name filters rely on ordinary coercion rather than an unnecessary explicit dereference.

The filtered iterator has a separate lifetime for its predicate captures. A
collected `&str` borrows the service, not the closure. Public regression tests
retain collected names after captured locals leave scope and inspect the FnMut
counter before using those names. This prevents an unnecessarily long mutable
borrow from becoming part of the public iterator contract.
