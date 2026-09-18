# PTR Rust Module and API Style

PTR uses Rust's type system as part of the architecture. Code organization must make semantic ownership, public contracts and backend replaceability visible from the source tree.

## 1. Crate layout

A crate should not grow as a single large `lib.rs`. Prefer responsibility modules:

```text
src/
  lib.rs          # thin facade: mod declarations + deliberate re-exports
  types.rs        # domain/value types owned by the crate
  error.rs        # crate-specific error enums
  ports.rs        # PTR-owned traits/contracts
  service.rs      # orchestration/use-case logic
  state.rs        # local state machine / state holder when applicable
  policy.rs       # deterministic policy/rules when applicable
  adapters/
    mod.rs
    <backend>.rs  # provider-native types confined here
```

The exact module set is per component; unused modules are not created only for symmetry.

### `lib.rs` rule

Prefer:

```rust
mod error;
mod ports;
mod service;
mod types;

pub use error::RuntimeError;
pub use ports::{RuntimeLedger, RuntimeModel};
pub use service::PtrRuntime;
pub use types::{RunRequest, RunResult};
```

Use `pub mod x;` only when callers are intentionally expected to address the module namespace itself. Otherwise keep modules private and re-export the stable contract.

## 2. Visibility

Visibility is an architectural decision.

- private — default for implementation details;
- `pub(super)` — parent-module implementation cooperation;
- `pub(crate)` — internal cross-module API inside one crate;
- `pub` — intentional cross-crate/public contract only.

Do not add `pub` to make tests or implementation convenient. Prefer crate-local tests or deliberate test helpers.

Public structs should not expose backend/provider-native fields. If a provider value must cross an adapter boundary, convert it to a PTR-owned type first.

## 3. Imports with `use`

Prefer imports over repeated long paths. Group them conceptually:

```rust
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use external_crate::ExternalType;

use ptr_types::{CapabilityId, Generation, Revision};

use crate::error::RuntimeError;
use crate::types::RunRequest;
```

Within an implementation, `Self::Variant` is preferred when matching the current enum.

Avoid wildcard imports in production code except for intentionally designed preludes. A facade may use explicit grouped re-exports; avoid `pub use module::*` once an API becomes stable.

## 4. Function parameters

Rust does not support labelled/named call arguments. PTR obtains the same readability through:

1. descriptive parameter names;
2. newtypes for semantically distinct scalar/string values;
3. typed parameter structs for multi-field calls;
4. enums instead of ambiguous boolean flags.

Prefer:

```rust
#[derive(Clone, Debug)]
pub struct RunRequest {
    pub request_id: RequestId,
    pub revision: Revision,
    pub input: String,
    pub budget: ReasoningBudget,
}

pub fn run(&mut self, request: RunRequest) -> Result<RunResult, RuntimeError> {
    // ...
}
```

over:

```rust
pub fn run(&mut self, id: String, rev: u64, text: String, deep: bool, max: usize) {
    // ...
}
```

Small functions with one to three obvious parameters remain normal Rust functions. Do not create parameter structs mechanically where they reduce clarity.

## 5. `derive`

Use derives to make value semantics explicit, not as decoration.

Common defaults:

- `Debug` for almost every public domain/config/error type;
- `Clone` where duplication is semantically valid;
- `Copy` only for genuinely small copy-value types;
- `PartialEq` / `Eq` when equality has clear meaning;
- `Hash` when a type is a valid hash key;
- `Ord` / `PartialOrd` only when ordering is meaningful;
- `Default` only when there is a real semantic default;
- serialization derives only at explicit persistence/wire/config boundaries.

Example:

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Generation(pub u64);
```

Do not derive `Default` merely to make construction convenient if a missing value would violate an invariant.

## 6. Enums and exhaustive `match`

Use enums for closed semantic state spaces and lifecycle states.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendState {
    Ready,
    Degraded,
    Unavailable,
}

pub fn can_route(state: BackendState) -> bool {
    match state {
        BackendState::Ready => true,
        BackendState::Degraded => true,
        BackendState::Unavailable => false,
    }
}
```

Prefer exhaustive matches when all variants matter. Avoid catch-all `_` when it would hide a newly added semantic variant. A wildcard is acceptable for intentionally irrelevant data or non-exhaustive external enums.

Use data-carrying variants when the states need different payloads:

```rust
pub enum LookupResult<T> {
    Hit(T),
    Miss,
    Stale { generation: Generation },
}
```

## 7. `Option<T>`, `Result<T, E>` and generics

Use `Option<T>` only when absence is a valid state.

- `Option<Generation>` — valid when an object may have no live generation;
- not `Option<bool>` when a three-state enum is clearer.

Use `Result<T, E>` for fallible operations with actionable failure semantics. Prefer typed error enums over free-form strings at stable boundaries.

Use generics when behavior is genuinely type-independent:

```rust
pub struct TypedValue<T> {
    pub value: T,
    pub generation: Generation,
}

pub trait Verifier<T> {
    fn verify(&self, candidate: &T) -> VerificationReport;
}
```

Prefer meaningful generic names for multi-parameter APIs (`TInput`, `TOutput`, `TBackend`) when bare `T`/ `E` would be ambiguous.

## 8. Collections

Choose a collection by semantics:

- `HashMap<K, V>` — fast unordered lookup where iteration order is irrelevant;
- `BTreeMap<K, V>` — deterministic ordered keys, stable iteration or range queries;
- `HashSet<T>` — unordered membership;
- `BTreeSet<T>` — deterministic ordered membership;
- `Vec<T>` — ordered sequence;
- `VecDeque<T>` — queue/deque;
- dedicated newtype/index structure — when the collection itself has domain meaning.

Do not use a `HashMap<String, String>` as a substitute for a domain model. Collections contain typed values; they do not replace them.

For reproducible tests, manifests, snapshots and hashes, deterministic collections are usually preferred unless ordering is explicitly normalized before serialization/hashing.

## 9. Traits / ports and adapters

PTR owns contracts. Backends implement them.

```rust
pub trait ObjectStore {
    fn put(&mut self, request: PutObject) -> Result<ObjectRef, StorageError>;
    fn get(&self, id: &ArtifactId) -> Result<Option<Object>, StorageError>;
}
```

Provider-specific implementations belong in `adapters::<provider>` or feature-gated sibling modules. Backend-native types remain private to the adapter whenever possible.

Prefer capability descriptors over backend-name branching:

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub streaming: bool,
    pub resumable: bool,
    pub structured_events: bool,
}
```

A backend name is metadata; it is not semantic routing authority.

## 10. Constructors and mutation

Use constructors when an invariant must be checked. Direct public fields are acceptable for transparent value records with no invalid state.

Prefer immutable access and explicit mutation methods. Mutation that affects lifecycle, authority, revisions or generations must use a named operation rather than direct field replacement.

## 11. Error design, expected values and failure propagation

Stable crate APIs expose typed error enums. A mismatch carries **typed expected and actual values** whenever those values are meaningful:

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LookupError {
    BackendUnavailable {
        backend: String,
    },
    StaleGeneration {
        expected: Generation,
        actual: Generation,
    },
    InvalidState {
        expected: BackendState,
        actual: BackendState,
    },
}
```

This is preferred over `Err("generation mismatch".into())`: callers can exhaustively `match`, tests can assert exact values, and tracing can record structured fields.

Rust has no standard `Expected<T>` success type analogous to C++ `std::expected`; PTR uses `Result<T, E>`. Use `Option<T>` when absence is valid and not itself an error.

### Propagation

- use `?` to propagate errors while preserving typed semantics;
- convert provider errors at the adapter boundary;
- add PTR context at the boundary where it becomes meaningful;
- do not silently convert an error to `None`, a default, or an empty collection;
- retries must be explicit and only for errors classified as retryable;
- cancellation/timeouts get explicit variants rather than generic backend strings;
- stable public errors should expose a code/variant and typed fields; human-readable detail is supplementary.

### `unwrap`, `expect` and panic

Production/library paths should not use `unwrap()` for recoverable input, I/O, backend, parsing, lifecycle or concurrency failures.

`expect("...")` is allowed only when the preceding logic establishes an invariant and the message states that invariant, for example after an append method that guarantees a committed event exists. Prefer removing the panic entirely if the invariant can cheaply be represented as a `Result`.

Tests, examples, benchmarks and build scripts may use `unwrap`/`expect` when failure should abort the test/tool, but messages should remain diagnostic.

`panic!` is reserved for impossible internal states, explicit failpoints and process-level startup policies where returning an error is not possible or useful.

## 12. Structured tracing

PTR uses structured tracing, not ad-hoc logging, for runtime behavior. The architecture-facing contract lives in `ptr-observe`; concrete `tracing`, OpenTelemetry and exporter layers remain replaceable.

Recommended span hierarchy:

```text
request
  model / route
    operator / pod / search / verifier
      effect / commit / projection
```

Trace fields should use stable PTR names and typed values converted only at the recording boundary. Useful fields include request ID, revision, generation, capability, Pod/backend ID, effect, commit index, operation, outcome, error code and latency.

For expected/actual failures, record both values:

```text
error.code = "stale_generation"
expected.generation = 8
actual.generation = 7
```

Rules:

- never emit secrets, raw private evidence, credentials or full prompts by default;
- `error!` means a failed operation requiring attention;
- `warn!` means degraded/retry/fallback/disputed state;
- `info!` means lifecycle/business-significant transitions;
- `debug!` means diagnostic execution detail;
- `trace!` is very high-volume local detail;
- errors should be recorded once at the ownership boundary rather than repeatedly at every `?`;
- observability failure must not bypass or block hard safety checks;
- trace data is telemetry, not causal authority.

Use `println!` only for intentional CLI/stdout protocols, benchmark machine output, Cargo build-script directives, or user-facing terminal output.

## 13. Function grouping

Inside modules, keep a predictable order when practical:

1. imports;
2. constants/type aliases;
3. public types/enums;
4. public traits;
5. inherent `impl` blocks;
6. public free functions;
7. crate-private/private helpers;
8. feature-gated adapters;
9. tests.

Within an `impl`, prefer constructor/accessors first, then primary operations, then private helpers.

Large groups of unrelated free functions are a signal to create a module/type/trait.

## 14. Tests

Unit tests live next to private implementation when they need private access. Cross-module and public-contract tests live in `tests/`.

Tests should exercise enum variants and match branches, `None`/ `Some`, success/error `Result` paths, collection invariants and generic implementations where they carry architecture semantics.

## 15. Stability rule

PTR is still in architecture discovery. Public Rust visibility does not automatically mean "frozen forever". Before v0 contract freeze:

- keep provider details private;
- keep module boundaries replaceable;
- expose the smallest useful API;
- record open API/type decisions in `research/catalogs/`;
- avoid compatibility commitments that have not been evaluated.
