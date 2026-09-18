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

## 9. Traits, bounds, implementors and adapters

PTR owns contracts. Backends implement them.

```rust
pub trait ObjectStore: Send + Sync {
    type Reader<'a>: std::io::Read + 'a
    where
        Self: 'a;

    fn put(&mut self, request: PutObject) -> Result<ObjectRef, StorageError>;

    fn get(&self, id: &ArtifactId) -> Result<Option<Object>, StorageError>;
}
```

### Trait rules

- traits describe a semantic contract, not a provider API;
- use associated types when an implementation owns a related type family;
- use generic parameters when the caller chooses the type;
- use supertraits such as `Send + Sync` only when the contract genuinely requires them;
- object safety is deliberate when `dyn Trait` is part of the architecture;
- sealed traits are acceptable when external implementation would violate invariants;
- marker traits are used sparingly and must carry a clear semantic meaning;
- prefer several focused traits over one large trait with unrelated methods.

### Bounds

Bounds belong as close as possible to the operation that needs them.

Prefer:

```rust
pub fn execute<TInput, TOutput, TBackend>(
    backend: &TBackend,
    input: TInput,
) -> Result<TOutput, ServiceError>
where
    TBackend: Backend<TInput, TOutput>,
    TInput: Send,
    TOutput: Send,
{
    backend.execute(input)
}
```

over placing `Clone + Send + Sync + 'static + Debug` on every generic type "just in case".

Use `where` clauses when bounds are non-trivial or relational. Avoid `'static` unless ownership/thread/task requirements actually need it.

### Implementors

Implementors are part of the replaceability model and should be easy to discover:

```rust
pub struct InMemoryStore {
    // ...
}

impl ObjectStore for InMemoryStore {
    // ...
}
```

Reference implementations belong in PTR-owned modules; provider implementations belong in `adapters::<provider>` or feature-gated sibling modules.

A trait should document:

1. semantic contract;
2. required invariants;
3. expected failure behavior;
4. concurrency/object-safety expectations;
5. known reference implementor;
6. backend implementors/adapters.

The API inventory tool should expose both the trait declaration and `impl Trait for Type` relationships.

Provider-specific implementations keep backend-native types private whenever possible.

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

## 10. Lifetimes and relations

Lifetimes express real borrowing relationships; they are not added for style.

Prefer explicit relational names when more than one lifetime exists:

```rust
pub struct SnapshotView<'snapshot> {
    pub revision: Revision,
    pub bytes: &'snapshot [u8],
}

pub struct QueryContext<'snapshot, 'query>
where
    'snapshot: 'query,
{
    pub snapshot: &'query SnapshotView<'snapshot>,
    pub query: &'query str,
}
```

Rules:

- use elision for simple one-input-reference cases;
- name lifetimes semantically (`'snapshot`, `'request`, `'store`) when relationships matter;
- use `'a` only for small/local APIs where the relation is obvious;
- encode outlives relations with `'long: 'short` only when required;
- prefer owned values/IDs/handles across async task, process, network or persistence boundaries;
- avoid self-referential structures unless there is a measured need and a clearly proven invariant;
- avoid returning references tied to locks/guards when an owned snapshot/handle is safer;
- `'static` means no borrowed lifetime dependency, not "lives forever".

For async APIs, do not force borrowed data across suspension points unless the lifetime relationship is intentional and testable.

## 11. Constructors and mutation

Use constructors when an invariant must be checked. Direct public fields are acceptable for transparent value records with no invalid state.

Prefer immutable access and explicit mutation methods. Mutation that affects lifecycle, authority, revisions or generations must use a named operation rather than direct field replacement.

## 12. Error design, expected values and failure propagation

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

## 13. Structured tracing

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

## 14. Function grouping

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

## 15. Check, validate and test functions

Use function names to distinguish semantics:

- `is_*`, `has_*`, `can_*` — infallible boolean predicate with no diagnostic detail required;
- `check_*` — verify one condition/invariant and return `Result<(), E>`;
- `validate_*` — validate a complete value/config/request and return `Result<(), E>` or a typed validation report;
- `verify_*` — evidence/semantic verification, normally returning a domain verification type rather than a boolean;
- `try_*` — attempt an operation where failure is expected and represented by `Result`/`Option`;
- `ensure_*` — internal precondition helper returning `Result<(), E>`, used sparingly;
- `test_*` is reserved for test helper naming; Rust test functions themselves should name the behavior, e.g. `stale_generation_returns_expected_and_actual`.

Example:

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    MissingField {
        field: &'static str,
        message: &'static str,
    },
    OutOfRange {
        field: &'static str,
        expected: &'static str,
        actual: usize,
        message: &'static str,
    },
}

pub fn check_backend_name(name: &str) -> Result<(), ValidationError> {
    if name.trim().is_empty() {
        return Err(ValidationError::MissingField {
            field: "backend",
            message: "backend name must not be empty",
        });
    }
    Ok(())
}

pub fn validate_request(request: &ExecuteRequest) -> Result<(), ValidationError> {
    check_backend_name(&request.backend)?;
    Ok(())
}
```

Validation failures use named fields and messages. Do not return `false` when the caller needs to know why validation failed.

Checks should be:

- deterministic;
- side-effect free unless their name/documentation says otherwise;
- individually testable;
- composed with `?`;
- traced once at the ownership boundary when failure matters operationally.

## 16. Tests

Unit tests live next to private implementation when they need private access. Cross-module and public-contract tests live in `tests/`.

Tests should exercise enum variants and match branches, `None`/ `Some`, success/error `Result` paths, collection invariants and generic implementations where they carry architecture semantics. The full unit/integration/common-module layout is defined in [TESTING.md](TESTING.md).

## 17. Stability rule

PTR is still in architecture discovery. Public Rust visibility does not automatically mean "frozen forever". Before v0 contract freeze:

- keep provider details private;
- keep module boundaries replaceable;
- expose the smallest useful API;
- record open API/type decisions in `research/catalogs/`;
- avoid compatibility commitments that have not been evaluated.
