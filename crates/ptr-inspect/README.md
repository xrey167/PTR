# ptr-inspect — Structured Runtime Introspection

> **Role:** Allows generic inspection and rendering of typed Rust values without collapsing the runtime into untyped JSON.  
> **Maturity:** see the generated status below; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-10-14  
**Code footprint:** 8 Rust source files · 2126 nonblank source lines · 2 integration-test files · 46 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- Generic InspectNode tree with lossless integers (Int(i128), UInt(u128)), float Number, Bytes (length, FNV-1a 64 fingerprint, 32-byte prefix), Map and Enum nodes
- Inspectable trait
- Secret<T> that always renders Redacted, prints Secret([redacted]) through Debug for any T, and keeps its value in a private field reached only through expose/into_inner
- Bounded pre-order Visitor/walk with InspectPath locations, enter/leave callbacks for struct, list, map and enum, and typed WalkError for depth and node-count limits
- Inspectable impls for bool, all integers, floats, char, str/String, (), Option, Result, Vec, slices, arrays, BTreeMap/HashMap/BTreeSet/HashSet (hash collections sorted by rendered key), tuples up to 4, references, Box, Rc and Arc
- Inspectable impls for ptr-types revisions, generations, timestamps, string identifiers, Probability, SemanticRole, EpistemicState and the other fieldless semantic enums, ConfidenceTarget/Estimate, ProvenanceRef, TypedValue, Epistemic and ActionIr
- inspect_struct! declarative macro listing the fields of a user struct
- Structural diff of two trees by struct field name, map key and list index (index-wise, no LCS) reporting Added, Removed, Changed and TypeChanged; two redaction markers compare equal and a redacted-versus-plaintext mismatch is reported with both sides redacted
- Private<T> (redaction marker with type name and optional size class) and RawEvidence<T> (length and FNV-1a 64 fingerprint, truncated preview only under PreviewPolicy::Allow, capped at 256 bytes), both with non-leaking Debug
- Key-based redaction passes (redact_keys exact, redact_keys_containing and redact_sensitive by fragment, case-insensitive) over struct fields, enum fields and text-keyed map entries
- Deterministic indented tree renderer, single-line renderer (width, depth and item limits) and a hand-written JSON renderer with control-character escaping and non-finite floats as strings

### Missing for the target architecture

- Valuable bridge (no valuable dependency or feature exists)
- Tracing bridge
- ptrctl inspect command wiring (the renderers exist; bins/ptrctl does not call them)
- Inspectable impls for ptr-types pod addressing, checkpoint, codebook and slot-encoding types
- List alignment (LCS) in diff and bounds on the size of a tree while it is being built

### Next milestones

- Expose read-only ptrctl inspect command over render_tree and render_json
- Implement Valuable adapter behind a feature flag if the dependency is added to the lockfile
- Add inspect impls for the remaining ptr-types structures

### Linked experiments

- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`

### Technology evaluations

- [introspection](../../evaluations/components/introspection/README.md) — `open`

### Decision records

- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)
- [ADR-0010-component-docs-as-code.md](../../research/decisions/ADR-0010-component-docs-as-code.md)

### Current automated checks

- Debug and alternate Debug of a Secret, alone and nested in a derived Debug, never contain the value
- Secret, Private and RawEvidence content is absent from Debug, tree, compact and JSON output and from diffs
- Visitor order, depth and node limits, every diff kind and JSON escaping are covered by tests
- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Typed Rust value"] --> B["ptr-inspect\nStructured Runtime Introspection"]
    B --> C["Redacted tree / diff"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-inspect.mmd`](../../docs/diagrams/components/ptr-inspect.mmd)

**Upstream:** ptr-types  
**Downstream:** none yet (no crate or binary consumes it)

## Mission

Allows generic inspection and rendering of typed Rust values without collapsing the runtime into untyped JSON.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- inspection tree
- value visitors
- state diffs
- redaction-aware rendering
- CLI/debug adapters

## Explicit non-responsibilities

- serialization authority
- semantic verification
- telemetry transport

## Data flow

| Direction | Contract |
|---|---|
| Input | Inspectable domain objects |
| Output | Structured inspection tree, diff or redacted render |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- tokio-rs/valuable candidate
- custom PTR visitor layer
- serde bridge only at explicit external boundaries

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Secret<T>, Private<T> and RawEvidence<T> never render plaintext by default.
2. Inspection does not confer semantic validity.
3. Unknown runtime types remain safely traversable or opaque.

These invariants should be executable wherever possible through unit, property, lifecycle or chaos tests.

## Failure model

The component must fail closed for semantic or effect-safety violations. Infrastructure failures should surface as typed errors that preserve request, revision, generation and provenance context. Retries must be idempotent whenever the operation may cross a process or network boundary.

## Performance model

Measure before optimizing. Benchmarks should record at least latency distribution, throughput, allocations/resident memory, queue depth or working-set size where relevant, and the cost of verification. Performance optimizations may not bypass generation, revision, capability or evidence checks.

## Security and privacy

- Treat external inputs and backend outputs as untrusted until validated.
- Do not put raw secrets/private evidence into generic tracing or inspection.
- Preserve provenance on every promotion from raw/possible evidence to stronger semantic state.
- External effects pass through `ptr-security` even if this component already performed local validation.

## Technology evaluation

- [introspection](../../evaluations/components/introspection/README.md)

A new candidate should be added with a reproducible benchmark and failure-semantics analysis rather than replacing the default ad hoc.

## Tests required before production use

- Contract/unit tests for all domain transitions.
- Invalid, stale-generation and stale-revision cases.
- Cancellation/retry behavior.
- Property tests for invariants where practical.
- Cross-backend equivalence if more than one backend exists.
- Observability and redaction checks.

## Related architecture

- [System architecture](../../docs/architecture/00-system.md)
- [Technical architecture](../../docs/TECHNICAL_ARCHITECTURE.md)
- [Component contracts](../../docs/COMPONENT_CONTRACTS.md)
- [Global invariants](../../docs/INVARIANTS.md)

