# ptr-verifier — Verifier Fabric

> **Role:** Combines deterministic, executable, statistical and learned checks into explicit verification reports.  
> **Maturity:** see the generated status below; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `scaffold`  
**Last reviewed:** 2026-10-10  
**Code footprint:** 6 Rust source files · 4184 nonblank source lines · 2 integration-test files · 65 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- Pass/Fail/Disputed/Unknown statuses
- Finding and VerificationReport with level/score/findings
- Generic Verifier<T> contract
- NamedVerifier + VerifierFabric aggregate reports with Fail > Disputed > Unknown > Pass precedence
- VerifierFabric implements Verifier<T>: its level is the WEAKEST level of its reports (a weaker verifier must not let the fabric satisfy a stronger requirement; the runtime's SemanticGrant applies the same rule, ADR-0020), its status is the worst one, and any hard finding forces Fail so a deterministic contradiction dominates learned scores; an empty fabric is Unknown/Unverified
- VerifierFabric implements Default for any T (a derive would have required T: Default, which ActionIr does not implement), so a fabric of ActionContractVerifier plus further verifiers can be built for ActionIr
- ActionContractVerifier (named action-contract) is a deterministic Verifier<ActionIr> over an ActionContract (target, operation, capability, input type, effect, payload size, PayloadEncoding): each violation is one hard finding with a stable code (target, operation, capability, input-type, effect, payload-size, payload-utf8), Fail/Deterministic/score 0, else Pass/Deterministic/score 1
- Four deterministic task verifiers implement Verifier<String> (Verifier<Vec<FactClaim>> for facts) and NamedVerifier, all at Deterministic level, Pass only on exact success (score 1) and bounded in input size (oversize input is a hard finding, never a panic): ArithmeticVerifier (task = integer expression with + - * / and parentheses, evaluated by an independent bounded exact-rational i128 evaluator, no eval; candidate = integer or a/b; undecidable task is Unknown), JsonSchemaVerifier (in-house pinned JSON Schema subset type/required/properties/additionalProperties/enum/minimum/maximum/minLength/maxLength/items, strict RFC 8259 parser with depth and byte bounds, numbers are exact decimals (no float: 2^53+1 differs from 2^53, 1e-400 is positive), objects are indexed so duplicate-key and equality checks are O(n log n), enum capped at 256 members, one finding per violated keyword with an RFC 6901 pointer cut at 256 bytes, finding cap of 32 with a json.truncated marker only when a further violation exists), CodeTestVerifier (candidate = program in an embedded pure integer language run under hard fuel (CPU; names resolved to frame slots at parse time so each step is O(1), a call pays its frame size), call-depth, memory and output bounds with no I/O, tests supplied by the task and hidden by default: a failing hidden test's finding names the id only, a visible test states expected and got; timeouts and resource exhaustion are findings), FactVerifier (FactVerifier::new takes the snapshot and the live revision, without_staleness_check is the documented host-duty constructor; claims checked read-only against one ptr-semdb SemanticSnapshot; names the snapshot revision in each finding and in verify_bound; a missing fact is Unknown, a stale snapshot is Unknown and judges nothing)
- One finding-code table (codes::CODE_TABLE, mirrored in the codes module doc and a test that keeps the two equal): every code is hard or soft and hard codes map to a RepairRole (format, value, schema, code, resource, fact); ptr-router::RepairRoleAssignments builds the FindingRouter registry and the runtime Roster from it. Codes: arith.format arith.oversize arith.value arith.bounds (soft arith.task); json.parse json.oversize json.depth json.type json.required json.enum json.range json.length json.extra (soft json.truncated json.schema); code.syntax code.oversize code.test_failed code.timeout code.resource (soft code.task); fact.contradicted fact.malformed fact.oversize (soft fact.unknown fact.stale_snapshot)

### Missing for the target architecture

- Revision/generation deterministic verifier (the runtime's authorize_action still owns that check)
- Source/evidence promotion verifier
- Compiler/GritQL/Sentrux adapters (CodeTestVerifier runs only the embedded integer language, not real-language test suites)
- Process-isolated execution of real-language candidates (ptr-exec offers isolates and bounded mailboxes, not process isolation)
- ArithmeticVerifier coverage limit: every reduced intermediate must stay within 10^30 and every unreduced product or sum within i128, so a legitimate task such as 10^20 * 10^20 / 10^20 is Unknown (arith.task), never a wrong Pass; a big-integer evaluator would lift it
- Structured snapshot revision through the Verifier trait (VerificationReport has no field for it; it is in each fact finding message and in FactVerifier::verify_bound)
- Calibration of the task verifiers on real task distributions (they are deterministic; their coverage of real tasks is unmeasured)
- Calibration/statistical and learned semantic verifiers

### Next milestones

- Add a revision/generation verifier fed with live runtime state
- Extend F001-style evidence to real generators (F001, completed, drove the fabric and the ActionContractVerifier through the ptr-feedback RepairLoop)

### Linked experiments

- [F001](../../experiments/feedback/F001-generate-verify-repair/README.md) — `frozen`
- [Q002](../../experiments/retrieval/Q002-evidence-promotion/README.md) — `frozen`
- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`

### Technology evaluations

- [code-quality-verifier](../../evaluations/components/code-quality-verifier/README.md) — `open`

### Decision records

- [ADR-0002-authority-hierarchy.md](../../research/decisions/ADR-0002-authority-hierarchy.md)
- [ADR-0012-hard-effect-boundary.md](../../research/decisions/ADR-0012-hard-effect-boundary.md)
- [ADR-0020-only-the-runtime-merges-an-agent-branch.md](../../research/decisions/ADR-0020-only-the-runtime-merges-an-agent-branch.md)

### Current automated checks

- unit tests per task verifier: positive and negative case for every finding code, oversize input, adversarial input (deep nesting, huge numbers, infinite loops and unbounded recursion in code), determinism (same input, identical report)
- unit tests for the review fixes: i128::MIN intermediates never panic (debug build, overflow checks on), exact Int/Float and enum comparison beyond 2^53, 1e-400 is not zero, 32 vs 33 violations with and without the truncation marker, pointer truncation, enum cap, hidden/visible test messages, call frames charged as fuel, maximal recursion and nesting on a 512 KiB stack, 007 and -0 accepted as numerals
- unit test: the code table has unique, short codes, hard iff it has a repair role, and the module doc table lists exactly the table
- crates/ptr-runtime/tests/vcw_real_verifiers.rs: VcwLoop over each real verifier with stub RolePods: wrong candidate, finding, routed to the table's repair role, repaired and accepted; unrepairable task Exhausted; every soft code (arith.task, json.schema, code.task, fact.unknown, fact.stale_snapshot) is Unknown, classified as the Unknown fallback, regenerates and is never repaired or accepted; fact.malformed/fact.oversize go to format-repair, code.resource to resource-repair, arith.bounds to value-repair
- verifier-fabric deterministic failure precedence test
- unit test: a fabric can be built for a type without a Default
- unit tests: one hard finding per ActionContractVerifier code, several findings at once, fabric weakest-level rule, hard finding forcing Fail, empty fabric Unknown
- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Candidate + evidence"] --> B["ptr-verifier\nVerifier Fabric"]
    B --> C["Pass / Fail / Disputed / Unknown"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-verifier.mmd`](../../docs/diagrams/components/ptr-verifier.mmd)

**Upstream:** ptr-semdb, ptr-types  
**Downstream:** ptr-runtime, ptr-pods, ptr-router, ptr-feedback, ptr-branch, ptr-podwire, ptr-server, ptr-bench

## Mission

Combines deterministic, executable, statistical and learned checks into explicit verification reports.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- verifier composition
- verification levels
- findings and evidence
- action validation hooks
- semantic and probabilistic checks

## Explicit non-responsibilities

- candidate generation
- state persistence
- search indexing

## Data flow

| Direction | Contract |
|---|---|
| Input | Candidate<T>, evidence, action, source or execution result |
| Output | VerificationReport with status, level, confidence and findings |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Compiler/tests/GritQL/Sentrux-style executable checks
- source evidence verification
- calibration checks
- learned semantic verifier only when deterministic checks are insufficient

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Deterministic contradictions dominate learned approval.
2. Unknown is a valid outcome.
3. Verifier confidence is not silently converted to hard truth.

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

- [code-quality-verifier](../../evaluations/components/code-quality-verifier/README.md)

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

