# PTR Type System

PTR's type system is part of the **model architecture**, not only Rust infrastructure. Its purpose is to give the neural core, semantic runtime, verifier, memory and action boundary a shared vocabulary that is richer than tokens and safer than free-form strings.

The central rule is that different semantic axes stay **orthogonal**. A value can be a goal, claim or constraint independently of whether it is unknown, hypothetical or observed; independently of whether its uncertainty is a point estimate or distribution; and independently of whether it is live, revoked, verified or committed.

## 1. Shared cognitive type axes — `ptr-types`

`ptr-types` owns small provider-independent types that multiple architectural layers must interpret identically.

### Semantic role

Implemented shared roles:

- `Goal`
- `Constraint`
- `Claim`
- `Evidence`
- `Resource`
- `Capability`
- `Relation`
- `Procedure`
- `Action`

These describe **what a representation means**, not how certain it is.

### Epistemic state

Implemented shared epistemic states:

- `Unknown`
- `Assumed`
- `Hypothesis`
- `Observed`
- `Inferred`
- `Verified`

These describe the epistemic standing of a value. They are independent of its semantic role.

### Uncertainty representation

Implemented uncertainty shapes:

- `Point`
- `Interval`
- `Distribution`

A distribution is therefore not a slot role. For example, a model slot may be:

```text
role        = Claim
epistemic   = Hypothesis
uncertainty = Distribution
```

### Reasoning operator

`ReasoningOperator` is shared between model, model API, router and training:

- semantic
- deductive
- probabilistic
- statistical
- temporal
- causal
- search
- optimization
- simulation
- symbolic
- external Pod

Operator identity must never degrade to provider/tool strings at an architectural boundary.

### Other cross-cutting axes

`ptr-types` also owns:

- `Probability`
- `VerificationLevel`
- `Validity`
- `Revision`
- `Generation`
- effects and capability identity
- provenance/evidence identity
- shared semantic IDs

These are not the primary purpose of the crate; they support the cognitive type system.

## 2. Neural representation — `ptr-core`

`ptr-core` owns the **representation of shared types inside the model**.

A `SemanticSlot` currently carries:

```text
role: SemanticRole
epistemic: EpistemicState
uncertainty: UncertaintyKind
generation: Generation
validity: Validity
confidence: Probability
value: latent vector
provenance: ...
```

The latent vector/tensor itself remains a `ptr-core` concern. Tensor layouts, embedding tables, typed-attention biases, masks, recurrence state and model-family-specific representations do **not** belong in `ptr-types`.

This separation allows experiments such as:

- separate role embeddings versus fused embeddings;
- epistemic attention bias;
- uncertainty-aware attention;
- lifecycle-generation masks;
- provenance-aware attention;
- operator prediction heads;
- multi-objective losses for role, epistemic state, uncertainty and routing.

## 3. Do not collapse the axes

The previous prototype `SlotKind` mixed:

- semantic roles: Goal, Constraint, Resource, Capability;
- epistemic states: Known, Observed, Hypothesis, Unknown;
- uncertainty representation: Distribution.

That representation is intentionally being removed from the target architecture.

Likewise, `Known` is not a safe single universal type label. A neural representation may predict high confidence or verified-looking content, but that does not create runtime authority.

## 4. Model proposal versus authority

PTR separates at least these concepts:

1. **model representation/proposal** — neural state or decoded typed proposal;
2. **verification result** — evidence that a proposal passed some checks;
3. **runtime-valid semantic state** — accepted into the semantic runtime;
4. **committed authority** — durable ledger/consensus-backed state where applicable.

Confidence and verification are data. They do not grant capabilities or effect permissions.

A model output may describe a source that was committed, but the model output itself does not inherit committed authority.

## 5. Runtime semantic types

The target semantic vocabulary includes generic types such as:

```rust
Goal<T>
Constraint<T>
Claim<T>
Evidence<T>
Relation<S, P, O>
Resource<T>
Procedure<T>
ActionIntent<T>
Estimate<T>
Interval<T>
Distribution<T>
```

These remain partly open because their exact generic shape affects SemDB, memory, training and serialization. They belong in the shared semantic kernel only when at least two architectural layers need identical semantics.

Component-specific aggregate types remain with their owners:

| Type | Owner |
|---|---|
| `SemanticSlot` / latent workspace | `ptr-core` |
| `ModelRequest`, `ModelEvent`, continuation | `ptr-model-api` |
| typed semantic keys/snapshots | `ptr-semdb` |
| `SemanticCapsule`, procedures, episodes | `ptr-memory` |
| search hits/retrieval plans | `ptr-search` |
| verification reports/findings | `ptr-verifier` |
| authorization decisions/receipts | `ptr-security` |
| Pod manifests/leases | `ptr-pods` |

This prevents `ptr-types` from becoming a god crate while still making it the semantic foundation of the architecture.

## 6. Model/event boundary

Model events must use shared typed identities whenever semantics cross a crate boundary.

For example:

```rust
ModelEvent::OperatorRequested {
    operator: ReasoningOperator,
}
```

is preferred over:

```rust
operator: String
```

The same rule applies to capabilities, type IDs, lifecycle coordinates and future semantic-role metadata.

## 7. Training implications

The type system is directly trainable/evaluable.

Potential supervised targets include:

- semantic role classification;
- epistemic-state prediction;
- uncertainty-shape prediction;
- calibrated confidence;
- operator routing;
- lifecycle validity awareness;
- provenance retention;
- typed ActionIR generation;
- verifier outcomes.

Ablations should compare fused versus separated type axes and measure whether each axis improves OOD semantic fidelity, calibration, routing quality, lifecycle correctness or sample efficiency.

## 8. Typestate and Rust static types

Rust typestate remains a second, complementary type layer.

Example:

`PodLease<Ready> -> PodLease<Revoked>`

Only `Ready` exposes invocation.

Compile-time typestate is used where Rust can prevent an invalid transition entirely. Runtime semantic types are used where values originate from model/user/world state and therefore require validation.

## 9. Probability

Probability is data, not permission.

A high probability of a mutation being useful does not grant mutation capability. A calibrated hypothesis does not become committed state by crossing a numeric threshold.

## 10. Ownership rule

A type belongs in `ptr-types` when all of the following are true:

1. it expresses PTR semantics rather than provider implementation;
2. multiple architectural layers must interpret it identically;
3. it is small enough to remain independent of storage/network/model frameworks;
4. moving it behind one component would create stringly-typed or duplicated cross-boundary semantics.

A type stays in a component crate when it is primarily that component's aggregate, representation or execution detail.
