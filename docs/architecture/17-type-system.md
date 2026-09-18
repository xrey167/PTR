# PTR Type System

The type system exists at three layers:

1. **Rust static types** — IDs, effects, capabilities, typestate.
2. **Runtime semantic types** — Goal<T>, Constraint<T>, Evidence<T>, Distribution<T>, Action<T>.
3. **Neural typed slots** — latent representations tagged/conditioned by semantic and epistemic kind.

These layers are related but not identical. Neural “Known” is still a model proposal until verified/runtime-authoritative.

## Typestate examples

`PodLease<Ready> -> PodLease<Revoked>` and only Ready exposes invoke.

## Probability

Probability is data, not permission. A high probability of a mutation being useful does not grant Mutation capability.
