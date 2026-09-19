# Roadmap

## Active sequence — one cognitive component at a time

The next work follows [Step 01: cognitive contract](architecture/20-cognitive-contract-step-01.md),
not a repository-wide type/backend refactor. The first increment adds explicit
confidence targets and representation/API tests to ptr-types. Codebook, semantic
payload integration, A0 corrections and real training remain separate subsequent
gates. After each code change, update its component metadata, generated README
and status, architecture contract and test evidence before widening scope.

1. Cognitive contract and reference cases (first ptr-types increment).
2. Minimal semantic values and versioned cognitive codebook.
3. One complete typed source/snapshot/model/observation path.
4. A0 mechanism correctness and isolated ablations.
5. Real training with matched baselines and held-out/OOD evaluation.
6. Learned-path integration into the existing runtime.

The broader phases below describe the target architecture; they are not a reason
to finish every infrastructure slot before validating the cognitive/model path.
The full Definition of Done remains unchanged.

## Phase A — Rust Foundation
`ptr-types`, `ptr-protocol`, `ptr-semdb`, `ptr-exec`, `ptr-ledger`, `ptr-state`.

## Phase B — Knowledge Runtime
Memory capsules, search planner/backends, Pods, verifier fabric and storage contracts.

## Phase C — Baseline Model Integration
External inference backend, ModelEvent stream, typed ingress datasets and end-to-end benchmark harness.

## Phase D — PTR-Core A0
Semantic slots, typed attention, epistemic workspace, latent recurrence, operator router and action/verifier heads.

## Phase E — Learning Loops
Synthetic data, SFT, verifier training, preference/RL, DSPy teachers and GEPA outer-loop optimization.

## Phase F — Distributed Lifecycle
Consensus, durable ledger, node transport, event streaming, snapshots, compaction barriers and fail-point chaos testing.

## Phase G — PTR-Diff
Parallel structured reasoning / diffusion architecture only after A0 has measurable evidence over controlled baselines.
