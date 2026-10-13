# Roadmap

## Active sequence — one cognitive component at a time

The next work follows [Step 01: cognitive contract](architecture/20-cognitive-contract-step-01.md),
not a repository-wide type/backend refactor. Confidence targets, the narrow A0 bias correction, scoped local execution and
semantic replay/integrity recovery now have bounded implementation contracts.
Minimum semantic values and a versioned codebook now exist
([document 26](architecture/26-cognitive-codebook.md)). The next cognitive increment is
connecting actual semantic payloads to a model end to end, followed by controlled
training against matched baselines. After each code change, update its component metadata, generated README
and status, architecture contract and test evidence before widening scope.

1. Cognitive contract and reference cases (first ptr-types increment).
2. Minimal semantic values and versioned cognitive codebook.
3. One complete typed source/snapshot/model/observation path.
4. A0 mechanism correctness and isolated ablations.
5. Real training with matched baselines and held-out/OOD evaluation.
6. Learned-path integration into the existing runtime.

The research-integration work packages (pod linking, reflex and failover, retrieval filters and wire deadline, training integrity and layering, serving and promotion) are tracked in [RESEARCH_INTEGRATION_PLAN_20261013.md](RESEARCH_INTEGRATION_PLAN_20261013.md).
Status (2026-10-13): steps 1 and 2 are done; step 4 ran as preregistered A0 studies
whose isolating contrasts are INCONCLUSIVE with no positive claim (`STATUS.md`);
step 3 (G1) and step 6 are decided "not now" until an A0 study passes its gates
(ADR-0022, S8); step 5 has not started.

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

### F.1 Mesh transport and protected tunnel lifecycle
- Mesh membership, invitations and revocation.
- Iroh direct/relay route selection with relay disabled by default.
- Persistent MeshEndpoint and route revisions.
- WireGuard userspace executor, followed by Windows Wintun and macOS Network Extension adapters.
- Runtime admission, fencing, recovery and PodDirectory/PodWire integration.
- Direct, relay, reconnect and stale-writer tests.

Current implementation status:
- `ptr-types` carries validated `MeshEndpointBinding`; PodDirectory rejects
  mismatched mesh peer/network bindings.
- PodWire addressed V3 frames carry the mesh binding and the admitted binding
  rejects mismatches before invocation; V1 and legacy unaddressed V2 remain
  compatible.
- `ptr-ledger` persists mesh/tunnel lifecycle events with stable tag 15 and
  `ptr-state`/`ptr-runtime` validate and project them.
- `ptr-net` provides the fenced `WireguardUserspaceExecutor<D>` contract and
  deterministic device-backed tests. Two platform devices exist behind features:
  `LinuxWireguardDevice` (`wireguard-uapi-backend`, the WireGuard userspace UAPI
  plus the host `ip` utility, linted and unit-tested in CI against a mock UAPI
  socket only) and `WintunDevice` (`wintun-backend`, a TUN interface with no
  WireGuard crypto dataplane, only compiled on CI's Windows leg). No macOS Network Extension adapter exists. None is
  claimed as hardware-validated here.

## Phase G — PTR-Diff
Parallel structured reasoning / diffusion architecture only after A0 has measurable evidence over controlled baselines.
