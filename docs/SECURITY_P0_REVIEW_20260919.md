# Security and P0 architecture review — 2026-09-19

Base: `efe8386d71f03997ce5ea2cc93bba66ff43342ba`. This document records
implementation scope, not a substitute for exact-commit CI evidence. Source,
lockfiles, scanner versions and advisory-db revision are retained in CI artifacts.

## Corrected baseline

The earlier issue #12 / merge review incorrectly identified Unicode-3.0 as an
unapproved license. It was already in deny.toml. The actual baseline rejected
ISC, BSD-3-Clause, Zlib, Unlicense, CDLA-Permissive-2.0 and specific MPL packages.
Root audit also found eight vulnerability reports; isolated A0 carried obsolete
bincode/paste, and fuzz lacked a committed independent lockfile.

## P0 findings and implemented repairs

| Finding | Repair | Evidence boundary |
|---|---|---|
| Hard-action freshness/capability checks were configurable away | Mutation, External and Irreversible always check revision, live generation, tombstones, capability and effect permission | Trusted runtime current-state input; no principal/resource authorization claim |
| Public lifecycle setter bypassed commit/replay | Setter is private; committed transitions validated before append and during replay | Default reference runtime only |
| Older or revoked generations could be reactivated by a new activation event | Monotonic activation/supersession with exact old-generation checks and permanent generation tombstones | Future generations still require explicit activation; deletion of neural weights is not claimed |
| Capsule IDs could impersonate procedure/constraint namespaces or cross projects | Reserved prefixes rejected; a capsule's project binding cannot change | Not a replacement for a complete scoped target type system |
| Multiple FileLedger writers could assign overlapping indices | Nonblocking OS advisory lock held for handle lifetime, including recovery/truncation | Cooperating local processes; no malicious filesystem/remote filesystem guarantee |
| An ambiguous failed append could be followed by another write | Writer poisons before write/failpoints and clears only after durable+memory success; drop/reopen required | Panic/interleaving tests are not hardware power-loss proof |
| Component metadata check passed when git comparison failed; HEAD^ missed earlier commits | Comparison failure is fatal; CI uses the complete PR/push range | Functional source/metadata coupling, not semantic documentation correctness |
| A0 query-only attention bias canceled in softmax | Rank-one query/key-dependent bias; cancellation control and nonzero-gradient regression | Mechanism correctness, not superior reasoning/learning |
| Training manifest omitted isolated A0 graph | Schema 3 includes A0 Cargo.toml and Cargo.lock in the run fingerprint | Run preparation remains a dry-run; no full training backend is added |

## Dependency and build decisions

Iroh moves from 0.35 to pinned 1.2.0, eliminating old Hickory/webpki/time dependency
paths. It remains optional: the core Rust 1.85 contract and the Iroh-feature
Rust 1.91 contract are tested separately. Direct loopback transport remains
relay-disabled, ALPN-separated and peer-authenticated.

Raft 0.7 / raft-engine 0.4.2 need explicit compatibility patches because their
published parent versions do not expose a clean legacy protobuf migration.
Original archives/licenses and upstream file hashes remain in vendor/ORIGINS.json.
The PTR Raft adapter is prost-only: legacy protobuf2 compatibility implementations
and parser/build dependencies are removed, not renamed or advisory-ignored.
The engine migrates to real protobuf 3.7.2 / Prometheus 0.14. Wire roundtrip,
malformed-input and consensus/storage tests must pass before merge.

A0 remains isolated, now pinned to Burn 0.22.0-pre.3 and Rust 1.95. This is an
explicit prerelease migration, not a production framework selection. It removes
the bincode2 recording dependency and uses the upstream device-dispatch API and
Flex CPU backend; autodiff is explicitly enabled on the training device.
The old integer metadata codebook/checkpoint format was never stable. Prior
synthetic measurements are not automatically comparable across this change.

Transitive paste consumers are patched to the maintained pastey 0.2.3 package
using Cargo dependency aliases. Actual package identity changes; no fake paste
version or RustSec exception is used. Added source patches have a maintenance
cost and must be retired when compatible upstream releases adopt the changes.

## License-policy decision

The explicitly listed permissive license families in deny.toml are accepted with
redistribution notices and attribution obligations. BSL-1.0 means Boost Software
License 1.0, not Business Source License. Unknown source registries/git sources,
unknown licenses and broad copyleft grants remain rejected.

MPL-2.0 is NOT globally allowed. Only `colored =3.1.1` is a version-scoped exception for the A0 dependency graph. Before
redistributing binaries, preserve their notices and provide the corresponding
MPL-covered source and modifications; keep those files under MPL. This technical
policy decision is not legal clearance for a proprietary distribution. Release
packaging must carry third-party notices/source availability. All upstream
vendored licenses are retained. Advisory ignores remain empty.

## Still open P0 architecture gates

1. P0.1 now binds reference-process execution to opaque sessions, exact
   capsule/project grants and registered verifiers/executors; see
   [the implementation contract](architecture/21-scoped-execution.md). The
   embedding host still authenticates the principal. Network/session admission,
   scoped Pure/Read Pods, durable audit/idempotency and remote fences remain open.
   A diagnostic allow receipt is still NOT an execution token.
2. SemDB payloads/revisions and inference checkpoints are not durably reconstructed
   by the lifecycle-only FileLedger. Restart tests do not prove full knowledge or
   neural-state consistency. Semantic payload admission remains incomplete.
3. File framing still lacks authenticated/checksummed records; snapshots are not
   fully serialized/restored. Multi-node consensus, transport and storage composition
   remains a prototype, distinct from the single-node harness.
4. Neural validity IDs are learned embeddings, not hard lifecycle masks. A shared,
   versioned cognitive codebook, actual payload path and independently verified
   stale-state closure remain prerequisites before scientific superiority claims.

A green build/security policy result does not close these architecture gates or
satisfy the overall PTR research/product Definition of Done. Never relabel the
above items as done merely because automated checks pass.
