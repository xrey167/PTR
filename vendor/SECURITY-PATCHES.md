# Dependency compatibility patches — review candidate

Original crate names/versions are retained; there are no forged version numbers
or RustSec ignore entries. ORIGINS.json pins published archives and every original
file. This directory is an explicit PTR maintenance responsibility, not a claim
that upstream has released these patches.

* raft 0.7.0: protobuf-codec dependency is optional, matching upstream; prost
  remains selected. Replace abandoned fxhash with maintained rustc-hash.
* raft-proto 0.7.0: optional protobuf dependency; protobuf-build 0.15.1 supports
  selecting prost without dragging in protobuf 2.
* raft-engine 0.4.2: migrate real protobuf dependency/API to 3.7.2 and the
  corresponding Prometheus 0.14 API. No log-payload or protocol format waiver.

Compilation, ledger round-trip, replay and corruption tests must pass on the
actual selected features before integration. Further compiler-driven migrations
are recorded in the PR review. All upstream sources and licenses are retained.
