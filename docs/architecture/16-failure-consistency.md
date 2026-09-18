# Failure & Consistency Model

## Failure classes

- process crash/restart;
- storage/fsync failure;
- network loss, delay, duplication and partition;
- stale semantic revision;
- stale object generation;
- overloaded bounded queue;
- unavailable model/search/Pod backend;
- verifier disagreement;
- corrupted/untrusted input;
- partially applied derived projection.

## Required recovery

Authoritative state rebuilds from validated snapshot + committed log. Derived indexes may be dropped and rebuilt.

## Chaos tests

Insert failpoints at commit/materialize/index/cache/effect boundaries. After every restart, assert lifecycle and authority invariants.

## Consistency

Authority is strongly ordered; projections are allowed to lag. Lag is safe only because reads cross-check live generation/commit metadata before promotion.
