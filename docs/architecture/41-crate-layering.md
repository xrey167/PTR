# 41 — Crate layering

Status: implemented as a CI check; the layer table is derived from the real dependency graph.

PTR keeps its domain semantics independent of any provider (invariant 10). That claim is
only worth something if it is checked, so the dependency graph of the root workspace is
validated against a table that lives in data, not in prose.

- Table: [`research/catalogs/crate-layers.toml`](../../research/catalogs/crate-layers.toml)
- Check: `python scripts/check_crate_layering.py` (CI job `repository-invariants`,
  `make ci-repository-invariants`)
- Tests: `scripts/tests/test_check_crate_layering.py`

## Rules

1. **Direction.** A crate depends only on crates of the same or a lower layer rank.
2. **Backend ownership.** A backend crate (`iroh`, `tokio-postgres`, `raft`, `raft-engine`,
   `opendal`, `candle-*`, `turso`, `tracing`, `wintun`, `wireguard-uapi`, and the reserved
   `burn*`, `libsql`) may be a direct dependency only of the owner crates the table names.
   A pattern ending in `*` matches by prefix. An empty owner list reserves the name: no
   workspace crate may use it.
3. **Placement.** Every workspace member under `crates/` or `bins/` is placed in exactly
   one layer. A new crate fails the check until someone decides where it belongs.
4. **Scope.** Normal and build dependencies count, optional and target-specific ones
   included, because a feature that is off today is on in some deployment.
   `dev-dependencies` are exempt. Excluded workspaces (`model/burn-a0`, `fuzz`, `vendor/*`)
   are out of scope.

The check reads `cargo +stable metadata --locked --no-deps --format-version 1`. The declared
dependencies are all the rules need, so nothing is resolved or downloaded.

## Layers

| Rank | Layer | Crates |
|---|---|---|
| 0 | kernel | `ptr-types` |
| 1 | foundation | leaf services that need only the kernel: `ptr-ledger`, `ptr-memory`, `ptr-semdb`, `ptr-security`, `ptr-storage`, `ptr-protocol`, `ptr-model-api`, `ptr-lineage`, and others |
| 2 | domain | `ptr-verifier`, `ptr-search`, `ptr-labeling`, `ptr-net` |
| 3 | composition | `ptr-pods`, `ptr-branch`, `ptr-feedback`, `ptr-fastmem`, `ptr-cluster` |
| 4 | projection | `ptr-router`, `ptr-state`, `ptr-worker` |
| 5 | orchestration | `ptr-runtime`, `ptr-pg` |
| 6 | surface | `ptr-server`, `ptr-podwire`, `ptr-execwire` |
| 7 | binary | `ptrd`, `ptrctl`, `ptr-bench` |

The table records what the graph is, not what an earlier sketch hoped for. `ptr-state`
depends on `ptr-pods`, and `ptr-podwire` and `ptr-execwire` call `ptr-runtime`, so they sit
above it. `ptr-lineage` has no `ptr-ledger` dependency by design, and the table keeps it that way.

## Open items

Places where the table records the graph as it is, and the target picture is still undecided:

- `ptr-podwire` and `ptr-execwire` sit above `ptr-runtime` because they call it. Whether the
  wire adapters should instead depend on a runtime-side port is open.
- `ptr-state` sits above `ptr-pods`, which it depends on for pod types in projections.
- `ptr-pg` shares the orchestration layer with `ptr-runtime` instead of sitting below it.
- `ptr-bench -> tokio-postgres` is the one listed exception (see below).

Later packages build on these choices. In particular `ptr-lineage` is a foundation crate with
no `ptr-ledger` dependency, so adapter promotion proposals are committed by `ptr-runtime`, not
by `ptr-lineage`.

## Exceptions are a ratchet

A known violation is listed under `[[allowed_exception]]` with `from`, `to` and a reason.
The check fails when a listed exception no longer matches a violation, so the list can only
shrink. The unit test pins the set of exceptions that may exist; adding one means editing
the test as well as the catalog, which makes the decision visible in review.

Today there is one: `ptr-bench` opens its own `tokio-postgres` connections for the
PostgreSQL experiment harness. It should move behind a `ptr-pg` test-support API.

## Changing the table

Moving a crate up a layer is a deliberate act and needs a sentence in the pull request.
Moving it down is free as long as its dependencies allow it. Adding a backend crate to a
workspace member means adding a `[[backend_owner]]` entry first.
