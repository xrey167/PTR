# Iroh 0.35 MSRV adapter — rejected

The Iroh 0.35 adapter proved PTR ALPN and authenticated peer-identity semantics on Rust 1.85, but retaining its older dependency graph in 2026 introduces unacceptable network-boundary security debt.

The repository security scan found vulnerable and obsolete transitive dependencies in that graph. Pinning older networking/TLS/DNS dependencies solely to retain the PTR core MSRV is not an acceptable production strategy.

Decision: keep the PTR core workspace on Rust 1.85 for now, but evaluate current Iroh 1.2.0 in an isolated package using Iroh's Rust 1.91 MSRV and its own lock/security CI.
