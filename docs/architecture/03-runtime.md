# Rust Execution Runtime

PTR execution uses isolated state machines with bounded typed mailboxes, explicit backpressure, supervision, cancellation, timeout and replay semantics. Tokio is expected to remain the default low-level async I/O substrate, but runtime semantics are PTR-owned and benchmarkable against alternatives.

No core subsystem should depend on unbounded hidden queues. Refused work returns ownership to the caller.
