# Observability and Introspection

`tracing` is the primary structured telemetry candidate. PTR standardizes field names for request, project, revision, generation, candidate, Pod, capability, effect and commit index. `valuable` is reserved for object-safe structured runtime inspection with explicit redaction. Async backtraces are debug/hang tooling rather than always-on semantics. NeMo Relay and OpenTelemetry are export targets.
