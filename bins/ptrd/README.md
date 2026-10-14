# ptrd

The PTR daemon: loads the layered configuration (file, then `PTR_*` environment, then CLI),
opens the durable runtime and serves `ptr-server` over HTTP. See `config.toml` and
`tests/` for what is exercised.

## Opt-in observability flags

Both are off by default; without them `ptrd` behaves exactly as before. Invalid values are a
startup error (exit code 2), never a panic.

- `--trace-file PATH` appends one OTLP/JSON-shaped span line per request to `PATH` (created if
  missing). The export is redacted: it never contains the request text, and client request ids
  appear only as `sha256:<16 hex>` digests. It is a plain file
  writer, not an OpenTelemetry collector client. Every traced route is written, including
  `GET /health`, so health probes add lines too.
- `--trace-max-bytes N` caps the trace file (default 8 MiB, must be non-zero, requires
  `--trace-file`). A span that would exceed the cap is dropped and counted, not written.
- `--ingress-diagnostics` adds an advisory `ingress_diagnostics` field to `POST /v1/requests`
  responses. The runtime still receives the raw text; nothing is blocked or rewritten by it.

These map to `ptr_server::ServerOptions` and are applied with `ServerState::with_options`
before serving. They are CLI flags only; there is no `ptr-config` key for them.
