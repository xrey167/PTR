# ptr-worker

A small worker host that serves deterministic CPU reference Pods over the generic control
protocol of `ptr-protocol`, through a pair of byte streams (stdin/stdout).

```text
client --[u32 BE length | ControlFrame]--> ptr-worker serve --stdio
                                              |  ProtocolSession (responder: Hello, Decl, Call/Reply,
                                              |  Cancel, Bye + drain, idempotency cache)
                                              v
                                        PodRegistry (project "worker"): echo, crc32, sum, counter
```

## Usage

```text
ptr-worker                  print usage, exit 0
ptr-worker pods             list the built-in Pods (name<TAB>contract summary)
ptr-worker serve [--stdio]  serve one session on stdin/stdout
```

Stdout is the protocol channel and carries nothing else; diagnostics go to stderr and never
contain call payloads. Exit codes: 0 clean (Bye exchanged), 1 protocol, session or I/O failure
(including a truncated or oversize frame), 2 usage error, 3 the peer closed the stream without
Bye.

## What it does

- `StdioTransport` implements `ptr_protocol::session::FrameTransport` over any `Read`/`Write`
  pair. Frame = `u32` big-endian length + the codec's encoded `ControlFrame`; the prefix is
  checked against `MAX_FRAME_BYTES` before the body is read. Typed `StdioError` for oversize,
  EOF mid-frame, I/O and codec failures.
- The worker answers a Hello as responder, then serves `Call` frames. The call payload
  (`worker.call.v1`) is `u16 BE name length | pod name | input`; the reply (`worker.reply.v1`)
  is `status u8 (0 ok, 1 error) | output or error code`. Unknown Pods, malformed calls and Pod
  failures are error replies, not session failures. Input is capped at 64 KiB.
- Duplicate calls (same key and payload) are answered by the session's idempotency cache and not
  executed again. Queued calls run when no further complete frame is buffered, so a `Cancel`
  delivered in the same read as its `Call` drops the call (Ack); a later `Cancel` is ignored.
  `Bye` is answered with `Bye`, queued calls drain first.
- Bounds: session window of 8 calls, 64 remembered idempotency keys, one frame at most 1 MiB,
  at most 4096 terms for `sum`.
- Built-in Pods (written against `ptr_pods::DynPod`): `echo`, `crc32` (IEEE), `sum` (checked
  i64), `counter` (process-local execution count, used to show that a cached duplicate did not
  execute).

## What it does not do

- No process isolation beyond being a separate OS process: no sandbox, seccomp or resource
  limits, and a Pod runs in the worker's own address space.
- No network transport (no TCP, no Iroh, no TLS, no authentication); the peer is whoever holds
  the pipes.
- No model Pods: no weights, no Candle/Qwen executors, no GPU path.
- One session per process, one call executing at a time; no concurrency, no deadlines, no
  Pod lifecycle or lease management.
- `ptr-podwire` and the runtime do not call this binary yet.
