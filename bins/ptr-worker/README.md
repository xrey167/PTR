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
ptr-worker serve --channel-fd <n> [--confined]
                            serve one session over an inherited SOCK_SEQPACKET channel
                            (Linux, feature data-plane)
ptr-worker guardian --channel-fd <n> --backend fake|cuda
                            hold the staging buffers of the CUDA carriers for the supervisor
                            (Linux, feature data-plane; `cuda` needs cuda-ipc or cuda-ipc-core)
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

## The Pod data plane (stage S1, Linux, feature `data-plane`)

`docs/architecture/42-pod-data-plane.md` stage S1: a Pod worker is a separate, confined process,
and tensors travel between workers by handle, not through the authority process. This binary is
both ends of that plane.

```text
PodHost (supervisor side, lib: host)  --spawn/poll/reap-->  ptr_fabric::Supervisor
   |  admit / consume / fence / reap + quota release              | fork: cgroup, netns, caps, uid, no_new_privs, exec
   |                                                              v
   +-- Channel (SOCK_SEQPACKET, ControlFrame + SCM_RIGHTS) --> ptr-worker serve --channel-fd 3 --confined
                                                                  1. install_worker_seccomp()   (first action)
                                                                  2. key frame, then challenge  (sign domain|session|nonce)
                                                                  3. ProtocolSession (Pods + data calls)
```

- **Worker (`serve --channel-fd`, module `channel_serve`).** Takes the inherited descriptor,
  installs the seccomp allowlist first when `--confined`, receives the key frame (session id, the
  one stage identity it may serve, the secret key: chosen by the host, never by the worker) and
  signs the host's challenge. Then it serves the same `ProtocolSession` as the stdio form, one
  `ControlFrame` per frame, descriptors attached to frames. Besides Pod calls it serves data
  calls (`worker.data.v1`, module `data_wire`): *produce* (build the tensor's carrier, a sealed
  `memfd` or inline bytes, sign the handle for its bound stage only), *consume* (map the admitted
  region read-only, verify the digest, unmap, answer) and *probe* (report that `socket`, `fork`
  and the other forbidden calls are refused). The worker holds no ledger handle and no admission
  grant.
- **Host (`PodHost`, module `host`).** The glue between `ptr-fabric` (processes) and `ptr-runtime`
  (admission, byte quota), which is why this binary sits in the top layer of the crate layering.
  It derives the stage identity from its own `Placement`, generates the key pair, sends the secret
  as the first frame, challenges the worker and registers the producer with every `HandleTable`
  only after the signature verifies (a bad signature or no answer reaps the worker). It chooses
  `HostShm` only if `carrier_permitted` holds for both workers, else `Bytes`; measures the real
  footprint of a received region (`inspect_region`) before `HandleTable::admit`; forwards a
  descriptor only to the addressed consumer after `HandleTable::consume`; and treats `Consumed`
  as a message that frees nothing for a shared carrier. A lost worker is fenced
  (`unregister_producer`: its open handles can no longer be consumed) and reaped; only when
  **both** workers of a charged pair are reaped (`recycle` reaps the counterparts) does the host
  call `release_after_teardown` and `SharedHostQuota::release_pair`.
- **What the worker may not decide.** The host chooses the carrier. `finish_produce` refuses a
  handle whose carrier is not the chosen one, whose fields differ from the template in anything
  but the signature, or whose descriptor count is not exactly 1 (`HostShm`) or 0, and a `HostShm`
  region that is not a sealed `memfd` of exactly the tensor's size; the descriptor is closed and
  the worker fenced and reaped (`HostError::Violation`). `Produced` carries the chosen carrier,
  and `admit`/`consume` use it, never the handle's; a region is forwarded only for `HostShm`.
  A malformed or oversized reply, or a `Consumed` whose digest or length differs from the
  handle's, fences and reaps the worker too. `admit` and `consume` look at the supervisor first,
  so a lost worker is fenced without anyone polling; fencing is by session
  (`unregister_producer_session`), so reaping an old session after a respawn leaves the new one.
  `PodHost::expire` fans out to every table and must be called with the host's clock.
- **Seccomp is checked twice.** The kernel must show a filter (`Seccomp: 2`), and the worker's own
  `probe` must report `EPERM` for every forbidden call (a self-report of the image the host
  started, not a kernel proof); only then is the worker `Full`-confined.
- **S1 limits.** The tensor travels inside the produce call (one 64 KiB frame, about 60 KiB) and
  a worker keeps at most 64 exports; no `PDEATHSIG`, so a SIGKILLed supervisor leaves its workers
  and `ptr-worker-*` cgroups behind.
- **Failure policy.** Every wait has a deadline, every queue is bounded, every refusal is a typed
  `HostError`. A worker started without full confinement keeps working: its tensors go by `Bytes`.

Tests: `tests/pod_data_plane.rs` (feature `data-plane,fault-injection`) starts a real supervisor
and real workers: handle transfer by `HostShm` under full confinement or by `Bytes` (it prints
`PATH: ...`), a producer crash mid-flight (fence, open handle refused, group reaped, drain closes
the queued descriptor, ledger recredited only after both reaps), a consumer that lies about
`Consumed` and keeps its mapping, `EPERM` for `socket`/`fork` after seccomp, a bad key-challenge
signature and a silent worker. Tests that need root skip with a printed reason; under
`CI=true` they fail unless `PTR_WORKER_UNPRIVILEGED_CI=1`. CI job `pod-worker`
(`make ci-pod-worker`) builds the test binaries as the runner user and runs them under
`sudo -E` (`scripts/run_pod_worker_tests.py`). The `fault-injection` feature adds
`--fault bad-signature|silent|carrier-swap|tamper-field|region-unsealed|region-wrong-size|region-not-memfd` to the worker, and the "keeps its mapping after `Consumed`" hook, for those tests and is never part of a release build.

## The CUDA staging lifecycle (stage S3, Linux, feature `data-plane`)

`docs/architecture/42-pod-data-plane.md` rules 3, 3a and 3b: the staging buffers a `CudaIpc` or
`DevicePeer` export is made from are owned by the supervisor and held by a **guardian** process,
not by a worker.

```text
StagingHost (lib: staging_host) --StagingPool + ExportRegistry-->  RemoteBackend --Channel--> ptr-worker guardian
   |  reap_worker / watch / shutdown / recycle                               (holds a StagingBackend: fake or CUDA;
   v                                                                          reads only this channel; not seccomp)
PodHost  --Supervisor-->  workers (own cgroup, uid, netns, seccomp)
```

- **Guardian (`guardian --channel-fd <n> --backend fake|cuda`, module `guardian`).** Serves
  `ptr_fabric::serve_guardian` on the inherited channel: allocate, fill, export (a descriptor
  rides with the reply), scrub, release. It accepts no worker input (a request with a descriptor is
  refused), runs in a cgroup and network namespace of its own with no capabilities and
  `no_new_privs`, and is **not** seccomp-confined: it is the supervisor's own code and needs the
  driver calls the worker allowlist forbids. `fake` is the memfd-backed CPU double used by the tests
  and on a host without a GPU; `cuda` needs a build with `cuda-ipc` (toolkit-free, dynamic loading)
  or `cuda-ipc-core` and has not run on a GPU.
- **Host (`StagingHost`, module `staging_host`).** Owns the guardian (`ptr_fabric::GuardianHost`),
  the pool over it and the export registry. `issue_export` leases a buffer and issues the record;
  `admit_import` consumes the record once; `reap_worker` reaps through the `PodHost` and then tells
  the pool (which frees a buffer only after **both** its domains were reaped and makes it reusable
  after its consumer's reap) and the registry. **Order:** `shutdown` reaps every worker domain
  first and the guardian last, logging each step with its time and whether the guardian was still
  alive; `watch` notices a dead guardian, terminates and reaps every worker whose buffers it held
  (a typed `StagingEvent::GuardianLost` names them), drops the pool, charges and records, and
  reaps the guardian's remains last. The loss is stored, so a worker reap that fails is retried by the next `watch`; a guardian that stops answering (request timeout) is killed and counts as lost. End a `StagingHost` with `shutdown` (a drop that still backs workers only warns). A producer-quota refusal comes with `recovery_plan`, and
  `recycle` reaps the producer and the listed consumers, oldest export first.
- **Not built.** The staging host is not in `PodHost`'s produce/admit/consume path (no CUDA handle
  is produced or consumed through it yet), there is no adapter from the pool's per-session
  `FootprintSink` to `SharedHostQuota`, and the CUDA backend has never run on a GPU.

Tests: `tests/staging_guardian.rs` (feature `data-plane`, runs unprivileged and as root) uses a
real guardian process and real workers; the GPU test in it is `#[ignore]`. CI: `pod-worker`
(`scripts/run_pod_worker_tests.py`) and `fabric-cuda-check`.

## What it does not do

- Without the `data-plane` feature (and on every non-Linux target) no process isolation beyond
  being a separate OS process: no sandbox, seccomp or resource limits, and a Pod runs in the
  worker's own address space. With it, confinement is the supervisor's and is only as good as the
  host provides (see `crates/ptr-fabric`); there are no resource limits (memory, CPU) beyond the
  cgroup membership.
- No network transport (no TCP, no Iroh, no TLS); the peer is whoever holds the pipes (stdio) or
  the supervisor that spawned the worker (channel, authenticated by the key challenge).
- No model Pods: no weights, no Candle/Qwen executors, no GPU path.
- One session per process, one call executing at a time; no concurrency, no deadlines, no
  Pod lifecycle or lease management.
- `ptr-podwire`, `ptrd` and the runtime's request path do not call this binary or `PodHost` yet;
  a Windows named-pipe channel and a scheduler are later work (doc 42 stage S1 scope guard).
