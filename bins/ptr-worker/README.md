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
ptr-worker serve --channel-fd <n> [--confined] [--staging fake|cuda]
                            serve one session over an inherited SOCK_SEQPACKET channel
                            (Linux, feature data-plane); --staging serves the CudaIpc staged
                            calls with the import that pairs with the guardian's --backend
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
- **S1 limits.** The tensor travels inside the produce call (one 64 KiB frame, about 60 KiB; the
  `CudaIpc` path below carries it in a `memfd` instead) and
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
`--fault bad-signature|silent|carrier-swap|tamper-field|region-unsealed|region-wrong-size|region-not-memfd|staging-wrong-size|staging-wrong-device|staging-corrupt` to the worker, and the "keeps its mapping after `Consumed`" hook, for those tests and is never part of a release build.

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
- **In the `PodHost` path (CudaIpc).** See the next section. What is not built: the producer side
  of the CUDA staging path (a worker writing through an imported VMM descriptor; `ptr-fabric` has
  only the guardian's fill, at most about 64 KiB per call), the `FootprintSink` is not an adapter
  onto `SharedHostQuota` (by decision, see below), and the CUDA backend has never run on a GPU.

Tests: `tests/staging_guardian.rs` (feature `data-plane`, runs unprivileged and as root; the
guardian-loss tests need `fault-injection`) uses a real guardian process and real workers; the
GPU test in it is `#[ignore]`. CI: `pod-worker` (`scripts/run_pod_worker_tests.py`) and
`fabric-cuda-check`.

## The CudaIpc carrier through `PodHost` (doc 42 rules 1, 3, 3a, 3b; CPU double, Linux)

```text
start_produce_staged   lease (issue_export) -> template locator CudaIpc{device, Vmm{0, footprint}}
   |                   -> ProduceStaged: [staging descriptor, sealed memfd with the tensor] --> producer
   v                      the producer writes the tensor through the buffer, signs the template
finish_produce         handle == template (locator included), no descriptor comes back
   v
admit_staged           ImportClaim{producer, consumer, parent_operation_id, device, alloc_size}
   |                   -> admit_import (record consumed)  -- BEFORE -->  HandleTable::admit
   v                   (footprint = the record's size; ledger charged here, once)
consume_staged         fresh descriptor + export record -> the addressed consumer only
                       -> ConsumeStaged: copy out, unmap, digest on the copy -> Consumed (frees nothing)
```

- **The host chooses.** `carrier_between_staged` answers `CudaIpc` only if the `StagingHost` is
  passed in and usable, both workers were placed on the same `Some(device)` and started with the
  guardian's import (`Placement::{device, staging}`, `StagingHost::import()`), and
  `carrier_permitted(CudaIpc)` holds for both (full confinement); otherwise it answers exactly
  what `carrier_between` answers (`HostShm`, then `Bytes`). The unstaged `start_produce`, `admit`
  and `consume` never choose it and refuse a `CudaIpc` handle (`HostError::StagingRequired`).
- **No 64 KiB limit on this path.** The tensor reaches the producer in a sealed `memfd` (the
  second descriptor of the call), not in the control frame, and the producer writes it into the
  staging buffer itself, so neither the frame size nor the guardian's fill limit applies. A
  produced locator is checked against the template exactly: a worker that signs another device or
  allocation size is a violation (fenced, reaped).
- **Admission order.** A locator that finds no live record for its producer session, consumer
  session, operation, device and size is `HostError::Staging(Registry(NoRecord))` and the table is
  never reached; a replay or a second admission finds the record spent. If the table then refuses
  the handle the record stays spent and the buffer stays leased until its workers are reaped (the
  pool's rule, bounded by its limits).
- **Charging, exactly once (for admitted bytes).** The host-wide ledger (`SharedHostQuota`) charges *session pairs*; the
  pool's `FootprintSink` charges *single sessions*, so it cannot be an adapter onto the ledger.
  Decision: the ledger is charged once per admitted handle by `HandleTable::admit` with the
  record's footprint and released by the pair's verified teardown; the pool's sink is its own book
  of leased buffers (use `ChargeBook` to watch it), bounded by the pool's caps and released when
  the pool frees the buffer. `Consumed` and the reap of one side release nothing in either; both
  reaps release both. One difference is deliberate and tested: a buffer reused for a new
  consumer is one buffer in the pool and a second pair charge in the ledger until the producer is
  reaped (an over-count of admitted bytes); a lease that is never admitted is in the pool's
  book only. **The ledger bounds only admitted bytes**: leases that are never admitted are
  bounded by `PoolLimits` alone, so `PoolLimits` must be at or below the ledger's caps.
- **Reaps the host did itself** (a protocol violation) reach the pool through
  `StagingHost::sync_reaped`, which `watch` and every staged call run first. It frees only a
  session the host registers no more *and* remembers as reaped (`PodHost::is_reaped`, a bounded
  list: a forgotten session is freed late, at `shutdown`, never early); a direct `PodHost::reap`
  leaves the pool stale until the next `watch` or staged call. `StagingHost::issue_export_for`
  refuses a lease for a session the host does not have.
- **A CUDA import is not chosen yet.** While the CUDA producer cannot write through an imported
  descriptor, `carrier_between_staged` falls back for a `--staging cuda` host (each attempt would
  leave a device lease); two live leases of one key must be admitted in lease order
  (`HostError::StagedExportMismatch` otherwise).
- **Worker side.** `--staging fake` maps the staging `memfd` (`staging_import`); `--staging cuda`
  uses `ptr_fabric::import_and_copy` for the consumer and refuses the producer with
  `cuda-producer-write-not-built`. Nothing here has run on a GPU. A worker started without
  `--staging` refuses staged calls with `staging-not-served`.

Tests: `tests/cuda_ipc_flow.rs` (feature `data-plane,fault-injection`; the transfers need root and
skip with a reason otherwise, the fallback tests run everywhere): an end-to-end transfer with the
digest checked, a 1 MiB tensor, exact-once charging with a reading after each step in both reap
orders, a forged locator (size, device), a claim by another consumer or producer, a replay, the
unstaged calls refusing a `CudaIpc` handle, a consumer refusing corrupt bytes, reuse after the
consumer's reap with the tail scrubbed, a producer crash after producing and during the call,
and the fallbacks. Unit tests of the worker's staged calls are in `channel_serve`.

## The GPU seccomp profile (`--gpu-profile`, builds with `cuda-ipc-core`)

`serve --channel-fd <n> --confined --gpu-profile <device>` initialises CUDA on the device first
(`prepare_gpu_worker`; the filter refuses `open`), then installs the weaker GPU filter
(`install_gpu_filter`) and keeps the `GpuWorker` for its CUDA imports. A worker that cannot (no
driver, no such device, a build without the feature) answers the challenge with a startup refusal
(`gpu-no-driver`, `gpu-no-such-device`, `gpu-prepare-failed`, `gpu-filter-failed`,
`gpu-not-built`) and the host reports `HostError::WorkerStartup(code)` and reaps it. Every
confined worker answers `probe` with `probe_profile_calls`; the host classifies the filter
(`observed_profile`), requires it to equal `Placement::profile` (`ProfileMismatch`,
`ProfileUnproven` reap the worker), keeps it per worker (`PodHost::profile`), decides carriers with
`carrier_permitted_for(carrier, &WorkerConfinement)` and exposes what a reduced profile gives up
(`PodHost::profile_weakening`). The classification is a self-report of the image the host started.
Nothing of it has run on a GPU; the `#[ignore]` test
`a_gpu_profile_worker_runs_its_filter_and_is_classified_gpu_on_a_gpu` is in `docs/GPU_HANDOVER.md`
section 12.2 step 4.

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
