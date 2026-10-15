//! Carriers of the Pod data plane: how the bytes behind a `TensorHandle` travel.
//!
//! The handle (in `ptr-protocol`) says where a tensor is; a carrier here moves or maps the
//! bytes and checks them against the handle's commitment before they are used. Two carriers
//! exist: `Bytes` (inline in the handle, every platform) and `HostShm` (a sealed `memfd`,
//! Linux only, feature `shm-backend`).
//!
//! Two more Linux-only pieces serve the Pod worker process (doc 42 stage S1): `channel`
//! (feature `worker-channel`, framed messages with `SCM_RIGHTS` descriptors) and `confine`
//! (feature `worker-confine`, the host probe, the carrier policy and the seccomp allowlist);
//! `supervisor` (feature `worker-supervisor`) spawns a confined worker, watches it and reaps its
//! whole cgroup, then drains the old channel end. `confine` also has the **GPU profile**, a
//! second, weaker seccomp allowlist for a worker that must call the CUDA driver (`ioctl` widened,
//! everything that opens a new resource still refused), the typed `SeccompProfile` that travels
//! with the confinement, and, with `cuda-ipc-core`, `prepare_gpu_worker` and `GpuWorker`, which
//! initialise CUDA before the filter is installed. Nothing of the GPU profile has run on a GPU.
//!
//! `cuda` (feature `cuda-ipc-core`, or `cuda-ipc` for the toolkit-free dynamic-loading variant;
//! Linux only) holds the CUDA foundations of the `CudaIpc`, `DevicePeer` and `PinnedHost`
//! carriers: the driver probe, the footprint rule, the VMM staging export and import and the
//! pinned registration of a `HostShm` region. With `cuda-ipc` it loads `libcuda` at run time and
//! answers with a typed error when there is none. Nothing in it has run on a GPU.
//!
//! `staging` (feature `shm-backend`, the guardian protocol with `worker-channel`, the guardian
//! process handle with `worker-supervisor`) is the supervisor-owned staging lifecycle of those
//! carriers (doc 42 rules 3, 3a, 3b): a backend trait with a CPU fake and a CUDA backend, the
//! pool, the export registry and the guardian.

/// Whether the `cuda` module is compiled into this build: the feature `cuda-ipc-core` (which
/// `cuda-ipc` includes) on Linux.
pub const CUDA_IPC_COMPILED: bool = cfg!(all(feature = "cuda-ipc-core", target_os = "linux"));

mod bytes;
#[cfg(all(feature = "worker-channel", target_os = "linux"))]
mod channel;
#[cfg(all(feature = "worker-confine", target_os = "linux"))]
mod confine;
#[cfg(all(feature = "cuda-ipc-core", target_os = "linux"))]
mod cuda;
mod error;
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
mod export_record;
#[cfg(all(
    feature = "cuda-ipc-core",
    feature = "worker-confine",
    target_os = "linux"
))]
mod gpu_worker;
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
mod shm;
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
mod staging;
#[cfg(all(feature = "worker-supervisor", target_os = "linux"))]
mod supervisor;

pub use bytes::{export_bytes, import_bytes};
#[cfg(all(feature = "worker-channel", target_os = "linux"))]
pub use channel::{
    drain, drain_with_limit, Channel, ChannelError, DrainEnd, DrainReport, Frame,
    DRAIN_MAX_RECEIVES, MAX_FDS_PER_FRAME, MAX_FRAME_BODY_BYTES,
};
#[cfg(all(feature = "worker-confine", target_os = "linux"))]
pub use confine::{
    carrier_permitted, carrier_permitted_for, gpu_worker_allowlist, install_gpu_worker_seccomp,
    install_gpu_worker_seccomp_with, install_worker_seccomp, install_worker_seccomp_with,
    observed_profile, probe_forbidden_calls, probe_profile_calls, worker_allowlist, AllowedSyscall,
    AppliedMechanisms, Availability, ConfineError, Confinement, ConfinementProbe, DenyAction,
    ForbiddenCall, GpuAllowlistChange, GpuChangeKind, Mechanism, ObservedProfile, SeccompProfile,
    WorkerConfinement, GPU_ALLOWLIST_CHANGES, GPU_PROFILE_WEAKENING, PROFILE_PROBE_CALL,
};
#[cfg(all(feature = "cuda-ipc-core", target_os = "linux"))]
pub use cuda::{
    audit_export_padding, check_extent, check_import, footprint_legacy, footprint_memfd,
    footprint_vmm, import_and_copy, padding_is_zero, prepare_gpu_worker, CudaDriver, CudaError,
    CudaProbe, DeviceInfo, GpuPrepared, ImportExpectation, PaddingAudit, PeerLink,
    PinnedRegistration, StagingAllocation, StagingExport, LEGACY_STAGING_UNIT,
};
pub use error::{FabricError, Seal};
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
pub use export_record::ExportRecord;
#[cfg(all(
    feature = "cuda-ipc-core",
    feature = "worker-confine",
    target_os = "linux"
))]
pub use gpu_worker::GpuWorker;
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
pub use shm::{host_page_size, inspect_region, ShmExpectation, ShmExport, ShmMapping, ShmRegion};
#[cfg(all(feature = "worker-channel", target_os = "linux"))]
pub use staging::{
    serve_guardian, GuardianError, GuardianExit, GuardianLimits, GuardianSummary, RemoteAllocation,
    RemoteBackend, MAX_FILL_BYTES,
};
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
pub use staging::{
    AcquireRequest, AdmittedExport, BackendError, ChargeBook, ChargeRefused, ExportId,
    ExportRegistry, FakeAllocation, FakeBackend, FakeCounters, FootprintSink, ImportClaim, Lease,
    NoCharges, PairKey, PoolLimits, PoolRefusal, ReapReport, RecoveryPlan, RegistryRefusal, SlotId,
    SlotInfo, StagingBackend, StagingPool,
};
#[cfg(all(feature = "cuda-ipc-core", target_os = "linux"))]
pub use staging::{CudaAllocation, CudaBackend};
#[cfg(all(feature = "worker-supervisor", target_os = "linux"))]
pub use staging::{GuardianHost, GuardianHostConfig};
#[cfg(all(feature = "worker-supervisor", target_os = "linux"))]
pub use supervisor::{
    fill_random, ConfinePolicy, ExitInfo, LostCause, ReapFailure, ReapedWorker, SessionId,
    SpawnRequest, SpawnedWorker, Step, Supervisor, SupervisorConfig, SupervisorError,
    SupervisorEvent, WorkerObservation, DEFAULT_CHANNEL_FD, REQUIRED_REFUSED_CALLS,
};
