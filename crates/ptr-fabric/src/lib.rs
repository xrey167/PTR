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
//! whole cgroup, then drains the old channel end.
//!
//! `cuda` (feature `cuda-ipc`, Linux only) holds the CUDA foundations of the `CudaIpc`,
//! `DevicePeer` and `PinnedHost` carriers: the driver probe, the footprint rule, the VMM staging
//! export and import and the pinned registration of a `HostShm` region. It loads `libcuda` at
//! run time and answers with a typed error when there is none. Nothing in it has run on a GPU.

/// Whether the `cuda` module is compiled into this build: the feature `cuda-ipc` on Linux.
pub const CUDA_IPC_COMPILED: bool = cfg!(all(feature = "cuda-ipc", target_os = "linux"));

mod bytes;
#[cfg(all(feature = "worker-channel", target_os = "linux"))]
mod channel;
#[cfg(all(feature = "worker-confine", target_os = "linux"))]
mod confine;
#[cfg(all(feature = "cuda-ipc", target_os = "linux"))]
mod cuda;
mod error;
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
mod shm;
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
    carrier_permitted, install_worker_seccomp, install_worker_seccomp_with, probe_forbidden_calls,
    worker_allowlist, AllowedSyscall, AppliedMechanisms, Availability, ConfineError, Confinement,
    ConfinementProbe, DenyAction, ForbiddenCall, Mechanism,
};
#[cfg(all(feature = "cuda-ipc", target_os = "linux"))]
pub use cuda::{
    audit_export_padding, check_extent, check_import, footprint_legacy, footprint_memfd,
    footprint_vmm, import_and_copy, padding_is_zero, CudaDriver, CudaError, CudaProbe, DeviceInfo,
    ExportRecord, ImportExpectation, PaddingAudit, PeerLink, PinnedRegistration, StagingAllocation,
    StagingExport, LEGACY_STAGING_UNIT,
};
pub use error::{FabricError, Seal};
#[cfg(all(feature = "shm-backend", target_os = "linux"))]
pub use shm::{host_page_size, inspect_region, ShmExpectation, ShmExport, ShmMapping, ShmRegion};
#[cfg(all(feature = "worker-supervisor", target_os = "linux"))]
pub use supervisor::{
    fill_random, ConfinePolicy, ExitInfo, LostCause, ReapFailure, ReapedWorker, SessionId,
    SpawnRequest, SpawnedWorker, Step, Supervisor, SupervisorConfig, SupervisorError,
    SupervisorEvent, WorkerObservation, DEFAULT_CHANNEL_FD, REQUIRED_REFUSED_CALLS,
};
