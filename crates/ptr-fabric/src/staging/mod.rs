//! The supervisor-owned staging lifecycle of the CUDA carriers (doc 42 rules 3, 3a and 3b).
//! Linux only, feature `shm-backend`; the guardian protocol needs `worker-channel` and the
//! guardian process handle `worker-supervisor`.
//!
//! The staging buffers a `CudaIpc` or `DevicePeer` export is made from are owned by the
//! supervisor and held by its **guardian** process, never by a worker: freeing an exported
//! allocation before every importer has closed it is undefined behaviour, and a crashing
//! producer's context would free its own allocations. This module is the policy around them,
//! written against a [`StagingBackend`] so it runs on a CPU:
//!
//! - [`StagingBackend`], [`FakeBackend`] (memfd-backed, counts what happens to it) and, with
//!   `cuda-ipc-core`, `CudaBackend` over the VMM staging allocations;
//! - [`StagingPool`]: a pool per (producer session, consumer lane) that hands out a fresh buffer
//!   while the importing consumer lives, reuses a buffer only after that consumer was reaped (and
//!   zeroes all of it first), frees one only after both domains were reaped, charges every
//!   allocated footprint to producer and consumer through a [`FootprintSink`], bounds bytes per
//!   handle, pair, session and device, and plans producer quota recovery oldest export first;
//! - [`ExportRegistry`]: the supervisor's [`ExportRecord`]s, consumed once by admission;
//! - the guardian protocol ([`serve_guardian`], [`RemoteBackend`]) and, with
//!   `worker-supervisor`, [`GuardianHost`], the supervisor's handle on the guardian process.
//!
//! The order rules that join this to the worker supervisor (workers are reaped before the
//! guardian; a dead guardian ends every worker it backed) live in the host glue of
//! `ptr-worker`, where the quota and admission code is visible.

mod backend;
mod charges;
#[cfg(feature = "cuda-ipc-core")]
mod cuda_backend;
#[cfg(feature = "worker-channel")]
mod guardian;
#[cfg(feature = "worker-supervisor")]
mod guardian_host;
mod pool;
mod registry;

pub use backend::{BackendError, FakeAllocation, FakeBackend, FakeCounters, StagingBackend};
pub use charges::{ChargeBook, ChargeRefused, FootprintSink, NoCharges};
#[cfg(feature = "cuda-ipc-core")]
pub use cuda_backend::{CudaAllocation, CudaBackend};
#[cfg(feature = "worker-channel")]
pub use guardian::{
    serve_guardian, GuardianError, GuardianExit, GuardianLimits, GuardianSummary, RemoteAllocation,
    RemoteBackend, MAX_FILL_BYTES,
};
#[cfg(feature = "worker-supervisor")]
pub use guardian_host::{GuardianHost, GuardianHostConfig};
pub use pool::{
    AcquireRequest, Lease, PairKey, PoolLimits, PoolRefusal, ReapReport, RecoveryPlan, SlotId,
    SlotInfo, StagingPool,
};
pub use registry::{AdmittedExport, ExportId, ExportRegistry, ImportClaim, RegistryRefusal};
