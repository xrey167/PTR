//! The CUDA foundations of the `CudaIpc`, `DevicePeer` and `PinnedHost` carriers (doc 42 stage
//! S2), feature `cuda-ipc`, Linux only.
//!
//! - [`CudaDriver`]: proof that the driver is loaded; [`CudaDriver::probe`] measures what the
//!   host can do (devices, virtual memory management, POSIX-descriptor handles, allocation
//!   granularity, the ordered peer matrix). Without `libcuda` it answers [`CudaProbe::Absent`]
//!   and no `cudarc` call runs.
//! - [`footprint_vmm`], [`footprint_legacy`], [`footprint_memfd`]: the allocated footprint
//!   rule 3b charges.
//! - [`StagingAllocation`] / [`import_and_copy`]: the VMM staging export and its import.
//! - [`ExportRecord`](crate::ExportRecord) / [`ImportExpectation`]: the supervisor's record and the consumer's
//!   expectation, with pure checks.
//! - [`PinnedRegistration`]: a sealed `HostShm` mapping registered as pinned memory.
//! - [`prepare_gpu_worker`] / [`GpuPrepared`]: the driver initialised and warmed before a GPU
//!   worker installs its seccomp filter (the GPU profile of `confine`).
//!
//! The staging pool, the export registry and the guardian that owns the staging buffers are in
//! [`staging`](crate) (`StagingPool`, `ExportRegistry`, `CudaBackend`). Not built here: event
//! handles, the legacy `cuIpc*` form (never for a worker), the receiver-side device-to-device
//! copy into a device allocation, and peer-to-peer copies.

mod driver;
mod error;
mod footprint;
mod pinned;
mod prepare;
mod record;
mod vmm;

pub(crate) use driver::guarded;
pub use driver::{CudaDriver, CudaProbe, DeviceInfo, PeerLink};
pub use error::CudaError;
pub use footprint::{footprint_legacy, footprint_memfd, footprint_vmm, LEGACY_STAGING_UNIT};
pub use pinned::PinnedRegistration;
pub use prepare::{prepare_gpu_worker, GpuPrepared};
pub use record::{check_extent, check_import, padding_is_zero, ImportExpectation};
pub(crate) use vmm::allocation_granularity;
pub use vmm::{
    audit_export_padding, import_and_copy, PaddingAudit, StagingAllocation, StagingExport,
};
