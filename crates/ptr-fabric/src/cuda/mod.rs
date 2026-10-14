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
//! - [`ExportRecord`] / [`ImportExpectation`]: the supervisor's record and the consumer's
//!   expectation, with pure checks.
//! - [`PinnedRegistration`]: a sealed `HostShm` mapping registered as pinned memory.
//!
//! Not built here (the guardian package): staging-buffer ownership by a guardian process, a
//! pool, event handles, the legacy `cuIpc*` form (never for a worker), the receiver-side
//! device-to-device copy into a device allocation, and peer-to-peer copies.

mod driver;
mod error;
mod footprint;
mod pinned;
mod record;
mod vmm;

pub use driver::{CudaDriver, CudaProbe, DeviceInfo, PeerLink};
pub use error::CudaError;
pub use footprint::{footprint_legacy, footprint_memfd, footprint_vmm, LEGACY_STAGING_UNIT};
pub use pinned::PinnedRegistration;
pub use record::{check_extent, check_import, padding_is_zero, ExportRecord, ImportExpectation};
pub use vmm::{
    audit_export_padding, import_and_copy, PaddingAudit, StagingAllocation, StagingExport,
};
