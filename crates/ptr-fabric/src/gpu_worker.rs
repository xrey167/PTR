//! The GPU worker: CUDA initialised, then the GPU seccomp filter (doc 42, the GPU worker
//! profile). Linux only, features `cuda-ipc-core` and `worker-confine`.
//!
//! The order is the point. The filter refuses `open`, and the driver opens its device nodes
//! lazily, so a worker must finish [`prepare_gpu_worker`](crate::prepare_gpu_worker) first and
//! only then install the filter. [`GpuPrepared::install_gpu_filter`] is the one place that does
//! both in that order, and the [`GpuWorker`] it returns is the proof that the filter is in
//! force: its methods are the only way to import an export or register a pinned mapping
//! afterwards, and each refuses what the preparation did not cover (another device, a path whose
//! warm-up failed) with a typed error before any driver call.
//!
//! The confinement is weaker than the CPU worker's: see [`SeccompProfile::Gpu`] and
//! [`GPU_PROFILE_WEAKENING`]. Nothing here has run on a GPU.

use std::os::fd::OwnedFd;

use crate::confine::{install_gpu_worker_seccomp_with, ConfineError, DenyAction, SeccompProfile};
use crate::cuda::{import_and_copy, CudaError, GpuPrepared, ImportExpectation, PinnedRegistration};
use crate::{ExportRecord, ShmMapping};

impl GpuPrepared {
    /// Install the GPU-profile seccomp filter with the default deny action (`EPERM`) and return
    /// the [`GpuWorker`] that may use the prepared device afterwards. Irreversible. The filter is
    /// installed on every thread of the process, the driver's own helper threads included.
    pub fn install_gpu_filter(self) -> Result<GpuWorker, ConfineError> {
        self.install_gpu_filter_with(DenyAction::Errno)
    }

    /// Like [`install_gpu_filter`](Self::install_gpu_filter) with a chosen deny action.
    pub fn install_gpu_filter_with(self, deny: DenyAction) -> Result<GpuWorker, ConfineError> {
        install_gpu_worker_seccomp_with(deny)?;
        Ok(GpuWorker { prepared: self })
    }
}

/// A worker whose GPU seccomp filter is installed: the proof that CUDA was initialised first.
#[derive(Debug)]
pub struct GpuWorker {
    prepared: GpuPrepared,
}

impl GpuWorker {
    /// The seccomp profile this worker runs under, for the report to the host.
    pub fn profile(&self) -> SeccompProfile {
        SeccompProfile::Gpu
    }

    /// The one device this worker may use.
    pub fn device(&self) -> u32 {
        self.prepared.device()
    }

    /// Import a VMM export and copy the tensor out ([`import_and_copy`]) on the prepared device.
    ///
    /// Refused with a typed error, before any driver call, when the expectation names another
    /// device than the prepared one (a context on a device that was not prepared would have to
    /// open its node, which the filter refuses), and with the warm-up's own error when the VMM
    /// path could not be warmed.
    pub fn import_and_copy(
        &self,
        fd: OwnedFd,
        record: &ExportRecord,
        expected: &ImportExpectation,
    ) -> Result<Vec<u8>, CudaError> {
        if expected.consumer_device != self.device() || expected.source_device != self.device() {
            return Err(CudaError::Unsupported {
                what: "a filtered GPU worker uses only the device prepared before the filter",
            });
        }
        self.prepared.vmm_warm_up().clone()?;
        import_and_copy(&self.prepared.driver, fd, record, expected)
    }

    /// Register a sealed `HostShm` mapping as pinned memory ([`PinnedRegistration::register`])
    /// on the prepared device. Refused with the warm-up's own error when that path could not be
    /// warmed.
    pub fn register_pinned<'a>(
        &self,
        mapping: &'a ShmMapping,
    ) -> Result<PinnedRegistration<'a>, CudaError> {
        self.prepared.pinned_warm_up().clone()?;
        PinnedRegistration::register(&self.prepared.driver, self.device(), mapping)
    }
}
