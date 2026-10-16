//! Preparing a GPU worker: everything the CUDA driver opens lazily, done **before** the GPU
//! seccomp filter is installed (doc 42, the GPU worker profile).
//!
//! The GPU filter (`install_gpu_worker_seccomp`, feature `worker-confine`) refuses
//! `open` and `openat`, so a driver that has not opened its device nodes yet cannot do it
//! afterwards. [`prepare_gpu_worker`] therefore loads the driver, runs `cuInit`, retains the
//! primary context of one device (which opens `/dev/nvidia<N>` and registers it with
//! `/dev/nvidia-uvm`), and then exercises the paths a worker uses later once, so that whatever
//! they open or map lazily is open or mapped:
//!
//! 1. the context itself: a small device allocation, a memset, a copy back and a synchronise;
//! 2. the VMM path of the `CudaIpc` carrier: create, fill, export, import, map and copy a
//!    one-granule staging allocation inside this process;
//! 3. the pinned-host path of the `PinnedHost` carrier: register a sealed `HostShm` mapping and
//!    unregister it.
//!
//! Steps 2 and 3 are **soft**: a device without VMM export or without read-only host
//! registration still yields a [`GpuPrepared`], and the failure is kept
//! ([`GpuPrepared::vmm_warm_up`], [`GpuPrepared::pinned_warm_up`]); the filtered worker then
//! refuses that path with the same typed error instead of failing inside the driver. Without a
//! driver the call returns [`CudaError::NoDriver`] and nothing else runs.
//!
//! # What still fails after the filter
//!
//! Expected to fail once the GPU filter is on, because the call needs a new file, socket or
//! descriptor (all of this is **unverified**: nothing here has run on a GPU, and the list comes
//! from the driver's documented behaviour, not from a trace):
//!
//! * the first context of a device that was not prepared (`cuDevicePrimaryCtxRetain` opens that
//!   device's node), and so peer access to a second device;
//! * loading another library: PTX JIT compilation (`libnvidia-ptxjitcompiler`,
//!   `libnvidia-nvvm`), `libcudart`, `libcublas` and the like are `dlopen`ed on first use; only
//!   kernels that are already compiled for the device run;
//! * queries the driver answers from `/proc` or `/sys` (NUMA affinity, `/proc/driver/nvidia`)
//!   that no step above has asked once;
//! * anything that creates a descriptor of its own: `eventfd`, `pipe2` or `timerfd` inside the
//!   driver.
//!
//! `strace -f` on `tests/gpu_worker.rs` names the refused call if a card shows one more; adding
//! it is a reviewed change of `GPU_ALLOWLIST_CHANGES`, not a runtime option.

use std::ffi::c_void;

use cudarc::driver::sys;

use super::driver::{check, guarded, CudaDriver, DeviceContext};
use super::record::ImportExpectation;
use super::vmm::{import_and_copy, StagingAllocation};
use super::{CudaError, PinnedRegistration};
use crate::{ShmExpectation, ShmExport, ShmMapping};

/// The size of the context warm-up buffer.
const WARM_BYTES: usize = 256;

/// A CUDA driver that is loaded and initialised, with the primary context of one device
/// retained and its lazy paths warmed. Created before the GPU filter; consumed by
/// `install_gpu_filter` (feature `worker-confine`), which returns the `GpuWorker` that may use
/// it afterwards. Holding it keeps the context retained for as long as it lives.
pub struct GpuPrepared {
    // Read by `GpuWorker` (feature `worker-confine`) only.
    #[cfg_attr(not(feature = "worker-confine"), allow(dead_code))]
    pub(crate) driver: CudaDriver,
    pub(crate) _context: DeviceContext,
    pub(crate) device: u32,
    pub(crate) vmm: Result<(), CudaError>,
    pub(crate) pinned: Result<(), CudaError>,
}

impl GpuPrepared {
    /// The device whose context is retained. The only one a filtered worker may use.
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Whether the VMM create, export, import, map and copy round trip ran without error. An
    /// `Err` here means `CudaIpc` imports are refused by the filtered worker.
    pub fn vmm_warm_up(&self) -> &Result<(), CudaError> {
        &self.vmm
    }

    /// Whether registering and unregistering a sealed `HostShm` mapping ran without error. An
    /// `Err` here means pinned registration is refused by the filtered worker.
    pub fn pinned_warm_up(&self) -> &Result<(), CudaError> {
        &self.pinned
    }
}

impl std::fmt::Debug for GpuPrepared {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuPrepared")
            .field("device", &self.device)
            .field("vmm", &self.vmm)
            .field("pinned", &self.pinned)
            .finish()
    }
}

/// Load the driver, initialise it, retain the primary context of `device` and warm the paths a
/// GPU worker uses later (see the module docs). Call it in the worker **before** the GPU filter.
///
/// Returns [`CudaError::NoDriver`] on a host without a usable driver (no `libcuda`, or `cuInit`
/// fails), [`CudaError::NoSuchDevice`] for an ordinal the driver does not have, and a
/// [`CudaError::Driver`] if the context or its first allocation fails. Never panics: a missing
/// driver entry point is [`CudaError::MissingEntryPoint`].
pub fn prepare_gpu_worker(device: u32) -> Result<GpuPrepared, CudaError> {
    let driver = CudaDriver::open()?;
    guarded(|| prepare_inner(driver, device))
}

fn prepare_inner(driver: CudaDriver, device: u32) -> Result<GpuPrepared, CudaError> {
    // The ordinal must exist before anything is retained.
    driver.device(device)?;
    let context = DeviceContext::retain(&driver, device)?;
    warm_context(&context)?;
    let vmm = guarded(|| warm_vmm(&driver, device));
    let pinned = guarded(|| warm_pinned(&driver, device));
    Ok(GpuPrepared {
        driver,
        _context: context,
        device,
        vmm,
        pinned,
    })
}

/// A device allocation, a memset, a copy back and a synchronise on the retained context.
fn warm_context(context: &DeviceContext) -> Result<(), CudaError> {
    let _current = context.enter()?;
    let mut pointer: sys::CUdeviceptr = 0;
    // SAFETY: `pointer` is a valid out pointer and `WARM_BYTES` is a small non-zero size; the
    // context is current.
    check("cuMemAlloc_v2", unsafe {
        sys::cuMemAlloc_v2(&mut pointer, WARM_BYTES)
    })?;
    let outcome = (|| {
        // SAFETY: `[pointer, pointer + WARM_BYTES)` was just allocated on the current context.
        check("cuMemsetD8_v2", unsafe {
            sys::cuMemsetD8_v2(pointer, 0xa5, WARM_BYTES)
        })?;
        let mut host = [0u8; WARM_BYTES];
        // SAFETY: the destination is `WARM_BYTES` writable bytes and the source range is the
        // allocation above.
        check("cuMemcpyDtoH_v2", unsafe {
            sys::cuMemcpyDtoH_v2(host.as_mut_ptr().cast::<c_void>(), pointer, WARM_BYTES)
        })?;
        // SAFETY: a plain call on the current context.
        check("cuCtxSynchronize", unsafe { sys::cuCtxSynchronize() })?;
        if host.iter().all(|byte| *byte == 0xa5) {
            Ok(())
        } else {
            Err(CudaError::Unsupported {
                what: "the warm-up buffer read back other bytes than were written",
            })
        }
    })();
    // SAFETY: `pointer` came from `cuMemAlloc_v2` above and is freed once; a failure cannot
    // change the outcome that is returned.
    let _ = unsafe { sys::cuMemFree_v2(pointer) };
    outcome
}

/// The VMM round trip in this process: one granule, created, filled, exported, imported and
/// copied back, then released.
fn warm_vmm(driver: &CudaDriver, device: u32) -> Result<(), CudaError> {
    let mut allocation = StagingAllocation::create(driver, device, 1)?;
    let digest = allocation.fill(&[0x5a])?;
    let export = allocation.export(allocation.record_for(0, 0, 0))?;
    let expected = ImportExpectation {
        byte_len: 1,
        content_digest: digest,
        consumer_session: 0,
        operation: 0,
        source_device: device,
        consumer_device: device,
        locator_alloc_size: allocation.footprint(),
    };
    let copy = import_and_copy(driver, export.fd, &export.record, &expected)?;
    if copy == [0x5a] {
        Ok(())
    } else {
        Err(CudaError::DigestMismatch)
    }
}

/// Register and unregister a small sealed `HostShm` mapping.
fn warm_pinned(driver: &CudaDriver, device: u32) -> Result<(), CudaError> {
    let bytes = [0x3cu8; 64];
    let export = ShmExport::create(&bytes).map_err(|error| CudaError::Io(error.to_string()))?;
    let expectation = ShmExpectation {
        byte_len: export.byte_len(),
        offset: 0,
        content_digest: export.content_digest(),
    };
    let mapping = ShmMapping::map(export.into_fd(), &expectation)
        .map_err(|error| CudaError::Io(error.to_string()))?;
    let registration = PinnedRegistration::register(driver, device, &mapping)?;
    drop(registration);
    Ok(())
}
