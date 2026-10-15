//! The `PinnedHost` carrier's pinned buffer (doc 42 section 2): an existing sealed `HostShm`
//! mapping, registered with the driver (`cuMemHostRegister`) so the GPU can copy to and from
//! it. The pinned buffer **is** the `HostShm` region: nothing is copied into a second buffer,
//! and the region keeps its seals and its digest check.
//!
//! The registration is read-only (`CU_MEMHOSTREGISTER_READ_ONLY`), which the sealed, `PROT_READ`
//! mapping requires; a device without read-only host registration is refused with a typed
//! error. Nothing here has run on a GPU.

use std::ffi::c_void;

use cudarc::driver::sys;

use super::driver::{check, guarded, guarded_drop, CudaDriver, DeviceContext};
use super::footprint::footprint_memfd;
use super::CudaError;
use crate::{host_page_size, ShmMapping};

/// A `HostShm` mapping registered as pinned memory. Borrows the mapping, so the mapping cannot
/// be unmapped while it is registered; unregisters on drop.
pub struct PinnedRegistration<'a> {
    mapping: &'a ShmMapping,
    context: DeviceContext,
}

impl<'a> PinnedRegistration<'a> {
    /// Register `mapping` as portable, read-only pinned host memory in the primary context of
    /// `device`. Refuses a device that cannot register read-only memory.
    pub fn register(
        driver: &CudaDriver,
        device: u32,
        mapping: &'a ShmMapping,
    ) -> Result<Self, CudaError> {
        guarded(|| Self::register_inner(driver, device, mapping))
    }

    fn register_inner(
        driver: &CudaDriver,
        device: u32,
        mapping: &'a ShmMapping,
    ) -> Result<Self, CudaError> {
        let handle = driver.device(device)?;
        let read_only = driver.attribute(
            handle,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
        )?;
        if read_only == 0 {
            return Err(CudaError::Unsupported {
                what: "the device cannot register read-only host memory",
            });
        }
        let context = DeviceContext::retain(driver, device)?;
        let guard = context.enter()?;
        let bytes = mapping.as_bytes();
        // The kernel maps whole pages, so the page-rounded length lies inside the mapping and
        // is what the driver pins anyway; registering it makes the registered range equal to
        // the footprint that rule 3b charges.
        let registered = footprint_memfd(bytes.len() as u64, host_page_size())
            .and_then(|length| usize::try_from(length).ok())
            .ok_or(CudaError::Overflow)?;
        // SAFETY: the pointer and length are those of a live read-only mapping owned by
        // `mapping`, which this value borrows for its whole life, so the range stays mapped
        // until `Drop` unregisters it; READ_ONLY matches the mapping's PROT_READ, and the
        // context is current.
        check("cuMemHostRegister_v2", unsafe {
            sys::cuMemHostRegister_v2(
                bytes.as_ptr().cast_mut().cast::<c_void>(),
                registered,
                sys::CU_MEMHOSTREGISTER_READ_ONLY | sys::CU_MEMHOSTREGISTER_PORTABLE,
            )
        })?;
        drop(guard);
        Ok(Self { mapping, context })
    }

    /// The registered length (the tensor length).
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Always false: an empty tensor has no mapping.
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// The host memory the registration costs: the length rounded up to whole pages (rule 3b).
    pub fn footprint_bytes(&self) -> Option<u64> {
        footprint_memfd(self.mapping.len() as u64, host_page_size())
    }
}

impl Drop for PinnedRegistration<'_> {
    fn drop(&mut self) {
        guarded_drop(|| {
            let _guard = self.context.enter().ok();
            // SAFETY: the range was registered by `register` and is still mapped (the borrow of
            // the mapping outlives this value); a failure cannot be reported from a destructor.
            let _ = unsafe {
                sys::cuMemHostUnregister(
                    self.mapping.as_bytes().as_ptr().cast_mut().cast::<c_void>(),
                )
            };
        });
    }
}

impl std::fmt::Debug for PinnedRegistration<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedRegistration")
            .field("len", &self.mapping.len())
            .finish()
    }
}
