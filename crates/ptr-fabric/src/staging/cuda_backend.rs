//! The CUDA staging backend: [`StagingBackend`] over the VMM staging allocations of the `cuda`
//! module. Feature `cuda-ipc-core`, Linux only. Held by the guardian process, never by a worker.
//!
//! Nothing here has run on a GPU. Two consequences of the `cuda` API it sits on:
//!
//! * a staging allocation is exported only after it was filled once (the digest comes from the
//!   fill), so the guardian fills it from bytes the supervisor sends; a producer that writes
//!   through an imported descriptor is a later step;
//! * an allocation cannot be zeroed in place for the next tensor, so [`scrub`](CudaBackend::scrub)
//!   creates a new, fully zeroed allocation of the same footprint and **keeps the old one in a
//!   retired list** until the buffer is released: the producer may still have the old memory
//!   imported (only the consumer was reaped), and dropping it would leave memory resident that
//!   nothing is charged for. `scrub_retains` is therefore true, and the pool counts the extra
//!   footprint against the device cap, the pair and both sessions *before* the fresh allocation
//!   is made, refusing the reuse when it does not fit. Each reuse of a CUDA buffer thus costs one
//!   more footprint until the producer is reaped; the retired allocations go with the buffer.

use std::os::fd::OwnedFd;

use super::backend::{BackendError, StagingBackend};
use crate::cuda::{
    allocation_granularity, footprint_vmm, guarded, CudaDriver, CudaError, StagingAllocation,
};
use crate::ExportRecord;

impl From<CudaError> for BackendError {
    fn from(error: CudaError) -> Self {
        BackendError::new(error.to_string())
    }
}

/// A VMM staging allocation held for the pool.
pub struct CudaAllocation {
    inner: StagingAllocation,
    /// Allocations a scrub replaced. The driver keeps their memory alive for as long as a
    /// producer's import of them lives, so they stay held (and charged by the pool, see
    /// `scrub_retains`) until the buffer is released, which happens only after the producer was
    /// reaped.
    retired: Vec<StagingAllocation>,
}

impl CudaAllocation {
    /// The allocation of the `cuda` module behind this buffer.
    pub fn staging(&self) -> &StagingAllocation {
        &self.inner
    }
}

impl std::fmt::Debug for CudaAllocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(formatter)
    }
}

/// The backend over the CUDA driver. Holds the [`CudaDriver`] proof, so it exists only on a
/// host where the driver loaded.
#[derive(Clone, Copy, Debug)]
pub struct CudaBackend {
    driver: CudaDriver,
}

impl CudaBackend {
    /// Open the driver, or [`CudaError::NoDriver`] on a host without one.
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            driver: CudaDriver::open()?,
        })
    }

    /// A backend over an already opened driver.
    pub fn with_driver(driver: CudaDriver) -> Self {
        Self { driver }
    }
}

impl StagingBackend for CudaBackend {
    type Allocation = CudaAllocation;

    fn footprint_for(&mut self, device: u32, len: u64) -> Result<u64, BackendError> {
        if len == 0 {
            return Err(CudaError::EmptyTensor.into());
        }
        let granularity = guarded(|| allocation_granularity(&self.driver, device))?;
        footprint_vmm(len, granularity).ok_or_else(|| CudaError::Overflow.into())
    }

    fn allocate(&mut self, device: u32, len: u64) -> Result<(Self::Allocation, u64), BackendError> {
        let inner = StagingAllocation::create(&self.driver, device, len)?;
        let footprint = inner.footprint();
        Ok((
            CudaAllocation {
                inner,
                retired: Vec::new(),
            },
            footprint,
        ))
    }

    fn fill(
        &mut self,
        allocation: &mut Self::Allocation,
        bytes: &[u8],
    ) -> Result<[u8; 32], BackendError> {
        Ok(allocation.inner.fill(bytes)?)
    }

    fn export(
        &mut self,
        allocation: &Self::Allocation,
        record: &ExportRecord,
    ) -> Result<OwnedFd, BackendError> {
        Ok(allocation.inner.export(*record)?.fd)
    }

    fn scrub(&mut self, allocation: &mut Self::Allocation, len: u64) -> Result<(), BackendError> {
        let device = allocation.inner.device();
        let footprint = allocation.inner.footprint();
        let fresh = StagingAllocation::create(&self.driver, device, len)?;
        if fresh.footprint() != footprint {
            return Err(BackendError::new(
                "the scrubbed allocation has another footprint than the one it replaces",
            ));
        }
        let old = std::mem::replace(&mut allocation.inner, fresh);
        allocation.retired.push(old);
        Ok(())
    }

    fn scrub_retains(&self) -> bool {
        true
    }

    fn release(&mut self, allocation: Self::Allocation) -> Result<(), BackendError> {
        drop(allocation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_a_host_without_a_driver_the_backend_does_not_open() {
        // On a host with a driver this opens, and the test has nothing more to say; the GPU
        // lifecycle is in tests/staging_cuda.rs (ignored without a GPU).
        match CudaBackend::open() {
            Ok(_) => {}
            Err(error) => assert!(matches!(error, CudaError::NoDriver { .. }), "{error:?}"),
        }
    }
}
