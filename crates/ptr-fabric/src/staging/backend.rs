//! The staging backend: what actually holds a staging buffer and hands out a descriptor for it.
//!
//! The pool and the guardian protocol are written against [`StagingBackend`], so the whole
//! lifecycle (fresh buffer per handle, reuse after the consumer is reaped, free after both are)
//! runs on a CPU with [`FakeBackend`]; `CudaBackend` (feature `cuda-ipc-core`) puts the same
//! trait over the VMM staging allocations of the `cuda` module.

use std::fmt;
use std::fs::File;
use std::os::fd::OwnedFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ptr_protocol::content_digest;
use rustix::fs::{fcntl_add_seals, memfd_create, MemfdFlags, SealFlags};

use crate::{host_page_size, ExportRecord};

/// Why a backend call failed. The text is the backend's own; nothing else is promised.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendError {
    pub reason: String,
}

impl BackendError {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "staging backend: {}", self.reason)
    }
}

impl std::error::Error for BackendError {}

/// A holder of staging buffers. Every buffer is isolated (holds one tensor and nothing else),
/// is zero outside the tensor span, and is owned by the backend until [`release`](Self::release):
/// a worker never owns one.
pub trait StagingBackend {
    /// One buffer. Dropping it without [`release`](Self::release) must also free it.
    type Allocation;

    /// The footprint an allocation of `len` bytes on `device` would have: `len` rounded up to
    /// the backend's unit (the VMM granularity, the page size). Pure; the pool charges this
    /// before it allocates, and [`allocate`](Self::allocate) must agree with it.
    fn footprint_for(&mut self, device: u32, len: u64) -> Result<u64, BackendError>;

    /// A fresh, entirely zeroed buffer for a `len` byte tensor, and its allocated footprint.
    fn allocate(&mut self, device: u32, len: u64) -> Result<(Self::Allocation, u64), BackendError>;

    /// Write the tensor into the buffer (exactly its length, once per fill) and return its
    /// SHA-256. The producer may instead write through the descriptor it is given.
    fn fill(
        &mut self,
        allocation: &mut Self::Allocation,
        bytes: &[u8],
    ) -> Result<[u8; 32], BackendError>;

    /// A new descriptor of the buffer for the one session the record names. Each call yields a
    /// new descriptor; the buffer stays allocated.
    fn export(
        &mut self,
        allocation: &Self::Allocation,
        record: &ExportRecord,
    ) -> Result<OwnedFd, BackendError>;

    /// Make the buffer fresh for the next tensor of `len` bytes: every byte zero, padding
    /// included, so a reused buffer never exposes the previous tensor's tail. The footprint is
    /// unchanged (`len` is within it).
    fn scrub(&mut self, allocation: &mut Self::Allocation, len: u64) -> Result<(), BackendError>;

    /// Whether [`scrub`](Self::scrub) leaves the memory it replaced resident until
    /// [`release`](Self::release) (a backend that cannot zero in place makes a new allocation
    /// and keeps the old one, because a producer that still has it imported keeps it alive). The
    /// pool then counts one more footprint against the device cap and the pair for every scrub,
    /// and refuses the reuse when that does not fit.
    fn scrub_retains(&self) -> bool {
        false
    }

    /// Free the buffer. Only called once both isolation domains that could hold it have been
    /// reaped. An error means the memory may still be held (the backend is unreachable): the
    /// pool then keeps the charges until the backend is gone and
    /// [`StagingPool::abandon_all`](super::StagingPool::abandon_all) runs.
    fn release(&mut self, allocation: Self::Allocation) -> Result<(), BackendError>;
}

/// What a [`FakeBackend`] has done, shared with every clone of its [`FakeCounters`].
#[derive(Debug, Default)]
struct FakeState {
    live: AtomicU64,
    live_bytes: AtomicU64,
    allocated: AtomicU64,
    exports: AtomicU64,
    scrubs: AtomicU64,
    released: AtomicU64,
}

/// A readable view of what a [`FakeBackend`] did. Cloneable, so a test keeps a copy after the
/// backend moved into a pool.
#[derive(Clone, Debug, Default)]
pub struct FakeCounters {
    state: Arc<FakeState>,
}

impl FakeCounters {
    /// Allocations that exist now.
    pub fn live(&self) -> u64 {
        self.state.live.load(Ordering::SeqCst)
    }

    /// Footprint bytes of the allocations that exist now.
    pub fn live_bytes(&self) -> u64 {
        self.state.live_bytes.load(Ordering::SeqCst)
    }

    /// Allocations ever made.
    pub fn allocated(&self) -> u64 {
        self.state.allocated.load(Ordering::SeqCst)
    }

    /// Descriptors ever exported.
    pub fn exports(&self) -> u64 {
        self.state.exports.load(Ordering::SeqCst)
    }

    /// Scrubs (zeroings before reuse) ever done.
    pub fn scrubs(&self) -> u64 {
        self.state.scrubs.load(Ordering::SeqCst)
    }

    /// Allocations ever released.
    pub fn released(&self) -> u64 {
        self.state.released.load(Ordering::SeqCst)
    }
}

/// One fake buffer: a `memfd` of exactly the footprint, so a descriptor of it behaves like a
/// real shared buffer (a producer can write through it, a consumer can read it).
#[derive(Debug)]
pub struct FakeAllocation {
    file: Option<File>,
    /// Memory a retaining scrub replaced, still resident until the allocation is released.
    retired: Vec<File>,
    device: u32,
    len: u64,
    footprint: u64,
    counters: FakeCounters,
}

impl FakeAllocation {
    /// The device ordinal the buffer is on.
    pub fn device(&self) -> u32 {
        self.device
    }

    /// The tensor length the buffer was last prepared for.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// A zero-length buffer does not exist.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The allocated footprint.
    pub fn footprint(&self) -> u64 {
        self.footprint
    }

    /// Read the whole buffer back (footprint bytes), for tests.
    pub fn read_all(&self) -> Result<Vec<u8>, BackendError> {
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| BackendError::new("released"))?;
        let mut bytes = vec![0u8; usize_of(self.footprint)?];
        file.read_exact_at(&mut bytes, 0)
            .map_err(|error| BackendError::new(format!("read: {error}")))?;
        Ok(bytes)
    }
}

impl Drop for FakeAllocation {
    fn drop(&mut self) {
        if self.file.take().is_some() {
            let count = 1 + self.retired.len() as u64;
            self.retired.clear();
            self.counters.state.live.fetch_sub(count, Ordering::SeqCst);
            self.counters
                .state
                .live_bytes
                .fetch_sub(self.footprint * count, Ordering::SeqCst);
            self.counters
                .state
                .released
                .fetch_add(count, Ordering::SeqCst);
        }
    }
}

/// A zeroed `memfd` of exactly `footprint` bytes whose size can no longer change
/// (`F_SEAL_SHRINK | F_SEAL_GROW`), like a device allocation: a worker that holds a descriptor
/// of it can write through it but cannot truncate it under another importer's mapping (that
/// would be a `SIGBUS` for the importer). It stays writable, so the producer can fill it.
fn new_fake_buffer(footprint: u64) -> Result<File, BackendError> {
    let fd = memfd_create(
        "ptr-fabric-fake-staging",
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .map_err(|error| BackendError::new(format!("memfd_create: {error}")))?;
    let file = File::from(fd);
    file.set_len(footprint)
        .map_err(|error| BackendError::new(format!("ftruncate: {error}")))?;
    fcntl_add_seals(&file, SealFlags::SHRINK | SealFlags::GROW)
        .map_err(|error| BackendError::new(format!("seal: {error}")))?;
    Ok(file)
}

fn usize_of(value: u64) -> Result<usize, BackendError> {
    usize::try_from(value).map_err(|_| BackendError::new("size does not fit this host"))
}

/// A CPU stand-in for the CUDA backend: each allocation is a `memfd` rounded up to `unit`
/// bytes, zeroed, with a counter of everything that happens to it. For tests and for guardian
/// runs on a host without a GPU; it makes no claim about device memory.
#[derive(Debug)]
pub struct FakeBackend {
    unit: u64,
    counters: FakeCounters,
    fail_allocations: bool,
    fail_scrubs: bool,
    retain_on_scrub: bool,
}

impl FakeBackend {
    /// A backend whose footprint unit is the host page size.
    pub fn new() -> Self {
        Self::with_unit(host_page_size())
    }

    /// A backend whose footprint unit is `unit` bytes (a stand-in for the VMM granularity).
    /// A zero unit is treated as 1.
    pub fn with_unit(unit: u64) -> Self {
        Self {
            unit: unit.max(1),
            counters: FakeCounters::default(),
            fail_allocations: false,
            fail_scrubs: false,
            retain_on_scrub: false,
        }
    }

    /// The counters; clone them to keep watching after the backend moved.
    pub fn counters(&self) -> FakeCounters {
        self.counters.clone()
    }

    /// Make every further scrub fail, to test the pool's handling of a buffer it cannot zero.
    pub fn fail_scrubs(&mut self, fail: bool) {
        self.fail_scrubs = fail;
    }

    /// Make a scrub behave like the CUDA one: a new zeroed buffer, with the old memory kept
    /// resident (and counted) until the allocation is released.
    pub fn retain_on_scrub(&mut self, retain: bool) {
        self.retain_on_scrub = retain;
    }

    /// Make every further allocation fail, to test the pool's rollback.
    pub fn fail_allocations(&mut self, fail: bool) {
        self.fail_allocations = fail;
    }
}

impl Default for FakeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StagingBackend for FakeBackend {
    type Allocation = FakeAllocation;

    fn footprint_for(&mut self, _device: u32, len: u64) -> Result<u64, BackendError> {
        if len == 0 {
            return Err(BackendError::new("an empty tensor has no buffer"));
        }
        len.div_ceil(self.unit)
            .checked_mul(self.unit)
            .ok_or_else(|| BackendError::new("size overflow"))
    }

    fn allocate(&mut self, device: u32, len: u64) -> Result<(Self::Allocation, u64), BackendError> {
        if self.fail_allocations {
            return Err(BackendError::new("allocation refused (injected)"));
        }
        let footprint = self.footprint_for(device, len)?;
        let file = new_fake_buffer(footprint)?;
        let state = &self.counters.state;
        state.live.fetch_add(1, Ordering::SeqCst);
        state.live_bytes.fetch_add(footprint, Ordering::SeqCst);
        state.allocated.fetch_add(1, Ordering::SeqCst);
        Ok((
            FakeAllocation {
                file: Some(file),
                retired: Vec::new(),
                device,
                len,
                footprint,
                counters: self.counters.clone(),
            },
            footprint,
        ))
    }

    fn fill(
        &mut self,
        allocation: &mut Self::Allocation,
        bytes: &[u8],
    ) -> Result<[u8; 32], BackendError> {
        if bytes.len() as u64 != allocation.len {
            return Err(BackendError::new(
                "the tensor length differs from the buffer's",
            ));
        }
        let file = allocation
            .file
            .as_ref()
            .ok_or_else(|| BackendError::new("released"))?;
        file.write_all_at(bytes, 0)
            .map_err(|error| BackendError::new(format!("write: {error}")))?;
        Ok(content_digest(bytes))
    }

    fn export(
        &mut self,
        allocation: &Self::Allocation,
        record: &ExportRecord,
    ) -> Result<OwnedFd, BackendError> {
        if record.device != allocation.device {
            return Err(BackendError::new("the record names another device"));
        }
        if record.size != allocation.footprint {
            return Err(BackendError::new("the record names another size"));
        }
        let file = allocation
            .file
            .as_ref()
            .ok_or_else(|| BackendError::new("released"))?;
        let fd = file
            .try_clone()
            .map_err(|error| BackendError::new(format!("dup: {error}")))?;
        self.counters.state.exports.fetch_add(1, Ordering::SeqCst);
        Ok(OwnedFd::from(fd))
    }

    fn scrub(&mut self, allocation: &mut Self::Allocation, len: u64) -> Result<(), BackendError> {
        if self.fail_scrubs {
            return Err(BackendError::new("scrub refused (injected)"));
        }
        if len == 0 || len > allocation.footprint {
            return Err(BackendError::new("the tensor does not fit the buffer"));
        }
        if self.retain_on_scrub {
            let fresh = new_fake_buffer(allocation.footprint)?;
            if let Some(old) = allocation.file.replace(fresh) {
                allocation.retired.push(old);
            }
            let state = &self.counters.state;
            state.live.fetch_add(1, Ordering::SeqCst);
            state
                .live_bytes
                .fetch_add(allocation.footprint, Ordering::SeqCst);
            state.allocated.fetch_add(1, Ordering::SeqCst);
            allocation.len = len;
            state.scrubs.fetch_add(1, Ordering::SeqCst);
            return Ok(());
        }
        let file = allocation
            .file
            .as_ref()
            .ok_or_else(|| BackendError::new("released"))?;
        let zeros = [0u8; 4096];
        let mut offset = 0u64;
        while offset < allocation.footprint {
            let chunk = zeros.len().min(usize_of(allocation.footprint - offset)?);
            file.write_all_at(&zeros[..chunk], offset)
                .map_err(|error| BackendError::new(format!("zero: {error}")))?;
            offset += chunk as u64;
        }
        allocation.len = len;
        self.counters.state.scrubs.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn scrub_retains(&self) -> bool {
        self.retain_on_scrub
    }

    fn release(&mut self, allocation: Self::Allocation) -> Result<(), BackendError> {
        drop(allocation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(allocation: &FakeAllocation) -> ExportRecord {
        ExportRecord {
            producer_session: 1,
            consumer_session: 2,
            operation: 3,
            device: allocation.device(),
            size: allocation.footprint(),
        }
    }

    #[test]
    fn the_footprint_rounds_up_to_the_unit() {
        let mut backend = FakeBackend::with_unit(4096);
        assert_eq!(backend.footprint_for(0, 1), Ok(4096));
        assert_eq!(backend.footprint_for(0, 4096), Ok(4096));
        assert_eq!(backend.footprint_for(0, 4097), Ok(8192));
        assert!(backend.footprint_for(0, 0).is_err());
        assert!(backend.footprint_for(0, u64::MAX).is_err());
    }

    #[test]
    fn counters_follow_allocate_export_scrub_and_release() {
        let mut backend = FakeBackend::with_unit(4096);
        let counters = backend.counters();
        let (mut allocation, footprint) = backend.allocate(0, 100).expect("allocate");
        assert_eq!(footprint, 4096);
        assert_eq!((counters.live(), counters.live_bytes()), (1, 4096));
        let digest = backend.fill(&mut allocation, &[7u8; 100]).expect("fill");
        assert_eq!(digest, content_digest(&[7u8; 100]));
        let fd = backend
            .export(&allocation, &record(&allocation))
            .expect("export");
        drop(fd);
        assert_eq!(counters.exports(), 1);
        let whole = allocation.read_all().expect("read");
        assert!(whole[..100].iter().all(|byte| *byte == 7));
        assert!(whole[100..].iter().all(|byte| *byte == 0));
        backend.scrub(&mut allocation, 50).expect("scrub");
        assert!(allocation
            .read_all()
            .expect("read")
            .iter()
            .all(|byte| *byte == 0));
        assert_eq!(counters.scrubs(), 1);
        backend.release(allocation).expect("release");
        assert_eq!(
            (counters.live(), counters.live_bytes(), counters.released()),
            (0, 0, 1)
        );
    }

    #[test]
    fn a_dropped_allocation_is_counted_as_released_once() {
        let mut backend = FakeBackend::new();
        let counters = backend.counters();
        let (allocation, _) = backend.allocate(0, 1).expect("allocate");
        drop(allocation);
        assert_eq!((counters.live(), counters.released()), (0, 1));
    }

    #[test]
    fn a_wrong_record_or_length_is_refused() {
        let mut backend = FakeBackend::with_unit(4096);
        let (mut allocation, _) = backend.allocate(0, 10).expect("allocate");
        let mut other = record(&allocation);
        other.device = 1;
        assert!(backend.export(&allocation, &other).is_err());
        let mut other = record(&allocation);
        other.size += 1;
        assert!(backend.export(&allocation, &other).is_err());
        assert!(backend.fill(&mut allocation, &[0u8; 11]).is_err());
        assert!(backend.scrub(&mut allocation, 0).is_err());
        assert!(backend.scrub(&mut allocation, 4097).is_err());
    }

    #[test]
    fn injected_failure_allocates_nothing() {
        let mut backend = FakeBackend::new();
        backend.fail_allocations(true);
        assert!(backend.allocate(0, 1).is_err());
        assert_eq!(backend.counters().live(), 0);
    }
}
