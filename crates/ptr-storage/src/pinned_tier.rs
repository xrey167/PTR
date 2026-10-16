//! `PinnedHostTierBackend`: a `TierBackend` over page-locked host memory with a fixed budget
//! (doc 42 stage S3, Linux only).
//!
//! Every chunk lives in its own anonymous private mapping of page-rounded size, locked with
//! `mlock`. The budget is charged with that page-rounded footprint, not the logical length.

use crate::tier::{
    digest, ChunkDescriptor, ChunkReceipt, StorageTier, TierBackend, TierBackendId,
    TierCapabilities, TierError, TierFuture, TierHealth, TierIntegrityReport,
};
use ptr_types::Digest;
use rustix::mm::{mlock, mmap_anonymous, munlock, munmap, MapFlags, ProtFlags};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

/// One locked anonymous mapping, exclusively owned and immutable after the initial write.
struct PinnedMapping {
    ptr: NonNull<c_void>,
    /// Page-rounded mapping length, at least one page.
    footprint: usize,
    /// Logical chunk length.
    length: usize,
}

// SAFETY: the mapping is owned exclusively by this value (no other alias exists), is written
// once before the value is shared, and is only read afterwards through `&self`, so moving it to
// or sharing it between threads cannot race. It is unmapped exactly once, in `Drop`.
unsafe impl Send for PinnedMapping {}
// SAFETY: see `Send`; shared access is read-only.
unsafe impl Sync for PinnedMapping {}

impl PinnedMapping {
    fn new(bytes: &[u8], footprint: usize) -> Result<Self, TierError> {
        // SAFETY: a null hint with a fresh anonymous private mapping aliases nothing; the length
        // is a non-zero multiple of the page size.
        let raw = unsafe {
            mmap_anonymous(
                std::ptr::null_mut(),
                footprint,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::PRIVATE,
            )
        }
        .map_err(|error| {
            TierError::Backend(format!("mmap of {footprint} bytes failed: {error}"))
        })?;
        let ptr = NonNull::new(raw)
            .ok_or_else(|| TierError::Backend("mmap returned a null mapping".into()))?;
        // SAFETY: `ptr..ptr+footprint` is the mapping created above and valid for reads.
        if let Err(error) = unsafe { mlock(ptr.as_ptr(), footprint) } {
            // SAFETY: unmapping the mapping created above; no reference to it exists yet.
            let _ = unsafe { munmap(ptr.as_ptr(), footprint) };
            return Err(TierError::MemoryLock(format!(
                "mlock of {footprint} bytes failed: {error}"
            )));
        }
        let mapping = Self {
            ptr,
            footprint,
            length: bytes.len(),
        };
        // SAFETY: the mapping is writable, exclusively owned and at least `bytes.len()` long
        // (`footprint >= length`); the source slice cannot overlap a fresh mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapping.ptr.as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(mapping)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is live for the lifetime of `self`, initialised for `length`
        // bytes by `new`, never written again, and `length <= footprint`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<u8>(), self.length) }
    }
}

impl Drop for PinnedMapping {
    fn drop(&mut self) {
        // SAFETY: the mapping is owned by `self`, no slice borrowed from it outlives `&self`,
        // and this is the only place that unlocks and unmaps it.
        unsafe {
            let _ = munlock(self.ptr.as_ptr(), self.footprint);
            let _ = munmap(self.ptr.as_ptr(), self.footprint);
        }
    }
}

#[derive(Default)]
struct PinnedState {
    chunks: BTreeMap<Digest, PinnedMapping>,
    used: usize,
}

/// A tier over `mlock`ed host memory with a fixed byte budget. `CpuTierBackend` stays the
/// unpinned test backend.
pub struct PinnedHostTierBackend {
    id: TierBackendId,
    budget: usize,
    page: usize,
    state: Mutex<PinnedState>,
}

impl PinnedHostTierBackend {
    /// `budget_bytes` is charged with page-rounded footprints. A budget that is not a multiple
    /// of the page size can never be filled to the last byte.
    pub fn new(id: TierBackendId, budget_bytes: usize) -> Result<Self, TierError> {
        if id.0.trim().is_empty() || id.0 != id.0.trim() {
            return Err(TierError::InvalidBackendId);
        }
        Ok(Self {
            id,
            budget: budget_bytes,
            page: rustix::param::page_size(),
            state: Mutex::new(PinnedState::default()),
        })
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget
    }

    /// Bytes currently charged: the sum of the page-rounded footprints of the stored chunks.
    pub fn used_bytes(&self) -> usize {
        // A poisoned lock still holds the accounting of the mappings it owns.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .used
    }

    pub fn page_size(&self) -> usize {
        self.page
    }

    /// Page-rounded footprint charged for a chunk of `length` bytes (at least one page), or
    /// `None` when the rounded size does not fit in `usize`.
    pub fn footprint_for(&self, length: usize) -> Option<usize> {
        length.div_ceil(self.page).max(1).checked_mul(self.page)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, PinnedState>, TierError> {
        self.state
            .lock()
            .map_err(|_| TierError::Backend("pinned tier lock poisoned".into()))
    }
}

impl TierBackend for PinnedHostTierBackend {
    fn identity(&self) -> TierBackendId {
        self.id.clone()
    }

    fn capabilities(&self) -> TierCapabilities {
        TierCapabilities {
            tier: StorageTier::Cpu,
            persistent: false,
            range_reads: false,
            conditional_create: true,
        }
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        Box::pin(async move {
            chunk.validate_bytes(&bytes)?;
            let mut state = self.lock_state()?;
            if let Some(existing) = state.chunks.get(&chunk.digest) {
                if existing.bytes() != bytes.as_slice() {
                    return Err(TierError::ImmutableConflict);
                }
            } else {
                let footprint = bytes
                    .len()
                    .div_ceil(self.page)
                    .max(1)
                    .checked_mul(self.page)
                    .ok_or_else(|| TierError::Backend("chunk footprint overflows".into()))?;
                let available = self.budget.saturating_sub(state.used);
                if footprint > available {
                    return Err(TierError::BudgetExceeded {
                        requested: footprint as u64,
                        available: available as u64,
                    });
                }
                let mapping = PinnedMapping::new(&bytes, footprint)?;
                state.used += footprint;
                state.chunks.insert(chunk.digest, mapping);
            }
            Ok(ChunkReceipt {
                backend: self.id.clone(),
                digest: chunk.digest,
                length: chunk.length,
            })
        })
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(async move {
            let state = self.lock_state()?;
            let bytes = state
                .chunks
                .get(&chunk.digest)
                .ok_or(TierError::UnknownChunk)?
                .bytes()
                .to_vec();
            chunk
                .validate_bytes(&bytes)
                .map_err(|_| TierError::CorruptChunk)?;
            Ok(bytes)
        })
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let state = self.lock_state()?;
            let bytes = state
                .chunks
                .get(&chunk.digest)
                .ok_or(TierError::UnknownChunk)?
                .bytes();
            Ok(TierIntegrityReport {
                digest: digest(bytes),
                length: bytes.len() as u64,
                valid: chunk.validate_bytes(bytes).is_ok(),
            })
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        Box::pin(async move {
            let mut state = self.lock_state()?;
            if let Some(mapping) = state.chunks.remove(&chunk.digest) {
                state.used -= mapping.footprint;
            }
            Ok(())
        })
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        Box::pin(async { Ok(TierHealth::Healthy) })
    }
}
