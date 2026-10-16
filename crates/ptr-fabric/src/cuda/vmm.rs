//! The VMM staging export and its import (doc 42 rules 3a and 3b), the CUDA form a Pod worker
//! may receive.
//!
//! The supervisor side ([`StagingAllocation`]) creates one `cuMemCreate` allocation per export,
//! sized to the granularity-rounded footprint and holding nothing else, zeroes all of it
//! (padding included) before any write, fills it once, and exports it as a POSIX descriptor.
//! The consumer side ([`import_and_copy`]) imports the descriptor, reserves and maps exactly the
//! **recorded** size read-only, copies `byte_len` bytes into memory it owns, **releases the
//! import before it returns**, and verifies `content_digest` on the copy.
//!
//! **No allocation is ever reused.** Every export gets a fresh allocation; there is no pool in
//! this module. Dropping a [`StagingAllocation`] releases the creator's mapping and handle, but
//! the driver keeps the physical memory alive for as long as an exported descriptor or an import
//! of it exists, so freeing is a decision of the caller: it keeps the value (and charges its
//! footprint to the producer and consumer quotas) until both isolation domains have been
//! reaped (rule 3b). The guardian that owns that decision is a later package.
//!
//! Nothing here has run on a GPU.

use std::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

use cudarc::driver::sys;
use ptr_protocol::{content_digest, CudaExport, MAX_HANDLE_BYTES};

use super::driver::{check, guarded, guarded_drop, CudaDriver, DeviceContext};
use super::footprint::footprint_vmm;
use super::record::{check_import, padding_is_zero, ExportRecord, ImportExpectation};
use super::CudaError;

/// A `CUmemLocation` naming the device `id`. The struct changed shape between `cudarc`'s
/// bindings (`{ type_, id }` up to CUDA 13.1, `{ type_, union { id, .. } }` from 13.2), and a
/// build that names no fixed version gets whichever its toolkit (or the newest) selects. Every
/// shape is a 4-byte type followed by a 4-byte union or integer whose first member is the
/// device ordinal, so write the ordinal there; `device_location_layout_holds` pins the sizes.
fn device_location(id: i32) -> Result<sys::CUmemLocation, CudaError> {
    // The newest layout holds the device in a one-byte member of the union next to the plain
    // integer; no host has 256 devices, and an ordinal that would not fit it is refused rather
    // than truncated.
    if !(0..=255).contains(&id) {
        return Err(CudaError::NoSuchDevice {
            ordinal: u32::try_from(id).unwrap_or(u32::MAX),
        });
    }
    // SAFETY: `CUmemLocation` is plain C data and the all-zero pattern is a valid value of it
    // (location type 0 is the invalid-location enumerator, the union is integers).
    let mut location: sys::CUmemLocation = unsafe { std::mem::zeroed() };
    location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    let offset = std::mem::size_of::<sys::CUmemLocationType>();
    // SAFETY: the struct is 8 bytes (checked by the layout test); bytes `offset..offset + 4`
    // belong to the ordinal (a `c_int`, or the first member of the union that holds it), and
    // the field is 4-byte aligned after a 4-byte enum.
    unsafe {
        (&mut location as *mut sys::CUmemLocation)
            .cast::<u8>()
            .add(offset)
            .cast::<std::ffi::c_int>()
            .write(id);
    }
    Ok(location)
}

/// The allocation properties of a POSIX-descriptor-exportable pinned allocation on `ordinal`.
fn allocation_prop(ordinal: u32) -> Result<sys::CUmemAllocationProp, CudaError> {
    let id = i32::try_from(ordinal).map_err(|_| CudaError::Overflow)?;
    Ok(sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes:
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        location: device_location(id)?,
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    })
}

/// The minimum allocation granularity of `ordinal` for a POSIX-descriptor-exportable
/// allocation: every VMM size is a multiple of it.
pub(crate) fn allocation_granularity(driver: &CudaDriver, ordinal: u32) -> Result<u64, CudaError> {
    // The device must exist; the property names it by ordinal.
    driver.device(ordinal)?;
    let prop = allocation_prop(ordinal)?;
    let mut granularity: usize = 0;
    // SAFETY: `granularity` is a valid out pointer and `prop` a fully initialised property
    // struct that lives across the call.
    check("cuMemGetAllocationGranularity", unsafe {
        sys::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    })?;
    if granularity == 0 {
        return Err(CudaError::Unsupported {
            what: "the driver reported a zero allocation granularity",
        });
    }
    Ok(granularity as u64)
}

fn access_desc(
    ordinal: u32,
    flags: sys::CUmemAccess_flags,
) -> Result<sys::CUmemAccessDesc, CudaError> {
    let id = i32::try_from(ordinal).map_err(|_| CudaError::Overflow)?;
    Ok(sys::CUmemAccessDesc {
        location: device_location(id)?,
        flags,
    })
}

/// A VMM allocation handle plus the address range it is mapped at, undone step by step on drop.
/// Used by both sides: the creator keeps it for the life of the export, the importer for the
/// span of one copy.
struct VmmMapping {
    context: DeviceContext,
    handle: Option<sys::CUmemGenericAllocationHandle>,
    range: Option<(sys::CUdeviceptr, usize)>,
    mapped: bool,
}

impl VmmMapping {
    fn new(context: DeviceContext) -> Self {
        Self {
            context,
            handle: None,
            range: None,
            mapped: false,
        }
    }

    /// Reserve `size` bytes of address space, map `handle` into it and grant `access` to
    /// `access_device`.
    fn map(
        &mut self,
        size: usize,
        access_device: u32,
        access: sys::CUmemAccess_flags,
    ) -> Result<sys::CUdeviceptr, CudaError> {
        let handle = self.handle.ok_or(CudaError::Unsupported {
            what: "map without an allocation handle",
        })?;
        let mut base: sys::CUdeviceptr = 0;
        // SAFETY: `base` is a valid out pointer; alignment 0 and a zero hint address let the
        // driver choose; the context of `self` is current at the call sites.
        check("cuMemAddressReserve", unsafe {
            sys::cuMemAddressReserve(&mut base, size, 0, 0, 0)
        })?;
        self.range = Some((base, size));
        // SAFETY: `[base, base + size)` was just reserved, `handle` is a live allocation of at
        // least `size` bytes (checked by the callers against the granularity-rounded size) and
        // the offset is zero.
        check("cuMemMap", unsafe {
            sys::cuMemMap(base, size, 0, handle, 0)
        })?;
        self.mapped = true;
        let desc = access_desc(access_device, access)?;
        // SAFETY: the range is mapped; `desc` is one fully initialised descriptor and the count
        // says one.
        check("cuMemSetAccess", unsafe {
            sys::cuMemSetAccess(base, size, &desc, 1)
        })?;
        Ok(base)
    }
}

impl Drop for VmmMapping {
    fn drop(&mut self) {
        // A missing driver entry point panics inside cudarc; a destructor must not unwind, so
        // the whole teardown is caught (see `guarded_drop`).
        guarded_drop(|| self.teardown());
    }
}

impl VmmMapping {
    fn teardown(&mut self) {
        // The context is made current best effort; the VMM calls below do not need more than
        // the driver being initialised. A failure cannot be reported from a destructor, and
        // each step is attempted regardless of the one before.
        let _guard = self.context.enter().ok();
        if let Some((base, size)) = self.range.take() {
            if self.mapped {
                // SAFETY: the range was mapped by `map` and not unmapped since.
                let _ = unsafe { sys::cuMemUnmap(base, size) };
            }
            // SAFETY: the range was reserved by `map` and is no longer mapped.
            let _ = unsafe { sys::cuMemAddressFree(base, size) };
        }
        if let Some(handle) = self.handle.take() {
            // SAFETY: the handle came from `cuMemCreate` or `cuMemImportFromShareableHandle`
            // and is released once; the range that used it is gone.
            let _ = unsafe { sys::cuMemRelease(handle) };
        }
    }
}

/// The producer-side staging allocation: one `cuMemCreate` allocation holding exactly one
/// tensor and nothing else, owned by the supervisor (or its guardian), never by a worker.
///
/// Created zeroed over its whole footprint, filled once, exported once per consumer. Not
/// `Sync`: use it from one thread at a time.
pub struct StagingAllocation {
    mapping: VmmMapping,
    device: u32,
    byte_len: u64,
    footprint: u64,
    base: sys::CUdeviceptr,
    digest: Option<[u8; 32]>,
}

impl StagingAllocation {
    /// Create a staging allocation for a `len` byte tensor on `device`: `cuMemCreate` with a
    /// POSIX-descriptor handle type, sized to `len` rounded up to the allocation granularity,
    /// mapped into the creator read-write, and **all of it zeroed** (padding included) before
    /// anything can be written. Refuses an empty tensor, a device without VMM or descriptor
    /// export, and a size that overflows.
    pub fn create(driver: &CudaDriver, device: u32, len: u64) -> Result<Self, CudaError> {
        guarded(|| Self::create_inner(driver, device, len))
    }

    fn create_inner(driver: &CudaDriver, device: u32, len: u64) -> Result<Self, CudaError> {
        if len == 0 {
            return Err(CudaError::EmptyTensor);
        }
        let granularity = require_vmm(driver, device)?;
        let footprint = footprint_vmm(len, granularity).ok_or(CudaError::Overflow)?;
        let size = usize::try_from(footprint).map_err(|_| CudaError::Overflow)?;
        let context = DeviceContext::retain(driver, device)?;
        let mut mapping = VmmMapping::new(context);
        let _current = mapping.context.enter()?;

        let prop = allocation_prop(device)?;
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: `handle` is a valid out pointer, `size` is a non-zero multiple of the
        // granularity, `prop` is fully initialised and flags must be zero.
        check("cuMemCreate", unsafe {
            sys::cuMemCreate(&mut handle, size, &prop, 0)
        })?;
        mapping.handle = Some(handle);
        let base = mapping.map(
            size,
            device,
            sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        )?;
        // Zero the whole allocation, tensor span and padding alike: a fresh allocation is not
        // guaranteed to read as zero, and the consumer must never see producer or foreign data
        // past the signed span.
        // SAFETY: `[base, base + size)` is mapped read-write on this device.
        check("cuMemsetD8_v2", unsafe {
            sys::cuMemsetD8_v2(base, 0, size)
        })?;
        // SAFETY: a plain call on the current context.
        check("cuCtxSynchronize", unsafe { sys::cuCtxSynchronize() })?;
        Ok(Self {
            mapping,
            device,
            byte_len: len,
            footprint,
            base,
            digest: None,
        })
    }

    /// Write the tensor into the allocation. Exactly once, and exactly `byte_len` bytes; the
    /// padding stays zero. Returns the SHA-256 of the tensor, for the handle's
    /// `content_digest`.
    pub fn fill(&mut self, bytes: &[u8]) -> Result<[u8; 32], CudaError> {
        guarded(|| self.fill_inner(bytes))
    }

    fn fill_inner(&mut self, bytes: &[u8]) -> Result<[u8; 32], CudaError> {
        if self.digest.is_some() {
            return Err(CudaError::RecordMismatch {
                field: "allocation already filled",
            });
        }
        if bytes.len() as u64 != self.byte_len {
            return Err(CudaError::ExtentMismatch {
                needed: bytes.len() as u64,
                available: self.byte_len,
            });
        }
        let _current = self.mapping.context.enter()?;
        // SAFETY: the source slice is `byte_len` readable bytes; the destination is the start
        // of the mapped read-write range of `footprint >= byte_len` bytes.
        check("cuMemcpyHtoD_v2", unsafe {
            sys::cuMemcpyHtoD_v2(self.base, bytes.as_ptr().cast::<c_void>(), bytes.len())
        })?;
        // SAFETY: a plain call on the current context.
        check("cuCtxSynchronize", unsafe { sys::cuCtxSynchronize() })?;
        let digest = content_digest(bytes);
        self.digest = Some(digest);
        Ok(digest)
    }

    /// The tensor length.
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// The allocated footprint: the length rounded up to the granularity. What the quotas are
    /// charged (rule 3b).
    pub fn footprint(&self) -> u64 {
        self.footprint
    }

    /// The device ordinal that holds the allocation.
    pub fn device(&self) -> u32 {
        self.device
    }

    /// The digest of the filled tensor, `None` before [`fill`](Self::fill).
    pub fn content_digest(&self) -> Option<[u8; 32]> {
        self.digest
    }

    /// The record for an export of this allocation to `consumer_session` for `operation`.
    pub fn record_for(
        &self,
        producer_session: u64,
        consumer_session: u64,
        operation: u64,
    ) -> ExportRecord {
        ExportRecord {
            producer_session,
            consumer_session,
            operation,
            device: self.device,
            size: self.footprint,
        }
    }

    /// Export the filled allocation as a POSIX descriptor (`cuMemExportToShareableHandle`),
    /// close-on-exec. The record must name this device and this footprint. Each call yields a
    /// new descriptor of the same allocation; the allocation is never reused for another
    /// tensor.
    pub fn export(&self, record: ExportRecord) -> Result<StagingExport, CudaError> {
        guarded(|| self.export_inner(record))
    }

    fn export_inner(&self, record: ExportRecord) -> Result<StagingExport, CudaError> {
        if self.digest.is_none() {
            return Err(CudaError::RecordMismatch {
                field: "allocation not filled",
            });
        }
        if record.device != self.device {
            return Err(CudaError::RecordMismatch { field: "device" });
        }
        if record.size != self.footprint {
            return Err(CudaError::RecordMismatch { field: "size" });
        }
        let handle = self.mapping.handle.ok_or(CudaError::Unsupported {
            what: "export without an allocation handle",
        })?;
        let _current = self.mapping.context.enter()?;
        let mut raw: std::ffi::c_int = -1;
        // SAFETY: for a POSIX-descriptor handle type the driver stores one `int` through the
        // pointer; `raw` is one. The handle was created with that handle type.
        check("cuMemExportToShareableHandle", unsafe {
            sys::cuMemExportToShareableHandle(
                (&mut raw as *mut std::ffi::c_int).cast::<c_void>(),
                handle,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
        })?;
        if raw < 0 {
            return Err(CudaError::Io(
                "the driver returned no descriptor".to_owned(),
            ));
        }
        // SAFETY: the driver returned a new descriptor that nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(raw as RawFd) };
        rustix::io::fcntl_setfd(&fd, rustix::io::FdFlags::CLOEXEC)
            .map_err(|error| CudaError::Io(error.to_string()))?;
        Ok(StagingExport {
            fd,
            recorded_size: record.size,
            footprint: self.footprint,
            record,
        })
    }
}

impl std::fmt::Debug for StagingAllocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StagingAllocation")
            .field("device", &self.device)
            .field("byte_len", &self.byte_len)
            .field("footprint", &self.footprint)
            .field("filled", &self.digest.is_some())
            .finish()
    }
}

/// One export of a staging allocation: the descriptor to pass with `SCM_RIGHTS` and the record
/// the supervisor keeps.
#[derive(Debug)]
pub struct StagingExport {
    /// The VMM descriptor. Pass it to the consumer only through the supervisor's channel.
    pub fd: OwnedFd,
    /// The allocation size the importer maps and checks the extent against.
    pub recorded_size: u64,
    /// The allocated footprint charged to the quotas.
    pub footprint: u64,
    /// The export record.
    pub record: ExportRecord,
}

impl StagingExport {
    /// The locator's export for a handle: the VMM form with the descriptor slot the caller
    /// passes `fd` in and the recorded allocation size.
    pub fn cuda_export(&self, fd_slot: u32) -> CudaExport {
        CudaExport::Vmm {
            fd_slot,
            alloc_size: self.recorded_size,
        }
    }
}

fn require_vmm(driver: &CudaDriver, ordinal: u32) -> Result<u64, CudaError> {
    let device = driver.device(ordinal)?;
    let vmm = driver.attribute(
        device,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
    )?;
    if vmm == 0 {
        return Err(CudaError::Unsupported {
            what: "the device has no virtual memory management",
        });
    }
    let posix = driver.attribute(
        device,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
    )?;
    if posix == 0 {
        return Err(CudaError::Unsupported {
            what: "the device cannot export an allocation as a POSIX descriptor",
        });
    }
    allocation_granularity(driver, ordinal)
}

/// Import `fd` under `record`, map exactly the recorded size read-only for the consumer's
/// device, copy `copy_len` bytes into a `Vec`, and release the import before returning.
fn import_and_read(
    driver: &CudaDriver,
    fd: OwnedFd,
    record: &ExportRecord,
    expected: &ImportExpectation,
    copy_len: u64,
) -> Result<Vec<u8>, CudaError> {
    check_import(record, expected)?;
    super::record::check_extent(0, copy_len, record.size)?;
    // The extent of a `cuMemCreate` allocation is the recorded size, which must be a multiple
    // of the allocation granularity of the device that created it.
    let granularity = require_vmm(driver, expected.source_device)?;
    if !record.size.is_multiple_of(granularity) {
        return Err(CudaError::RecordMismatch { field: "size" });
    }
    // The recorded size is bounded by the footprint of the largest legal handle, so a hostile
    // or corrupt record cannot make the importer reserve and copy an unbounded range.
    let max_size = footprint_vmm(MAX_HANDLE_BYTES, granularity).ok_or(CudaError::Overflow)?;
    if record.size > max_size {
        return Err(CudaError::ExtentMismatch {
            needed: record.size,
            available: max_size,
        });
    }
    if expected.consumer_device != expected.source_device {
        require_vmm(driver, expected.consumer_device)?;
        // Peer access is probed, never assumed: the consumer's device must be able to access
        // the source device's memory, else the refusal is typed before any import.
        if !driver.can_access_peer(expected.consumer_device, expected.source_device)? {
            return Err(CudaError::Unsupported {
                what: "the consumer device has no peer access to the source device",
            });
        }
    }
    let size = usize::try_from(record.size).map_err(|_| CudaError::Overflow)?;
    let copy = usize::try_from(copy_len).map_err(|_| CudaError::Overflow)?;

    let context = DeviceContext::retain(driver, expected.consumer_device)?;
    let mut mapping = VmmMapping::new(context);
    let current = mapping.context.enter()?;
    let mut handle: sys::CUmemGenericAllocationHandle = 0;
    // SAFETY: for a POSIX-descriptor handle type the driver reads the descriptor number from
    // the pointer value itself; `fd` is open for the call. The driver duplicates what it needs,
    // so `fd` is closed by its own drop afterwards.
    check("cuMemImportFromShareableHandle", unsafe {
        sys::cuMemImportFromShareableHandle(
            &mut handle,
            fd.as_raw_fd() as usize as *mut c_void,
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
    })?;
    drop(fd);
    mapping.handle = Some(handle);
    // Read-only for the importer: it has no reason to write the staging buffer.
    let base = mapping.map(
        size,
        expected.consumer_device,
        sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
    )?;
    let mut bytes = vec![0u8; copy];
    // SAFETY: the destination is `copy` writable bytes; the source range is mapped and at least
    // `copy` bytes long (`copy_len <= record.size` was checked above against the recorded size).
    check("cuMemcpyDtoH_v2", unsafe {
        sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast::<c_void>(), base, copy)
    })?;
    // SAFETY: a plain call on the current context.
    check("cuCtxSynchronize", unsafe { sys::cuCtxSynchronize() })?;
    // Unmap and release the import before the bytes are used or returned: the consumer keeps
    // no mapping of the producer's staging buffer.
    drop(current);
    drop(mapping);
    Ok(bytes)
}

/// Import a VMM export and copy the tensor out (the consumer side of rule 3a).
///
/// The record is checked against the expectation first (consumer session, operation, device,
/// size, extent), then the descriptor is imported with `cuMemImportFromShareableHandle`, exactly
/// `record.size` bytes are reserved and mapped read-only for the consumer's device, `byte_len`
/// bytes are copied into receiver-owned memory, the import is unmapped and released, and only
/// then is `content_digest` verified on the copy. On any failure no bytes are returned and no
/// mapping remains.
pub fn import_and_copy(
    driver: &CudaDriver,
    fd: OwnedFd,
    record: &ExportRecord,
    expected: &ImportExpectation,
) -> Result<Vec<u8>, CudaError> {
    let bytes = guarded(|| import_and_read(driver, fd, record, expected, expected.byte_len))?;
    if content_digest(&bytes) != expected.content_digest {
        return Err(CudaError::DigestMismatch);
    }
    Ok(bytes)
}

/// What an overread audit found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddingAudit {
    /// The bytes past the tensor span inside the export.
    pub padding_len: u64,
    /// Whether every one of them is zero.
    pub all_zero: bool,
}

/// The overread test of E009 (a consumer that maps the whole export): copy the whole recorded
/// extent, verify the digest of the tensor prefix, and report whether the padding is all zero.
/// A producer-side bug that left data in the padding shows as `all_zero == false`.
pub fn audit_export_padding(
    driver: &CudaDriver,
    fd: OwnedFd,
    record: &ExportRecord,
    expected: &ImportExpectation,
) -> Result<PaddingAudit, CudaError> {
    let whole = guarded(|| import_and_read(driver, fd, record, expected, record.size))?;
    let byte_len = usize::try_from(expected.byte_len).map_err(|_| CudaError::Overflow)?;
    let prefix = whole.get(..byte_len).ok_or(CudaError::ExtentMismatch {
        needed: expected.byte_len,
        available: whole.len() as u64,
    })?;
    if content_digest(prefix) != expected.content_digest {
        return Err(CudaError::DigestMismatch);
    }
    Ok(PaddingAudit {
        padding_len: record.size - expected.byte_len,
        all_zero: padding_is_zero(&whole, byte_len),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_location_layout_holds() {
        // Every `cudarc` binding shape is a 4-byte type followed by a 4-byte ordinal.
        assert_eq!(std::mem::size_of::<sys::CUmemLocationType>(), 4);
        assert_eq!(std::mem::size_of::<sys::CUmemLocation>(), 8);
        let location = device_location(3).expect("a small ordinal");
        // Under the fixed 12.8 bindings the struct has the field itself: read it as such.
        #[cfg(feature = "cuda-ipc")]
        assert_eq!(location.id, 3);
        assert!(device_location(256).is_err() && device_location(-1).is_err());
        assert_eq!(
            location.type_,
            sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE
        );
        // SAFETY: reading back the 4 bytes after the type of an 8-byte plain-data struct.
        let ordinal = unsafe {
            (&location as *const sys::CUmemLocation)
                .cast::<u8>()
                .add(4)
                .cast::<std::ffi::c_int>()
                .read()
        };
        assert_eq!(ordinal, 3);
    }
}
