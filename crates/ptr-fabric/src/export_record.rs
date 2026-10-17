//! The supervisor's export record (doc 42 rule 3a): one per staging export, naming the producer
//! session, the one consumer session, the operation, the device and the allocation size.
//!
//! The record is plain data and does not depend on a GPU: the staging pool mints it, the export
//! registry keeps it until the one admission that consumes it, and the CUDA import (feature
//! `cuda-ipc-core`) checks it against the handle it was given.

/// What the supervisor recorded when it minted one VMM export. The importer checks the record
/// against the handle it was given; the record, not the handle and not the driver, names the
/// size of the allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExportRecord {
    /// The session that produced the tensor.
    pub producer_session: u64,
    /// The one session that may import the export.
    pub consumer_session: u64,
    /// The operation the export is for (the handle's `parent_operation_id`).
    pub operation: u64,
    /// The ordinal of the device that holds the allocation.
    pub device: u32,
    /// The size of the allocation: the footprint, a multiple of the allocation granularity.
    pub size: u64,
}
