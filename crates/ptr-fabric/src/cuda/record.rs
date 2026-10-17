//! The export record and the importer's expectation, as pure data and pure checks (doc 42 rule
//! 3a). The supervisor mints an [`ExportRecord`] for each CUDA export; the consumer's import
//! trusts its recorded size for the extent check, because `cuMemGetAddressRange` is documented
//! only for `cuMemAlloc` allocations, not for a `cuMemCreate` allocation mapped with `cuMemMap`.

use ptr_protocol::{Carrier, CarrierLocator, CudaExport, TensorHandle, MAX_HANDLE_BYTES};

use super::CudaError;
pub use crate::export_record::ExportRecord;

/// What an importer expects of an export, taken from an admitted handle and from its own
/// identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImportExpectation {
    /// The tensor length the handle commits to.
    pub byte_len: u64,
    /// The digest the copy must have.
    pub content_digest: [u8; 32],
    /// The session of the importing worker.
    pub consumer_session: u64,
    /// The operation of the handle.
    pub operation: u64,
    /// The device that holds the allocation (`device` of `CudaIpc`, `source_device` of
    /// `DevicePeer`).
    pub source_device: u32,
    /// The device of the importer. Equal to `source_device` for `CudaIpc`, different for
    /// `DevicePeer`.
    pub consumer_device: u32,
    /// The allocation size the locator states.
    pub locator_alloc_size: u64,
}

impl ImportExpectation {
    /// Read the expectation from a `CudaIpc` or `DevicePeer` handle with a VMM export. A legacy
    /// export is refused: it is a bearer capability and is never handed to a worker.
    pub fn from_handle(
        handle: &TensorHandle,
        consumer_session: u64,
        consumer_device: u32,
    ) -> Result<Self, CudaError> {
        let (source_device, export, peer) = match (&handle.carrier, &handle.locator) {
            (Carrier::CudaIpc, CarrierLocator::CudaIpc { device, export, .. }) => {
                (*device, export, false)
            }
            (
                Carrier::DevicePeer,
                CarrierLocator::DevicePeer {
                    source_device,
                    export,
                    ..
                },
            ) => (*source_device, export, true),
            _ => return Err(CudaError::WrongLocator),
        };
        let alloc_size = match export {
            CudaExport::Vmm { alloc_size, .. } => *alloc_size,
            CudaExport::Legacy { .. } => {
                return Err(CudaError::Unsupported {
                    what: "a legacy cuIpc export is never imported by a worker",
                })
            }
        };
        let expectation = Self {
            byte_len: handle.byte_len,
            content_digest: handle.content_digest,
            consumer_session,
            operation: handle.parent_operation_id,
            source_device,
            consumer_device,
            locator_alloc_size: alloc_size,
        };
        // The carrier decides whether the importer sits on the same device, so a handle whose
        // carrier and devices disagree is refused here, before any driver call.
        let same_device = expectation.source_device == expectation.consumer_device;
        if same_device == peer {
            return Err(CudaError::RecordMismatch {
                field: "consumer_device",
            });
        }
        Ok(expectation)
    }
}

/// `offset + len` must lie inside `size`, overflow-checked. The extent check of rule 3a.
pub fn check_extent(offset: u64, len: u64, size: u64) -> Result<(), CudaError> {
    let end = offset.checked_add(len).ok_or(CudaError::Overflow)?;
    if end > size {
        return Err(CudaError::ExtentMismatch {
            needed: end,
            available: size,
        });
    }
    Ok(())
}

/// Check a record against an expectation, before any descriptor is touched: the record names
/// this consumer session, this operation and the device the locator names, its size equals the
/// size the locator states, and the tensor (offset 0) lies inside that size.
pub fn check_import(record: &ExportRecord, expected: &ImportExpectation) -> Result<(), CudaError> {
    if expected.byte_len == 0 {
        return Err(CudaError::EmptyTensor);
    }
    if expected.byte_len > MAX_HANDLE_BYTES {
        return Err(CudaError::ExtentMismatch {
            needed: expected.byte_len,
            available: MAX_HANDLE_BYTES,
        });
    }
    if record.consumer_session != expected.consumer_session {
        return Err(CudaError::RecordMismatch {
            field: "consumer_session",
        });
    }
    if record.operation != expected.operation {
        return Err(CudaError::RecordMismatch { field: "operation" });
    }
    if record.device != expected.source_device {
        return Err(CudaError::RecordMismatch { field: "device" });
    }
    if record.size != expected.locator_alloc_size {
        return Err(CudaError::RecordMismatch { field: "size" });
    }
    check_extent(0, expected.byte_len, record.size)
}

/// Whether every byte of `whole` from `byte_len` on is zero: the staging padding must not carry
/// any producer data. False when `whole` is shorter than `byte_len`.
pub fn padding_is_zero(whole: &[u8], byte_len: usize) -> bool {
    whole
        .get(byte_len..)
        .is_some_and(|padding| padding.iter().all(|byte| *byte == 0))
}

#[cfg(test)]
mod tests {
    use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
    use ptr_types::TypeId;

    use super::*;

    const GRAN: u64 = 2 << 20;

    fn record() -> ExportRecord {
        ExportRecord {
            producer_session: 10,
            consumer_session: 20,
            operation: 7,
            device: 0,
            size: GRAN,
        }
    }

    fn expectation() -> ImportExpectation {
        ImportExpectation {
            byte_len: 4096,
            content_digest: [9; 32],
            consumer_session: 20,
            operation: 7,
            source_device: 0,
            consumer_device: 0,
            locator_alloc_size: GRAN,
        }
    }

    fn element(name: &str) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation: 1,
            digest: [1; 32],
            fence_epoch: 0,
        }
    }

    fn handle(carrier: Carrier, locator: CarrierLocator) -> TensorHandle {
        TensorHandle {
            carrier,
            locator,
            dtype: BoundaryDtype::F32,
            shape: vec![1024],
            byte_len: 4096,
            upstream_chain: UpstreamChain {
                stage: element("stage"),
                base_model: element("model"),
                adapters: vec![],
                experts: vec![],
            },
            placement_epoch: 1,
            signed_absolute_deadline_ms: 10_000,
            handle_nonce: 1,
            parent_operation_id: 7,
            consumer: "consumer".to_owned(),
            payload_type: TypeId::from("ptr.split.activation.v1"),
            content_digest: [9; 32],
            signature: [0; 64],
        }
    }

    fn vmm() -> CudaExport {
        CudaExport::Vmm {
            fd_slot: 0,
            alloc_size: GRAN,
        }
    }

    #[test]
    fn a_matching_record_is_accepted() {
        assert_eq!(check_import(&record(), &expectation()), Ok(()));
    }

    #[test]
    fn each_field_of_the_record_is_checked() {
        let mut other = record();
        other.consumer_session = 21;
        assert_eq!(
            check_import(&other, &expectation()),
            Err(CudaError::RecordMismatch {
                field: "consumer_session"
            })
        );
        let mut other = record();
        other.operation = 8;
        assert_eq!(
            check_import(&other, &expectation()),
            Err(CudaError::RecordMismatch { field: "operation" })
        );
        let mut other = record();
        other.device = 1;
        assert_eq!(
            check_import(&other, &expectation()),
            Err(CudaError::RecordMismatch { field: "device" })
        );
        let mut other = record();
        other.size = 2 * GRAN;
        assert_eq!(
            check_import(&other, &expectation()),
            Err(CudaError::RecordMismatch { field: "size" })
        );
    }

    #[test]
    fn the_extent_uses_the_recorded_size() {
        let mut small = record();
        small.size = 4096;
        let mut expected = expectation();
        expected.locator_alloc_size = 4096;
        expected.byte_len = 4097;
        assert_eq!(
            check_import(&small, &expected),
            Err(CudaError::ExtentMismatch {
                needed: 4097,
                available: 4096
            })
        );
        expected.byte_len = 4096;
        assert_eq!(check_import(&small, &expected), Ok(()));
    }

    #[test]
    fn extent_arithmetic_is_overflow_checked() {
        assert_eq!(check_extent(0, 10, 10), Ok(()));
        assert_eq!(
            check_extent(1, 10, 10),
            Err(CudaError::ExtentMismatch {
                needed: 11,
                available: 10
            })
        );
        assert_eq!(
            check_extent(u64::MAX, 1, u64::MAX),
            Err(CudaError::Overflow)
        );
    }

    #[test]
    fn an_empty_or_oversize_tensor_is_refused() {
        let mut expected = expectation();
        expected.byte_len = 0;
        assert_eq!(
            check_import(&record(), &expected),
            Err(CudaError::EmptyTensor)
        );
        expected.byte_len = MAX_HANDLE_BYTES + 1;
        assert!(matches!(
            check_import(&record(), &expected),
            Err(CudaError::ExtentMismatch { .. })
        ));
    }

    #[test]
    fn padding_must_be_zero() {
        let mut whole = vec![1u8; 8];
        whole.extend([0u8; 8]);
        assert!(padding_is_zero(&whole, 8));
        assert!(!padding_is_zero(&whole, 7));
        assert!(padding_is_zero(&whole, 16));
        assert!(!padding_is_zero(&whole, 17));
        whole[15] = 1;
        assert!(!padding_is_zero(&whole, 8));
    }

    #[test]
    fn from_handle_reads_a_vmm_cuda_ipc_handle() {
        let locator = CarrierLocator::CudaIpc {
            device: 0,
            export: vmm(),
            event_handle: [0; 64],
        };
        let expected =
            ImportExpectation::from_handle(&handle(Carrier::CudaIpc, locator), 20, 0).expect("vmm");
        assert_eq!(expected, expectation());
    }

    #[test]
    fn from_handle_reads_a_device_peer_handle_only_for_another_device() {
        let locator = CarrierLocator::DevicePeer {
            source_device: 0,
            export: vmm(),
            event_handle: [0; 64],
        };
        let peer = handle(Carrier::DevicePeer, locator);
        let expected = ImportExpectation::from_handle(&peer, 20, 1).expect("peer");
        assert_eq!((expected.source_device, expected.consumer_device), (0, 1));
        assert_eq!(
            ImportExpectation::from_handle(&peer, 20, 0),
            Err(CudaError::RecordMismatch {
                field: "consumer_device"
            })
        );
    }

    #[test]
    fn from_handle_refuses_cuda_ipc_for_another_device() {
        let locator = CarrierLocator::CudaIpc {
            device: 0,
            export: vmm(),
            event_handle: [0; 64],
        };
        assert_eq!(
            ImportExpectation::from_handle(&handle(Carrier::CudaIpc, locator), 20, 1),
            Err(CudaError::RecordMismatch {
                field: "consumer_device"
            })
        );
    }

    #[test]
    fn from_handle_refuses_a_legacy_export_and_other_carriers() {
        let legacy = CarrierLocator::CudaIpc {
            device: 0,
            export: CudaExport::Legacy {
                mem_handle: [7; 64],
            },
            event_handle: [0; 64],
        };
        assert!(matches!(
            ImportExpectation::from_handle(&handle(Carrier::CudaIpc, legacy), 20, 0),
            Err(CudaError::Unsupported { .. })
        ));
        let bytes = CarrierLocator::Bytes { inline: vec![0; 4] };
        assert_eq!(
            ImportExpectation::from_handle(&handle(Carrier::Bytes, bytes), 20, 0),
            Err(CudaError::WrongLocator)
        );
        // A handle whose carrier tag and locator variant disagree.
        let locator = CarrierLocator::CudaIpc {
            device: 0,
            export: vmm(),
            event_handle: [0; 64],
        };
        assert_eq!(
            ImportExpectation::from_handle(&handle(Carrier::DevicePeer, locator), 20, 1),
            Err(CudaError::WrongLocator)
        );
    }
}
