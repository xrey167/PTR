//! The tensor handle codec through the public API of `ptr-protocol`.

use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{
    content_digest, handle_digest, handle_payload, handle_signing_bytes, verify_content, Carrier,
    CarrierLocator, CudaExport, HandleError, TensorHandle, MAX_HANDLE_BYTES, MAX_HANDLE_DIMS,
    MAX_INLINE_BYTES, TENSOR_HANDLE_TYPE,
};
use ptr_types::TypeId;

fn element(name: &str, generation: u64) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation,
        digest: [generation as u8; 32],
        fence_epoch: 0,
    }
}

fn handle(locator: CarrierLocator) -> TensorHandle {
    let digest = match &locator {
        CarrierLocator::Bytes { inline } => content_digest(inline),
        CarrierLocator::HostShm { .. }
        | CarrierLocator::CudaIpc { .. }
        | CarrierLocator::DevicePeer { .. }
        | CarrierLocator::PinnedHost { .. }
        | CarrierLocator::NvmeFile { .. }
        | CarrierLocator::Remote { .. } => [6; 32],
    };
    TensorHandle {
        carrier: locator.carrier(),
        locator,
        dtype: BoundaryDtype::F32,
        shape: vec![4, 8],
        byte_len: 128,
        upstream_chain: UpstreamChain {
            stage: element("stage", 1),
            base_model: element("model", 1),
            adapters: vec![element("adapter", 2)],
            experts: vec![],
        },
        placement_epoch: 3,
        signed_absolute_deadline_ms: 10_000,
        handle_nonce: 42,
        parent_operation_id: 9,
        consumer: "consumer".to_owned(),
        payload_type: TypeId::from("ptr.split.activation.v1"),
        content_digest: digest,
        signature: [7; 64],
    }
}

fn shm() -> TensorHandle {
    handle(CarrierLocator::HostShm {
        fd_slot: 1,
        offset: 64,
    })
}

#[test]
fn every_serialisable_carrier_round_trips_through_the_public_codec() {
    let locators = [
        CarrierLocator::Bytes {
            inline: vec![9; 128],
        },
        CarrierLocator::HostShm {
            fd_slot: 1,
            offset: 64,
        },
        CarrierLocator::CudaIpc {
            device: 0,
            export: CudaExport::Legacy {
                mem_handle: [1; 64],
            },
            event_handle: [2; 64],
        },
        CarrierLocator::CudaIpc {
            device: 0,
            export: CudaExport::Vmm {
                fd_slot: 3,
                alloc_size: 4096,
            },
            event_handle: [2; 64],
        },
        CarrierLocator::DevicePeer {
            source_device: 1,
            export: CudaExport::Legacy {
                mem_handle: [8; 64],
            },
            event_handle: [2; 64],
        },
        CarrierLocator::DevicePeer {
            source_device: 1,
            export: CudaExport::Vmm {
                fd_slot: 4,
                alloc_size: 128,
            },
            event_handle: [2; 64],
        },
        CarrierLocator::PinnedHost {
            region: 2,
            offset: 0,
        },
        CarrierLocator::NvmeFile {
            chunk_digest: [3; 32],
        },
        CarrierLocator::Remote {
            frame_id: 5,
            frame_digest: [4; 32],
            chunk_total: 2,
        },
    ];
    for locator in locators {
        let original = handle(locator);
        let bytes = original.encode().expect("a well-formed handle encodes");
        assert_eq!(TensorHandle::decode(&bytes), Ok(original.clone()));
        let payload = handle_payload(&original).expect("payload");
        assert_eq!(payload.type_id.0, TENSOR_HANDLE_TYPE);
        assert_eq!(payload.bytes, bytes);
    }
}

#[test]
fn an_in_process_handle_has_no_wire_form() {
    let mut local = shm();
    local.carrier = Carrier::InProcess;
    assert_eq!(local.encode(), Err(HandleError::InProcessNotEncodable));
}

#[test]
fn a_locator_that_is_not_the_carriers_is_refused() {
    let mut mismatched = shm();
    mismatched.carrier = Carrier::CudaIpc;
    assert_eq!(
        mismatched.encode(),
        Err(HandleError::LocatorMismatch {
            carrier: Carrier::CudaIpc,
            locator: Carrier::HostShm,
        })
    );
}

#[test]
fn malformed_bytes_are_refused_with_a_named_error() {
    let bytes = shm().encode().expect("encodes");
    let mut bad_magic = bytes.clone();
    bad_magic[7] = b'9';
    assert_eq!(TensorHandle::decode(&bad_magic), Err(HandleError::BadMagic));
    assert_eq!(
        TensorHandle::decode(&bytes[..bytes.len() - 5]),
        Err(HandleError::Truncated)
    );
    let mut trailing = bytes;
    trailing.extend_from_slice(&[0, 0]);
    assert_eq!(
        TensorHandle::decode(&trailing),
        Err(HandleError::TrailingBytes)
    );
}

#[test]
fn shape_byte_len_and_bound_violations_are_typed() {
    let mut mismatched = shm();
    mismatched.byte_len = 127;
    assert_eq!(
        mismatched.encode(),
        Err(HandleError::ShapeMismatch {
            expected: 128,
            found: 127,
        })
    );
    let mut deep = shm();
    deep.shape = vec![1; MAX_HANDLE_DIMS + 1];
    deep.byte_len = 4;
    assert_eq!(deep.encode(), Err(HandleError::OverBound("dims")));
    let mut huge = shm();
    huge.shape = vec![MAX_HANDLE_BYTES];
    huge.byte_len = MAX_HANDLE_BYTES * 4;
    assert_eq!(huge.encode(), Err(HandleError::OverBound("byte_len")));
    let inline_len = MAX_INLINE_BYTES as usize + 4;
    let mut inline = handle(CarrierLocator::Bytes {
        inline: vec![0; inline_len],
    });
    inline.shape = vec![inline_len as u64 / 4];
    inline.byte_len = inline_len as u64;
    assert_eq!(inline.encode(), Err(HandleError::OverBound("inline")));
}

#[test]
fn the_consumer_verifies_the_mapped_bytes_against_the_commitment() {
    let bytes = vec![5u8; 128];
    let mut committed = shm();
    committed.content_digest = content_digest(&bytes);
    assert_eq!(verify_content(&committed, &bytes), Ok(()));
    let mut tampered = bytes.clone();
    tampered[0] = 6;
    assert_eq!(
        verify_content(&committed, &tampered),
        Err(HandleError::ContentMismatch)
    );
    assert_eq!(
        verify_content(&committed, &bytes[..100]),
        Err(HandleError::ContentLength {
            expected: 128,
            found: 100
        })
    );
    let mut other = committed.clone();
    other.content_digest = [0; 32];
    assert_ne!(
        handle_signing_bytes(&other),
        handle_signing_bytes(&committed)
    );
}

#[test]
fn the_signature_covers_every_field_and_the_digest_covers_the_signature() {
    let base = shm();
    let signed = handle_signing_bytes(&base);
    let mut other_nonce = base.clone();
    other_nonce.handle_nonce += 1;
    assert_ne!(handle_signing_bytes(&other_nonce), signed);
    let mut other_epoch = base.clone();
    other_epoch.placement_epoch += 1;
    assert_ne!(handle_signing_bytes(&other_epoch), signed);
    let mut other_deadline = base.clone();
    other_deadline.signed_absolute_deadline_ms += 1;
    assert_ne!(handle_signing_bytes(&other_deadline), signed);
    let mut other_chain = base.clone();
    other_chain.upstream_chain.base_model.generation += 1;
    assert_ne!(handle_signing_bytes(&other_chain), signed);

    let mut resigned = base.clone();
    resigned.signature = [8; 64];
    assert_eq!(handle_signing_bytes(&resigned), signed);
    assert_ne!(handle_digest(&resigned), handle_digest(&base));
}

#[test]
fn a_remote_locator_needs_a_chunk_count_within_the_frame_bound() {
    let remote = |chunk_total: u32| {
        handle(CarrierLocator::Remote {
            frame_id: 5,
            frame_digest: [4; 32],
            chunk_total,
        })
    };
    assert!(remote(1).encode().is_ok());
    assert_eq!(remote(0).encode(), Err(HandleError::ZeroChunkTotal));
    assert_eq!(
        remote(ptr_protocol::activation::MAX_CHUNKS_PER_FRAME + 1).encode(),
        Err(HandleError::OverBound("chunk_total"))
    );
}

#[test]
fn the_payload_type_is_signed_bounded_and_round_trips() {
    let base = shm();
    let decoded = TensorHandle::decode(&base.encode().expect("encodes")).expect("decodes");
    assert_eq!(
        decoded.payload_type,
        TypeId::from("ptr.split.activation.v1")
    );
    let mut other = base.clone();
    other.payload_type = TypeId::from("ptr.other.slot.v1");
    assert_ne!(handle_signing_bytes(&other), handle_signing_bytes(&base));
    let mut long = base.clone();
    long.payload_type = TypeId("t".repeat(257));
    assert_eq!(long.encode(), Err(HandleError::OverBound("identity")));
    let mut empty = base;
    empty.payload_type = TypeId::from("");
    assert_eq!(empty.encode(), Err(HandleError::BadIdentity));
}

#[test]
fn a_vmm_export_must_hold_the_tensor_and_the_peer_locator_has_no_raw_pointer() {
    let peer = |alloc_size: u64| {
        handle(CarrierLocator::DevicePeer {
            source_device: 1,
            export: CudaExport::Vmm {
                fd_slot: 4,
                alloc_size,
            },
            event_handle: [2; 64],
        })
    };
    assert!(peer(128).encode().is_ok());
    assert_eq!(peer(0).encode(), Err(HandleError::ZeroAllocSize));
    assert_eq!(
        peer(127).encode(),
        Err(HandleError::AllocTooSmall {
            byte_len: 128,
            alloc_size: 127
        })
    );
    // A different export kind is a different signed message.
    let legacy = handle(CarrierLocator::DevicePeer {
        source_device: 1,
        export: CudaExport::Legacy {
            mem_handle: [0; 64],
        },
        event_handle: [2; 64],
    });
    assert_ne!(
        handle_signing_bytes(&legacy),
        handle_signing_bytes(&peer(128))
    );
}
